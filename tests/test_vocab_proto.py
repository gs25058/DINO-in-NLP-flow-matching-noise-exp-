"""Word-prototype heads: legacy path untouched, untied decoder, freeze, RNG, space, diagnostics."""

import math

import pytest
import torch
from torch.nn import functional as F
from transformers import BertForMaskedLM

from src import vocab_proto as vp
from src.evaluate import apply_postprocess
from src.model import DINOHead, DinoTextModel, EMATeacher
from src.train import build_param_groups, head_options, load_config

NAME = "bert-base-uncased"


@pytest.fixture(scope="module")
def vocab_model():
    torch.manual_seed(0)
    return DinoTextModel(NAME, 768, 30522, head_type="vocab")


def _state(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_no_head_keys_is_the_legacy_random_head_bit_for_bit():
    assert head_options({"bottleneck_dim": 256, "logit_dim": 8192}, None, 0) == {}
    torch.manual_seed(3)
    legacy = DinoTextModel(NAME, 256, 8192)
    after_legacy = torch.random.get_rng_state()
    torch.manual_seed(3)
    explicit = DinoTextModel(NAME, 256, 8192, **head_options({"head_type": "random"}, None, 0))
    assert isinstance(legacy.head, DINOHead)
    assert torch.equal(after_legacy, torch.random.get_rng_state())
    a, b = _state(legacy), _state(explicit)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_vocab_head_reproduces_pretrained_mlm_logits_on_pooled_vectors(vocab_model):
    mlm = BertForMaskedLM.from_pretrained(NAME).eval()
    pooled = torch.randn(4, 768)
    with torch.no_grad():
        expected = mlm.cls.predictions(pooled)
        out = vocab_model.head(pooled)
    assert torch.allclose(out.logits, expected, atol=1e-4, rtol=1e-5)
    assert torch.allclose(out.bottleneck.norm(dim=-1), torch.ones(4))
    assert vocab_model.head.expand is vocab_model.head.decoder
    assert not any(k.startswith("head.expand") for k in vocab_model.state_dict())


def test_decoder_is_untied_and_head_updates_never_touch_input_embeddings():
    torch.manual_seed(0)
    model = DinoTextModel(NAME, 768, 30522, head_type="vocab")
    embeddings = model.get_input_embeddings().weight
    decoder = model.head.decoder.weight
    assert decoder.data_ptr() != embeddings.data_ptr()
    assert torch.equal(decoder, embeddings)  # same values at init, separate storage
    before = embeddings.detach().clone()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=1.0)
    model.head(torch.randn(2, 768)).logits.square().mean().backward()
    assert embeddings.grad is None
    optimizer.step()
    assert torch.equal(embeddings, before) and not torch.equal(decoder, before)
    decoder_before = decoder.detach().clone()
    with torch.no_grad():
        embeddings.add_(1.0)
    assert torch.equal(decoder, decoder_before)


def test_frozen_head_has_no_optimizer_group_and_its_teacher_copy_stays_bitwise():
    torch.manual_seed(0)
    model = DinoTextModel(NAME, 768, 30522, head_type="vocab", head_freeze=True)
    assert not any(p.requires_grad for p in model.head.parameters())
    groups = build_param_groups(model, 1e-4, 1e-3, 0.1)
    grouped = {id(p) for g in groups for p in g["params"]}
    assert not any(id(p) in grouped for p in model.head.parameters())
    assert all(id(p) in grouped for p in model.backbone.parameters())
    teacher = EMATeacher(model, 0.9)
    initial = {k: v.clone() for k, v in teacher.model.head.state_dict().items()}
    with torch.no_grad():
        next(model.backbone.parameters()).add_(1.0)
    teacher.update(model)
    assert all(torch.equal(initial[k], v) for k, v in teacher.model.head.state_dict().items())
    assert not torch.equal(next(teacher.model.backbone.parameters()), next(model.backbone.parameters()))


def test_head_construction_does_not_shift_the_global_rng(tmp_path):
    torch.manual_seed(5)
    DinoTextModel(NAME, 768, 30522, head_type="vocab")
    a = torch.rand(3)
    centers = torch.randn(16, 768)
    path = tmp_path / "km.pt"
    torch.save({"centers": centers}, path)
    torch.manual_seed(5)
    model = DinoTextModel(NAME, 768, 16, head_type="vocab_kmeans", kmeans_path=str(path))
    assert torch.equal(a, torch.rand(3))
    assert torch.allclose(model.head.decoder.weight, F.normalize(centers, dim=-1))
    assert torch.equal(model.head.decoder.bias, torch.zeros(16))
    mlm = BertForMaskedLM.from_pretrained(NAME)
    assert not torch.allclose(model.head.transform[0].weight, mlm.cls.predictions.transform.dense.weight)
    assert model.head.transform[0].weight.std().item() == pytest.approx(0.02, rel=0.05)


def test_invalid_head_options_rejected(tmp_path):
    with pytest.raises(ValueError):
        DinoTextModel(NAME, 256, 8192, head_type="other")
    with pytest.raises(ValueError):
        DinoTextModel(NAME, 256, 8192, head_freeze=True)
    with pytest.raises(ValueError):
        DinoTextModel(NAME, 256, 30522, head_type="vocab")


def test_space_bounds_teacher_never_softer_and_anchors_are_t13():
    base = load_config("configs/vocab_proto/base.yaml")
    for condition in vp.CONDITIONS:
        points = vp.anchors(base, condition)
        for point in points:
            cfg = vp.configure(base, condition, point, kmeans_path="/k.pt")
            loss = cfg["loss"]
            assert loss["warmup_teacher_temp"] <= loss["teacher_temp"] <= loss["student_temp"] + 1e-12
            for key in vp.COMMON:
                if key != "train.grad_clip" and not key.startswith("_"):
                    section, name = key.split(".")
                    assert cfg[section][name] == pytest.approx(base[section][name])
            assert cfg["augment"]["t_max"] == pytest.approx(base["augment"]["t_max"])
        assert points[0]["train.grad_clip"] == pytest.approx(base["train"]["grad_clip"])
        assert points[1]["train.grad_clip"] == 3.0
        with pytest.raises(ValueError):
            vp.configure(base, condition, {**points[0], "loss.student_temp": 9.0}, kmeans_path="/k.pt")
    c3 = vp.configure(base, "C3", vp.anchors(base, "C3")[0], kmeans_path="/k.pt")
    assert c3["loss"]["teacher_temp"] == pytest.approx(base["loss"]["teacher_temp"])
    assert c3["train"]["head_lr"] == pytest.approx(base["train"]["head_lr"])
    c1 = vp.configure(base, "C1", {**vp.anchors(base, "C1")[0], "_head_lr_mult": 2.0})
    assert c1["train"]["head_lr"] == pytest.approx(2 * c1["train"]["lr"])


def test_copy_rate_pair_kl_and_postprocess_are_exact_on_toys():
    ids = torch.tensor([[101, 5, 6, 102], [101, 7, 102, 0]])
    special = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1]]).bool()
    sets = vp._sentence_token_sets(ids, special, None)
    assert sets == [{5, 6}, {7}]
    scores = torch.full((2, 20), -1.0)
    scores[0, [5, 6, 1]] = 5.0
    assert vp.copy_rate(scores, sets, k=3) == [2 / 3, 0.0]
    labels = torch.arange(200) // 2
    assert vp._sentence_token_sets(ids, special, labels) == [{2, 3}, {3}]
    logits = torch.randn(4, 9)
    log_p = F.log_softmax(logits / 0.7, -1)
    p = log_p.exp()
    pairs = [(p[i] * (log_p[i] - log_p[j])).sum() for i in range(4) for j in range(4) if i != j]
    assert vp.pair_kl(logits, 0.7).item() == pytest.approx(torch.stack(pairs).mean().item(), rel=1e-5)
    z = F.normalize(torch.randn(50, 6), dim=-1)
    mean = z.double().mean(0)
    _, _, vh = torch.linalg.svd(z.double() - mean, full_matrices=False)
    fit = {"mean": mean, "components": vh[:2]}
    out = apply_postprocess(z, fit, "center_pc2").double()
    assert torch.allclose(out @ vh[:2].t(), torch.zeros(50, 2, dtype=torch.double), atol=1e-6)
    assert torch.allclose(out.norm(dim=-1), torch.ones(50, dtype=torch.double))


def test_condition_k_matches_entropy_normalization():
    assert vp.CONDITIONS["C1"]["logit_dim"] == 30522
    assert math.log(vp.CONDITIONS["C3"]["logit_dim"]) == pytest.approx(9.0109, abs=1e-4)
