"""Corrected baseline and legacy checkpoint compatibility, without downloads."""
import itertools
import random

import pytest
import torch
from transformers import BertConfig, BertModel

from scripts import train_simcse_baseline as simcse
from scripts.eval_sts7 import load_eval_model, resolve_pooling


@pytest.fixture
def tiny_bert(monkeypatch):
    cfg = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=48, vocab_size=40)
    monkeypatch.setattr(simcse.AutoConfig, "from_pretrained", lambda _: cfg)
    monkeypatch.setattr(simcse.BertModel, "from_pretrained",
                        lambda _, **kwargs: BertModel(**kwargs))
    return cfg


def test_corrected_pooler_and_mlp_initialization(tiny_bert):
    torch.manual_seed(3)
    model = simcse.SimCSEModel("offline", 0.1)
    assert model.backbone.pooler is None
    assert torch.count_nonzero(model.mlp[0].bias) == 0
    assert model.mlp[0].weight.std().item() == pytest.approx(0.02, abs=0.002)


def test_epoch_batches_keep_all_samples_including_last_partial():
    stream = simcse.sentence_batches(list(range(10)), 4, rng=random.Random(7))
    first = list(itertools.islice(stream, 3))
    second = list(itertools.islice(stream, 3))
    assert [len(b) for b in first] == [4, 4, 2]
    assert sorted(sum(first, [])) == sorted(sum(second, [])) == list(range(10))
    assert first != second


def test_singleton_partial_batch_has_finite_loss_and_gradients():
    z = torch.randn(1, 8, requires_grad=True)
    loss, aux = simcse.info_nce(z, z, 0.05)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert all(torch.isfinite(torch.tensor(v)) for v in aux.values())


def test_legacy_simcse_pooler_keys_are_accepted_but_missing_encoder_is_not(tiny_bert):
    old = simcse.SimCSEModel("offline", 0.1)
    state = old.state_dict()
    state["backbone.pooler.dense.weight"] = torch.eye(32)
    state["backbone.pooler.dense.bias"] = torch.zeros(32)
    checkpoint = {"model_cfg": {"simcse": True, "backbone": "offline", "dropout": 0.1},
                  "state_dict": state}
    model, backbone, kind = load_eval_model({}, checkpoint, "student", "cpu")
    assert backbone == "offline" and kind == "simcse"
    assert torch.equal(model.mlp[0].weight, old.mlp[0].weight)
    assert resolve_pooling("auto", kind) == "cls"
    with pytest.raises(ValueError, match="teacher"):
        load_eval_model({}, checkpoint, "teacher", "cpu")
    del state["backbone.embeddings.word_embeddings.weight"]
    with pytest.raises(RuntimeError, match="missing"):
        load_eval_model({}, checkpoint, "student", "cpu")


def test_pretrained_and_dino_checkpoint_still_use_mean_pooling(monkeypatch):
    from scripts import eval_sts7
    monkeypatch.setattr(eval_sts7, "DinoTextModel", lambda *args: torch.nn.Linear(2, 2))
    cfg = {"model": {"backbone": "offline", "head": {"bottleneck_dim": 2, "logit_dim": 2}}}
    pretrained, backbone, kind = load_eval_model(cfg, None, "student", "cpu")
    assert backbone == "offline" and kind == "pretrained"
    assert resolve_pooling("auto", kind) == "last"
    checkpoint = {"model_cfg": cfg["model"], "state_dict": pretrained.state_dict(),
                  "teacher_state_dict": pretrained.state_dict()}
    loaded, _, kind = load_eval_model(cfg, checkpoint, "teacher", "cpu")
    assert torch.equal(loaded.weight, pretrained.weight)
    assert resolve_pooling("auto", kind) == "last"
    assert resolve_pooling("first_last", "simcse") == "first_last"
