import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts.run_coviso_relax import fixed_decision, verdict
from src import train
from src.model import DINOHead
from src.regularization import backbone_grad_norm, resolve_coviso_weights


def scores(best, last):
    return {42: {"best_sts": best, "last_sts": last}, 43: {"best_sts": best + .001, "last_sts": last + .001}}


def test_decay_requires_individual_term_evidence_and_never_test_scores():
    metrics = {"control": scores(.74, .73), "both1": scores(.75, .74),
               "cov1": scores(.745, .735), "mean1": scores(.739, .729)}
    decision = fixed_decision(metrics)
    assert decision["constant_candidate"] == "both1"
    assert decision["supported_decay_terms"] == ["cov"]
    metrics["cov1"] = scores(.739, .729)
    assert fixed_decision(metrics)["supported_decay_terms"] == []


def test_confirmation_distinguishes_inconclusive_and_rejection():
    control = scores(.74, .73)
    assert verdict(control, scores(.741, .731), (42,43))["verdict"] == "판정불가"
    assert verdict(control, scores(.739, .729), (42,43))["verdict"] == "기각"
    assert verdict(control, scores(.745, .72), (42,43))["verdict"] == "판정불가"


def test_fixed_configs_change_only_intended_loss_weights():
    base = train.load_config("configs/coviso_relax/control.yaml")
    assert base["loss"]["cov_iso_lambda"] == 5.874093568592943
    expected = {"both1": (1,1), "cov1": (5.874093568592943,1), "mean1": (1,5.874093568592943)}
    for arm, weights in expected.items():
        cfg = train.load_config(f"configs/coviso_relax/{arm}.yaml")
        assert resolve_coviso_weights(cfg["loss"], 0, 4000) == weights
        for key in base.keys() - {"loss", "run_name"}:
            assert cfg[key] == base[key]
    assert base["train"]["max_steps"] == 4000 and base["train"]["batch_size"] == 32
    assert base["train"]["tf32"] and base["train"]["batch_views"]
    assert base["augment"]["token_latent_views"] == 0


def test_gradient_diagnostics_do_not_mutate_parameter_gradients_or_rng():
    parameter = nn.Parameter(torch.randn(3, 4))
    parameter.grad = torch.ones_like(parameter)
    original = parameter.grad.clone()
    state = torch.get_rng_state()
    assert backbone_grad_norm(parameter.square().sum(), [parameter]) > 0
    assert torch.equal(original, parameter.grad)
    assert torch.equal(state, torch.get_rng_state())


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(16, 8)
        self.linear = nn.Linear(8, 8)


class TinyStudent(nn.Module):
    def __init__(self, *_):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = DINOHead(8, 4, 12)

    def get_input_embeddings(self):
        return self.backbone.embedding

    def forward(self, inputs_embeds, attention_mask, embed_push=None):
        hidden = self.backbone.linear(inputs_embeds)
        pooled = (hidden * attention_mask[...,None]).mean(1)
        return F.normalize(pooled, dim=-1), self.head(pooled), hidden, pooled


class Tokenizer:
    def __call__(self, sentences, **_):
        n = len(sentences)
        return {"input_ids": torch.tensor([[1, int(s) % 10 + 3, 2] for s in sentences]),
                "attention_mask": torch.ones(n,3,dtype=torch.long),
                "special_tokens_mask": torch.tensor([[1,0,1]] * n)}


class Writer:
    def add_scalar(self, *_):
        pass


def test_single_training_loop_split_equal_legacy_and_best_last_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "DinoTextModel", TinyStudent)
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", lambda _: Tokenizer())
    monkeypatch.setattr(train, "load_sentences", lambda _: [str(i) for i in range(12)])
    monkeypatch.setattr(train, "effective_rank_metrics", lambda *_: (7.0,.2))
    monkeypatch.setattr(train.wandb, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train.wandb, "finish", lambda: None)
    torch.save({"mean":torch.zeros(8), "std":torch.ones(8)}, tmp_path / "stats.pt")
    outputs, counts = [], []
    original_update = train.CovIsoPenalty._update_ema
    def update(self, *args):
        counts.append(1)
        return original_update(self, *args)
    monkeypatch.setattr(train.CovIsoPenalty, "_update_ema", update)
    for split in (False, True):
        torch.manual_seed(18)
        random.seed(18)
        cfg = train.load_config("configs/coviso_relax/control.yaml")
        cfg = copy.deepcopy(cfg)
        cfg["run_name"] = f"tiny_{split}"
        cfg["model"]["head"].update(bottleneck_dim=4, logit_dim=12)
        cfg["train"].update(batch_size=4, max_steps=4, tf32=False, record_provenance=False,
                             record_components=split, batch_views=True, component_grad_steps=[0,1,2,3] if split else [])
        cfg["data"]["embed_stats_path"] = "stats.pt"
        cfg["loss"].update(cov_iso_start_step=1)
        if split:
            cfg["loss"].update(cov_iso_mean_lambda=cfg["loss"]["cov_iso_lambda"],
                               cov_iso_cov_lambda=cfg["loss"]["cov_iso_lambda"])
        cfg["eval"].update(every_steps=2, log_every_steps=1, spaces=["embedding"])
        sequence = iter([.7,.73,.72])
        monkeypatch.setattr(train, "sts_b_dev_metrics_by_space", lambda *_: {"embedding": (next(sequence),.2,-2.0)})
        train._run(cfg, "cpu", 4, train.setup_logger(cfg["run_name"]), Writer())
        checkpoint = tmp_path / "checkpoints" / cfg["run_name"]
        last = torch.load(checkpoint / "last.pt", weights_only=True)
        best = torch.load(checkpoint / "best.pt", weights_only=True)
        assert best["loop_step"] == 2 and best["sts_b_dev"] == .73
        assert last["loop_step"] == 3 and last["sts_b_dev"] == .72
        outputs.append(last["state_dict"])
        if split:
            log = tmp_path / "results/logs" / cfg["run_name"]
            rows = [json.loads(line) for line in (log / "components.jsonl").read_text().splitlines()]
            assert rows[0]["weighted_mean"] == rows[0]["weighted_cov"] == 0
            assert rows[1]["grad_mean_backbone"] > 0 and rows[1]["grad_cov_backbone"] > 0
            assert json.loads((log / "completion.json").read_text())["completed_steps"] == 4
    assert len(counts) == 2 * 4 * 3
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
