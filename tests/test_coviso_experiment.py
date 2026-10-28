"""Compare the real Trainer against measured pre-refactor tiny-BERT trajectories."""

import hashlib
import json
import logging
import random
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel

from src import model, train
from src.joint_search import config_from_parameters
from src.loss import backbone_grad_norm

REFERENCE = json.loads((Path(__file__).parent / "fixtures/r18_cpu_reference.json").read_text())


class Tokenizer:
    def __call__(self, sentences, **kwargs):
        ids = torch.tensor(
            [[1, int(s) % 20 + 4, (int(s) * 3) % 20 + 4, 2, 0, 0] for s in sentences]
        )
        return {
            "input_ids": ids,
            "attention_mask": (ids != 0).long(),
            "special_tokens_mask": ((ids == 1) | (ids == 2) | (ids == 0)).long(),
        }


class Writer:
    def add_scalar(self, *args):
        pass


def digest(tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize(
    "case",
    REFERENCE["cases"],
    ids=[f"{r['name']}_s{r['seed']}" for r in REFERENCE["cases"]],
)
def test_full_training_preserves_outputs_gradients_ema_checkpoints_and_rng(
    case, tmp_path, monkeypatch
):
    """Four 60-step CPU validation runs; no pretrained downloads or STS test scores."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", lambda _: Tokenizer())
    monkeypatch.setattr(
        model.AutoConfig,
        "from_pretrained",
        lambda _: BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=32,
            max_position_embeddings=32,
        ),
    )
    monkeypatch.setattr(model.AutoModel, "from_pretrained", lambda _, config: BertModel(config))
    # The diagnostic data values are deliberately fixed to test selection/ties.
    from src.evaluate import STSDevMetrics

    monkeypatch.setattr(
        train,
        "sts_b_dev_metrics_by_space",
        lambda *args: {s: STSDevMetrics(0.7, 0.2, -2.0) for s in train.EVAL_SPACES},
    )
    monkeypatch.setattr(train, "effective_rank_metrics", lambda *args: (7.0, 0.2))
    sentences = tmp_path / "sentences.jsonl"
    sentences.write_text("".join(json.dumps({"text": str(i)}) + "\n" for i in range(24)))
    stats = tmp_path / "stats.pt"
    torch.save(
        {"mean": torch.arange(16) * 0.001, "std": torch.arange(16) * 0.001 + 0.04},
        stats,
    )
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps({"git": {"commit": REFERENCE["source_commit"]}}))
    cfg = config_from_parameters(
        train.load_config("configs/coviso_relax/control.yaml"), case["params"]
    )
    cfg.update(
        seed=case["seed"],
        run_name=f"{case['name']}_{case['seed']}",
        experiment={"source_manifest": str(source)},
    )
    cfg["model"]["head"].update(bottleneck_dim=8, logit_dim=13)
    cfg["train"].update(
        batch_size=4, stop_after_steps=60, tf32=False, component_grad_steps=[0, 56, 59]
    )
    schedules = [
        [
            train.cosine_ramp(
                s,
                cfg["loss"]["warmup_teacher_temp"],
                cfg["loss"]["teacher_temp"],
                max(1, int(cfg["loss"]["teacher_temp_warmup_frac"] * 4000)),
            ),
            train.cosine_ramp(
                s,
                cfg["train"]["momentum_start"],
                cfg["train"]["momentum_end"],
                max(1, int(cfg["train"]["momentum_ramp_frac"] * 4000)),
            ),
        ]
        for s in range(4000)
    ]
    assert digest([torch.tensor(schedules, dtype=torch.float64)]) == case["schedule_sha256"]
    cfg["data"].update(
        sentences_path=str(sentences),
        rank_eval_path=str(sentences),
        embed_stats_path=str(stats),
    )
    forwards, gradients, updates = [], [], []
    original_model = train.DinoTextModel

    def make_model(*args):
        m = original_model(*args)
        m.register_forward_hook(
            lambda _m, _i, out: forwards.append(digest([out.embedding, out.logits]))
        )
        return m

    monkeypatch.setattr(train, "DinoTextModel", make_model)
    original_step = torch.optim.AdamW.step

    def step(optimizer, *args, **kwargs):
        gradients.append(
            digest(
                [p.grad for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
            )
        )
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    from src.loss import CovIsoPenalty

    original_update = CovIsoPenalty._update_ema

    def update(self, *args):
        updates.append(1)
        original_update(self, *args)

    monkeypatch.setattr(CovIsoPenalty, "_update_ema", update)
    try:
        torch.manual_seed(case["seed"])
        random.seed(case["seed"])
        trainer = train.Trainer(cfg, "cpu", logging.getLogger("reference"), Writer(), root=tmp_path)
        trainer.run()
        path = tmp_path / "checkpoints" / cfg["run_name"]
        last = torch.load(path / "last.pt", weights_only=True)
        best = torch.load(path / "best.pt", weights_only=True)
        assert forwards == case["forwards"]
        assert gradients == case["gradients"]
        assert digest(last["state_dict"].values()) == case["student_state"]
        assert digest(last["teacher_state_dict"].values()) == case["teacher_state"]
        assert digest(last["cov_iso_state_dict"].values()) == case["covariance_state"]
        assert len(updates) == 60 * 3
        assert digest([torch.get_rng_state()]) == case["rng"]
        assert hashlib.sha256(repr(random.getstate()).encode()).hexdigest() == case["python_rng"]
        assert best["best_step"] == 0 and last["loop_step"] == 59
        log = tmp_path / "results/logs" / cfg["run_name"]
        actual = [json.loads(line) for line in (log / "components.jsonl").read_text().splitlines()]
        for row, expected in zip(actual, case["logs"], strict=True):
            assert {key: row[key] for key in expected} == expected
        assert actual[0]["weighted_mean"] == actual[0]["weighted_cov"] == 0
        assert actual[-1]["grad_mean_backbone"] > 0 and actual[-1]["grad_cov_backbone"] > 0
        assert json.loads((log / "completion.json").read_text())["status"] == "smoke_complete"
    finally:
        torch.set_num_threads(previous_threads)


def test_gradient_diagnostics_leave_parameter_gradients_and_rng_untouched():
    parameter = torch.nn.Parameter(torch.randn(3, 4))
    parameter.grad = torch.ones_like(parameter)
    gradient = parameter.grad.clone()
    state = torch.get_rng_state()
    assert backbone_grad_norm(parameter.square().sum(), [parameter]) > 0
    assert torch.equal(parameter.grad, gradient) and torch.equal(torch.get_rng_state(), state)
