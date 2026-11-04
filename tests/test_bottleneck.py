import copy
import logging
import random
import signal
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import torch
from torch import nn

from src.augment import FlowNoiseAug
from src.bottleneck import (
    ARMS,
    CONFIRM_SEEDS,
    anchor_parameters,
    calibrate,
    common_space,
    trial_config,
)
from src.loss import DINOObjective, bottleneck_cosine_loss
from src.model import DINOHead, DinoTextModel, EMATeacher
from src.provenance import append_record, prepare, read_json, write_json
from src.train import Trainer, load_config


class Tokenizer:
    def __call__(self, sentences, **kwargs):
        ids = torch.tensor([[1, int(s) % 12 + 3, 2] for s in sentences])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "special_tokens_mask": torch.tensor([[1, 0, 1]] * len(sentences)),
        }


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 8)
        self.linear = nn.Linear(8, 8)
        self.config = SimpleNamespace(hidden_size=8)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask):
        return SimpleNamespace(last_hidden_state=self.linear(inputs_embeds))


def model():
    result = DinoTextModel.__new__(DinoTextModel)
    nn.Module.__init__(result)
    result.backbone = Backbone()
    result.head = DINOHead(8, 256, 13)
    return result


def test_cosine_loss_averages_sentences_and_views_and_detaches_teacher():
    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    students = [
        teacher.detach().clone().requires_grad_(),
        (-teacher.detach()).requires_grad_(),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True),
    ]
    loss = bottleneck_cosine_loss(teacher, students)
    assert loss.item() == 1.0
    loss.backward()
    assert teacher.grad is None and all(s.grad is not None for s in students)
    for student in students:
        torch.testing.assert_close(student.grad, -teacher.detach() / 6, rtol=0, atol=0)


def test_cached_256d_bottleneck_is_normalized_and_reused_once():
    student = model()
    calls = []
    hook = student.head.mlp.register_forward_hook(lambda *_: calls.append(1))
    output = student(
        student.get_input_embeddings()(torch.tensor([[1, 4, 2], [1, 5, 2]])), torch.ones(2, 3)
    )
    assert len(calls) == 1 and output.bottleneck.shape == (2, 256)
    torch.testing.assert_close(output.bottleneck.norm(dim=-1), torch.ones(2))
    assert torch.equal(output.logits, student.head.expand(output.bottleneck))
    hook.remove()


def test_aux_gradient_reaches_backbone_and_mlp_but_not_teacher_or_expand():
    torch.manual_seed(42)
    student = model()
    teacher = EMATeacher(student, 0.997)
    mask = torch.ones(4, 3)
    inputs = student.get_input_embeddings()(
        torch.tensor([[1, 4, 2], [1, 5, 2], [1, 6, 2], [1, 7, 2]])
    )
    target = teacher(inputs, mask)
    outputs = [
        student(inputs + torch.randn_like(inputs) * scale, mask) for scale in (0.2, 0.5, 0.8)
    ]
    bottleneck_cosine_loss(target.bottleneck, [out.bottleneck for out in outputs]).backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in student.backbone.parameters())
    assert all(p.grad is not None for p in student.head.mlp.parameters())
    assert all(p.grad is None for p in student.head.expand.parameters())
    assert all(p.grad is None for p in teacher.model.parameters())


def test_combined_objective_adds_only_cosine_and_does_not_duplicate_ema_updates():
    cfg = load_config("configs/bottleneck_cosine/base.yaml")["loss"]
    cfg["bneck_cosine_lambda"] = 0.7
    augmented = DINOObjective(cfg, 8, 13)
    off = copy.deepcopy(augmented)
    off.bneck_weight = 0.0
    teacher = model()
    student = model()
    mask = torch.ones(3, 3)
    x = torch.randn(3, 3, 8)
    target = teacher(x, mask)
    outputs = [student(x + 0.2 * i, mask) for i in (1, 2, 3)]
    args = (target.logits, [o.embedding for o in outputs], [o.logits for o in outputs], 56, 0.1)
    kwargs = {
        "teacher_bottleneck": target.bottleneck,
        "student_bottlenecks": [o.bottleneck for o in outputs],
    }
    positive = augmented(*args, **kwargs)
    zero = off(*args, **kwargs)
    assert torch.equal(positive.total, zero.total + 0.7 * positive.bneck)
    assert all(torch.equal(a, b) for a, b in zip(augmented.buffers(), off.buffers()))


def test_calibration_preserves_all_states_and_defines_gradient_scaled_log_range():
    torch.manual_seed(42)
    random.seed(42)
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = load_config("configs/bottleneck_cosine/base.yaml")
    trainer.device = "cpu"
    trainer.student = model()
    trainer.teacher = EMATeacher(trainer.student, 0.997)
    trainer.tokenizer = Tokenizer()
    trainer.sentences = [str(i) for i in range(40)]
    trainer.augment = FlowNoiseAug(
        torch.zeros(8),
        torch.ones(8),
        num_student_views=3,
        t_lo=0.39,
        t_start=0.46,
        t_max=0.88,
        warmup_steps=100,
    )
    trainer.objective = DINOObjective(trainer.cfg["loss"], 8, 13)
    trainer.optimizer = torch.optim.AdamW(trainer.student.parameters(), lr=1e-4)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    result = calibrate(trainer)
    assert all(result["state_proof"].values()) and result["bottleneck_dim"] == 256
    assert result["lambda_range"][0] * result["grad_bneck_unweighted_backbone"] / result[
        "grad_dino_backbone"
    ] == pytest.approx(0.01)
    assert result["lambda_range"][1] * result["grad_bneck_unweighted_backbone"] / result[
        "grad_dino_backbone"
    ] == pytest.approx(1.0)
    assert result["lambda_anchor"] == pytest.approx(
        (result["lambda_range"][0] * result["lambda_range"][1]) ** 0.5
    )


def test_shared_search_space_is_identical_and_covariance_noise_are_fixed():
    base = load_config("configs/bottleneck_cosine/base.yaml")
    cal = {"lambda_range": [0.01, 1.0], "lambda_anchor": 0.1}
    expected = {"train.lr", "train.head_lr", "loss.teacher_temp"}
    assert set(common_space(base)) == expected
    for arm in ARMS:
        study = optuna.create_study()
        study.enqueue_trial(anchor_parameters(base, arm, cal))
        trial = study.ask()
        cfg = trial_config(trial, base, arm, cal)
        assert cfg["augment"] == base["augment"]
        assert (
            cfg["loss"]["cov_iso_mean_lambda"]
            == cfg["loss"]["cov_iso_cov_lambda"]
            == 5.874093568592943
        )
        assert cfg["train"]["max_steps"] == 4000
        assert set(trial.params) == expected | (
            {"loss.bneck_cosine_lambda"} if arm == "bottleneck" else set()
        )
        for key in expected:
            section, name = key.split(".")
            assert cfg[section][name] == base[section][name]
    assert 42 not in CONFIRM_SEEDS and len(set(CONFIRM_SEEDS)) == 3


def test_model_selection_uses_768d_even_if_bottleneck_is_better(tmp_path, monkeypatch):
    import src.train as train
    from src.evaluate import STSDevMetrics

    trainer = Trainer.__new__(Trainer)
    trainer.student = model()
    trainer.tokenizer = Tokenizer()
    trainer.device = "cpu"
    trainer.rank_sentences = ["1"]
    trainer.writer = SimpleNamespace(add_scalar=lambda *_: None)
    trainer.logger = logging.getLogger("selection")
    trainer.run_dir = tmp_path
    trainer.best_sts = 0.0
    trainer.best_step = -1
    saved = []
    trainer.save_checkpoint = lambda *args: saved.append(args)
    monkeypatch.setattr(
        train,
        "sts_b_dev_metrics_by_space",
        lambda *_: {
            "embedding": STSDevMetrics(0.6, 0.2, -2.0),
            "bottleneck": STSDevMetrics(0.95, 0.2, -2.0),
            "head": STSDevMetrics(0.99, 0.2, -2.0),
        },
    )
    monkeypatch.setattr(train, "effective_rank_metrics", lambda *_: (100.0, 0.02))
    assert trainer.evaluate(250) == 0.6 and trainer.best_sts == 0.6
    assert saved == [("best", 250, 0.6)]
    assert read_json_lines(tmp_path / "evaluations.jsonl")[0]["sts_b_dev_bottleneck"] == 0.95


def read_json_lines(path):
    from src.provenance import read_records

    return read_records(path)


def test_entire_paired_protocol_freezes_selection_before_test_and_has_pure_aux_off(
    tmp_path, monkeypatch
):
    from scripts import run_bottleneck as pipeline

    campaign = tmp_path / "campaign"
    workspace = tmp_path / "workspace"
    manifest = prepare(
        campaign,
        pipeline.ROOT,
        ["auto", "auto"],
        tmp_path / "report.md",
        experiment_dir="bottleneck_cosine",
    )
    manifest.update(
        workspace=str(workspace),
        trials_per_arm=3,
        common_space=common_space(load_config("configs/bottleneck_cosine/base.yaml")),
    )
    write_json(campaign / "source_manifest.json", manifest)
    calls = []

    def execute(self, gpu, label, command, **kwargs):
        calls.append(label)
        if "--calibration-child" in command:
            write_json(
                campaign / "calibration.json",
                {
                    "first_indices": [0, 1],
                    "initial_student_sha256": "init42",
                    "lambda_anchor": 0.1,
                    "lambda_equivalent": 1.0,
                    "lambda_range": [0.01, 1.0],
                    "grad_dino_backbone": 1.0,
                    "grad_bneck_unweighted_backbone": 1.0,
                    "state_proof": {"unchanged": True},
                },
            )
        elif "-m" in command:
            cfg = load_config(command[command.index("--config") + 1])
            seed = cfg["seed"]
            steps = 60 if "--stop-after-steps" in command else 4000
            log = campaign / "results/logs" / cfg["run_name"]
            log.mkdir(parents=True)
            score = 0.73 + (0.01 if cfg["loss"]["bneck_cosine_lambda"] else 0) + seed * 1e-6
            write_json(
                log / "completion.json",
                {
                    "status": "smoke_complete" if steps == 60 else "complete",
                    "completed_steps": steps,
                    "best_sts": score,
                    "last_sts": score - 0.001,
                    "best_loop_step": 0,
                    "last_loop_step": steps - 1,
                },
            )
            data = {"sentences_path": {"sha256": "same"}}
            write_json(
                log / "provenance.json", {"initial_student_sha256": f"init{seed}", "data": data}
            )
            write_json(
                log / "batch_order.json",
                {"sha256": f"{seed}_{steps}", "first_indices": [0, 1], "seed": seed},
            )
            append_record(
                log / "evaluations.jsonl",
                {"loop_step": steps - 1, "sts_b_dev": score - 0.001, "sts_b_dev_bottleneck": 0.9},
            )
            append_record(log / "components.jsonl", {"loop_step": steps - 1, "bneck_cosine": 0.9})
        else:
            assert (campaign / "selection.json").exists()
            assert "aux_off" in read_json(campaign / "final_runs.json")
            name = command[command.index("--runs") + 1]
            output = Path(command[command.index("--out") + 1])
            scores = {"avg_7task": 0.7}
            # A better 256d diagnostic alone must never produce a positive
            # conclusion for the unchanged 768d primary representation.
            diagnostic = {"avg_7task": 0.99 if label.startswith("eval_bottleneck_") else 0.5}
            write_json(
                output, {"runs": {name: scores}, "bottleneck_diagnostic": {name: diagnostic}}
            )

    monkeypatch.setattr(pipeline.Children, "run", execute)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        pipeline.run(campaign, manifest)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    assert read_json(campaign / "state.json")["stage"] == "complete"
    for arm in ARMS:
        study = optuna.load_study(
            study_name=f"{campaign.name}_{arm}", storage=f"sqlite:///{campaign / f'{arm}.db'}"
        )
        assert len(study.trials) == 3 and all(t.state.name == "COMPLETE" for t in study.trials)
        assert (
            study.trials[0].params["train.lr"]
            == load_config("configs/bottleneck_cosine/base.yaml")["train"]["lr"]
        )
    assert sum(x.startswith("train_confirm_") for x in calls) == 18
    assert sum(x.startswith("train_aux_off_") for x in calls) == 3
    assert sum(x.startswith("eval_") for x in calls) == 18
    conclusions = [
        line
        for line in (campaign / "report.md").read_text().splitlines()
        if line.startswith("**주 평가:")
    ]
    assert len(conclusions) == 2
    assert all("'verdict': '기각'" in line for line in conclusions)
