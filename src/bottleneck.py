"""Paired bottleneck experiment: narrow shared search and non-mutating calibration."""

import copy
import hashlib
import math
import random

import torch

from src.loss import CovIsoPenalty, backbone_grad_norm, bottleneck_cosine_loss
from src.provenance import model_fingerprint
from src.train import BatchIndexSampler

ARMS = ("baseline", "bottleneck")
SEARCH_SEED = 42
CONFIRM_SEEDS = (101, 102, 103)
SAMPLER_SEED = 20261104


def common_space(base):
    return {
        "train.lr": (base["train"]["lr"] * 0.5, base["train"]["lr"] * 2.0, True),
        "train.head_lr": (base["train"]["head_lr"] * 0.5, base["train"]["head_lr"] * 2.0, True),
        "loss.teacher_temp": (0.10, 0.16, False),
    }


def anchor_parameters(base, arm, calibration):
    point = {
        key: float(base[section][name])
        for key in common_space(base)
        for section, name in (key.split("."),)
    }
    if arm == "bottleneck":
        point["loss.bneck_cosine_lambda"] = calibration["lambda_anchor"]
        if "bneck_iso_lambda" in base["loss"]:
            point["loss.bneck_iso_lambda"] = calibration["iso_lambda_anchor"]
    return point


def trial_config(trial, base, arm, calibration):
    cfg = copy.deepcopy(base)
    for key, (low, high, log) in common_space(base).items():
        section, name = key.split(".")
        cfg[section][name] = trial.suggest_float(key, low, high, log=log)
    cfg["loss"]["bneck_cosine_lambda"] = (
        trial.suggest_float("loss.bneck_cosine_lambda", *calibration["lambda_range"], log=True)
        if arm == "bottleneck"
        else 0.0
    )
    if "bneck_iso_lambda" in base["loss"]:
        cfg["loss"]["bneck_iso_lambda"] = (
            trial.suggest_float("loss.bneck_iso_lambda", *calibration["iso_lambda_range"], log=True)
            if arm == "bottleneck" else 0.0
        )
    return cfg


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def calibrate(trainer):
    """At initialization, read CE/cosine gradients on one fixed training minibatch.

    No criterion forward, persistent center/covariance updates, backward accumulation,
    optimizer/scheduler steps, or teacher updates. Restore all RNG states.
    """
    python_rng = random.getstate()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if str(trainer.device).startswith("cuda") else None
    student_before = model_fingerprint(trainer.student)
    teacher_before = model_fingerprint(trainer.teacher.model)
    criterion_before = model_fingerprint(trainer.objective)
    optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
    scheduler_before = copy.deepcopy(trainer.scheduler.state_dict())
    gradients_before = [
        None if p.grad is None else p.grad.clone() for p in trainer.student.parameters()
    ]
    try:
        sampler = BatchIndexSampler(len(trainer.sentences), trainer.cfg["train"]["batch_size"])
        indices = sampler.next()
        texts = [trainer.sentences[i] for i in indices]
        views = trainer.forward_views(texts, 0)
        ce, _ = trainer.objective.dino(
            views.teacher.logits,
            views.student_logits,
            trainer.cfg["loss"]["warmup_teacher_temp"],
            trainer.cfg["loss"]["student_temp"],
            update_center=False,
        )
        cosine = bottleneck_cosine_loss(views.teacher.bottleneck, views.student_bottlenecks)
        parameters = [p for p in trainer.student.backbone.parameters() if p.requires_grad]
        g_dino = backbone_grad_norm(ce, parameters)
        g_bneck = backbone_grad_norm(cosine, parameters)
        if not all(math.isfinite(x) and x > 0 for x in (g_dino, g_bneck)):
            raise ValueError("calibration requires finite, positive CE and bottleneck gradients")
        equivalent = g_dino / g_bneck
        result = {
            "calibration_step": 0,
            "seed": trainer.cfg["seed"],
            "first_indices": indices,
            "batch_text_sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
            "initial_student_sha256": student_before,
            "L_DINO": ce.item(),
            "L_bneck": cosine.item(),
            "grad_dino_backbone": g_dino,
            "grad_bneck_unweighted_backbone": g_bneck,
            "lambda_equivalent": equivalent,
            "lambda_range": [equivalent * 0.01, equivalent],
            "lambda_anchor": equivalent * 0.1,
            "expected_initial_gradient_ratios": [0.01, 1.0],
            "bottleneck_dim": views.teacher.bottleneck.shape[-1],
        }
        if "bneck_iso_lambda" in trainer.cfg["loss"]:
            # Mimic the three sequential EMA blends on disposable statistics only.
            # The live objective's moments must remain untouched by calibration.
            temporary = CovIsoPenalty(
                result["bottleneck_dim"], trainer.cfg["loss"]["cov_ema_momentum"]
            ).to(trainer.device)
            terms = [temporary(b) for b in views.student_bottlenecks]
            iso = torch.stack([t.total for t in terms]).mean()
            g_iso = backbone_grad_norm(iso, parameters)
            if not math.isfinite(g_iso) or g_iso <= 0:
                raise ValueError("calibration requires finite, positive bottleneck iso gradient")
            iso_equivalent = g_dino / g_iso
            mlp = [p for p in trainer.student.head.mlp.parameters() if p.requires_grad]
            result.update(
                L_bneck_iso=iso.item(),
                grad_bneck_iso_unweighted_backbone=g_iso,
                iso_lambda_equivalent=iso_equivalent,
                iso_lambda_range=[0.01 * iso_equivalent, iso_equivalent],
                iso_lambda_anchor=0.1 * iso_equivalent,
                iso_calibration="fresh disposable moments, three sequential views; activation ignored only for measurement",
                grad_dino_mlp=backbone_grad_norm(ce, mlp),
                grad_bneck_unweighted_mlp=backbone_grad_norm(cosine, mlp),
                grad_bneck_iso_unweighted_mlp=backbone_grad_norm(iso, mlp),
            )
    finally:
        random.setstate(python_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    proof = {
        "student_unchanged": student_before == model_fingerprint(trainer.student),
        "teacher_unchanged": teacher_before == model_fingerprint(trainer.teacher.model),
        "criterion_buffers_unchanged": criterion_before == model_fingerprint(trainer.objective),
        "optimizer_unchanged": tree_equal(optimizer_before, trainer.optimizer.state_dict()),
        "scheduler_unchanged": tree_equal(scheduler_before, trainer.scheduler.state_dict()),
        "parameter_grads_unchanged": all(
            tree_equal(a, p.grad) for a, p in zip(gradients_before, trainer.student.parameters())
        ),
        "python_rng_restored": python_rng == random.getstate(),
        "torch_rng_restored": torch.equal(cpu_rng, torch.get_rng_state()),
        "cuda_rng_restored": cuda_rng is None
        or tree_equal(cuda_rng, torch.cuda.get_rng_state_all()),
    }
    if not all(proof.values()):
        raise RuntimeError(f"calibration mutated state: {proof}")
    return {**result, "state_proof": proof}
