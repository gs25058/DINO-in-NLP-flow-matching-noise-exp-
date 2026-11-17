"""Single-factor interventions change only their factor, only from from_step, and draw no RNG."""

from types import SimpleNamespace

import pytest
import torch

from src.intervention import KINDS, Intervention


def fake_trainer():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    return SimpleNamespace(
        objective=SimpleNamespace(dino=SimpleNamespace(center_momentum=0.9), mean_weight=2.0, cov_weight=3.0),
        augment=SimpleNamespace(t_lo=0.3, t_start=0.32, t_max=0.4),
        optimizer=optimizer, scheduler=scheduler)


@pytest.mark.parametrize("kind", KINDS)
def test_nothing_changes_before_from_step_and_only_the_named_factor_after(kind):
    extra = {"t_lo": 0.47, "t_max": 0.9} if kind == "noise" else {}
    iv = Intervention({"kind": kind, "from_step": 1000, **extra})
    trainer = fake_trainer()
    rng = torch.random.get_rng_state()
    iv.begin(trainer, 999)
    assert iv.teacher_temp(999, 0.2, {"student_temp": 1.1}) == 0.2
    assert iv.teacher_momentum(999, 0.999) == 0.999
    assert trainer.objective.dino.center_momentum == 0.9 and trainer.objective.mean_weight == 2.0
    assert (trainer.augment.t_lo, trainer.augment.t_max) == (0.3, 0.4)
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-3
    iv.begin(trainer, 1000)
    after = {
        "sharpen_off": iv.teacher_temp(1000, 0.2, {"student_temp": 1.1}) == 1.1,
        "center_freeze": trainer.objective.dino.center_momentum == 1.0,
        "iso_off": trainer.objective.mean_weight == 0.0 and trainer.objective.cov_weight == 0.0,
        "teacher_freeze": iv.teacher_momentum(1000, 0.999) == 1.0,
        "noise": (trainer.augment.t_lo, trainer.augment.t_start, trainer.augment.t_max) == (0.47, 0.9, 0.9),
        "lr_zero": trainer.optimizer.param_groups[0]["lr"] == 0.0 and trainer.scheduler.base_lrs == [0.0],
    }
    assert after.pop(kind)
    # Every other factor is untouched.
    assert iv.teacher_temp(1000, 0.2, {"student_temp": 1.1}) == (1.1 if kind == "sharpen_off" else 0.2)
    assert not any(after.values())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_frozen_center_and_teacher_are_exact_under_unit_momentum():
    center = torch.randn(5)
    before = center.clone()
    center.mul_(1.0).add_(torch.randn(5), alpha=0.0)
    assert torch.equal(center, before)


def test_invalid_interventions_rejected():
    with pytest.raises(ValueError):
        Intervention({"kind": "other", "from_step": 10})
    with pytest.raises(ValueError):
        Intervention({"kind": "noise", "from_step": 10, "t_lo": 0.9, "t_max": 0.5})
    with pytest.raises(ValueError):
        Intervention({"kind": "iso_off", "from_step": 10, "t_lo": 0.1})
    with pytest.raises(ValueError):
        Intervention({"kind": "iso_off", "from_step": 0})
