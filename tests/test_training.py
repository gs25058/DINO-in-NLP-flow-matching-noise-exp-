import pytest

from src.loss import DINOObjective
from src.train import cosine_ramp, load_config


def test_cosine_warmup_and_momentum_boundaries_match_current_schedules():
    cfg = load_config("configs/coviso_relax/control.yaml")
    for initial, final, duration in (
        (cfg["loss"]["warmup_teacher_temp"], cfg["loss"]["teacher_temp"], 315),
        (cfg["train"]["momentum_start"], cfg["train"]["momentum_end"], 1630),
    ):
        assert cosine_ramp(-2, initial, final, duration) == initial
        assert cosine_ramp(0, initial, final, duration) == initial
        assert cosine_ramp(duration, initial, final, duration) == final
        assert cosine_ramp(duration + 1, initial, final, duration) == final
        assert cosine_ramp(duration / 2, initial, final, duration) == pytest.approx(
            (initial + final) / 2, abs=1e-12
        )
        values = [cosine_ramp(step, initial, final, duration) for step in range(duration + 1)]
        assert all(a <= b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_component_weights_reject_invalid_values(value):
    cfg = load_config("configs/coviso_relax/control.yaml")["loss"]
    cfg["cov_iso_mean_lambda"] = value
    with pytest.raises(ValueError, match="nonnegative"):
        DINOObjective(cfg, 8, 12)
