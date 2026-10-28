import pytest

from src.collapse_guard import CollapseGuard
from scripts.measure_guard_refs import measure_refs


def config(**overrides):
    return dict(enabled=True, h_p_bar_t_ref=9.0, batch_kl_ref=0.03,
                eff_rank_ref=300, start_frac=0.2, patience=3, **overrides)


def test_disabled_guard_requires_no_reference_or_changes():
    guard = CollapseGuard({}, 100)
    assert guard.observe_train(99, 0, 0) is None
    assert guard.observe_rank(99, 0) is None


def test_enabled_guard_demands_explicit_base_references():
    with pytest.raises(ValueError, match="reference"):
        CollapseGuard({"enabled": True}, 100)


def test_warmup_and_consecutive_strikes_prevent_false_alarms():
    guard = CollapseGuard(config(), 100)
    for step in range(20):
        assert guard.observe_train(step, 0, 0) is None
    assert guard.observe_train(20, 8.5, 0) is None
    assert guard.observe_train(21, 9.0, 0.03) is None
    assert guard.strikes == 0
    assert guard.observe_train(22, 8.5, 0) is None
    assert guard.observe_train(23, 8.5, 0) is None
    assert guard.observe_train(24, 8.5, 0) is not None
    assert guard.observe_rank(19, 100) is None
    assert guard.observe_rank(20, 100) is not None


def test_measurement_uses_own_complete_runs_and_reports_provenance(tmp_path):
    paths = []
    for seed in (42, 43):
        path = tmp_path / f"base_s{seed}.log"
        path.write_text("[step 20] loss=1 H_p_bar_t=9 batch_KL=0.02\n"
                        "[step 50] loss=1 H_p_bar_t=9 batch_KL=0.04\n"
                        "[step 99] EVAL sts_b_dev=0.7 eff_rank=300\n")
        paths.append(path)
    refs = measure_refs(paths, 100)
    assert refs["h_p_bar_t_ref"] == 9
    assert refs["eff_rank_ref"] == 300
    assert refs["batch_kl_ref"] == pytest.approx(0.04 / 1.47)
    assert len(refs["reference_logs"]) == 2
    paths[1].write_text("[step 50] loss=1 H_p_bar_t=9 batch_KL=0.04\n")
    with pytest.raises(ValueError, match="incomplete"):
        measure_refs(paths, 100)
