"""Optuna 탐색 공간 검증.

flow_noise_t는 t ~ U(t_lo, t_hi(step))의 제약(t_lo < t_hi <= 1.0)을 파라미터화로
보장한다 - 무효 조합이 나오면 trial이 통째로 낭비되므로 여기서 못 박아 둔다.
"""
import importlib.util
from pathlib import Path

import optuna
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tune", ROOT / "scripts" / "tune.py")
tune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tune)


def _sample(space_name, n=200):
    """탐색 공간을 실제 Optuna 샘플러로 n번 뽑아 config에 적용한 결과를 돌려준다."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    out = []
    for _ in range(n):
        trial = study.ask()
        cfg = {"augment": {"t_lo": 0.35, "t_start": 0.5, "t_max": 0.5,
                           "warmup_frac": 0.0, "num_student_views": 3}}
        pseudo = {}
        for dotted, (kind, kwargs) in tune.SEARCH_SPACES[space_name].items():
            value = getattr(trial, f"suggest_{kind}")(dotted, **kwargs)
            if dotted.startswith("_"):
                pseudo[dotted] = value
            else:
                tune.set_by_path(cfg, dotted, value)
        if space_name in tune.DERIVED:
            tune.DERIVED[space_name](cfg, pseudo)
        study.tell(trial, 0.0)
        out.append(cfg["augment"])
    return out


def test_flow_noise_t_always_produces_valid_range():
    for a in _sample("flow_noise_t"):
        assert 0.0 <= a["t_lo"] < a["t_start"] <= a["t_max"] <= 1.0, a
        assert a["t_lo"] < a["t_max"], a           # sample_t의 폭이 0이면 노이즈가 상수가 된다


def test_flow_noise_t_covers_current_champion_region():
    """현 챔피언(t_lo=0.35, t_hi=0.5)이 탐색 범위 안에 들어 있어야 비교가 성립한다."""
    lo_rng = tune.SEARCH_SPACES["flow_noise_t"]["augment.t_lo"][1]
    span_rng = tune.SEARCH_SPACES["flow_noise_t"]["_t_span"][1]
    assert lo_rng["low"] <= 0.35 <= lo_rng["high"]
    assert span_rng["low"] <= 0.15 <= span_rng["high"]      # 0.5 - 0.35


def test_flow_noise_t_reaches_the_controller_equilibrium():
    """r10b 제어기가 평형한 t~0.63 부근도 도달 가능해야 한다(이 study의 동기)."""
    reach = [a for a in _sample("flow_noise_t") if a["t_lo"] <= 0.63 <= a["t_max"]]
    assert len(reach) > 0


def test_pseudo_params_never_leak_into_config():
    """밑줄 키가 config에 그대로 박히면 train.py가 조용히 무시해 탐색이 무의미해진다."""
    for a in _sample("flow_noise_t"):
        assert not any(k.startswith("_") for k in a), a


def test_pseudo_params_and_derived_hooks_agree():
    """불변식: 의사 파라미터(_로 시작)를 쓰는 공간은 정확히 DERIVED 훅이 있는 공간이다.

    한쪽만 있으면 조용히 깨진다 - 훅 없이 의사 파라미터를 두면 그 축이 config에 반영되지 않은
    채 탐색만 낭비되고, 반대면 훅이 존재하지 않는 키를 읽어 KeyError가 난다.
    """
    with_pseudo = {n for n, s in tune.SEARCH_SPACES.items() if any(k.startswith("_") for k in s)}
    assert with_pseudo == set(tune.DERIVED), (with_pseudo, set(tune.DERIVED))


def test_every_non_pseudo_key_is_a_real_config_path():
    """점 경로의 최상위 섹션이 config 구조에 실제로 존재하는 이름인지 (오타 방지)."""
    allowed = {"loss", "train", "augment", "model", "eval", "data"}
    for name, space in tune.SEARCH_SPACES.items():
        for dotted in space:
            if dotted.startswith("_"):
                continue
            assert dotted.split(".")[0] in allowed, f"{name}: {dotted}"


def _sample_with_base(space_name, base, n=200):
    """base config(챔피언 상속값)를 주고 탐색 공간을 적용한 결과를 돌려준다."""
    import copy
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    out = []
    for _ in range(n):
        trial = study.ask()
        cfg = copy.deepcopy(base)
        pseudo = {}
        for dotted, (kind, kwargs) in tune.SEARCH_SPACES[space_name].items():
            value = getattr(trial, f"suggest_{kind}")(dotted, **kwargs)
            if dotted.startswith("_"):
                pseudo[dotted] = value
            else:
                tune.set_by_path(cfg, dotted, value)
        if space_name in tune.DERIVED:
            tune.DERIVED[space_name](cfg, pseudo)
        study.tell(trial, 0.0)
        out.append(cfg)
    return out


def test_r13_sharpen_ctrl_window_is_always_valid():
    base = {"loss": {"entropy_ctrl_start_frac": 0.45, "entropy_ctrl_end_frac": 0.85,
                     "entropy_ctrl_delta": 0.1, "entropy_ctrl_gain": 0.3}}
    for c in _sample_with_base("r13_sharpen_ctrl", base):
        l = c["loss"]
        assert 0.0 < l["entropy_ctrl_start_frac"] < l["entropy_ctrl_end_frac"] <= 1.0, l
        assert 0.02 <= l["entropy_ctrl_delta"] <= 0.8, l


def test_r13_view_corr_scales_t_without_inverting_it():
    base = {"augment": {"t_lo": 0.392, "t_start": 0.615, "t_max": 0.881}}
    for c in _sample_with_base("r13_view_corr", base):
        a = c["augment"]
        assert 0.0 <= a["t_lo"] < a["t_max"] <= 1.0, a
        assert a["t_start"] <= a["t_max"], a
        assert 0.05 <= a["noise_corr_rho"] <= 1.0, a


def test_r13_view_cutoff_stays_in_range():
    base = {"augment": {"cutoff_span_frac": 0.1, "cutoff_prob": 1.0, "cutoff_mode": "drop"}}
    for c in _sample_with_base("r13_view_cutoff", base):
        a = c["augment"]
        assert 0.02 <= a["cutoff_span_frac"] <= 0.35
        assert 0.3 <= a["cutoff_prob"] <= 1.0
        assert a["cutoff_mode"] in ("drop", "mask")


def test_r13_spaces_have_derived_hooks_where_needed():
    assert "r13_sharpen_ctrl" in tune.DERIVED and "r13_view_corr" in tune.DERIVED
    assert "r13_view_cutoff" not in tune.DERIVED      # 의사 파라미터가 없다


def test_r14_long_budget_brackets_the_champion_and_keeps_temps_ordered():
    """재튜닝 공간은 현 챔피언 값을 안쪽에 포함해야 하고(비교가 성립), teacher 온도는 warmup 시작값보다 커야 한다."""
    space = tune.SEARCH_SPACES["r14_long_budget"]
    champion = {"loss.cov_iso_lambda": 5.874, "train.momentum_end": 0.9985, "train.lr": 6.876e-5,
                "train.head_lr": 5.825e-4, "loss.teacher_temp": 0.1348}
    for key, val in champion.items():
        kind, kw = space[key]
        assert kw["low"] < val < kw["high"], (key, val, kw)
    assert space["loss.teacher_temp"][1]["low"] > 0.0494          # warmup_teacher_temp보다 위
    assert 0.999 < space["train.momentum_end"][1]["high"]          # P-14c의 0.999를 포함
    assert "r14_long_budget" not in tune.DERIVED


def test_trials_launch_with_the_current_interpreter():
    """uv run으로 띄우면 venv 없는 worktree에서 trial마다 venv를 새로 만든다 - sys.executable을 써야 한다."""
    src = (ROOT / "scripts" / "tune.py").read_text(encoding="utf-8")
    assert "[sys.executable, \"-m\", \"src.train\"" in src
    assert "[\"uv\", \"run\", \"python\", \"-m\", \"src.train\"" not in src
