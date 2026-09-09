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


def test_existing_spaces_have_no_pseudo_params():
    """기존 공간들은 파생 기제 도입 전과 동일하게 동작해야 한다."""
    for name, space in tune.SEARCH_SPACES.items():
        if name == "flow_noise_t":
            continue
        assert not any(k.startswith("_") for k in space), name
        assert name not in tune.DERIVED, name
