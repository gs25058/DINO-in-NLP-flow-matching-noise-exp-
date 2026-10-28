"""Bounded joint search around the measured R18 control (performance, not ablation)."""
import copy
import math

import optuna

SPACE = {
    "loss.cov_iso_mean_lambda": (0.05, 8.0, True),
    "loss.cov_iso_cov_lambda": (0.05, 8.0, True),
    "train.lr": (2e-5, 2e-4, True),
    "train.head_lr": (1e-4, 2e-3, True),
    "train.grad_clip": (1.0, 6.0, False),
    "train.weight_decay": (0.01, 0.3, True),
    "train.warmup_frac": (0.01, 0.2, False),
    "train.momentum_end": (0.998, 0.9999, False),
    "train.momentum_ramp_frac": (0.2, 1.0, False),
    "loss.teacher_temp": (0.08, 0.18, False),
    "loss.warmup_teacher_temp": (0.03, 0.08, False),
    "loss.student_temp": (0.1, 0.25, False),
    "loss.teacher_temp_warmup_frac": (0.02, 0.3, False),
    "loss.center_momentum": (0.8, 0.99, False),
    "loss.cov_ema_momentum": (0.75, 0.99, False),
    "augment.t_lo": (0.0, 0.6, False),
    "_t_span": (0.05, 0.6, False),
    "_t_start_frac": (0.1, 1.0, False),
    "augment.warmup_frac": (0.0, 0.15, False),
}


def distributions():
    return {key: optuna.distributions.FloatDistribution(lo, hi, log=log)
            for key, (lo, hi, log) in SPACE.items()}


def baseline_parameters(cfg):
    result = {}
    for key in SPACE:
        if key.startswith("_"):
            continue
        section, name = key.split(".")
        value = cfg[section].get(name)
        if value is None and name in ("cov_iso_mean_lambda", "cov_iso_cov_lambda"):
            value = cfg["loss"]["cov_iso_lambda"]
        result[key] = float(value)
    aug = cfg["augment"]
    result["_t_span"] = aug["t_max"] - aug["t_lo"]
    result["_t_start_frac"] = (aug["t_start"] - aug["t_lo"]) / result["_t_span"]
    return result


def config_from_parameters(base, params):
    cfg = copy.deepcopy(base)
    if set(params) != set(SPACE):
        raise ValueError("joint search parameters must match the declared space")
    for key, value in params.items():
        lo, hi, _ = SPACE[key]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"parameter outside declared search space: {key}")
        if not key.startswith("_"):
            section, name = key.split(".")
            cfg[section][name] = float(value)
    aug = cfg["augment"]
    aug["t_max"] = min(0.99, aug["t_lo"] + params["_t_span"])
    aug["t_start"] = aug["t_lo"] + (aug["t_max"] - aug["t_lo"]) * params["_t_start_frac"]
    return cfg


def suggest_config(trial, base):
    params = {key: trial.suggest_float(key, lo, hi, log=log) for key, (lo, hi, log) in SPACE.items()}
    return config_from_parameters(base, params)
