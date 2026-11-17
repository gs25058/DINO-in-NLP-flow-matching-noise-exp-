"""Phase B of the late-drop study: schedule the two causal factors on top of the vocab_proto space.

Phase A found two causes of the late dev drop: teacher sharpening and the EMA teacher tracking
the student. Four new axes schedule them; every other axis is the vocab_proto space unchanged.

- ``_late_relax`` u ∈ [0, 1]: late teacher/student ratio = r + (1 − r)·u, r the tuned base ratio
  (u = 0 keeps the sharpening, u = 1 removes it as the ``sharpen_off`` intervention did).
- ``_late_start_frac``: relaxation start (fraction of training).
- ``_late_len_frac``: cosine relaxation length (0 = switch at once).
- ``train.teacher_freeze_frac``: teacher EMA momentum 1 from this fraction (1.0 = never).
"""

from src import vocab_proto as vp

CONDITIONS = ("C1", "C2")
SCHEDULE = {
    "_late_relax": (0.0, 1.0, False),
    "_late_start_frac": (0.05, 0.75, False),
    "_late_len_frac": (0.0, 0.5, False),
    "train.teacher_freeze_frac": (0.1, 1.0, False),
}
NEUTRAL = {"_late_relax": 0.0, "_late_start_frac": 0.75, "_late_len_frac": 0.0, "train.teacher_freeze_frac": 1.0}
# Phase A switch step 1000 of 4000.
SWITCH = 0.25


def space(condition):
    return {**vp.space(condition), **SCHEDULE}


def configure(base, condition, params, kmeans_path=None):
    declared = space(condition)
    if set(params) != set(declared):
        raise ValueError("parameters must match the declared space")
    for key, value in params.items():
        lo, hi, _ = declared[key]
        if not lo <= value <= hi:
            raise ValueError(f"parameter outside declared search space: {key}={value}")
    cfg = vp.configure(base, condition, {k: params[k] for k in vp.space(condition)}, kmeans_path)
    # Neutral values add no keys, so a neutral point is the vocab_proto configuration bit for bit.
    if params["_late_relax"] > 0:
        loss = cfg["loss"]
        # 1 − (1 − r)(1 − u) is exactly 1 at u = 1 (the sharpen_off intervention).
        loss["teacher_temp_late_ratio"] = 1.0 - (1.0 - params["_teacher_ratio"]) * (1.0 - params["_late_relax"])
        loss["teacher_temp_late_start_frac"] = params["_late_start_frac"]
        loss["teacher_temp_late_frac"] = params["_late_len_frac"]
    if params["train.teacher_freeze_frac"] < 1.0:
        cfg["train"]["teacher_freeze_frac"] = params["train.teacher_freeze_frac"]
    return cfg


def anchors(previous, condition):
    """Previous vocab_proto configurations, plain and with the two Phase A interventions as schedules."""
    sharpen_off = {"_late_relax": 1.0, "_late_start_frac": SWITCH, "_late_len_frac": 0.0,
                   "train.teacher_freeze_frac": 1.0}
    freeze = {**NEUTRAL, "train.teacher_freeze_frac": SWITCH}
    plan = {"C2": [("9", sharpen_off), ("9", freeze), ("3", NEUTRAL), ("3", sharpen_off), ("28", sharpen_off)],
            "C1": [("16", sharpen_off), ("16", freeze), ("29", NEUTRAL), ("29", sharpen_off), ("14", sharpen_off)]}
    return [{**previous[condition][n]["params"], **schedule} for n, schedule in plan[condition]]
