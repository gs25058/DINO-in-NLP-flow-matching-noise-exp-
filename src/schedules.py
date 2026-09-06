"""DINO 원본 방식의 teacher_temp / teacher momentum 스케줄 (R5 실험, METHOD.md §5 참고).

config에 관련 키가 없으면 train.py는 이 모듈을 쓰지 않고 기존 상수 동작을 그대로 유지한다
(기존 config 재실행 결과가 변하지 않아야 함).
"""
import math


def _ease(progress: float, shape: str) -> float:
    """진행률(0~1)을 스케줄 방식에 따라 0~1 보간 계수로 바꾼다.

    linear: 일정한 속도. cosine: S-curve(시작/끝이 완만하고 중간이 가파름).
    스케줄 '방식' 자체를 탐색 대상으로 올리기 위해 분리했다(r5_schedules_full study).
    """
    progress = min(max(progress, 0.0), 1.0)
    if shape == "linear":
        return progress
    if shape == "cosine":
        return (1 - math.cos(math.pi * progress)) / 2
    raise ValueError(f"unknown schedule shape: {shape!r} (linear|cosine)")


def teacher_temp_schedule(step: int, warmup_teacher_temp: float, teacher_temp: float, warmup_steps: int,
                          shape: str = "linear") -> float:
    """warmup_teacher_temp -> teacher_temp 증가, warmup_steps 이후 고정.

    shape 기본값 "linear"는 기존 동작과 bit-identical(기존 config 보존).
    """
    step = max(step, 0)
    progress = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
    return warmup_teacher_temp + (teacher_temp - warmup_teacher_temp) * _ease(progress, shape)


def teacher_momentum_schedule(step: int, momentum_start: float, momentum_end: float, ramp_steps: int,
                              shape: str = "cosine") -> float:
    """momentum_start -> momentum_end 증가, ramp_steps 이후 momentum_end로 고정.

    ramp_steps에 max_steps를 넘기면 전 구간 ramp(기존 동작). 더 작은 값을 넘기면 그 시점에
    ramp가 끝나고 이후로는 momentum_end가 유지된다 - 스케줄 '기간'을 탐색 대상으로 올리기 위함.
    shape 기본값 "cosine"은 기존 동작과 bit-identical.
    """
    step = min(max(step, 0), ramp_steps)
    progress = step / ramp_steps if ramp_steps > 0 else 1.0
    return momentum_start + (momentum_end - momentum_start) * _ease(progress, shape)


def koleo_lambda_schedule(step: int, lam_max: float, hold_steps: int, decay_steps: int, lam_min_ratio: float) -> float:
    """hold_steps까지 lam_max 유지, 이후 decay_steps에 걸쳐 lam_max*lam_min_ratio까지 cosine 감쇠,
    그 이후는 lam_max*lam_min_ratio로 고정 (R6 koleo 후반 정규화 과잉 완화 실험, koleo_schedule)."""
    step = max(step, 0)
    lam_min = lam_max * lam_min_ratio
    if step < hold_steps:
        return lam_max
    progress = min(1.0, (step - hold_steps) / decay_steps) if decay_steps > 0 else 1.0
    cos_in = (1 + math.cos(math.pi * progress)) / 2  # 1 -> 0
    return lam_min + (lam_max - lam_min) * cos_in


def resolve_koleo_lambda(step: int, lam_max: float, hold_steps: int | None, decay_steps: int | None,
                          lam_min_ratio: float | None) -> float:
    """hold_steps가 None이면(config에 koleo_hold_frac 미지정) 스케줄을 쓰지 않고 lam_max를 그대로
    반환한다 - 기존 상수 λ config의 동작을 그대로 보존하기 위한 진입점."""
    if hold_steps is None:
        return lam_max
    return koleo_lambda_schedule(step, lam_max, hold_steps, decay_steps, lam_min_ratio)
