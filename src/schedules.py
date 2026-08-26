"""DINO 원본 방식의 teacher_temp / teacher momentum 스케줄 (R5 실험, METHOD.md §5 참고).

config에 관련 키가 없으면 train.py는 이 모듈을 쓰지 않고 기존 상수 동작을 그대로 유지한다
(기존 config 재실행 결과가 변하지 않아야 함).
"""
import math


def teacher_temp_schedule(step: int, warmup_teacher_temp: float, teacher_temp: float, warmup_steps: int) -> float:
    """warmup_teacher_temp -> teacher_temp 선형 증가, warmup_steps 이후 고정."""
    step = max(step, 0)
    frac = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
    return warmup_teacher_temp + (teacher_temp - warmup_teacher_temp) * frac


def teacher_momentum_schedule(step: int, momentum_start: float, momentum_end: float, total_steps: int) -> float:
    """momentum_start -> momentum_end cosine 증가 (전체 학습 구간)."""
    step = min(max(step, 0), total_steps)
    progress = step / total_steps if total_steps > 0 else 1.0
    cos_out = (1 - math.cos(math.pi * progress)) / 2  # 0 -> 1
    return momentum_start + (momentum_end - momentum_start) * cos_out
