import math

from src.schedules import (
    koleo_lambda_schedule,
    resolve_koleo_lambda,
    teacher_momentum_schedule,
    teacher_temp_schedule,
)


def test_teacher_temp_schedule_start_end_and_monotonic():
    warmup_teacher_temp, teacher_temp, warmup_steps = 0.082, 0.11, 300

    assert math.isclose(teacher_temp_schedule(0, warmup_teacher_temp, teacher_temp, warmup_steps), warmup_teacher_temp)
    assert math.isclose(teacher_temp_schedule(warmup_steps, warmup_teacher_temp, teacher_temp, warmup_steps), teacher_temp)
    assert math.isclose(
        teacher_temp_schedule(warmup_steps * 10, warmup_teacher_temp, teacher_temp, warmup_steps), teacher_temp
    )

    prev = -math.inf
    for step in range(0, warmup_steps * 2, 5):
        v = teacher_temp_schedule(step, warmup_teacher_temp, teacher_temp, warmup_steps)
        assert v >= prev - 1e-9
        assert v <= teacher_temp + 1e-9
        prev = v


def test_teacher_momentum_schedule_start_end_and_monotonic():
    momentum_start, momentum_end, total_steps = 0.997, 0.9995, 1500

    assert math.isclose(teacher_momentum_schedule(0, momentum_start, momentum_end, total_steps), momentum_start)
    assert math.isclose(teacher_momentum_schedule(total_steps, momentum_start, momentum_end, total_steps), momentum_end)

    prev = -math.inf
    for step in range(0, total_steps, 10):
        v = teacher_momentum_schedule(step, momentum_start, momentum_end, total_steps)
        assert v >= prev - 1e-9
        assert v <= momentum_end + 1e-9
        prev = v


def test_teacher_temp_schedule_default_shape_is_linear_bit_identical():
    # shape 인자를 안 주면 기존(선형) 동작과 정확히 같아야 한다 - 기존 config 보존
    lo, hi, ws = 0.082, 0.11, 450
    for step in [0, 1, 100, 225, 449, 450, 1000]:
        assert teacher_temp_schedule(step, lo, hi, ws) == teacher_temp_schedule(step, lo, hi, ws, "linear")
        # 선형 공식과 직접 대조
        expected = lo + (hi - lo) * min(1.0, step / ws)
        assert math.isclose(teacher_temp_schedule(step, lo, hi, ws), expected)


def test_teacher_temp_schedule_cosine_shape():
    lo, hi, ws = 0.082, 0.11, 450

    # 양 끝값은 방식과 무관하게 동일
    assert math.isclose(teacher_temp_schedule(0, lo, hi, ws, "cosine"), lo)
    assert math.isclose(teacher_temp_schedule(ws, lo, hi, ws, "cosine"), hi)
    assert math.isclose(teacher_temp_schedule(ws * 3, lo, hi, ws, "cosine"), hi)

    # 중간 지점에서는 선형과 달라야 하고(=S-curve), 단조 증가여야 한다
    mid_lin = teacher_temp_schedule(ws // 4, lo, hi, ws, "linear")
    mid_cos = teacher_temp_schedule(ws // 4, lo, hi, ws, "cosine")
    assert mid_cos < mid_lin  # cosine은 초반이 완만

    prev = -math.inf
    for step in range(0, ws + 1, 5):
        v = teacher_temp_schedule(step, lo, hi, ws, "cosine")
        assert v >= prev - 1e-12
        assert lo - 1e-12 <= v <= hi + 1e-12
        prev = v


def test_teacher_momentum_schedule_default_shape_and_full_ramp_bit_identical():
    # ramp_steps=total_steps + shape 기본값이면 기존 동작과 정확히 같아야 한다
    ms, me, total = 0.997, 0.9995, 1500
    for step in [0, 1, 300, 750, 1499, 1500]:
        expected = ms + (me - ms) * (1 - math.cos(math.pi * min(step, total) / total)) / 2
        assert math.isclose(teacher_momentum_schedule(step, ms, me, total), expected)
        assert teacher_momentum_schedule(step, ms, me, total) == teacher_momentum_schedule(
            step, ms, me, total, "cosine"
        )


def test_teacher_momentum_schedule_short_ramp_holds_at_end_value():
    # ramp를 전 구간보다 짧게 주면 그 시점에 끝나고 이후로는 momentum_end 유지
    ms, me, ramp = 0.997, 0.9995, 500
    assert math.isclose(teacher_momentum_schedule(0, ms, me, ramp), ms)
    assert math.isclose(teacher_momentum_schedule(ramp, ms, me, ramp), me)
    for step in [ramp + 1, 1000, 1499]:
        assert math.isclose(teacher_momentum_schedule(step, ms, me, ramp), me)

    # 짧은 ramp는 같은 step에서 전 구간 ramp보다 항상 앞서간다(더 큰 momentum)
    for step in [100, 250, 400]:
        assert teacher_momentum_schedule(step, ms, me, ramp) >= teacher_momentum_schedule(step, ms, me, 1500)


def test_schedule_shape_rejects_unknown_value():
    import pytest

    with pytest.raises(ValueError, match="unknown schedule shape"):
        teacher_temp_schedule(10, 0.082, 0.11, 450, "quadratic")
    with pytest.raises(ValueError, match="unknown schedule shape"):
        teacher_momentum_schedule(10, 0.997, 0.9995, 1500, "quadratic")


def test_koleo_lambda_schedule_hold_then_decay_then_floor():
    lam_max, hold_steps, decay_steps, lam_min_ratio = 0.4, 300, 600, 0.25
    lam_min = lam_max * lam_min_ratio

    # hold 구간: lam_max 그대로
    for step in [0, 1, 150, 299]:
        assert math.isclose(koleo_lambda_schedule(step, lam_max, hold_steps, decay_steps, lam_min_ratio), lam_max)

    # 감쇠 구간 끝(step == hold_steps + decay_steps)에서 정확히 바닥값
    assert math.isclose(
        koleo_lambda_schedule(hold_steps + decay_steps, lam_max, hold_steps, decay_steps, lam_min_ratio), lam_min
    )

    # 감쇠 구간 이후로도 계속 바닥값 유지
    for step in [hold_steps + decay_steps + 1, hold_steps + decay_steps + 1000]:
        assert math.isclose(koleo_lambda_schedule(step, lam_max, hold_steps, decay_steps, lam_min_ratio), lam_min)


def test_koleo_lambda_schedule_decay_is_monotonic_decreasing():
    lam_max, hold_steps, decay_steps, lam_min_ratio = 0.4, 300, 600, 0.25

    prev = math.inf
    for step in range(hold_steps, hold_steps + decay_steps + 1, 10):
        v = koleo_lambda_schedule(step, lam_max, hold_steps, decay_steps, lam_min_ratio)
        assert v <= prev + 1e-9
        assert v >= lam_max * lam_min_ratio - 1e-9
        prev = v


def test_koleo_lambda_schedule_full_decay_to_zero():
    # lam_min_ratio=0.0 (koleo_sched_b: 후반 완전 소거) - 바닥값이 정확히 0
    lam_max, hold_steps, decay_steps, lam_min_ratio = 0.4, 300, 600, 0.0
    assert math.isclose(
        koleo_lambda_schedule(hold_steps + decay_steps, lam_max, hold_steps, decay_steps, lam_min_ratio), 0.0,
        abs_tol=1e-9,
    )


def test_resolve_koleo_lambda_unset_schedule_matches_constant_lambda():
    # koleo_hold_frac 미지정(hold_steps=None) -> train.py의 기존 상수 λ 동작과 bit-identical해야 함
    lam_max = 0.2227
    for step in [0, 1, 300, 749, 1499]:
        assert resolve_koleo_lambda(step, lam_max, None, None, None) == lam_max


def test_resolve_koleo_lambda_set_schedule_matches_koleo_lambda_schedule():
    lam_max, hold_steps, decay_steps, lam_min_ratio = 0.4, 300, 600, 0.25
    for step in [0, 300, 450, 900, 1499]:
        assert resolve_koleo_lambda(step, lam_max, hold_steps, decay_steps, lam_min_ratio) == koleo_lambda_schedule(
            step, lam_max, hold_steps, decay_steps, lam_min_ratio
        )
