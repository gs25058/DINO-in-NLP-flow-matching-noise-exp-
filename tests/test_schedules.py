import math

from src.schedules import teacher_momentum_schedule, teacher_temp_schedule


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
