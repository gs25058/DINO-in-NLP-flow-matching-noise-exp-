import math

import torch

from src.augment import FlowNoiseAug

DIM = 8


def _make_aug(*, t_lo, t_start, t_max, warmup_steps=10, num_student_views=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(DIM, generator=g) * 2.0
    sigma = torch.rand(DIM, generator=g) * 2.0 + 0.5
    return FlowNoiseAug(
        mu,
        sigma,
        num_student_views=num_student_views,
        t_lo=t_lo,
        t_start=t_start,
        t_max=t_max,
        warmup_steps=warmup_steps,
    )


def _make_batch(B, L, dim=DIM, seed=1):
    torch.manual_seed(seed)
    token_embeds = torch.randn(B, L, dim)
    special_mask = torch.zeros(B, L, dtype=torch.bool)
    special_mask[:, 0] = True  # CLS
    special_mask[:, -1] = True  # SEP/pad 자리
    return token_embeds, special_mask


def test_identity_at_t_zero():
    # t_lo=t_start=t_max=0 이면 U(t_lo, t_hi(step))가 항상 0으로 붕괴 -> 항등이어야 함
    aug = _make_aug(t_lo=0.0, t_start=0.0, t_max=0.0)
    token_embeds, special_mask = _make_batch(B=4, L=6)

    out = aug(token_embeds, special_mask, step=0)

    assert torch.allclose(out.teacher_embeds, token_embeds, atol=1e-5)
    for v in out.student_embeds:
        assert torch.allclose(v, token_embeds, atol=1e-5)


def test_variance_at_t_one():
    # t_lo=t_start=t_max=1 이면 t가 항상 1로 붕괴 -> 표준화 공간에서 순수 eps ~ N(0,I)
    aug = _make_aug(t_lo=1.0, t_start=1.0, t_max=1.0, num_student_views=1)
    B, L = 8, 6
    token_embeds, special_mask = _make_batch(B=B, L=L)
    # 반복 배치로 통계량 확보 (같은 원본 문장, 독립적인 eps 샘플링)
    n_rep = 500
    token_embeds = token_embeds.repeat(n_rep, 1, 1)
    special_mask = special_mask.repeat(n_rep, 1)

    out = aug(token_embeds, special_mask, step=0)
    v = out.student_embeds[0]
    standardized = (v - aug.mu) / aug.sigma

    noise_positions = ~special_mask  # [B*n_rep, L]
    selected = standardized[noise_positions]  # [(B*n_rep)*content_tokens, D]
    var = selected.var(dim=0, unbiased=True)

    assert torch.allclose(var, torch.ones(DIM), atol=0.2)


def test_no_noise_at_special_or_padding():
    # 최대 노이즈(t=1)에서도 special_mask 위치는 원본과 동일해야 함
    aug = _make_aug(t_lo=1.0, t_start=1.0, t_max=1.0)
    token_embeds, special_mask = _make_batch(B=4, L=6)

    out = aug(token_embeds, special_mask, step=0)

    for v in out.student_embeds:
        assert torch.allclose(v[special_mask], token_embeds[special_mask], atol=1e-5)
    assert torch.allclose(out.teacher_embeds[special_mask], token_embeds[special_mask], atol=1e-5)


def test_curriculum_monotonic_and_bounded():
    t_start, t_max, warmup_steps = 0.05, 0.5, 100
    aug = _make_aug(t_lo=0.02, t_start=t_start, t_max=t_max, warmup_steps=warmup_steps)

    prev = -1.0
    for step in range(0, warmup_steps * 2, 5):
        t_hi = aug.t_hi(step)
        assert t_hi >= prev - 1e-9, "curriculum must be monotonically non-decreasing"
        assert t_hi <= t_max + 1e-9, "curriculum must not exceed t_max"
        prev = t_hi

    assert math.isclose(aug.t_hi(0), t_start, rel_tol=1e-6)
    assert math.isclose(aug.t_hi(warmup_steps), t_max, rel_tol=1e-6)
    assert math.isclose(aug.t_hi(warmup_steps * 10), t_max, rel_tol=1e-6)


def test_weak_view_changes_only_first_view_and_preserves_rng_teacher_and_gradients():
    import copy
    aug = _make_aug(t_lo=.39, t_start=.46, t_max=.88)
    weak = copy.deepcopy(aug)
    weak.weak_view = {'t_lo': 0., 't_max': .15}
    x, special = _make_batch(4, 6)
    x.requires_grad_()
    state = torch.get_rng_state()
    original = aug(x, special, 100)
    after_original = torch.get_rng_state()
    torch.set_rng_state(state)
    changed = weak(x, special, 100)
    assert torch.equal(after_original, torch.get_rng_state())
    assert torch.equal(changed.teacher_embeds, original.teacher_embeds)
    assert not torch.equal(changed.student_embeds[0], original.student_embeds[0])
    for a, b in zip(changed.student_embeds[1:], original.student_embeds[1:]):
        assert torch.equal(a, b)
    for view in changed.student_embeds:
        assert torch.equal(view[special], x[special])
    # Recover exactly the t and epsilon draws; validate the weak interpolation.
    torch.set_rng_state(state)
    t = .15 * torch.rand(4)
    eps = torch.randn_like(x)
    expected = ((1-t[:, None, None]) * ((x-aug.mu)/aug.sigma) + t[:, None, None]*eps)*aug.sigma+aug.mu
    expected = torch.where(special.unsqueeze(-1), x, expected)
    assert torch.equal(changed.student_embeds[0], expected)
    changed.student_embeds[0].sum().backward()
    expected_grad = torch.where(special.unsqueeze(-1), torch.ones_like(x), (1-t[:, None, None]).expand_as(x))
    torch.testing.assert_close(x.grad, expected_grad)


def test_disabled_weak_view_is_bitwise_legacy_and_equal_bounds_reproduce_all_views():
    aug = _make_aug(t_lo=.39, t_start=.46, t_max=.88)
    x, special = _make_batch(4, 6)
    state = torch.get_rng_state()
    expected = aug(x, special, 0)
    aug.weak_view = {'t_lo': .39, 't_max': .46}
    torch.set_rng_state(state)
    actual = aug(x, special, 0)
    assert all(torch.equal(a,b) for a,b in zip(expected.student_embeds,actual.student_embeds))


def test_invalid_weak_view_configuration_rejected():
    import pytest
    for weak in ({'t_lo':-.1,'t_max':.1}, {'t_lo':.2,'t_max':.1}, {'t_max':.1}):
        with pytest.raises(ValueError):
            FlowNoiseAug(torch.zeros(8),torch.ones(8),num_student_views=3,
                         t_lo=.39,t_start=.46,t_max=.88,warmup_steps=100,weak_view=weak)
