import math

import pytest
import torch

from src.augment import FlowNoiseAug


DIM = 8


def _make_aug(*, mode="anchor", t_lo, t_start, t_max, warmup_steps=10, delta_t=0.1,
              num_student_views=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(DIM, generator=g) * 2.0
    sigma = torch.rand(DIM, generator=g) * 2.0 + 0.5
    return FlowNoiseAug(
        mu, sigma,
        mode=mode, num_student_views=num_student_views,
        t_lo=t_lo, t_start=t_start, t_max=t_max,
        warmup_steps=warmup_steps, delta_t=delta_t,
    )


def _make_batch(B, L, dim=DIM, seed=1):
    torch.manual_seed(seed)
    token_embeds = torch.randn(B, L, dim)
    special_mask = torch.zeros(B, L, dtype=torch.bool)
    special_mask[:, 0] = True          # CLS
    special_mask[:, -1] = True         # SEP/pad 자리
    return token_embeds, special_mask


def test_identity_at_t_zero():
    # t_lo=t_start=t_max=0 이면 U(t_lo, t_hi(step))가 항상 0으로 붕괴 -> 항등이어야 함
    aug = _make_aug(mode="anchor", t_lo=0.0, t_start=0.0, t_max=0.0)
    token_embeds, special_mask = _make_batch(B=4, L=6)

    out = aug(token_embeds, special_mask, step=0)

    assert torch.allclose(out.teacher_embeds, token_embeds, atol=1e-5)
    for v in out.student_embeds:
        assert torch.allclose(v, token_embeds, atol=1e-5)


def test_variance_at_t_one():
    # t_lo=t_start=t_max=1 이면 t가 항상 1로 붕괴 -> 표준화 공간에서 순수 eps ~ N(0,I)
    aug = _make_aug(mode="anchor", t_lo=1.0, t_start=1.0, t_max=1.0, num_student_views=1)
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
    aug = _make_aug(mode="anchor", t_lo=1.0, t_start=1.0, t_max=1.0)
    token_embeds, special_mask = _make_batch(B=4, L=6)

    out = aug(token_embeds, special_mask, step=0)

    for v in out.student_embeds:
        assert torch.allclose(v[special_mask], token_embeds[special_mask], atol=1e-5)
    assert torch.allclose(
        out.teacher_embeds[special_mask], token_embeds[special_mask], atol=1e-5
    )


def test_curriculum_monotonic_and_bounded():
    t_start, t_max, warmup_steps = 0.05, 0.5, 100
    aug = _make_aug(mode="anchor", t_lo=0.02, t_start=t_start, t_max=t_max,
                     warmup_steps=warmup_steps)

    prev = -1.0
    for step in range(0, warmup_steps * 2, 5):
        t_hi = aug.t_hi(step)
        assert t_hi >= prev - 1e-9, "curriculum must be monotonically non-decreasing"
        assert t_hi <= t_max + 1e-9, "curriculum must not exceed t_max"
        prev = t_hi

    assert math.isclose(aug.t_hi(0), t_start, rel_tol=1e-6)
    assert math.isclose(aug.t_hi(warmup_steps), t_max, rel_tol=1e-6)
    assert math.isclose(aug.t_hi(warmup_steps * 10), t_max, rel_tol=1e-6)


def test_consistency_mode_shares_eps_with_first_student_view():
    aug = _make_aug(mode="consistency", t_lo=0.3, t_start=0.3, t_max=0.3,
                     delta_t=0.1, num_student_views=2)
    token_embeds, special_mask = _make_batch(B=4, L=6)

    out = aug(token_embeds, special_mask, step=0)

    assert torch.allclose(out.t_teacher, out.t_students[0] - 0.1, atol=1e-6)
    # teacher가 student view 0과 같은 eps를 공유하는지 -> 표준화 공간에서 재구성해 비교
    t_t = out.t_teacher.view(-1, 1, 1)
    t_s = out.t_students[0].view(-1, 1, 1)
    x_hat = out.x0_std
    eps_shared = out.eps[0]

    x_t_teacher_std = (1 - t_t) * x_hat + t_t * eps_shared
    reconstructed_teacher = torch.where(
        special_mask.unsqueeze(-1), token_embeds, x_t_teacher_std * aug.sigma + aug.mu
    )
    assert torch.allclose(reconstructed_teacher, out.teacher_embeds, atol=1e-5)
