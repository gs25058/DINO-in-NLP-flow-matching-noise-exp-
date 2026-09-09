"""r10 BYOL식 alignment 항 + t 난이도 제어기 단위 테스트."""
import math

import torch
import torch.nn.functional as F

from src.augment import FlowNoiseAug
from src.loss import Predictor, TCtrl, align_loss


def _vecs(n=4, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_align_loss_zero_when_prediction_matches_teacher():
    z_t = _vecs(4, 8, 0)
    q_out = z_t.clone() * 3.0            # 방향만 같으면 크기 무관(출력을 L2 정규화하므로)
    raw, w, _ = align_loss([q_out], z_t, [torch.full((4,), 0.9)], focal_gamma=0.0)
    assert math.isclose(float(raw), 0.0, abs_tol=1e-6)
    assert math.isclose(float(w), 0.0, abs_tol=1e-6)


def test_align_loss_focal_gamma_zero_equals_unweighted():
    torch.manual_seed(1)
    z_t = _vecs(6, 8, 1)
    raw, w, aux = align_loss([torch.randn(6, 8)], z_t, [torch.rand(6)], focal_gamma=0.0)
    assert torch.allclose(raw, w)
    assert aux["focal_w_max"] == 1.0


def test_align_loss_focal_upweights_hard_positives():
    z_t = _vecs(2, 8, 2)
    q = torch.randn(2, 8)
    easy_hard = [torch.tensor([0.99, 0.10])]      # 두 번째가 어려움
    _, w_focal, aux = align_loss([q], z_t, easy_hard, focal_gamma=1.0)
    _, w_flat, _ = align_loss([q], z_t, easy_hard, focal_gamma=0.0)
    assert aux["focal_w_max"] > 1.0               # 배치 평균 1 정규화 -> 최대는 1 초과
    assert not torch.allclose(w_focal, w_flat)


def test_align_loss_averages_over_views():
    z_t = _vecs(3, 8, 3)
    cos_raw = [torch.full((3,), 0.5), torch.full((3,), 0.5)]
    raw, _, _ = align_loss([z_t.clone(), -z_t.clone()], z_t, cos_raw, focal_gamma=0.0)
    assert math.isclose(float(raw), 1.0, abs_tol=1e-5)   # (0 + 2)/2


def test_tctrl_moves_only_outside_band_and_clips():
    c = TCtrl(0.85, 0.90, 0.01, 0.1, t_min=0.2, t_max=0.9, t_init=0.5)
    c.update(0.87); assert c.t_ctrl == 0.5          # 밴드 안 -> 그대로
    c.update(0.95); assert math.isclose(c.t_ctrl, 0.51)   # 너무 쉬움 -> 증가
    c.update(0.50); assert math.isclose(c.t_ctrl, 0.50)   # 너무 어려움 -> 감소
    for _ in range(200):
        c.update(0.99)
    assert c.t_ctrl == 0.9                          # 상한 clip
    for _ in range(400):
        c.update(0.10)
    assert c.t_ctrl == 0.2                          # 하한 clip


def test_tctrl_range_respects_bounds():
    c = TCtrl(0.85, 0.90, 0.01, 0.1, t_min=0.2, t_max=0.9, t_init=0.25)
    lo, hi = c.range()
    assert lo == 0.2 and math.isclose(hi, 0.35)
    c.t_ctrl = 0.88
    lo, hi = c.range()
    assert math.isclose(lo, 0.78) and hi == 0.9


def test_augment_t_range_overrides_curriculum_and_default_unchanged():
    d = 4
    aug = FlowNoiseAug(torch.zeros(d), torch.ones(d), mode="anchor", num_student_views=2,
                       t_lo=0.35, t_start=0.5, t_max=0.5, warmup_steps=1, delta_t=0.1)
    emb = torch.randn(5, 6, d)
    mask = torch.zeros(5, 6, dtype=torch.bool)

    torch.manual_seed(0)
    base = aug(emb, mask, step=10)
    for t_v in base.t_students:                      # 기존 동작: U(0.35, 0.5)
        assert float(t_v.min()) >= 0.35 - 1e-6 and float(t_v.max()) <= 0.5 + 1e-6

    torch.manual_seed(0)
    ranged = aug(emb, mask, step=10, t_range=(0.7, 0.8))
    for t_v in ranged.t_students:                    # 주입 범위로 대체
        assert float(t_v.min()) >= 0.7 - 1e-6 and float(t_v.max()) <= 0.8 + 1e-6

    torch.manual_seed(0)                             # None이면 기존과 bit-identical
    again = aug(emb, mask, step=10, t_range=None)
    for a, b in zip(base.t_students, again.t_students):
        assert torch.equal(a, b)


def test_predictor_shape_and_eig_diagnostic_range():
    q = Predictor(dim=8, hidden=32)
    assert q(torch.randn(3, 8)).shape == (3, 8)
    assert 0.0 < q.top1_eig_frac() <= 1.0
