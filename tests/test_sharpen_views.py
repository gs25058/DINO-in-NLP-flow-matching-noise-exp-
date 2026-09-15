"""R11(A) 후반 샤프닝 / R12(B) pooling을 견디는 뷰 단위 테스트.

두 계열 모두 "기본값에서 r8과 bit-identical"이 가장 중요한 성질이다 - 기준선을 재실행하지
않고 비교하므로 off 경로가 조금이라도 달라지면 비교 자체가 무효가 된다.
"""
import math

import pytest
import torch

from src.augment import FlowNoiseAug
from src.loss import EntropyCtrl
from src.schedules import teacher_temp_schedule

WU_T, T, WU = 0.0494, 0.1348, 118          # r8 실제 값
D = 4


# ---------------------------------------------------------------- A1: 3국면 스케줄
def _decay(step, **kw):
    return teacher_temp_schedule(step, WU_T, T, WU, "cosine", decay_start_step=675,
                                 decay_end_step=1275, teacher_temp_final=0.085, **kw)


def test_a1_without_final_is_bit_identical():
    for step in range(0, 1600, 7):
        for shape in ("linear", "cosine"):
            assert (teacher_temp_schedule(step, WU_T, T, WU, shape)
                    == teacher_temp_schedule(step, WU_T, T, WU, shape, teacher_temp_final=None))


def test_a1_four_phases():
    assert _decay(0) == pytest.approx(WU_T)
    assert _decay(WU) == pytest.approx(T)
    assert _decay(674) == pytest.approx(T)          # plateau 유지
    assert _decay(675) == pytest.approx(T)          # 하강 시작점은 plateau와 이어짐
    assert _decay(1275) == pytest.approx(0.085)
    assert _decay(1499) == pytest.approx(0.085)


def test_a1_decay_is_monotone():
    v = [_decay(s) for s in range(675, 1276, 10)]
    assert all(b <= a + 1e-12 for a, b in zip(v, v[1:])) and v[0] > v[-1]


def test_a1_rejects_decay_before_warmup_ends():
    with pytest.raises(ValueError, match="앞설 수 없다"):
        teacher_temp_schedule(0, WU_T, T, WU, teacher_temp_final=0.085, decay_start_step=WU - 1)


# ---------------------------------------------------------------- A2: 엔트로피 제어기
def _ctrl(**kw):
    kw = {"start_step": 675, "end_step": 1275, "delta": 0.3, "gain": 0.02,
          "tau_min": 0.05, "tau_max": 0.1348, "tau_init": 0.1348, **kw}
    return EntropyCtrl(**kw)


def test_a2_tau_unchanged_before_start():
    c = _ctrl()
    for s in range(0, 675, 25):
        c.observe(8.974)
        assert c.update(s) == pytest.approx(0.1348)
    assert c.h_anchor is None              # anchor는 인계 시점에야 잡힌다


def test_a2_anchor_is_taken_at_handover_and_target_descends():
    c = _ctrl()
    for _ in range(50):
        c.observe(8.974)
    c.update(675)
    assert c.h_anchor == pytest.approx(c.h_ema)
    assert c.target(675) == pytest.approx(c.h_anchor)             # 인계 순간 목표 = 현재값(연속)
    assert c.target(1275) == pytest.approx(c.h_anchor - 0.3)      # 종점에서 delta만큼 낮다
    assert c.target(1499) == pytest.approx(c.h_anchor - 0.3)      # 이후 유지
    mid = c.target((675 + 1275) // 2)
    assert c.h_anchor - 0.3 < mid < c.h_anchor


def test_a2_tau_falls_when_entropy_is_above_target():
    c = _ctrl()
    for _ in range(50):
        c.observe(8.974)
    c.update(675)
    taus = []
    for s in range(676, 900):
        c.observe(8.974)                   # H가 안 내려가면 제어기가 계속 조인다
        taus.append(c.update(s))
    assert all(b <= a + 1e-12 for a, b in zip(taus, taus[1:]))
    assert taus[-1] < 0.1348


def test_a2_rate_limit_and_clip():
    c = _ctrl(gain=100.0)                  # 과도한 이득으로 한 스텝에 튀게 만든다
    for _ in range(50):
        c.observe(9.5)
    prev = c.tau
    c.update(675)
    assert c.tau >= prev * (1 - 0.01) - 1e-12        # 스텝당 상대 변화 제한
    for s in range(676, 2000):
        c.observe(9.5)
        c.update(s)
    assert c.tau == pytest.approx(0.05)              # tau_min에서 clip


def test_a2_rejects_bad_ranges():
    with pytest.raises(ValueError, match="앞설 수 없다"):
        _ctrl(start_step=900, end_step=800)
    with pytest.raises(ValueError, match="tau 범위"):
        _ctrl(tau_min=0.2, tau_max=0.1)


# ---------------------------------------------------------------- B1: 상관 노이즈
def _aug(**kw):
    return FlowNoiseAug(torch.zeros(D), torch.ones(D), mode="anchor", num_student_views=2,
                        t_lo=0.35, t_start=0.5, t_max=0.5, warmup_steps=1, delta_t=0.1, **kw)


def _batch(B=3, L=6):
    emb = torch.randn(B, L, D)
    special = torch.zeros(B, L, dtype=torch.bool)
    special[:, 0] = True                 # CLS
    special[:, -1] = True                # SEP
    attn = torch.ones(B, L, dtype=torch.long)
    return emb, special, attn


def test_b1_rho_zero_is_bit_identical():
    emb, sp, _ = _batch()
    torch.manual_seed(0); base = _aug()(emb, sp, step=10)
    torch.manual_seed(0); same = _aug(noise_corr_rho=0.0)(emb, sp, step=10)
    for a, b in zip(base.student_embeds, same.student_embeds):
        assert torch.equal(a, b)


def test_b1_rho_one_shares_eps_across_tokens():
    emb, sp, _ = _batch()
    v = _aug(noise_corr_rho=1.0)(emb, sp, step=10)
    eps = v.eps[0]                        # [B, L, D]
    for b in range(eps.shape[0]):
        assert torch.allclose(eps[b], eps[b, :1].expand_as(eps[b]), atol=1e-6)


def test_b1_preserves_unit_variance():
    torch.manual_seed(0)
    aug = _aug(noise_corr_rho=0.5)
    emb = torch.randn(256, 12, D)
    sp = torch.zeros(256, 12, dtype=torch.bool)
    eps = aug(emb, sp, step=10).eps[0]
    assert eps.var().item() == pytest.approx(1.0, abs=0.05)


def test_b1_correlated_noise_survives_mean_pooling():
    """이 실험의 핵심 주장: iid는 pooling에서 소멸하고 상관 노이즈는 살아남는다."""
    torch.manual_seed(0)
    B, L = 128, 24
    iid = torch.randn(B, L, D).mean(dim=1).pow(2).mean()
    shared = torch.randn(B, 1, D).expand(B, L, D).mean(dim=1).pow(2).mean()
    assert shared > iid * 5               # 1/L 소멸 vs 보존


# ---------------------------------------------------------------- B2: span cutoff
def test_b2_off_is_bit_identical():
    emb, sp, attn = _batch()
    torch.manual_seed(0); base = _aug()(emb, sp, step=10, attention_mask=attn)
    torch.manual_seed(0); same = _aug(cutoff_span_frac=0.0)(emb, sp, step=10, attention_mask=attn)
    assert base.student_masks is None and same.student_masks is None
    for a, b in zip(base.student_embeds, same.student_embeds):
        assert torch.equal(a, b)


def test_b2_drop_zeroes_a_contiguous_span_of_non_special_tokens():
    emb, sp, attn = _batch(B=4, L=10)
    v = _aug(cutoff_span_frac=0.3)(emb, sp, step=10, attention_mask=attn)
    assert v.student_masks is not None and len(v.student_masks) == 2
    for m in v.student_masks:
        for b in range(m.shape[0]):
            dropped = (m[b] == 0).nonzero(as_tuple=True)[0]
            assert dropped.numel() >= 1
            assert dropped.max() - dropped.min() == dropped.numel() - 1   # 연속
            assert not sp[b][dropped].any()                               # 특수 토큰 불포함


def test_b2_span_length_follows_fraction():
    emb, sp, attn = _batch(B=4, L=22)          # 유효 토큰 20개
    v = _aug(cutoff_span_frac=0.25)(emb, sp, step=10, attention_mask=attn)
    for m in v.student_masks:
        for b in range(m.shape[0]):
            assert int((m[b] == 0).sum()) == 5   # 0.25 * 20


def test_b2_mask_mode_replaces_embeddings_and_keeps_mask():
    emb, sp, attn = _batch(B=3, L=10)
    marker = torch.full((D,), 7.0)
    v = _aug(cutoff_span_frac=0.3, cutoff_mode="mask", mask_embed=marker)(
        emb, sp, step=10, attention_mask=attn)
    for m in v.student_masks:
        assert int((m == 0).sum()) == 0          # mask 모드는 attention_mask를 건드리지 않는다


def test_b2_mask_mode_requires_mask_embed():
    emb, sp, attn = _batch()
    with pytest.raises(ValueError, match="mask_embed"):
        _aug(cutoff_span_frac=0.3, cutoff_mode="mask")(emb, sp, step=10, attention_mask=attn)


def test_b2_requires_attention_mask():
    emb, sp, _ = _batch()
    with pytest.raises(ValueError, match="attention_mask"):
        _aug(cutoff_span_frac=0.3)(emb, sp, step=10)


def test_b2_cutoff_prob_zero_leaves_everything():
    emb, sp, attn = _batch(B=8, L=10)
    v = _aug(cutoff_span_frac=0.3, cutoff_prob=0.0)(emb, sp, step=10, attention_mask=attn)
    for m in v.student_masks:
        assert int((m == 0).sum()) == 0
