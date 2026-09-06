import torch
import torch.nn.functional as F

from src.loss import CovIsoPenalty


def test_cov_iso_lambda_zero_leaves_total_loss_unchanged():
    """train.py 통합 계약: cov_iso_lambda<=0이면 모듈을 아예 호출하지 않는다 - koleo_lambda의
    기존 `if koleo_lambda > 0:` 게이트와 동일한 패턴. total_loss도, EMA 버퍼도 그대로여야 한다."""
    torch.manual_seed(0)
    d = 8
    z = F.normalize(torch.randn(16, d), dim=-1)
    cov_iso_lambda = 0.0
    penalty = CovIsoPenalty(embed_dim=d, momentum=0.99)
    base_loss = torch.tensor(1.2345)

    total_loss = base_loss
    if cov_iso_lambda > 0:
        l_iso, _ = penalty(z)
        total_loss = total_loss + cov_iso_lambda * l_iso

    assert torch.equal(total_loss, base_loss)
    assert torch.equal(penalty.mu_ema, torch.zeros(d))
    assert torch.equal(penalty.m2_ema, torch.eye(d) / d)


def test_cov_iso_cov_term_small_for_isotropic_large_for_rank1():
    torch.manual_seed(0)
    d = 8

    # isotropic: 큰 B의 정규화된 가우시안은 구면 위에 거의 균등 분포 -> 공분산이 I/d에 근접.
    penalty_iso = CovIsoPenalty(embed_dim=d, momentum=0.0)
    z_iso = F.normalize(torch.randn(4000, d), dim=-1)
    _, aux_iso = penalty_iso(z_iso)

    # rank-1: 모든 표본이 ±v 둘 중 하나 -> 공분산이 정확히 v v^T (rank 1), I/d와 크게 어긋남.
    penalty_rank1 = CovIsoPenalty(embed_dim=d, momentum=0.0)
    v = F.normalize(torch.randn(d), dim=-1)
    signs = torch.sign(torch.randn(2000))
    signs[signs == 0] = 1.0
    z_rank1 = signs.unsqueeze(1) * v.unsqueeze(0)
    _, aux_rank1 = penalty_rank1(z_rank1)

    # raw ||C_norm - I/d||_F^2 (D 곱셈 없음, results/analysis/coviso/report.md 스케일 조정
    # 참고): 등방이면 ~0, 정확한 rank-1(모든 표본이 ±v)이면 1 - 1/d에 근접.
    assert aux_iso["L_iso_cov"] < 0.1
    assert aux_rank1["L_iso_cov"] > aux_iso["L_iso_cov"] * 10
    assert aux_rank1["L_iso_cov"] > (1.0 - 1.0 / d) * 0.9


def test_cov_iso_mean_term_large_for_shifted_small_for_zero_mean():
    torch.manual_seed(1)
    d = 8
    b = 512

    # 공통 방향으로 쏠린 표본 -> mu_blend의 norm이 1에 가까움 -> L_iso_mean 큼.
    v = F.normalize(torch.randn(d), dim=-1)
    z_shifted = F.normalize(v.unsqueeze(0).repeat(b, 1) + 0.01 * torch.randn(b, d), dim=-1)
    penalty_shift = CovIsoPenalty(embed_dim=d, momentum=0.0)
    _, aux_shift = penalty_shift(z_shifted)

    # 대척점 쌍(x, -x) -> 배치 평균이 정확히 0 -> L_iso_mean ~0.
    half = b // 2
    base = torch.randn(half, d)
    z_zero_mean = F.normalize(torch.cat([base, -base], dim=0), dim=-1)
    penalty_zero = CovIsoPenalty(embed_dim=d, momentum=0.0)
    _, aux_zero = penalty_zero(z_zero_mean)

    assert aux_shift["L_iso_mean"] > 0.5
    assert aux_zero["L_iso_mean"] < 0.05


def test_cov_iso_gradient_flows_to_z():
    torch.manual_seed(2)
    d = 8
    z = F.normalize(torch.randn(32, d), dim=-1)
    z.requires_grad_(True)
    penalty = CovIsoPenalty(embed_dim=d, momentum=0.99)

    l_iso, _ = penalty(z)
    l_iso.backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_cov_iso_ema_buffers_update_only_under_no_grad():
    torch.manual_seed(3)
    d = 8
    z = F.normalize(torch.randn(32, d), dim=-1)
    z.requires_grad_(True)
    penalty = CovIsoPenalty(embed_dim=d, momentum=0.9)

    mu_before = penalty.mu_ema.clone()
    m2_before = penalty.m2_ema.clone()

    l_iso, _ = penalty(z)
    l_iso.backward()

    # 버퍼는 forward() 안에서 갱신됐어야 하고(더 이상 초기값이 아님), grad_fn이 전혀 없어야 한다
    # (no_grad로만 갱신됐다는 뜻 - autograd 그래프에 편입되지 않음).
    assert not torch.equal(penalty.mu_ema, mu_before)
    assert not torch.equal(penalty.m2_ema, m2_before)
    assert penalty.mu_ema.grad_fn is None
    assert penalty.m2_ema.grad_fn is None
    assert not penalty.mu_ema.requires_grad
    assert not penalty.m2_ema.requires_grad
