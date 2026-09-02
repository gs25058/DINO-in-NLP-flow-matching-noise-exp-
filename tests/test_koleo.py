import torch
import torch.nn.functional as F

from src.loss import koleo_loss


def test_koleo_larger_for_clustered_than_spread_embeddings():
    torch.manual_seed(0)
    B, D = 16, 8

    # 뭉친 임베딩: 한 점 근처에 작은 노이즈만 추가 -> 최근접 이웃 거리 d_i가 작음
    base = F.normalize(torch.randn(1, D), dim=-1)
    clustered = F.normalize(base + torch.randn(B, D) * 0.001, dim=-1)

    # 구면에 고르게 퍼진 임베딩 -> d_i가 더 큼
    spread = F.normalize(torch.randn(B, D), dim=-1)

    l_clustered = koleo_loss(clustered).item()
    l_spread = koleo_loss(spread).item()

    assert l_clustered > l_spread


def test_koleo_lambda_zero_does_not_change_total_loss():
    """train.py의 게이팅(if koleo_lambda > 0)과 동일하게, lambda=0이면 total_loss에 어떤
    영향도 주지 않아야 한다 (기존 config 재현성 보존)."""
    torch.manual_seed(1)
    z = F.normalize(torch.randn(8, 16), dim=-1)
    base_loss = torch.tensor(1.2345)
    koleo_lambda = 0.0

    l_koleo = koleo_loss(z)
    total = base_loss + koleo_lambda * l_koleo

    assert torch.equal(total, base_loss)


def test_koleo_no_nan_or_inf_with_duplicate_embeddings():
    torch.manual_seed(2)
    B, D = 10, 4
    z = F.normalize(torch.randn(B, D), dim=-1)
    z[1] = z[0]  # 정확히 중복인 문장 쌍 (d_i = 0)

    loss = koleo_loss(z)

    assert torch.isfinite(loss)


def test_koleo_gradient_pushes_points_apart():
    """두 점 토이 케이스: gradient descent 한 스텝 후 두 점 사이 거리가 실제로 늘어나는지 확인."""
    torch.manual_seed(3)
    theta = 0.05  # 아주 가까운 두 점(단위원 위)
    z = torch.tensor([[1.0, 0.0], [float(torch.cos(torch.tensor(theta))), float(torch.sin(torch.tensor(theta)))]],
                      requires_grad=True)
    dist_before = (z[0] - z[1]).detach().norm().item()

    loss = koleo_loss(z)
    (grad,) = torch.autograd.grad(loss, z)

    lr = 1e-3
    z_after = (z - lr * grad).detach()
    dist_after = (z_after[0] - z_after[1]).norm().item()

    assert dist_after > dist_before
