import pytest
import torch

from src.loss import DINOLoss, EmbedUniformPush, velocity_loss


def _uniformity(embeds: torch.Tensor) -> float:
    """evaluate.py의 _uniformity와 동일 정의(전체 off-diag 쌍), 테스트 검증용."""
    d2 = torch.cdist(embeds, embeds, p=2).pow(2)
    n = embeds.shape[0]
    off_diag = ~torch.eye(n, dtype=torch.bool)
    return torch.log(torch.exp(-2 * d2[off_diag]).mean().clamp_min(1e-12)).item()


def test_uniform_push_increases_marginal_entropy():
    torch.manual_seed(0)
    B, logit_dim = 16, 32
    logits = torch.randn(B, logit_dim)
    logits[:, 0] += 6.0  # prototype 0을 인위적으로 과대표현시킴

    loss_fn = DINOLoss(logit_dim=logit_dim, center_momentum=0.9,
                        centering="uniform_push", uniform_push_lr=1.0)
    _, aux_before = loss_fn(logits, [logits.clone()], teacher_temp=0.5, student_temp=0.5,
                             update_center=False)
    h_before = aux_before["H_p_bar_t"].item()

    # center 업데이트(EMA + uniform push) 1회 실행
    _, aux_push = loss_fn(logits, [logits.clone()], teacher_temp=0.5, student_temp=0.5, update_center=True)
    assert aux_push["push_grad_norm"] > 0.0

    _, aux_after = loss_fn(logits, [logits.clone()], teacher_temp=0.5, student_temp=0.5,
                            update_center=False)
    h_after = aux_after["H_p_bar_t"].item()

    assert h_after > h_before


def test_uniform_push_zero_lr_matches_plain_ema():
    """uniform_push_lr=0이면 centering='uniform_push'도 순수 EMA와 동일해야 함 (회귀 보존)."""
    torch.manual_seed(1)
    B, logit_dim = 8, 16
    logits = torch.randn(B, logit_dim)

    ema_loss = DINOLoss(logit_dim=logit_dim, center_momentum=0.9)
    push_loss = DINOLoss(logit_dim=logit_dim, center_momentum=0.9,
                          centering="uniform_push", uniform_push_lr=0.0)

    ema_loss(logits, [logits.clone()], teacher_temp=0.5, student_temp=0.5, update_center=True)
    push_loss(logits, [logits.clone()], teacher_temp=0.5, student_temp=0.5, update_center=True)

    assert torch.allclose(ema_loss.center, push_loss.center)


def test_embed_uniform_push_improves_uniformity():
    torch.manual_seed(0)
    B, embed_dim = 16, 8
    embeds = torch.nn.functional.normalize(torch.randn(B, embed_dim) * 0.05 + 1.0, p=2, dim=-1)
    unif_before = _uniformity(embeds)

    push = EmbedUniformPush(embed_dim=embed_dim, lr=1.0)
    grad_norm = push.step(embeds)
    assert grad_norm > 0.0

    shifted = torch.nn.functional.normalize(embeds + push.push, p=2, dim=-1)
    unif_after = _uniformity(shifted)
    assert unif_after < unif_before  # 더 음수 = 더 균일 (evaluate.py 관례와 동일)


def test_embed_uniform_push_zero_lr_is_noop():
    torch.manual_seed(1)
    B, embed_dim = 8, 4
    embeds = torch.randn(B, embed_dim)

    push = EmbedUniformPush(embed_dim=embed_dim, lr=0.0)
    grad_norm = push.step(embeds)

    assert grad_norm == 0.0
    assert torch.all(push.push == 0.0)


def test_kl_near_zero_when_teacher_equals_student():
    torch.manual_seed(0)
    B, logit_dim = 16, 32
    logits = torch.randn(B, logit_dim) * 3.0

    loss_fn = DINOLoss(logit_dim=logit_dim, center_momentum=0.9)
    same_temp = 0.5
    # center는 0으로 초기화된 채 갱신하지 않으므로 p_t == p_s (동일 logits, 동일 온도)
    loss, aux = loss_fn(
        logits, [logits.clone()], teacher_temp=same_temp, student_temp=same_temp,
        update_center=False,
    )

    assert aux["KL_pt_ps"].item() == pytest.approx(0.0, abs=1e-4)
    assert loss.item() == pytest.approx(aux["H_pt"].item(), abs=1e-4)


def test_entropy_increases_with_temperature():
    torch.manual_seed(1)
    B, logit_dim = 16, 32
    logits = torch.randn(B, logit_dim) * 3.0

    loss_fn = DINOLoss(logit_dim=logit_dim, center_momentum=0.9)
    _, aux_low = loss_fn(
        logits, [logits.clone()], teacher_temp=0.05, student_temp=0.05, update_center=False
    )
    _, aux_high = loss_fn(
        logits, [logits.clone()], teacher_temp=5.0, student_temp=5.0, update_center=False
    )

    assert aux_low["H_pt"].item() < aux_high["H_pt"].item()


def test_center_ema_update():
    logit_dim = 8
    loss_fn = DINOLoss(logit_dim=logit_dim, center_momentum=0.9)
    teacher_logits = torch.ones(4, logit_dim) * 2.0

    assert torch.allclose(loss_fn.center, torch.zeros(logit_dim))
    loss_fn(
        teacher_logits, [teacher_logits.clone()], teacher_temp=0.5, student_temp=0.5,
        update_center=True,
    )

    expected_center = 0.9 * torch.zeros(logit_dim) + 0.1 * teacher_logits.mean(dim=0)
    assert torch.allclose(loss_fn.center, expected_center, atol=1e-6)


def test_multi_view_ce_is_averaged_over_k():
    torch.manual_seed(2)
    B, logit_dim = 8, 16
    teacher_logits = torch.randn(B, logit_dim)
    s1 = torch.randn(B, logit_dim)
    s2 = torch.randn(B, logit_dim)

    loss_fn = DINOLoss(logit_dim=logit_dim, center_momentum=0.9)
    loss_multi, _ = loss_fn(
        teacher_logits, [s1, s2], teacher_temp=0.5, student_temp=0.5, update_center=False
    )
    loss1, _ = loss_fn(
        teacher_logits, [s1], teacher_temp=0.5, student_temp=0.5, update_center=False
    )
    loss2, _ = loss_fn(
        teacher_logits, [s2], teacher_temp=0.5, student_temp=0.5, update_center=False
    )

    assert loss_multi.item() == pytest.approx((loss1.item() + loss2.item()) / 2, abs=1e-5)


def test_velocity_loss_zero_when_prediction_matches_target():
    B, L, D = 2, 5, 4
    torch.manual_seed(3)
    eps = torch.randn(B, L, D)
    x_hat = torch.randn(B, L, D)
    special_mask = torch.zeros(B, L, dtype=torch.bool)
    special_mask[:, 0] = True

    target = eps - x_hat
    loss = velocity_loss(target, eps, x_hat, special_mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_velocity_loss_ignores_special_positions():
    B, L, D = 2, 5, 4
    torch.manual_seed(4)
    eps = torch.randn(B, L, D)
    x_hat = torch.randn(B, L, D)
    special_mask = torch.zeros(B, L, dtype=torch.bool)
    special_mask[:, 0] = True

    target = eps - x_hat
    v_pred = target.clone()
    v_pred[:, 0] += 100.0  # special 위치 오차는 무시되어야 함

    loss = velocity_loss(v_pred, eps, x_hat, special_mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
