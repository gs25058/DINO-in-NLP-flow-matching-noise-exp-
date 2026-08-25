import pytest
import torch

from src.loss import DINOLoss, velocity_loss


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
