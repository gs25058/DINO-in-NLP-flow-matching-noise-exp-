"""R18 iBOT식 토큰 수준 프로토타입 CE 테스트."""
import math

import pytest
import torch
import torch.nn.functional as F

from src.loss import DINOLoss, IBOTLambdaCtrl, IBOTTokenLoss
from src.model import DINOHead, DinoTextModel, EMATeacher, IBOTTokenHeads
from src.train import build_param_groups, ibot_enabled, token_latent_views

D, BOTTLENECK, K = 8, 4, 16


class _Backbone(torch.nn.Module):
    class _Cfg:
        hidden_size = D

    def __init__(self):
        super().__init__()
        self.config = self._Cfg()
        self.proj = torch.nn.Linear(D, D)

    def forward(self, inputs_embeds, attention_mask, output_hidden_states=False):
        out = type("O", (), {})()
        out.last_hidden_state = self.proj(inputs_embeds)
        out.hidden_states = None
        return out


def _model():
    torch.manual_seed(0)
    m = DinoTextModel.__new__(DinoTextModel)
    torch.nn.Module.__init__(m)
    m.backbone = _Backbone()
    m.head = DINOHead(D, BOTTLENECK, K)
    return m


def _hidden_and_mask(B=3, L=7, n_true=5):
    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(B, L, D, generator=g)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask.view(-1)[torch.randperm(B * L, generator=g)[:n_true]] = True
    return hidden, mask


def _cfg(**loss):
    return {"loss": dict(loss), "augment": {}}


# ------------------------------------------------------------------ 켜짐 조건 (ibot_lambda=0 동일)
def test_default_config_keeps_ibot_and_masking_off():
    assert not ibot_enabled(_cfg())
    assert not ibot_enabled(_cfg(ibot_lambda=0.0))
    assert token_latent_views(_cfg(ibot_lambda=0.0)) == 0      # 마스킹 뷰가 없으니 증강도 기존과 동일


def test_ibot_turns_on_masking_by_lambda_or_grad_ratio():
    assert ibot_enabled(_cfg(ibot_lambda=0.3)) and token_latent_views(_cfg(ibot_lambda=0.3)) == 1
    assert ibot_enabled(_cfg(ibot_grad_ratio=0.5)) and token_latent_views(_cfg(ibot_grad_ratio=0.5)) == 1


# ------------------------------------------------------------------ 토큰 head
def test_token_logits_shape_matches_masked_positions():
    student = _model()
    heads = IBOTTokenHeads(student, EMATeacher(student, 0.99), "shared")
    hidden, mask = _hidden_and_mask(n_true=5)
    assert heads.student_logits(hidden, mask).shape == (5, K)
    assert heads.teacher_logits(hidden, mask).shape == (5, K)


def test_shared_head_uses_sentence_heads_and_teacher_is_no_grad():
    student = _model()
    teacher = EMATeacher(student, 0.99)
    heads = IBOTTokenHeads(student, teacher, "shared")
    assert heads.trainable_modules() == []                     # 문장 head가 이미 optimizer에 있다
    hidden, mask = _hidden_and_mask()
    hidden.requires_grad_(True)
    s = heads.student_logits(hidden, mask)
    t = heads.teacher_logits(hidden, mask)
    assert s.requires_grad
    assert not t.requires_grad
    assert torch.allclose(s, student.head(hidden[mask]))
    assert torch.allclose(t, teacher.model.head(hidden[mask]))


def test_separate_head_shares_no_parameters_and_follows_ema():
    student = _model()
    teacher = EMATeacher(student, 0.9)
    heads = IBOTTokenHeads(student, teacher, "separate")
    sep = {p.data_ptr() for p in heads.head.parameters()}
    others = {p.data_ptr() for p in student.parameters()} | {p.data_ptr() for p in teacher.model.parameters()}
    tsep = {p.data_ptr() for p in heads.teacher_head.model.parameters()}
    assert sep.isdisjoint(others) and tsep.isdisjoint(others) and sep.isdisjoint(tsep)
    assert heads.trainable_modules() == [heads.head]
    assert all(not p.requires_grad for p in heads.teacher_head.model.parameters())

    # student 쪽 separate head를 움직이면 teacher 쪽은 문장 teacher의 momentum으로 따라간다
    before = [p.detach().clone() for p in heads.teacher_head.model.parameters()]
    with torch.no_grad():
        for p in heads.head.parameters():
            p.add_(1.0)
    teacher.momentum = 0.5
    heads.update()
    for b, t_p, s_p in zip(before, heads.teacher_head.model.parameters(), heads.head.parameters()):
        assert torch.allclose(t_p, 0.5 * b + 0.5 * s_p)


def test_separate_head_params_reach_optimizer_groups():
    student = _model()
    heads = IBOTTokenHeads(student, EMATeacher(student, 0.99), "separate")
    groups = build_param_groups(student, None, lr=1e-4, head_lr=1e-3, weight_decay=0.01,
                                exclude_ln_bias_wd=False, extra_heads=heads.trainable_modules())
    head_group = {id(p) for p in groups[1]["params"]}
    assert all(id(p) in head_group for p in heads.head.parameters())


def test_unknown_head_mode_raises():
    student = _model()
    with pytest.raises(ValueError, match="ibot_head"):
        IBOTTokenHeads(student, EMATeacher(student, 0.99), "tied")


# ------------------------------------------------------------------ 토큰 CE / center
def test_token_center_is_independent_of_sentence_center():
    torch.manual_seed(0)
    sent = DINOLoss(K, center_momentum=0.9)
    tok = IBOTTokenLoss(K, center_momentum=0.9)
    t_logits = [torch.randn(6, K)]
    tok(t_logits, [torch.randn(6, K)], teacher_temp=0.1, student_temp=0.2)
    assert torch.equal(sent.center, torch.zeros(K))            # 토큰 갱신이 문장 center를 건드리지 않는다
    assert torch.allclose(tok.token_center, 0.1 * t_logits[0].mean(dim=0))

    before = tok.token_center.clone()
    sent(torch.randn(4, K), [torch.randn(4, K)], teacher_temp=0.1, student_temp=0.2)
    assert torch.equal(tok.token_center, before)               # 반대 방향도 독립


def test_loss_uses_pre_update_center_and_decomposes_into_h_plus_kl():
    torch.manual_seed(0)
    tok = IBOTTokenLoss(K, center_momentum=0.8)
    tok.token_center.copy_(torch.randn(K))
    center = tok.token_center.clone()
    t, s = torch.randn(5, K), torch.randn(5, K)
    loss, aux = tok([t], [s], teacher_temp=0.1, student_temp=0.2)

    p_t = F.softmax((t - center) / 0.1, dim=-1)
    ce = -(p_t * F.log_softmax(s / 0.2, dim=-1)).sum(-1).mean()
    h = -(p_t * p_t.clamp_min(1e-12).log()).sum(-1).mean()
    assert torch.allclose(loss, ce, atol=1e-6)
    assert torch.allclose(aux["H_pt_tok"], h, atol=1e-6)
    assert torch.allclose(aux["KL_tok"], ce - h, atol=1e-6)
    p_bar = p_t.mean(0)
    assert torch.allclose(aux["H_p_bar_tok"], -(p_bar * p_bar.log()).sum(), atol=1e-5)
    assert torch.allclose(tok.token_center, 0.8 * center + 0.2 * t.mean(0))


def test_identical_distributions_give_zero_kl():
    tok = IBOTTokenLoss(K, center_momentum=0.9)
    logits = torch.randn(4, K)
    _, aux = tok([logits], [logits.clone()], teacher_temp=0.1, student_temp=0.1)
    assert abs(float(aux["KL_tok"])) < 1e-5


def test_without_token_center_nothing_is_subtracted_or_updated():
    tok = IBOTTokenLoss(K, center_momentum=0.9, use_token_center=False)
    tok.token_center.fill_(3.0)                                 # 켜져 있었다면 결과를 바꿀 값
    t, s = torch.randn(5, K), torch.randn(5, K)
    loss, _ = tok([t], [s], teacher_temp=0.1, student_temp=0.2)
    p_t = F.softmax(t / 0.1, dim=-1)
    assert torch.allclose(loss, -(p_t * F.log_softmax(s / 0.2, dim=-1)).sum(-1).mean(), atol=1e-6)
    assert torch.equal(tok.token_center, torch.full((K,), 3.0))


def test_gradient_flows_to_student_only():
    tok = IBOTTokenLoss(K, center_momentum=0.9)
    t = torch.randn(5, K, requires_grad=True)
    s = torch.randn(5, K, requires_grad=True)
    loss, _ = tok([t], [s], teacher_temp=0.1, student_temp=0.2)
    loss.backward()
    assert t.grad is None and s.grad is not None


def test_no_masked_tokens_gives_differentiable_zero_and_keeps_center():
    tok = IBOTTokenLoss(K, center_momentum=0.9)
    s = torch.zeros(0, K, requires_grad=True)
    loss, aux = tok([torch.zeros(0, K)], [s], teacher_temp=0.1, student_temp=0.2)
    assert float(loss) == 0.0 and aux["n_tok"] == 0 and math.isnan(float(aux["H_pt_tok"]))
    loss.backward()
    assert torch.equal(tok.token_center, torch.zeros(K))


# ------------------------------------------------------------------ 기울기 비율 제어
def test_grad_ratio_first_observation_jumps_to_target_then_ema():
    ctrl = IBOTLambdaCtrl(ratio=0.5, init_lambda=1.0)
    assert math.isclose(ctrl.observe(g_ce_norm=2.0, g_ibot_norm=4.0), 0.25)      # 0.5 * 2 / 4
    assert math.isclose(ctrl.observe(g_ce_norm=4.0, g_ibot_norm=1.0), 0.9 * 0.25 + 0.1 * 2.0)


def test_grad_ratio_lambda_is_clipped():
    hi = IBOTLambdaCtrl(ratio=0.5, init_lambda=1.0)
    assert hi.observe(g_ce_norm=1e6, g_ibot_norm=1.0) == 10.0
    lo = IBOTLambdaCtrl(ratio=0.5, init_lambda=1.0)
    assert lo.observe(g_ce_norm=1e-9, g_ibot_norm=1.0) == 0.01
    for _ in range(50):                                          # 계속 밀어도 범위 안
        assert 0.01 <= lo.observe(g_ce_norm=1e-9, g_ibot_norm=1.0) <= 10.0


def test_grad_ratio_skips_degenerate_observations():
    ctrl = IBOTLambdaCtrl(ratio=0.5, init_lambda=0.3)
    assert ctrl.observe(g_ce_norm=1.0, g_ibot_norm=0.0) == 0.3
    assert ctrl.observe(g_ce_norm=float("nan"), g_ibot_norm=1.0) == 0.3
    assert not ctrl.initialized


def test_grad_ratio_rejects_bad_settings():
    with pytest.raises(ValueError):
        IBOTLambdaCtrl(ratio=0.0, init_lambda=1.0)
    with pytest.raises(ValueError):
        IBOTLambdaCtrl(ratio=0.5, init_lambda=1.0, lam_min=1.0, lam_max=0.1)
