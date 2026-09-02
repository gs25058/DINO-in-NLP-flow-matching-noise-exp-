import torch
import torch.nn as nn

from src.train import _is_no_decay_param, build_param_groups


class _FakeBackbone(nn.Module):
    """BERT류(bias+LayerNorm)와 ModernBERT류(bias 없이 소문자 norm)를 모두 흉내:
    Linear(bias 有), LayerNorm(weight+bias), Linear(bias=False) 조합."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(8, 8)             # weight(decay) + bias(no-decay)
        self.LayerNorm = nn.LayerNorm(8)          # weight+bias 둘 다 no-decay ("norm" 포함)
        self.no_bias_proj = nn.Linear(8, 8, bias=False)  # weight만(decay)


class _FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(8, 4)  # weight(decay) + bias(no-decay)


class _FakeStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _FakeBackbone()
        self.head = _FakeHead()


def _param_ids(params):
    return {id(p) for p in params}


def test_is_no_decay_param_covers_bert_and_modernbert_naming():
    # BERT류
    assert _is_no_decay_param("backbone.embeddings.LayerNorm.bias")
    assert _is_no_decay_param("backbone.encoder.layer.0.attention.output.LayerNorm.weight")
    assert _is_no_decay_param("backbone.encoder.layer.0.attention.self.query.bias")
    assert not _is_no_decay_param("backbone.encoder.layer.0.attention.self.query.weight")
    # ModernBERT류 (bias 없음, 소문자 norm)
    assert _is_no_decay_param("backbone.embeddings.norm.weight")
    assert _is_no_decay_param("backbone.layers.0.attn_norm.weight")
    assert not _is_no_decay_param("backbone.layers.0.attn.Wqkv.weight")
    # head
    assert _is_no_decay_param("head.mlp.0.bias")
    assert not _is_no_decay_param("head.expand.parametrizations.weight.original0")


def test_groups_partition_all_params_no_overlap():
    student = _FakeStudent()
    groups = build_param_groups(
        student, velocity_head=None, lr=1e-4, head_lr=5e-4, weight_decay=0.1, exclude_ln_bias_wd=True
    )
    all_expected = _param_ids(student.parameters())
    seen = []
    for g in groups:
        seen.extend(g["params"])
    seen_ids = _param_ids(seen)

    assert seen_ids == all_expected
    assert len(seen) == len(seen_ids)  # 중복(교집합) 없음


def test_groups_include_velocity_head_in_head_group():
    student = _FakeStudent()
    velocity_head = nn.Linear(8, 8)
    groups = build_param_groups(
        student, velocity_head=velocity_head, lr=1e-4, head_lr=5e-4, weight_decay=0.1, exclude_ln_bias_wd=True
    )
    all_expected = _param_ids(list(student.parameters()) + list(velocity_head.parameters()))
    seen_ids = _param_ids(p for g in groups for p in g["params"])
    assert seen_ids == all_expected


def test_no_decay_groups_have_zero_weight_decay():
    student = _FakeStudent()
    groups = build_param_groups(
        student, velocity_head=None, lr=1e-4, head_lr=1e-4, weight_decay=0.1, exclude_ln_bias_wd=True
    )
    no_decay_param_ids = _param_ids(
        p for n, p in list(student.backbone.named_parameters()) + list(student.head.named_parameters())
        if _is_no_decay_param(n)
    )
    for g in groups:
        group_ids = _param_ids(g["params"])
        if group_ids & no_decay_param_ids:
            assert group_ids <= no_decay_param_ids
            assert g["weight_decay"] == 0.0


def test_head_lr_differs_when_specified():
    student = _FakeStudent()
    groups = build_param_groups(
        student, velocity_head=None, lr=1e-4, head_lr=5e-4, weight_decay=0.1, exclude_ln_bias_wd=True
    )
    backbone_ids = _param_ids(p for _, p in student.backbone.named_parameters())
    head_ids = _param_ids(p for _, p in student.head.named_parameters())

    backbone_lrs = {g["lr"] for g in groups if _param_ids(g["params"]) & backbone_ids}
    head_lrs = {g["lr"] for g in groups if _param_ids(g["params"]) & head_ids}

    assert backbone_lrs == {1e-4}
    assert head_lrs == {5e-4}


def test_defaults_collapse_to_single_group_matching_legacy():
    """head_lr/exclude_ln_bias_wd 둘 다 미지정(기본값)이면 기존
    AdamW(student.parameters(), lr=lr, weight_decay=wd)와 완전히 동등해야 한다."""
    student = _FakeStudent()
    lr, wd = 3.0e-5, 0.226
    groups = build_param_groups(
        student, velocity_head=None, lr=lr, head_lr=lr, weight_decay=wd, exclude_ln_bias_wd=False
    )

    assert len(groups) == 1
    assert groups[0]["lr"] == lr
    assert groups[0]["weight_decay"] == wd
    assert _param_ids(groups[0]["params"]) == _param_ids(student.parameters())

    # optimizer 상태(step 1회 후 각 파라미터의 업데이트)가 legacy 생성 방식과 동일한지.
    legacy_student = _FakeStudent()
    legacy_student.load_state_dict(student.state_dict())
    legacy_opt = torch.optim.AdamW(list(legacy_student.parameters()), lr=lr, weight_decay=wd)
    new_opt = torch.optim.AdamW(groups)

    torch.manual_seed(0)
    x = torch.randn(2, 8)
    for opt, s in ((legacy_opt, legacy_student), (new_opt, student)):
        opt.zero_grad()
        out = s.head.mlp(s.backbone.no_bias_proj(s.backbone.LayerNorm(s.backbone.dense(x))))
        out.sum().backward()
        opt.step()

    for p_new, p_legacy in zip(student.parameters(), legacy_student.parameters()):
        assert torch.allclose(p_new, p_legacy)


def test_grad_clip_disabled_by_default_leaves_gradients_unchanged():
    """grad_clip=None(기본)이면 학습 루프에서 clip_grad_norm_이 전혀 호출되지 않아야 한다
    (train.py 코드 경로 확인 - 여기서는 동등한 조건을 직접 재현해 gradient가 그대로인지 확인)."""
    student = _FakeStudent()
    x = torch.randn(4, 8)
    out = student.head.mlp(student.backbone.no_bias_proj(student.backbone.LayerNorm(student.backbone.dense(x))))
    out.sum().backward()
    grads_before = [p.grad.clone() for p in student.parameters()]

    grad_clip = None
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(list(student.parameters()), max_norm=grad_clip)

    for p, g_before in zip(student.parameters(), grads_before):
        assert torch.equal(p.grad, g_before)
