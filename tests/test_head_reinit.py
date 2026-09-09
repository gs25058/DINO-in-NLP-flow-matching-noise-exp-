"""R9 주기적 DINO head 리셋(train.head_reinit_every_steps / head_reinit_scope) 계약 테스트.

scope="sync":         student head + teacher head + center를 함께 step 0 상태로 되돌린다.
scope="student_only": student head만 되돌리고, teacher head와 center(=teacher logit의 EMA)는 보존한다.
두 scope 모두 head의 optimizer state는 비우고, backbone은 한 바이트도 건드리지 않는다.
"""
import copy

import torch
import torch.nn as nn

from src.loss import DINOLoss
from src.model import DINOHead, EMATeacher
from src.train import reinit_dino_head

IN_DIM, BOTTLENECK, LOGIT_DIM = 8, 4, 16


class _TinyStudent(nn.Module):
    """DinoTextModel과 같은 (backbone, head) 구조만 흉내 - HF 백본 다운로드 없이 테스트."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(IN_DIM, IN_DIM), nn.LayerNorm(IN_DIM))
        self.head = DINOHead(IN_DIM, BOTTLENECK, LOGIT_DIM)

    def forward(self, x):
        return self.head(self.backbone(x))


def _build():
    torch.manual_seed(0)
    student = _TinyStudent()
    teacher = EMATeacher(student, momentum=0.9)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    dino_loss = DINOLoss(LOGIT_DIM, center_momentum=0.9)
    return student, teacher, optimizer, dino_loss


def _train_a_few_steps(student, teacher, optimizer, dino_loss, n=3):
    for _ in range(n):
        x = torch.randn(4, IN_DIM)
        loss, _ = dino_loss(teacher.model(x).detach(), [student(x)], 0.1, 0.15)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        teacher.update(student)


def test_reset_parameters_changes_all_head_tensors():
    """weight_norm parametrization(expand.parametrizations.weight.original0/1)까지 포함해
    head state_dict의 모든 텐서가 실제로 새 값으로 바뀌어야 한다 - 조용히 no-op이 되는
    (계산된 .weight를 in-place 초기화하는) 실패 모드를 잡는다."""
    torch.manual_seed(0)
    head = DINOHead(IN_DIM, BOTTLENECK, LOGIT_DIM)
    before = {k: v.clone() for k, v in head.state_dict().items()}
    head.reset_parameters()
    after = head.state_dict()

    assert set(before) == set(after)
    assert any("parametrizations" in k for k in after), "weight_norm 파라미터가 state_dict에 있어야 함"
    for k in before:
        if before[k].numel() == 0 or torch.count_nonzero(before[k]) == 0:
            continue  # 상수 0으로 초기화되는 텐서(bias 등)는 값이 같아도 정상
        assert not torch.equal(before[k], after[k]), f"{k}가 재초기화되지 않았다"


def test_reinit_resets_head_teacher_center_and_optimizer_state_only():
    student, teacher, optimizer, dino_loss = _build()
    _train_a_few_steps(student, teacher, optimizer, dino_loss)

    backbone_before = copy.deepcopy(student.backbone.state_dict())
    t_backbone_before = copy.deepcopy(teacher.model.backbone.state_dict())
    head_before = copy.deepcopy(student.head.state_dict())
    backbone_params = list(student.backbone.parameters())
    assert torch.count_nonzero(dino_loss.center) > 0, "사전 조건: center가 이미 갱신돼 있어야 함"
    assert all(p in optimizer.state for p in backbone_params)

    reinit_dino_head(student, teacher, optimizer, dino_loss)

    # 1. student head는 새 값
    for k, v in student.head.state_dict().items():
        if torch.count_nonzero(head_before[k]) == 0:
            continue
        assert not torch.equal(head_before[k], v), f"student.head.{k}가 그대로다"
    # 2. teacher head == student head (step 0의 deepcopy 관계 복원)
    for k, v in teacher.model.head.state_dict().items():
        assert torch.equal(v, student.head.state_dict()[k])
    # 3. head optimizer state 제거, backbone optimizer state 보존
    assert not any(p in optimizer.state for p in student.head.parameters())
    assert all(p in optimizer.state for p in backbone_params)
    # 4. center는 0으로
    assert torch.count_nonzero(dino_loss.center) == 0
    # 5. backbone(student/teacher)은 불변
    for k, v in student.backbone.state_dict().items():
        assert torch.equal(backbone_before[k], v), f"student.backbone.{k}가 바뀌었다"
    for k, v in teacher.model.backbone.state_dict().items():
        assert torch.equal(t_backbone_before[k], v), f"teacher.backbone.{k}가 바뀌었다"


def test_student_only_scope_keeps_teacher_head_and_center():
    """scope="student_only": student head만 백지가 되고, teacher head와 center는 살아 있어야 한다.
    center는 teacher logit의 EMA라 teacher head와 한 몸 - teacher를 남기면서 center만 0으로
    만들면 살아 있는 teacher의 centering(붕괴 방지 기제)이 망가진다."""
    student, teacher, optimizer, dino_loss = _build()
    _train_a_few_steps(student, teacher, optimizer, dino_loss)

    s_head_before = copy.deepcopy(student.head.state_dict())
    t_head_before = copy.deepcopy(teacher.model.head.state_dict())
    center_before = dino_loss.center.clone()
    assert torch.count_nonzero(center_before) > 0

    reinit_dino_head(student, teacher, optimizer, dino_loss, scope="student_only")

    for k, v in student.head.state_dict().items():
        if torch.count_nonzero(s_head_before[k]) == 0:
            continue
        assert not torch.equal(s_head_before[k], v), f"student.head.{k}가 그대로다"
    for k, v in teacher.model.head.state_dict().items():
        assert torch.equal(t_head_before[k], v), f"teacher.head.{k}가 바뀌었다(보존돼야 함)"
    assert torch.equal(center_before, dino_loss.center), "center가 보존돼야 한다"
    assert not any(p in optimizer.state for p in student.head.parameters())


def test_student_only_produces_larger_kl_than_sync():
    """설계 의도 검증: 동기 리셋은 teacher==student라 KL이 공짜로 0에 가까워지지만,
    student만 리셋하면 학습된 teacher를 다시 맞춰야 해 KL이 확실히 커야 한다."""
    def kl_after(scope):
        student, teacher, optimizer, dino_loss = _build()
        _train_a_few_steps(student, teacher, optimizer, dino_loss, n=20)
        reinit_dino_head(student, teacher, optimizer, dino_loss, scope=scope)
        torch.manual_seed(123)
        x = torch.randn(16, IN_DIM)
        _, aux = dino_loss(teacher.model(x).detach(), [student(x)], 0.1, 0.15, update_center=False)
        return aux["KL_pt_ps"].item()

    kl_sync = kl_after("sync")
    kl_student_only = kl_after("student_only")
    assert kl_student_only > kl_sync, f"student_only={kl_student_only} <= sync={kl_sync}"


def test_unknown_scope_rejected():
    student, teacher, optimizer, dino_loss = _build()
    try:
        reinit_dino_head(student, teacher, optimizer, dino_loss, scope="teacher_only")
    except AssertionError:
        return
    raise AssertionError("알 수 없는 scope는 거부돼야 한다")


def test_teacher_stays_ema_linked_after_reinit():
    """리셋 후에도 teacher.update()가 student head를 정상적으로 따라가야 한다
    (load_state_dict가 파라미터 객체를 교체해 EMA zip이 어긋나는 실패 모드 방지)."""
    student, teacher, optimizer, dino_loss = _build()
    reinit_dino_head(student, teacher, optimizer, dino_loss)
    with torch.no_grad():
        for p in student.head.parameters():
            p.add_(1.0)
    teacher_before = copy.deepcopy(teacher.model.head.state_dict())
    teacher.update(student)
    moved = [
        not torch.equal(teacher_before[k], v)
        for k, v in teacher.model.head.state_dict().items()
        if teacher_before[k].numel() > 0
    ]
    assert any(moved), "teacher head가 EMA로 갱신되지 않았다"
    # 여전히 학습이 진행되어야 한다(리셋 직후 grad 흐름 확인)
    _train_a_few_steps(student, teacher, optimizer, dino_loss, n=1)


def test_reinit_step_condition_matches_config_semantics():
    """train.py 루프 조건과 동일한 식: step 0은 건너뛰고 이후 every N step마다,
    0/미지정이면 한 번도 리셋하지 않는다."""

    def fires(every, max_steps):
        return [s for s in range(max_steps) if every > 0 and s > 0 and s % every == 0]

    assert fires(1000, 3000) == [1000, 2000]
    assert fires(0, 3000) == []
    assert fires(500, 1500) == [500, 1000]
