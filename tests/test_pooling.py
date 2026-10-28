from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from src.model import DinoTextModel, EMATeacher, masked_mean_pool


class Backbone(nn.Module):
    def forward(self, inputs_embeds, attention_mask):
        return SimpleNamespace(last_hidden_state=inputs_embeds * 2)


def test_forward_exposes_named_mean_pool_and_logits_and_ignores_padding():
    model = DinoTextModel.__new__(DinoTextModel)
    nn.Module.__init__(model)
    model.backbone = Backbone()
    model.head = nn.Linear(4, 8)
    inputs = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    output = model(inputs, mask)
    expected = masked_mean_pool(inputs * 2, mask)
    torch.testing.assert_close(output.pooled, expected, rtol=0, atol=0)
    torch.testing.assert_close(output.embedding, F.normalize(expected, p=2, dim=-1), rtol=0, atol=0)
    torch.testing.assert_close(output.logits, model.head(expected), rtol=0, atol=0)
    changed = inputs.clone()
    changed[0, 3:] = 10000
    assert torch.equal(model(changed, mask).embedding, output.embedding)


def test_teacher_updates_shared_parameters_once_and_never_receives_gradients():
    student = nn.Module()
    student.backbone = nn.Linear(4, 4, bias=False)
    student.head = nn.Linear(4, 4, bias=False)
    student.head.weight = student.backbone.weight
    teacher = EMATeacher(student, 0.8)
    old = teacher.model.backbone.weight.clone()
    with torch.no_grad():
        student.backbone.weight.add_(1)
    teacher.update(student)
    expected = old * 0.8 + student.backbone.weight * 0.2
    torch.testing.assert_close(teacher.model.backbone.weight, expected, rtol=0, atol=1e-7)
    assert teacher.model.backbone.weight is teacher.model.head.weight
    assert not teacher.model.training and all(
        not p.requires_grad for p in teacher.model.parameters()
    )
