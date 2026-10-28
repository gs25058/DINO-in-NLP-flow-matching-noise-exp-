import torch
from torch import nn

from src.train import build_param_groups


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        # Match BERT's named normalization convention.
        self.backbone = nn.ModuleDict({"dense": nn.Linear(8, 8), "LayerNorm": nn.LayerNorm(8)})
        self.head = nn.Linear(8, 4)


def test_groups_preserve_parameter_order_coverage_learning_rates_and_decay():
    model = Student()
    groups = build_param_groups(model, 1e-4, 5e-4, 0.2)
    assert [(g["lr"], g["weight_decay"]) for g in groups] == [
        (1e-4, 0.2),
        (1e-4, 0),
        (5e-4, 0.2),
        (5e-4, 0),
    ]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert groups[0]["params"] == [model.backbone["dense"].weight]
    assert [id(p) for p in groups[1]["params"]] == [
        id(model.backbone["dense"].bias),
        id(model.backbone["LayerNorm"].weight),
        id(model.backbone["LayerNorm"].bias),
    ]


def test_adamw_applies_group_settings_to_actual_parameter_updates():
    model = Student()
    groups = build_param_groups(model, 1e-4, 5e-4, 0.2)
    before = {id(p): p.detach().clone() for p in model.parameters()}
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    torch.optim.AdamW(groups).step()
    for group in groups:
        for p in group["params"]:
            expected = before[id(p)] * (1 - group["lr"] * group["weight_decay"]) - group["lr"] / (
                1 + 1e-8
            )
            torch.testing.assert_close(p, expected, rtol=1e-6, atol=1e-7)
