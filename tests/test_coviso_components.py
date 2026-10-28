import copy

import pytest
import torch
from torch.nn import functional as F

from src.loss import CovIsoPenalty, DINOLoss, combine_coviso
from src.train import load_config


def expected_terms(z, mu, m2, momentum):
    mean = momentum * mu.detach() + (1 - momentum) * z.mean(0)
    second = momentum * m2.detach() + (1 - momentum) * (z.T @ z / z.shape[0])
    cov = second - torch.outer(mean, mean)
    return mean.square().sum(), (
        cov / cov.trace().clamp_min(1e-8) - torch.eye(z.shape[1]) / z.shape[1]
    ).square().sum()


def test_equal_weights_preserve_loss_gradients_and_three_updates():
    cfg = load_config("configs/coviso_relax/control.yaml")["loss"]
    lam = cfg["cov_iso_mean_lambda"]
    torch.manual_seed(17)
    views = [F.normalize(torch.randn(4, 8), dim=-1).requires_grad_() for _ in range(3)]
    module = CovIsoPenalty(8, cfg["cov_ema_momentum"])
    mu, second = module.mu_ema.clone(), module.m2_ema.clone()
    terms, expected = [], []
    for z in views:
        mean, cov = expected_terms(z, mu, second, module.momentum)
        expected.append(mean + cov)
        mu.mul_(module.momentum).add_(z.detach().mean(0), alpha=1 - module.momentum)
        second.mul_(module.momentum).add_(
            z.detach().T @ z.detach() / len(z), alpha=1 - module.momentum
        )
        terms.append(module(z))
    actual = combine_coviso(terms, lam, lam)
    reference = lam * torch.stack(expected).mean()
    assert torch.equal(actual, reference)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            torch.autograd.grad(actual, views, retain_graph=True),
            torch.autograd.grad(reference, views),
        )
    )
    torch.testing.assert_close(module.mu_ema, mu, rtol=0, atol=0)
    torch.testing.assert_close(module.m2_ema, second, rtol=0, atol=0)


@pytest.mark.parametrize("weights,key", [((0.0, 2.0), "cov"), ((2.0, 0.0), "mean")])
def test_zero_component_removes_its_gradient(weights, key):
    z = F.normalize(torch.randn(7, 8), dim=-1).requires_grad_()
    terms = CovIsoPenalty(8, 0.8)(z)
    (actual,) = torch.autograd.grad(combine_coviso([terms], *weights), z, retain_graph=True)
    (expected,) = torch.autograd.grad(2 * getattr(terms, key), z)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_previous_statistics_and_teacher_are_detached_and_updated_after_loss():
    module = CovIsoPenalty(8, 0.9)
    module.mu_ema.requires_grad_()
    module.m2_ema.requires_grad_()
    mu, second = module.mu_ema.clone(), module.m2_ema.clone()
    z = F.normalize(torch.randn(5, 8), dim=-1).requires_grad_()
    expected = expected_terms(z, mu, second, 0.9)
    terms = module(z)
    torch.testing.assert_close(terms.mean, expected[0])
    torch.testing.assert_close(terms.cov, expected[1])
    teacher, student = (
        torch.randn(5, 12, requires_grad=True),
        torch.randn(5, 12, requires_grad=True),
    )
    ce, _ = DINOLoss(12, 0.9)(teacher, [student], 0.1, 0.2)
    (ce + terms.total).backward()
    assert teacher.grad is None and module.mu_ema.grad is None and module.m2_ema.grad is None
    assert z.grad is not None and student.grad is not None


def test_reading_components_does_not_add_ema_updates():
    module = CovIsoPenalty(8, 0.9)
    clone = copy.deepcopy(module)
    count = []
    original = module._update_ema

    def update(*args):
        count.append(1)
        original(*args)

    module._update_ema = update
    for _ in range(3):
        z = F.normalize(torch.randn(5, 8), dim=-1)
        reference = clone(z)
        terms = module(z)
        assert torch.equal(reference.total, terms.total)
        assert torch.equal(terms.total, terms.mean + terms.cov)
    assert len(count) == 3
    assert torch.equal(module.mu_ema, clone.mu_ema) and torch.equal(module.m2_ema, clone.m2_ema)
