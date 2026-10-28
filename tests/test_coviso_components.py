import copy

import pytest
import torch
from torch.nn import functional as F

from src.loss import CovIsoPenalty, DINOLoss
from src.regularization import combine_coviso, resolve_coviso_weights
from src.train import load_config


def legacy_components(z, mu, m2, momentum):
    mean = momentum * mu.detach() + (1 - momentum) * z.mean(0)
    second = momentum * m2.detach() + (1 - momentum) * (z.T @ z / z.shape[0])
    cov = second - torch.outer(mean, mean)
    return mean.square().sum(), (cov / cov.trace().clamp_min(1e-8) - torch.eye(z.shape[1]) / z.shape[1]).square().sum()


def test_legacy_config_and_equal_weights_preserve_loss_gradients_and_three_updates():
    cfg = load_config("configs/r12_bert_noise_optuna_t35.yaml")["loss"]
    lam = cfg["cov_iso_lambda"]
    assert resolve_coviso_weights(cfg, 0, 4000) == (lam, lam)
    torch.manual_seed(17)
    views = [F.normalize(torch.randn(4, 8), dim=-1).requires_grad_() for _ in range(3)]
    module = CovIsoPenalty(8, cfg["cov_ema_momentum"])
    expected_mu, expected_m2 = module.mu_ema.clone(), module.m2_ema.clone()
    losses, means, covs, expected = [], [], [], []
    for z in views:
        mean, cov = legacy_components(z, expected_mu, expected_m2, module.momentum)
        expected.append(mean + cov)
        expected_mu.mul_(module.momentum).add_(z.detach().mean(0), alpha=1 - module.momentum)
        expected_m2.mul_(module.momentum).add_(z.detach().T @ z.detach() / len(z), alpha=1 - module.momentum)
        loss, aux = module(z, return_components=True)
        losses.append(loss)
        means.append(aux["mean_term"])
        covs.append(aux["cov_term"])
    actual = combine_coviso(losses, means, covs, lam, lam)
    legacy = lam * torch.stack(expected).mean()
    assert torch.equal(actual, legacy)
    assert all(torch.equal(a, b) for a, b in zip(torch.autograd.grad(actual, views, retain_graph=True),
                                               torch.autograd.grad(legacy, views)))
    torch.testing.assert_close(module.mu_ema, expected_mu)
    torch.testing.assert_close(module.m2_ema, expected_m2)


@pytest.mark.parametrize("weights,key", [((0.0, 2.0), "cov_term"), ((2.0, 0.0), "mean_term")])
def test_zero_component_removes_its_gradient(weights, key):
    z = F.normalize(torch.randn(7, 8), dim=-1).requires_grad_()
    loss, aux = CovIsoPenalty(8, 0.8)(z, return_components=True)
    weighted = combine_coviso([loss], [aux["mean_term"]], [aux["cov_term"]], *weights)
    actual, = torch.autograd.grad(weighted, z, retain_graph=True)
    expected, = torch.autograd.grad(2 * aux[key], z)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_previous_statistics_and_teacher_are_detached_and_updated_after_loss():
    module = CovIsoPenalty(8, 0.9)
    module.mu_ema.requires_grad_()
    module.m2_ema.requires_grad_()
    old_mu, old_m2 = module.mu_ema.clone(), module.m2_ema.clone()
    z = F.normalize(torch.randn(5, 8), dim=-1).requires_grad_()
    expected = legacy_components(z, old_mu, old_m2, 0.9)
    loss, aux = module(z, return_components=True)
    torch.testing.assert_close(aux["mean_term"], expected[0])
    torch.testing.assert_close(aux["cov_term"], expected[1])
    teacher_logits = torch.randn(5, 12, requires_grad=True)
    student_logits = torch.randn(5, 12, requires_grad=True)
    ce, _ = DINOLoss(12, 0.9)(teacher_logits, [student_logits], 0.1, 0.2)
    (ce + loss).backward()
    assert teacher_logits.grad is None
    assert module.mu_ema.grad is None and module.m2_ema.grad is None
    assert not aux["L_iso_mean"].requires_grad and not aux["L_iso_cov"].requires_grad
    assert z.grad is not None and student_logits.grad is not None


def test_component_access_does_not_add_ema_updates():
    module = CovIsoPenalty(8, 0.9)
    clone = copy.deepcopy(module)
    calls = []
    original = module._update_ema
    module._update_ema = lambda *args: (calls.append(1), original(*args))[-1]
    for _ in range(3):
        z = F.normalize(torch.randn(5, 8), dim=-1)
        old, _ = clone(z)
        new, _ = module(z, return_components=True)
        assert torch.equal(old, new)
    assert len(calls) == 3
    assert torch.equal(clone.mu_ema, module.mu_ema)
    assert torch.equal(clone.m2_ema, module.m2_ema)


def test_weight_fallback_null_override_zero_and_schedule_boundaries():
    cfg = {"cov_iso_lambda": 5.874093568592943, "cov_iso_mean_lambda": 0, "cov_iso_cov_lambda": None}
    assert resolve_coviso_weights(cfg, 0, 4000) == (0, cfg["cov_iso_lambda"])
    cfg.update(cov_iso_mean_lambda=1.0,
               cov_iso_mean_schedule={"start_frac": 0.5, "end_frac": 1.0, "min_ratio": 0.1, "shape": "cosine"})
    assert resolve_coviso_weights(cfg, -1, 4000)[0] == 1.0
    assert resolve_coviso_weights(cfg, 1999, 4000)[0] == 1.0
    assert resolve_coviso_weights(cfg, 2000, 4000)[0] == 1.0
    assert resolve_coviso_weights(cfg, 3999, 4000)[0] == pytest.approx(0.1)
    assert resolve_coviso_weights(cfg, 8000, 4000)[0] == pytest.approx(0.1)
    values = [resolve_coviso_weights(cfg, s, 4000)[0] for s in range(2000, 4000)]
    assert all(b <= a for a, b in zip(values, values[1:]))
    assert all(resolve_coviso_weights(cfg, s, 4000)[1] == cfg["cov_iso_lambda"] for s in (0, 2000, 3999))


def test_split_weights_reject_nonfinite_negative_and_invalid_schedule():
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            resolve_coviso_weights({"cov_iso_mean_lambda": value}, 0, 4000)
    with pytest.raises(ValueError):
        resolve_coviso_weights({"cov_iso_cov_schedule": {"start_frac": 0.9, "end_frac": 0.2}}, 0, 4000)
