"""Independent covariance-isotropy weights, preserving the legacy sum path."""
import math

import torch


def component_weight(initial, schedule, step, max_steps):
    if not math.isfinite(initial) or initial < 0:
        raise ValueError("covariance component weights must be finite and nonnegative")
    if schedule is None:
        return initial
    start_frac, end_frac = schedule.get("start_frac", 0.5), schedule.get("end_frac", 1.0)
    ratio = schedule.get("min_ratio", 0.1)
    if not 0 <= start_frac < end_frac <= 1 or not 0 <= ratio <= 1 or schedule.get("shape", "cosine") != "cosine":
        raise ValueError("invalid covariance component cosine schedule")
    start, end = int(start_frac * max_steps), min(max_steps - 1, int(end_frac * max_steps))
    if end <= start:
        raise ValueError("schedule needs at least two distinct boundary steps")
    progress = min(1.0, max(0.0, (step - start) / (end - start)))
    if progress == 0:
        return initial
    if progress == 1:
        return initial * ratio
    return initial * (ratio + (1 - ratio) * (1 + math.cos(math.pi * progress)) / 2)


def resolve_coviso_weights(loss_cfg, step, max_steps):
    legacy = loss_cfg.get("cov_iso_lambda", 0.0)
    result = []
    for component in ("mean", "cov"):
        value = loss_cfg.get(f"cov_iso_{component}_lambda")
        initial = legacy if value is None else value
        # Preserve the legacy nonpositive gate for old configurations.
        if value is None and initial < 0:
            initial = 0.0
        result.append(component_weight(float(initial), loss_cfg.get(f"cov_iso_{component}_schedule"), step, max_steps))
    return tuple(result)


def combine_coviso(losses, mean_terms, cov_terms, lambda_mean, lambda_cov):
    if lambda_mean == lambda_cov:
        return lambda_mean * torch.stack(losses).mean()
    return lambda_mean * torch.stack(mean_terms).mean() + lambda_cov * torch.stack(cov_terms).mean()


def backbone_grad_norm(loss, parameters):
    """Read a gradient norm without altering .grad, optimizer state, RNG or EMA."""
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squares = [g.detach().float().square().sum() for g in grads if g is not None]
    return torch.stack(squares).sum().sqrt().item() if squares else 0.0
