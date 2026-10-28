"""Teacher-prototype cross entropy and independently weighted distribution penalties."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class DINOLoss(nn.Module):
    """CE(softmax((teacher−center)/τt), softmax(student/τs)), averaged over views.

    Targets use the previous center. Update it once, after forming the targets;
    neither the teacher nor the stored center receives gradients.
    """

    def __init__(self, logit_dim, center_momentum):
        super().__init__()
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(logit_dim))

    def forward(
        self,
        teacher_logits,
        student_logits,
        teacher_temp,
        student_temp,
        update_center=True,
    ):
        p_t = F.softmax((teacher_logits.detach() - self.center) / teacher_temp, dim=-1)
        h_pt = -(p_t * torch.log(p_t.clamp_min(1e-12))).sum(dim=-1).mean()
        p_bar = p_t.mean(dim=0)
        h_p_bar = -(p_bar * torch.log(p_bar.clamp_min(1e-12))).sum()
        ce = [
            -(p_t * F.log_softmax(logits / student_temp, dim=-1)).sum(dim=-1).mean()
            for logits in student_logits
        ]
        loss = torch.stack(ce).mean()
        if update_center:
            with torch.no_grad():
                self.center.mul_(self.center_momentum).add_(
                    teacher_logits.mean(dim=0), alpha=1 - self.center_momentum
                )
        return loss, {
            "H_pt": h_pt.detach(),
            "KL_pt_ps": loss.detach() - h_pt.detach(),
            "H_p_bar_t": h_p_bar.detach(),
        }


@dataclass
class CovarianceTerms:
    total: torch.Tensor
    mean: torch.Tensor
    cov: torch.Tensor


class CovIsoPenalty(nn.Module):
    """||μ||² + ||C/tr(C)−I/D||²_F on normalized sentence embeddings.

    Blend detached *previous* EMA with this view's differentiable moments,
    then update the buffers. Three views mean three sequential updates, also
    before the regularizer is activated. No extra factor D in the covariance.
    """

    def __init__(self, embed_dim, momentum):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mu_ema", torch.zeros(embed_dim))
        self.register_buffer("m2_ema", torch.eye(embed_dim) / embed_dim)

    def forward(self, z) -> CovarianceTerms:
        dim, momentum = z.shape[-1], self.momentum
        batch_mu = z.mean(dim=0)
        batch_m2 = (z.t() @ z) / z.shape[0]
        mu = momentum * self.mu_ema.detach() + (1.0 - momentum) * batch_mu
        second = momentum * self.m2_ema.detach() + (1.0 - momentum) * batch_m2
        covariance = second - torch.outer(mu, mu)
        normalized = covariance / covariance.trace().clamp_min(1e-8)
        mean = mu.pow(2).sum()
        cov = (normalized - torch.eye(dim, device=z.device, dtype=z.dtype) / dim).pow(2).sum()
        self._update_ema(batch_mu.detach(), batch_m2.detach())
        return CovarianceTerms(mean + cov, mean, cov)

    @torch.no_grad()
    def _update_ema(self, batch_mu, batch_m2):
        self.mu_ema.mul_(self.momentum).add_(batch_mu, alpha=1.0 - self.momentum)
        self.m2_ema.mul_(self.momentum).add_(batch_m2, alpha=1.0 - self.momentum)


def combine_coviso(terms, lambda_mean, lambda_cov):
    # Keep λ·mean(mean_i+cov_i), including its rounding, for the equal-weight control.
    if lambda_mean == lambda_cov:
        return lambda_mean * torch.stack([t.total for t in terms]).mean()
    return (
        lambda_mean * torch.stack([t.mean for t in terms]).mean()
        + lambda_cov * torch.stack([t.cov for t in terms]).mean()
    )


@dataclass
class LossTerms:
    total: torch.Tensor
    dino: torch.Tensor
    mean: torch.Tensor
    cov: torch.Tensor
    lambda_mean: float
    lambda_cov: float
    active: bool
    diagnostics: dict

    def scalars(self):
        mean, cov = self.mean.item(), self.cov.item()
        return {
            "loss": self.total.item(),
            **{k: v.item() for k, v in self.diagnostics.items()},
            "L_DINO": self.dino.item(),
            "L_mean": mean,
            "L_cov": cov,
            "lambda_mean": self.lambda_mean,
            "lambda_cov": self.lambda_cov,
            "weighted_mean": self.lambda_mean * mean if self.active else 0.0,
            "weighted_cov": self.lambda_cov * cov if self.active else 0.0,
        }


class DINOObjective(nn.Module):
    """One objective for every trial: DINO + λmean·mean + λcov·covariance."""

    def __init__(self, cfg, embed_dim, logit_dim):
        super().__init__()
        self.dino = DINOLoss(logit_dim, cfg["center_momentum"])
        self.covariance = CovIsoPenalty(embed_dim, cfg["cov_ema_momentum"])
        self.mean_weight, self.cov_weight = (
            float(cfg[f"cov_iso_{name}_lambda"]) for name in ("mean", "cov")
        )
        if any(not math.isfinite(v) or v < 0 for v in (self.mean_weight, self.cov_weight)):
            raise ValueError("component weights must be finite and nonnegative")
        self.start_step, self.student_temp = (
            cfg["cov_iso_start_step"],
            cfg["student_temp"],
        )

    def forward(
        self, teacher_logits, student_embeddings, student_logits, step, teacher_temp
    ) -> LossTerms:
        terms = [self.covariance(z) for z in student_embeddings]
        dino, diagnostics = self.dino(
            teacher_logits, student_logits, teacher_temp, self.student_temp
        )
        mean = torch.stack([t.mean for t in terms]).mean()
        cov = torch.stack([t.cov for t in terms]).mean()
        active = step >= self.start_step
        total = dino + combine_coviso(terms, self.mean_weight, self.cov_weight) if active else dino
        return LossTerms(
            total,
            dino,
            mean,
            cov,
            self.mean_weight,
            self.cov_weight,
            active,
            diagnostics,
        )


def backbone_grad_norm(loss, parameters):
    """Read a norm without changing .grad, optimizer state, RNG, or EMA."""
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squares = [g.detach().float().square().sum() for g in grads if g is not None]
    return torch.stack(squares).sum().sqrt().item() if squares else 0.0


def batch_kl_diagnostic(student_logits, student_temp, n_pairs=256):
    """Random symmetric-KL pairs. Call at the fixed logging cadence: it consumes RNG."""
    probs = F.softmax(torch.cat(student_logits, dim=0) / student_temp, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    n = probs.shape[0]
    i = torch.randint(0, n, (n_pairs,), device=probs.device)
    j = torch.randint(0, n, (n_pairs,), device=probs.device)
    kl_ij = (probs[i] * (log_probs[i] - log_probs[j])).sum(dim=-1)
    kl_ji = (probs[j] * (log_probs[j] - log_probs[i])).sum(dim=-1)
    return ((kl_ij + kl_ji) / 2).mean()
