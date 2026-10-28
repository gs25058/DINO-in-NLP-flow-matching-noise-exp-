"""Independent flow-noise student views and an unchanged, clean teacher view."""

from dataclasses import dataclass

import torch


@dataclass
class NoiseViews:
    teacher_embeds: torch.Tensor
    student_embeds: list[torch.Tensor]


class FlowNoiseAug:
    def __init__(self, mu, sigma, *, num_student_views, t_lo, t_start, t_max, warmup_steps):
        self.mu, self.sigma = mu, sigma
        self.num_student_views = num_student_views
        self.t_lo, self.t_start, self.t_max = t_lo, t_start, t_max
        self.warmup_steps = warmup_steps

    def t_hi(self, step):
        progress = min(1.0, max(step, 0) / self.warmup_steps) if self.warmup_steps > 0 else 1.0
        return self.t_start + (self.t_max - self.t_start) * progress

    def __call__(self, token_embeds, special_mask, step) -> NoiseViews:
        """x̂=(x−μ)/σ; x'=[(1−t)x̂+tε]σ+μ. CLS/SEP/pad keep their input.

        Draw t[B], then ε[B,L,D], separately for each view. This order is part
        of the experiment's RNG trajectory, including at noise level zero.
        """
        batch, length, dim = token_embeds.shape
        device, dtype = token_embeds.device, token_embeds.dtype
        mu = self.mu.to(device=device, dtype=dtype)
        sigma = self.sigma.to(device=device, dtype=dtype)
        standardized = (token_embeds - mu) / sigma
        students = []
        for _ in range(self.num_student_views):
            t = self.t_lo + (self.t_hi(step) - self.t_lo) * torch.rand(
                batch, device=device, dtype=dtype
            )
            eps = torch.randn(batch, length, dim, device=device, dtype=dtype)
            t = t.view(batch, 1, 1)
            noised = ((1 - t) * standardized + t * eps) * sigma + mu
            students.append(torch.where(special_mask.unsqueeze(-1), token_embeds, noised))
        return NoiseViews(token_embeds, students)
