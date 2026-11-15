from dataclasses import dataclass

import torch


@dataclass
class NoiseViews:
    teacher_embeds: torch.Tensor
    student_embeds: list[torch.Tensor]


class FlowNoiseAug:
    def __init__(self, mu, sigma, *, num_student_views, t_lo, t_start, t_max, warmup_steps,
                 weak_view=None):
        self.mu, self.sigma = mu, sigma
        self.num_student_views = num_student_views
        self.t_lo, self.t_start, self.t_max = t_lo, t_start, t_max
        self.warmup_steps = warmup_steps
        self.weak_view = weak_view
        if weak_view is not None:
            if set(weak_view) != {"t_lo", "t_max"}:
                raise ValueError("weak_view requires t_lo and t_max")
            if num_student_views < 2 or not 0 <= weak_view["t_lo"] <= weak_view["t_max"] <= 1:
                raise ValueError("weak_view requires valid bounds and at least two student views")

    def t_hi(self, step):
        progress = min(1.0, max(step, 0) / self.warmup_steps) if self.warmup_steps > 0 else 1.0
        return self.t_start + (self.t_max - self.t_start) * progress

    def __call__(self, token_embeds, special_mask, step) -> NoiseViews:
        batch, length, dim = token_embeds.shape
        device, dtype = token_embeds.device, token_embeds.dtype
        mu = self.mu.to(device=device, dtype=dtype)
        sigma = self.sigma.to(device=device, dtype=dtype)
        standardized = (token_embeds - mu) / sigma
        students = []
        for view_index in range(self.num_student_views):
            lo, hi = self.t_lo, self.t_hi(step)
            if view_index == 0 and self.weak_view is not None:
                lo, hi = self.weak_view["t_lo"], self.weak_view["t_max"]
            t = lo + (hi - lo) * torch.rand(
                batch, device=device, dtype=dtype
            )
            eps = torch.randn(batch, length, dim, device=device, dtype=dtype)
            t = t.view(batch, 1, 1)
            noised = ((1 - t) * standardized + t * eps) * sigma + mu
            students.append(torch.where(special_mask.unsqueeze(-1), token_embeds, noised))
        return NoiseViews(token_embeds, students)
