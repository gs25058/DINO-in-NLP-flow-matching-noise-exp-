"""FlowNoiseAug: flow-matching style noise views for text embeddings.

METHOD.md §2–3 이 스펙. 이 파일이 프로젝트의 핵심이며,
학습 루프는 이 인터페이스 밖의 증강 세부를 알아서는 안 된다.
"""
from dataclasses import dataclass
import torch


@dataclass
class NoiseViews:
    """teacher/student에 넘길 inputs_embeds 뷰 묶음."""
    teacher_embeds: torch.Tensor          # [B, L, D]
    student_embeds: list[torch.Tensor]    # K x [B, L, D]
    t_teacher: torch.Tensor               # [B]
    t_students: list[torch.Tensor]        # K x [B]
    eps: list[torch.Tensor] | None        # velocity loss용 (표준화 공간의 ε)
    x0_std: torch.Tensor | None           # velocity loss용 (표준화된 원본 x̂)


class FlowNoiseAug:
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor, *,
                 mode: str, num_student_views: int,
                 t_lo: float, t_start: float, t_max: float,
                 warmup_steps: int, delta_t: float):
        """mu, sigma: [D], scripts/prepare_data.py가 계산해 캐시한 값."""
        assert mode in ("anchor", "consistency"), f"unknown mode: {mode}"
        self.mu = mu
        self.sigma = sigma
        self.mode = mode
        self.num_student_views = num_student_views
        self.t_lo = t_lo
        self.t_start = t_start
        self.t_max = t_max
        self.warmup_steps = warmup_steps
        self.delta_t = delta_t

    def t_hi(self, step: int) -> float:
        """curriculum: t_start -> t_max 선형 증가, warmup_steps 이후 고정.
        테스트: 단조 증가, t_max 초과 금지."""
        step = max(step, 0)
        frac = min(1.0, step / self.warmup_steps) if self.warmup_steps > 0 else 1.0
        return self.t_start + (self.t_max - self.t_start) * frac

    def __call__(self, token_embeds: torch.Tensor,
                 special_mask: torch.Tensor, step: int) -> NoiseViews:
        """token_embeds: word embedding lookup 출력 [B, L, D] (위치 임베딩 합산 전).
        special_mask: [B, L] bool, True = CLS/SEP/pad (노이즈 제외).

        절차 (METHOD.md §2.1):
          1. x̂ = (x - μ) / σ
          2. 뷰마다 t ~ U(t_lo, t_hi(step)) 문장당 1개, ε ~ N(0, I) 토큰별
          3. x_t = (1-t)·x̂ + t·ε  (special_mask 위치는 원본 유지)
          4. 역표준화 후 반환
        anchor 모드: teacher t=0 (원본 그대로).
        consistency 모드: teacher t = max(0, t_student - Δt), 같은 ε 공유.
        (K개 student 뷰 중 뷰 0과만 Δt로 인접 페어링 — teacher_embeds는 설계상 단일 뷰이므로
        나머지 student 뷰는 multi-crop적 추가 정보 스케일로만 기여한다.)

        테스트 필수 (CLAUDE.md 작업 순서 2):
          - t=0 → 항등
          - t=1 → 표준화 공간 분산 ≈ 1
          - special_mask 위치 불변
        """
        B, L, D = token_embeds.shape
        device, dtype = token_embeds.device, token_embeds.dtype
        mu = self.mu.to(device=device, dtype=dtype)
        sigma = self.sigma.to(device=device, dtype=dtype)

        x_hat = (token_embeds - mu) / sigma
        keep = special_mask.unsqueeze(-1)  # [B, L, 1]

        t_hi_now = self.t_hi(step)

        def sample_t() -> torch.Tensor:
            return self.t_lo + (t_hi_now - self.t_lo) * torch.rand(B, device=device, dtype=dtype)

        def make_view(t_vec: torch.Tensor, eps: torch.Tensor | None = None):
            if eps is None:
                eps = torch.randn(B, L, D, device=device, dtype=dtype)
            t_b = t_vec.view(B, 1, 1)
            x_t = (1 - t_b) * x_hat + t_b * eps
            x_noised = x_t * sigma + mu
            embeds = torch.where(keep, token_embeds, x_noised)
            return embeds, eps

        student_embeds: list[torch.Tensor] = []
        t_students: list[torch.Tensor] = []
        eps_list: list[torch.Tensor] = []
        for _ in range(self.num_student_views):
            t_s = sample_t()
            embeds_s, eps_s = make_view(t_s)
            student_embeds.append(embeds_s)
            t_students.append(t_s)
            eps_list.append(eps_s)

        if self.mode == "anchor":
            t_teacher = torch.zeros(B, device=device, dtype=dtype)
            teacher_embeds = token_embeds
        else:  # consistency
            t_teacher = (t_students[0] - self.delta_t).clamp(min=0.0)
            teacher_embeds, _ = make_view(t_teacher, eps=eps_list[0])

        return NoiseViews(
            teacher_embeds=teacher_embeds,
            student_embeds=student_embeds,
            t_teacher=t_teacher,
            t_students=t_students,
            eps=eps_list,
            x0_std=x_hat,
        )
