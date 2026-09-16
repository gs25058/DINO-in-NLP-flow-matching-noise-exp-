"""FlowNoiseAug: flow-matching style noise views for text embeddings.

METHOD.md §2–3 이 스펙. 이 파일이 프로젝트의 핵심이며,
학습 루프는 이 인터페이스 밖의 증강 세부를 알아서는 안 된다.
"""
from dataclasses import dataclass
import math

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
    # R12-B2 span cutoff가 켜졌을 때만 채워진다. None이면 학습 루프가 원본 attention_mask를
    # 그대로 쓰므로 기존 동작과 동일하다.
    student_masks: list[torch.Tensor] | None = None
    # R15 토큰 latent 예측: 뷰별 [MASK] 치환 위치 [B, L] bool. 마스킹하지 않는 뷰는 None,
    # 기능이 꺼져 있으면 리스트 자체가 None.
    student_token_masks: list[torch.Tensor | None] | None = None


class FlowNoiseAug:
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor, *,
                 mode: str, num_student_views: int,
                 t_lo: float, t_start: float, t_max: float,
                 warmup_steps: int, delta_t: float,
                 noise_corr_rho: float = 0.0,
                 cutoff_span_frac: float = 0.0, cutoff_prob: float = 1.0,
                 cutoff_mode: str = "drop", mask_embed: torch.Tensor | None = None,
                 token_latent_views: int = 0, mask_ratio: float = 0.15, mask_then_noise: bool = True):
        """mu, sigma: [D], scripts/prepare_data.py가 계산해 캐시한 값.

        R12 신규(둘 다 기본값이 off이며 그때 기존과 bit-identical):
          noise_corr_rho: 토큰 간 노이즈 상관. 0이면 현행(토큰별 독립).
          cutoff_span_frac: student 뷰마다 연속 span을 잘라내는 비율. 0이면 off.
        """
        assert mode in ("anchor", "consistency"), f"unknown mode: {mode}"
        assert 0.0 <= noise_corr_rho <= 1.0, f"noise_corr_rho는 [0,1]: {noise_corr_rho}"
        assert cutoff_mode in ("drop", "mask"), f"unknown cutoff_mode: {cutoff_mode}"
        assert 0 <= token_latent_views <= num_student_views, \
            f"token_latent_views({token_latent_views})는 0~num_student_views({num_student_views})"
        if token_latent_views > 0:
            assert 0.0 < mask_ratio < 1.0, f"mask_ratio는 (0,1): {mask_ratio}"
            assert mask_embed is not None, "token_latent_views > 0이면 mask_embed([MASK] 임베딩)가 필요하다"
        self.mu = mu
        self.sigma = sigma
        self.mode = mode
        self.num_student_views = num_student_views
        self.t_lo = t_lo
        self.t_start = t_start
        self.t_max = t_max
        self.warmup_steps = warmup_steps
        self.delta_t = delta_t
        self.noise_corr_rho = noise_corr_rho
        self.cutoff_span_frac = cutoff_span_frac
        self.cutoff_prob = cutoff_prob
        self.cutoff_mode = cutoff_mode
        self.mask_embed = mask_embed
        self.token_latent_views = token_latent_views
        self.mask_ratio = mask_ratio
        self.mask_then_noise = mask_then_noise

    def t_hi(self, step: int) -> float:
        """curriculum: t_start -> t_max 선형 증가, warmup_steps 이후 고정.
        테스트: 단조 증가, t_max 초과 금지."""
        step = max(step, 0)
        frac = min(1.0, step / self.warmup_steps) if self.warmup_steps > 0 else 1.0
        return self.t_start + (self.t_max - self.t_start) * frac

    def _apply_cutoff(self, embeds: torch.Tensor, special_mask: torch.Tensor,
                      attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """R12-B2: 문장마다 비특수 토큰 중 연속 span 하나를 잘라낸다.

        "drop"은 attention_mask를 0으로 만들어 attention과 pooling 양쪽에서 제외하고,
        "mask"는 해당 위치 임베딩을 [MASK]로 치환한다(pooling에는 포함된다).
        어느 쪽이든 CLS/SEP/pad는 건드리지 않으며, teacher는 항상 깨끗한 원문을 본다.

        토큰별 독립 노이즈와 달리 이 열화는 구조적이라 mean pooling에서 평균되어 사라지지
        않는다 - "pooled 수준에서 실제로 어려운 positive"를 만드는 것이 목적이다.
        (선례: Cutoff, Shen et al. 2020 / ConSERT 증강 계열. 배치 내 다른 문장은 관여하지
        않으므로 문장 정체성은 보존된다.)
        """
        B, L, _ = embeds.shape
        new_mask = attention_mask.clone()
        new_embeds = embeds
        if self.cutoff_mode == "mask":
            new_embeds = embeds.clone()
        valid = (~special_mask) & (attention_mask.bool())
        for b in range(B):
            idx = valid[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            if self.cutoff_prob < 1.0 and float(torch.rand(1)) >= self.cutoff_prob:
                continue
            n_valid = int(idx.numel())
            span = max(1, int(self.cutoff_span_frac * n_valid))
            span = min(span, n_valid)
            first, last = int(idx[0]), int(idx[-1])
            hi = last - span + 1
            start = first if hi <= first else int(torch.randint(first, hi + 1, (1,)))
            sl = slice(start, start + span)
            if self.cutoff_mode == "drop":
                new_mask[b, sl] = 0
            else:
                new_embeds[b, sl] = self.mask_embed.to(device=embeds.device, dtype=embeds.dtype)
        return new_embeds, new_mask

    def _choose_token_mask(self, special_mask: torch.Tensor,
                           attention_mask: torch.Tensor | None) -> torch.Tensor:
        """R15: 문장마다 비특수 토큰의 mask_ratio만큼을 무작위로 고른다 [B, L] bool.

        개수는 문장별 max(1, floor(ratio * n_valid + 0.5))로 정확히 맞춘다(유효 토큰이 0이면 0).
        확률적으로 뽑지 않고 개수를 고정하는 이유는, 짧은 문장에서 마스크가 0개가 되어 그 문장이
        목표에 기여하지 못하는 일을 막기 위해서다. CLS/SEP/pad는 절대 고르지 않는다.
        """
        valid = ~special_mask.bool()
        if attention_mask is not None:
            valid &= attention_mask.bool()
        n_valid = valid.sum(dim=1)
        counts = torch.clamp(torch.floor(n_valid.float() * self.mask_ratio + 0.5), min=1).long()
        counts = torch.where(n_valid > 0, torch.minimum(counts, n_valid), torch.zeros_like(counts))
        scores = torch.rand(valid.shape, device=valid.device)
        scores = scores.masked_fill(~valid, float("inf"))          # 무효 위치는 항상 순위 밖
        ranks = scores.argsort(dim=1).argsort(dim=1)
        return (ranks < counts.unsqueeze(1)) & valid

    def __call__(self, token_embeds: torch.Tensor,
                 special_mask: torch.Tensor, step: int,
                 attention_mask: torch.Tensor | None = None) -> NoiseViews:
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

        def sample_eps() -> torch.Tensor:
            """R12-B1: eps_i = sqrt(rho)*eps_shared + sqrt(1-rho)*eps_i^ind.

            rho=0이면 torch.randn 한 번과 완전히 동일한 소비 패턴이라 기존 run과 bit-identical이다.
            문장당 하나의 eps_shared를 모든 토큰이 공유하므로, mean pooling에서 1/L로 소멸하지
            않고 살아남는다 - 토큰별 독립 노이즈가 pooled 수준에서 사라지는 문제(오류 2)를
            정면으로 겨냥한 형태다. 계수는 단위 분산을 보존한다(rho + (1-rho) = 1).
            """
            if self.noise_corr_rho <= 0.0:
                return torch.randn(B, L, D, device=device, dtype=dtype)
            ind = torch.randn(B, L, D, device=device, dtype=dtype)
            shared = torch.randn(B, 1, D, device=device, dtype=dtype).expand(B, L, D)
            rho = self.noise_corr_rho
            return math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * ind

        def make_view(t_vec: torch.Tensor, eps: torch.Tensor | None = None,
                      x_src: torch.Tensor | None = None):
            """x_src를 주면 그 임베딩(예: [MASK] 치환본)을 표준화해 보간한다. None이면 공유 x_hat -
            기존 경로와 연산·난수 소비가 완전히 같다."""
            if eps is None:
                eps = sample_eps()
            t_b = t_vec.view(B, 1, 1)
            src_hat = x_hat if x_src is None else (x_src - mu) / sigma
            x_t = (1 - t_b) * src_hat + t_b * eps
            x_noised = x_t * sigma + mu
            embeds = torch.where(keep, token_embeds, x_noised)
            return embeds, eps

        student_embeds: list[torch.Tensor] = []
        t_students: list[torch.Tensor] = []
        eps_list: list[torch.Tensor] = []
        student_token_masks = [None] * self.num_student_views if self.token_latent_views > 0 else None
        for k in range(self.num_student_views):
            t_s = sample_t()
            if student_token_masks is not None and k < self.token_latent_views:
                # R15: 이 뷰는 비특수 토큰 일부를 [MASK]로 바꾼다. teacher는 깨끗한 원문이라
                # 마스크 위치의 teacher latent가 곧 회귀 목표다.
                tmask = self._choose_token_mask(special_mask, attention_mask)
                m = self.mask_embed.to(device=device, dtype=dtype)
                if self.mask_then_noise:
                    masked = torch.where(tmask.unsqueeze(-1), m, token_embeds)
                    embeds_s, eps_s = make_view(t_s, x_src=masked)
                else:
                    embeds_s, eps_s = make_view(t_s)
                    embeds_s = torch.where(tmask.unsqueeze(-1), m, embeds_s)
                student_token_masks[k] = tmask
            else:
                embeds_s, eps_s = make_view(t_s)
            student_embeds.append(embeds_s)
            t_students.append(t_s)
            eps_list.append(eps_s)

        student_masks = None
        if self.cutoff_span_frac > 0.0:
            if attention_mask is None:
                raise ValueError("cutoff_span_frac > 0 이면 attention_mask가 필요하다")
            if self.cutoff_mode == "mask" and self.mask_embed is None:
                raise ValueError('cutoff_mode="mask"면 mask_embed가 필요하다')
            student_masks = []
            for k in range(len(student_embeds)):
                e, m = self._apply_cutoff(student_embeds[k], special_mask, attention_mask)
                student_embeds[k] = e
                student_masks.append(m)

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
            student_masks=student_masks,
            student_token_masks=student_token_masks,
        )
