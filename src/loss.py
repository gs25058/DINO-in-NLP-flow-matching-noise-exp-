"""DINO self-distillation loss (centering/sharpening) + 보조 velocity loss.

METHOD.md §4.1, 식 (6)(7)(10)(11): 주 loss.
METHOD.md §4.2: 보조 velocity loss (R4에서만 사용).

학습 루프는 이 파일이 반환하는 (loss, aux_dict)만 소비한다 (CLAUDE.md 구현 원칙 4).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """teacher centering + student/teacher 비대칭 온도의 DINO cross-entropy.

        c <- m*c + (1-m)*mean_batch(l_t)                  (식 6)
        p_t = softmax((l_t - c) / tau_t), stop-grad         (식 7, 10)
        p_s = softmax(l_s / tau_s)                          (식 10)
        L = CE(p_t, p_s) = H(p_t) + KL(p_t || p_s)          (식 11, 12)

    K개 student 뷰가 있으면 CE를 뷰 평균한다.
    """

    def __init__(self, logit_dim: int, center_momentum: float):
        super().__init__()
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(logit_dim))

    @torch.no_grad()
    def _update_center(self, teacher_logits: torch.Tensor) -> None:
        batch_mean = teacher_logits.mean(dim=0)
        self.center.mul_(self.center_momentum).add_(batch_mean, alpha=1.0 - self.center_momentum)

    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: list[torch.Tensor],
        teacher_temp: float,
        student_temp: float,
        update_center: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        teacher_logits: [B, logit_dim] (단일 teacher 뷰)
        student_logits: K x [B, logit_dim]
        반환: (loss, aux_dict={"H_pt", "KL_pt_ps", "H_p_bar_t"})  배치·K뷰 평균, 로깅용(detached)
        """
        centered = teacher_logits.detach() - self.center
        p_t = F.softmax(centered / teacher_temp, dim=-1)
        log_p_t = torch.log(p_t.clamp_min(1e-12))
        h_pt = -(p_t * log_p_t).sum(dim=-1).mean()  # H(p_t), teacher 온도에만 의존

        # marginal usage entropy H(p_bar_t) (R5, METHOD.md §5): 8192개 prototype 사용 균형 진단.
        # H(p_t)(샘플별 평균)와 별개 지표. 상한 log(logit_dim); 0에 가까우면 소수 prototype만 사용(붕괴).
        p_bar_t = p_t.mean(dim=0)
        h_p_bar_t = -(p_bar_t * torch.log(p_bar_t.clamp_min(1e-12))).sum()

        ce_list = []
        for s_logits in student_logits:
            log_p_s = F.log_softmax(s_logits / student_temp, dim=-1)
            ce = -(p_t * log_p_s).sum(dim=-1).mean()  # H(p_t, p_s)
            ce_list.append(ce)

        loss = torch.stack(ce_list).mean()
        kl_pt_ps = loss.detach() - h_pt.detach()  # CE - H = KL(p_t||p_s)

        if update_center:
            self._update_center(teacher_logits)

        aux = {"H_pt": h_pt.detach(), "KL_pt_ps": kl_pt_ps, "H_p_bar_t": h_p_bar_t.detach()}
        return loss, aux


def batch_kl_diagnostic(
    student_logits: list[torch.Tensor], student_temp: float, n_pairs: int = 256
) -> torch.Tensor:
    """METHOD.md §5 batch-KL: 배치 내 모든 student 뷰 출력(N=B*K개) 중 무작위 n_pairs 샘플,
    대칭화 KL 평균. →0이면 조건부 붕괴 경보."""
    all_logits = torch.cat(student_logits, dim=0)  # [N, logit_dim]
    probs = F.softmax(all_logits / student_temp, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    n = probs.shape[0]
    idx_i = torch.randint(0, n, (n_pairs,), device=probs.device)
    idx_j = torch.randint(0, n, (n_pairs,), device=probs.device)
    p_i, p_j = probs[idx_i], probs[idx_j]
    lp_i, lp_j = log_probs[idx_i], log_probs[idx_j]
    kl_ij = (p_i * (lp_i - lp_j)).sum(dim=-1)
    kl_ji = (p_j * (lp_j - lp_i)).sum(dim=-1)
    return ((kl_ij + kl_ji) / 2).mean()


def velocity_loss(
    v_pred: torch.Tensor,
    eps: torch.Tensor,
    x_hat: torch.Tensor,
    special_mask: torch.Tensor,
) -> torch.Tensor:
    """METHOD.md §4.2: target v_i = eps_i - x_hat_i (표준화 공간), 비특수·비패딩 토큰만 평균.

    v_pred, eps, x_hat: [B, L, D]
    special_mask: [B, L] bool, True = CLS/SEP/pad (FlowNoiseAug와 동일 규약, 이 위치는 제외)
    """
    target = eps - x_hat
    per_token = ((v_pred - target) ** 2).mean(dim=-1)  # [B, L], 이미 /D
    keep = (~special_mask).to(per_token.dtype)
    denom = keep.sum().clamp_min(1.0)
    return (per_token * keep).sum() / denom
