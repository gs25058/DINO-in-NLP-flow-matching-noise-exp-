"""DINO self-distillation loss (centering/sharpening) + 보조 velocity loss.

METHOD.md §4.1, 식 (6)(7)(10)(11): 주 loss.
METHOD.md §4.2: 보조 velocity loss (R4에서만 사용).

학습 루프는 이 파일이 반환하는 (loss, aux_dict)만 소비한다 (CLAUDE.md 구현 원칙 4).
"""
import math

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

    centering="ema"(기본): 식 (6) 그대로.
    centering="uniform_push": 식 (6) EMA 업데이트에 더해, marginal usage entropy
    H(p_bar_t)를 높이는 방향(=KL(p_bar_t||uniform) 감소 방향)의 gradient를 center에
    한 스텝 추가로 더한다 (팀원 제안, centering-uniform-push 브랜치). uniform_push_lr=0
    이면 순수 EMA와 동일 - 기존 config 재현성 유지.
    """

    def __init__(self, logit_dim: int, center_momentum: float,
                 centering: str = "ema", uniform_push_lr: float = 0.0):
        super().__init__()
        assert centering in ("ema", "uniform_push"), f"unknown centering: {centering}"
        self.center_momentum = center_momentum
        self.centering = centering
        self.uniform_push_lr = uniform_push_lr
        self.register_buffer("center", torch.zeros(logit_dim))

    @torch.no_grad()
    def _update_center_ema(self, teacher_logits: torch.Tensor) -> None:
        batch_mean = teacher_logits.mean(dim=0)
        self.center.mul_(self.center_momentum).add_(batch_mean, alpha=1.0 - self.center_momentum)

    def _update_center_uniform_push(self, teacher_logits: torch.Tensor, teacher_temp: float) -> float:
        """식 (6) EMA 업데이트 + H(p_bar_t) 상승 방향 gradient step. push gradient norm 반환(로깅용)."""
        self._update_center_ema(teacher_logits)
        if self.uniform_push_lr <= 0:
            return 0.0
        c = self.center.detach().clone().requires_grad_(True)
        centered = teacher_logits.detach() - c
        p_t = F.softmax(centered / teacher_temp, dim=-1)
        p_bar = p_t.mean(dim=0)
        neg_entropy = (p_bar * torch.log(p_bar.clamp_min(1e-12))).sum()  # 최소화 = H(p_bar_t) 최대화
        (grad,) = torch.autograd.grad(neg_entropy, c)
        with torch.no_grad():
            self.center -= self.uniform_push_lr * grad
        return grad.norm().item()

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

        push_grad_norm = None
        if update_center:
            if self.centering == "uniform_push":
                push_grad_norm = self._update_center_uniform_push(teacher_logits, teacher_temp)
            else:
                self._update_center_ema(teacher_logits)

        # per-view KL(뷰별 t와 짝지어 로깅용, Part B diag_tbin_kl) + p_t/p_bar_t(diag_confidence용).
        # 기본 학습 경로에서는 소비하지 않음 - 존재해도 동작에 영향 없음.
        kl_per_view = [(ce.detach() - h_pt.detach()) for ce in ce_list]

        aux = {
            "H_pt": h_pt.detach(), "KL_pt_ps": kl_pt_ps, "H_p_bar_t": h_p_bar_t.detach(),
            "kl_per_view": kl_per_view, "p_t": p_t.detach(), "p_bar_t": p_bar_t.detach(),
        }
        if push_grad_norm is not None:
            aux["push_grad_norm"] = push_grad_norm
        return loss, aux


class EmbedUniformPush(nn.Module):
    """teacher 쪽 mean-pooled embedding(BERT 통과 직후, DINO head 이전)에 직접 가하는
    uniformity push. centering="uniform_push"(DINOLoss, logit 공간 8192차원, marginal
    usage entropy 기준)와 같은 구조를 embedding 공간(hidden_size차원, Wang & Isola
    uniformity 기준 - evaluate.py의 _uniformity와 동일 정의: log E[exp(-2||x-y||^2)])으로
    옮긴 실험 변형 (팀원 요청, centering-uniform-push 브랜치, R5 구조와 결합해 임시 비교).

    매 스텝: 이번 스텝 teacher embedding(push 반영分 포함, detached)으로 uniformity loss의
    push 방향 gradient를 구해 persistent 벡터를 한 스텝 갱신 -> 다음 스텝 teacher forward의
    mean-pooled 출력에 더해진다(model.py DinoTextModel.forward embed_push 인자).
    lr<=0이면 항상 zero-vector - model.forward에 전달돼도 항등(no-op).
    """

    def __init__(self, embed_dim: int, lr: float):
        super().__init__()
        self.lr = lr
        self.register_buffer("push", torch.zeros(embed_dim))

    def step(self, teacher_embedding: torch.Tensor) -> float:
        if self.lr <= 0:
            return 0.0
        e = self.push.detach().clone().requires_grad_(True)
        shifted = F.normalize(teacher_embedding.detach() + e, p=2, dim=-1)
        d2 = torch.cdist(shifted, shifted, p=2).pow(2)
        n = shifted.shape[0]
        off_diag = ~torch.eye(n, dtype=torch.bool, device=shifted.device)
        # evaluate.py _uniformity와 동일 정의(log E[exp(-2||x-y||^2)]), 배치 내 모든 off-diag 쌍 사용.
        unif_loss = torch.log(torch.exp(-2 * d2[off_diag]).mean().clamp_min(1e-12))
        (grad,) = torch.autograd.grad(unif_loss, e)
        with torch.no_grad():
            self.push -= self.lr * grad
        return grad.norm().item()


class CovIsoPenalty(nn.Module):
    """R7: 배치 공분산을 등방(isotropic)으로 미는 분포 통계량 정규화 (METHOD.md §9 - KoLeo와
    달리 쌍별 반발이 아니라 배치 요약 통계량 하나에만 페널티가 걸리는 형태라 허용 범주).

    z: L2 정규화된 student pooled 임베딩 [B, D] (koleo_loss와 동일 지점 - model.py
    DinoTextModel.forward의 첫 번째 반환값, head 출력 logits이 아님).

    평균 mu와 2차 모멘트 M2를 EMA로 추적하되, 그 EMA 값(detached)을 현재 배치의 (미분 가능한)
    평균/2차 모멘트와 같은 momentum으로 다시 한번 블렌드해 페널티를 계산한다 - 이렇게 하면
    gradient가 오직 "현재 배치가 EMA를 얼마나 밀 수 있는가"라는 (1-m) 몫을 통해서만 z로
    흐르고, 장기 통계는 EMA 절반이 안정적으로 붙잡아준다.
      - mu_blend 항(||mu_blend||^2)은 평가 후처리의 centering(평균을 원점으로)을,
      - C_norm 항(등방 목표 I/D와의 편차)은 PC 제거(스펙트럼 평탄화)를
    각각 학습 중 loss에 내장한 것이다.

    버퍼 갱신은 DINOLoss.center와 동일한 순서를 따른다: forward()는 먼저 (갱신 *전*의) EMA
    버퍼로 페널티를 계산하고, 그 다음에 이번 배치 통계를 no_grad로 버퍼에 반영한다.
    """

    def __init__(self, embed_dim: int, momentum: float):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mu_ema", torch.zeros(embed_dim))
        self.register_buffer("m2_ema", torch.eye(embed_dim) / embed_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        d = z.shape[-1]
        m = self.momentum
        batch_mu = z.mean(dim=0)
        batch_m2 = (z.t() @ z) / z.shape[0]

        mu_blend = m * self.mu_ema.detach() + (1.0 - m) * batch_mu
        m2_blend = m * self.m2_ema.detach() + (1.0 - m) * batch_m2
        cov = m2_blend - torch.outer(mu_blend, mu_blend)
        cov_norm = cov / cov.trace().clamp_min(1e-8)

        target = torch.eye(d, device=z.device, dtype=z.dtype) / d
        l_iso_mean = mu_blend.pow(2).sum()
        # 스케일 조정(캘리브레이션 dry-run, results/analysis/coviso/report.md 참고): 최초 스펙은
        # 여기에 *d를 곱했으나, raw ||C_norm - I/d||_F^2 자체가 이미 rank-1 근처에서 ~1
        # 스케일이라(테스트로 확인) *d를 추가로 곱하면 d=768(hidden_size)에서 최대 ~768까지
        # 뛴다 - λ=1.0 캘리브레이션에서 cov_iso 단독 backbone grad norm이 순수 DINO CE의
        # 300배를 넘어 즉시 rank-1 붕괴(eff_rank 1.6)를 일으키는 것으로 실측 확인됨. *d를
        # 제거해 raw Frobenius 값을 그대로 쓴다 - 이러면 rank-1에서 ~1, 등방에서 ~0로
        # ||mu||^2 ∈ [0,1] 항과 실제로 유사 스케일이 된다(원래 의도한 "D를 곱해 유사 스케일로"
        # 라는 목표는 D를 곱하지 않아야 달성됨 - 최초 스펙의 스케일 추정이 반대 방향이었음).
        l_iso_cov = (cov_norm - target).pow(2).sum()
        l_iso = l_iso_mean + l_iso_cov

        self._update_ema(batch_mu.detach(), batch_m2.detach())

        return l_iso, {"L_iso_mean": l_iso_mean.detach(), "L_iso_cov": l_iso_cov.detach()}

    @torch.no_grad()
    def _update_ema(self, batch_mu: torch.Tensor, batch_m2: torch.Tensor) -> None:
        self.mu_ema.mul_(self.momentum).add_(batch_mu, alpha=1.0 - self.momentum)
        self.m2_ema.mul_(self.momentum).add_(batch_m2, alpha=1.0 - self.momentum)


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


def koleo_loss(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """DINOv2 KoLeo(Kozachenko-Leonenko differential-entropy) 정규화 (r6_koleo 실험,
    METHOD.md 밖 - 학습 중 loss가 평가 후처리(centering+PC 제거)의 일을 대신 하게 만드는
    인과 확인용).

    z: L2 정규화된 student pooled 임베딩 [B, D] (model.py DinoTextModel.forward의 첫 번째
    반환값 embedding과 동일 공간 - head 출력 logits이 아님). 배치 내 각 샘플의 최근접
    이웃까지의 거리 d_i를 키우는(=서로 밀어내는) 방향으로 gradient를 준다.

    최근접 이웃 index는 no_grad로 코사인 유사도(이미 unit norm이므로 내적=코사인)의
    최댓값으로 찾고(자기 자신은 대각선을 -1로 채워 제외), 실제 거리는 z를 통해 미분 가능한
    L2 norm으로 다시 계산한다(DINOv2 원 구현과 동일 - index 선택 자체는 미분 대상이 아님).
    eps는 중복/거의 중복인 문장(d_i≈0)에서 log 발산을 막는다.
    """
    with torch.no_grad():
        dots = z @ z.t()
        n = z.shape[0]
        dots.view(-1)[:: n + 1].fill_(-1.0)  # 대각선(자기 자신) 제외
        nn_idx = dots.argmax(dim=1)
    distances = (z - z[nn_idx]).norm(p=2, dim=-1)
    return -torch.log(distances + eps).mean()


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


class EntropyCtrl:
    """R11-A2: teacher 엔트로피 H(p_t)를 목표 궤적에 맞춰 teacher 온도를 조절하는 폐루프 제어기.

    설계 좌표를 tau_t가 아니라 H_pt로 잡는 이유: 학습이 진행되면 logit 스케일이 자라므로
    같은 tau_t가 같은 "유효 날카로움"을 뜻하지 않는다. 엔트로피는 그 스케일 변화를 흡수한다.

    동작:
      - start 시점의 H_pt EMA를 H_anchor로 고정한다.
      - 목표 H_target은 [start, end] 구간에서 H_anchor -> H_anchor - delta 로 cosine 하강,
        이후 유지.
      - 매 스텝 err = H_ema - H_target 에 대해 tau <- clip(tau * exp(-gain * err), tau_min, tau_max).
        H가 목표보다 높으면(너무 뭉툭) err>0 이라 tau가 내려가 더 날카로워진다.
      - 스텝당 상대 변화는 max_step으로 제한한다(진동 억제).

    개루프(A1 decay)와 달리 "도달점"을 엔트로피로 지정하므로, 같은 H에 도달하는 두 경로의
    결과가 다르면 경로(속도) 의존성이 있다는 뜻이다 - 사전 등록 예측의 비교 항목이다.
    """

    def __init__(self, start_step: int, end_step: int, delta: float, gain: float,
                 tau_min: float, tau_max: float, tau_init: float,
                 max_step: float = 0.01, ema_momentum: float = 0.9):
        if end_step < start_step:
            raise ValueError(f"end_step({end_step})이 start_step({start_step})보다 앞설 수 없다")
        if not 0.0 < tau_min <= tau_max:
            raise ValueError(f"tau 범위가 잘못됨: [{tau_min}, {tau_max}]")
        self.start_step, self.end_step = start_step, end_step
        self.delta, self.gain = delta, gain
        self.tau_min, self.tau_max = tau_min, tau_max
        self.max_step, self.ema_momentum = max_step, ema_momentum
        self.tau = min(max(tau_init, tau_min), tau_max)
        self.h_ema: float | None = None
        self.h_anchor: float | None = None

    def observe(self, h_pt: float) -> None:
        """매 스텝 관측. 제어 시작 전에도 EMA는 돌려두어야 anchor가 안정된다."""
        self.h_ema = h_pt if self.h_ema is None else (
            self.ema_momentum * self.h_ema + (1.0 - self.ema_momentum) * h_pt
        )

    def target(self, step: int) -> float | None:
        """현재 step의 H_target. anchor가 잡히기 전이면 None."""
        if self.h_anchor is None:
            return None
        span = self.end_step - self.start_step
        progress = min(1.0, (step - self.start_step) / span) if span > 0 else 1.0
        progress = min(max(progress, 0.0), 1.0)
        ease = (1 - math.cos(math.pi * progress)) / 2          # cosine 하강
        return self.h_anchor - self.delta * ease

    def update(self, step: int) -> float:
        """이번 스텝에 쓸 tau_t를 돌려준다. start 이전이면 tau_init 그대로."""
        if step < self.start_step:
            return self.tau
        if self.h_anchor is None:
            self.h_anchor = self.h_ema                          # 인계 시점의 EMA로 고정
        err = self.h_ema - self.target(step)
        factor = math.exp(-self.gain * err)
        factor = min(max(factor, 1.0 - self.max_step), 1.0 + self.max_step)   # 스텝당 상대 변화 제한
        self.tau = min(max(self.tau * factor, self.tau_min), self.tau_max)
        return self.tau


class TokenLatentPredictor(nn.Module):
    """R15: student 마지막 층 토큰 hidden -> teacher 토큰 latent 목표 회귀용 2층 MLP (D->D->D, GELU)."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def token_latent_targets(hidden_states, top_k: int) -> torch.Tensor:
    """R15 teacher 목표: 상위 top_k 층 hidden state 평균 후 토큰별 instance norm (data2vec 관례).

    hidden_states: HF output_hidden_states 튜플 (임베딩 출력, 1층, ..., 마지막 층), 각 [B, L, D].
    정규화는 affine 없이 각 토큰 벡터를 D축으로 표준화한다(평균 0, 분산 1). 목표의 스케일이 층마다,
    학습 시점마다 달라지는 것을 막아 회귀가 스케일 맞추기로 흐르지 않게 한다.
    목표로 gradient가 흐르면 안 되므로 항상 detach한다(teacher가 no_grad여도 방어적으로).
    """
    if not 1 <= top_k <= len(hidden_states) - 1:
        raise ValueError(f"top_k({top_k})는 1 이상, 층 수({len(hidden_states) - 1}) 이하여야 한다")
    avg = torch.stack(tuple(hidden_states[-top_k:]), dim=0).mean(dim=0)
    return F.layer_norm(avg, (avg.shape[-1],)).detach()


def token_latent_loss(pred: torch.Tensor, target: torch.Tensor,
                      token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """마스크 위치에서만 smooth-L1(beta=1)을 평균한다. (loss, 마스크 위치 예측 코사인 평균)을 준다.

    pred/target: [B, L, D], token_mask: [B, L] bool. 코사인은 정확도 대리 지표라 no_grad로 잰다.
    마스크 위치가 하나도 없으면 0을 준다(학습 그래프는 유지).
    """
    sel = token_mask.bool()
    if not sel.any():
        zero = pred.sum() * 0.0
        return zero, zero.detach()
    p, y = pred[sel], target[sel]                       # [N, D]
    loss = F.smooth_l1_loss(p, y, beta=1.0, reduction="none").mean(dim=-1).mean()
    with torch.no_grad():
        cos = F.cosine_similarity(p, y, dim=-1).mean()
    return loss, cos


class IBOTTokenLoss(nn.Module):
    """R18 iBOT식 토큰 수준 프로토타입 CE. 마스크 위치에서 teacher 토큰 분포를 student가 맞춘다.

        c_tok <- m*c_tok + (1-m)*mean_tokens(l_t_tok)      문장 center와 별개 버퍼(iBOT 관례)
        p_t = softmax((l_t_tok - c_tok) / tau_t), stop-grad
        p_s = softmax(l_s_tok / tau_s)
        L = mean_{마스크 토큰} CE(p_t, p_s) = H(p_t) + KL(p_t || p_s)

    목표가 raw hidden(R15)이 아니라 학습 중인 head의 분포라 head가 움직이면 목표도 움직인다.
    use_token_center=False면 토큰 로짓을 centering하지 않는다(centering 기여를 떼어 보는 대조용).
    문장 DINOLoss와 같은 순서로, 이번 스텝 분포는 갱신 전 center로 계산하고 center는 그 뒤에 갱신한다.
    """

    def __init__(self, logit_dim: int, center_momentum: float, use_token_center: bool = True):
        super().__init__()
        self.center_momentum = center_momentum
        self.use_token_center = use_token_center
        self.register_buffer("token_center", torch.zeros(logit_dim))

    def forward(
        self,
        teacher_tok_logits: list[torch.Tensor],
        student_tok_logits: list[torch.Tensor],
        teacher_temp: float,
        student_temp: float,
        update_center: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """teacher/student: 마스킹 뷰마다 [N_mask_k, K] (뷰마다 마스크 위치가 달라 teacher도 뷰별이다).

        반환 loss는 뷰별 토큰 평균 CE의 뷰 평균. 마스크 토큰이 하나도 없으면 0(그래프 유지)이고
        center도 갱신하지 않는다.
        """
        assert len(teacher_tok_logits) == len(student_tok_logits)
        pairs = [(t, s) for t, s in zip(teacher_tok_logits, student_tok_logits) if t.shape[0] > 0]
        if not pairs:
            zero = sum((s.sum() * 0.0 for s in student_tok_logits), self.token_center.new_zeros(()))
            nan = self.token_center.new_tensor(float("nan"))
            return zero, {"H_pt_tok": nan, "KL_tok": nan, "H_p_bar_tok": nan, "n_tok": 0}

        center = self.token_center if self.use_token_center else torch.zeros_like(self.token_center)
        ce_list, h_list, all_p_t = [], [], []
        for t_logits, s_logits in pairs:
            p_t = F.softmax((t_logits.detach() - center) / teacher_temp, dim=-1)
            log_p_s = F.log_softmax(s_logits / student_temp, dim=-1)
            ce_list.append(-(p_t * log_p_s).sum(dim=-1).mean())
            h_list.append(-(p_t * torch.log(p_t.clamp_min(1e-12))).sum(dim=-1).mean())
            all_p_t.append(p_t)
        loss = torch.stack(ce_list).mean()
        h_pt = torch.stack(h_list).mean().detach()
        p_bar = torch.cat(all_p_t, dim=0).mean(dim=0)
        h_p_bar = -(p_bar * torch.log(p_bar.clamp_min(1e-12))).sum()

        if update_center and self.use_token_center:
            with torch.no_grad():
                batch_mean = torch.cat([t for t, _ in pairs], dim=0).detach().mean(dim=0)
                self.token_center.mul_(self.center_momentum).add_(batch_mean, alpha=1.0 - self.center_momentum)

        aux = {
            "H_pt_tok": h_pt, "KL_tok": loss.detach() - h_pt, "H_p_bar_tok": h_p_bar.detach(),
            "n_tok": sum(t.shape[0] for t, _ in pairs),
        }
        return loss, aux


class IBOTLambdaCtrl:
    """R18 기울기 비율 균형: backbone 기울기 노름 비 ||lambda*g_ibot|| / ||g_ce||를 ratio에 맞춘다.

    R15 병리(문장 CE 기울기가 스스로 줄어든 뒤 고정 lambda의 보조항이 student 기울기를 지배)에 대한 대응.
    관측할 때마다 목표 lambda* = ratio * ||g_ce|| / ||g_ibot||(g_ibot은 lambda를 곱하기 전)를 구해
    lambda <- m*lambda + (1-m)*lambda* 로 따라가고 [lam_min, lam_max]로 자른다. 첫 관측은 EMA 없이
    lambda*를 바로 쓴다(초깃값 편향 방지). g_ibot이 0이면(마스크 토큰 없음) 이번 관측은 건너뛴다.
    """

    def __init__(self, ratio: float, init_lambda: float, momentum: float = 0.9,
                 lam_min: float = 0.01, lam_max: float = 10.0):
        if ratio <= 0:
            raise ValueError(f"ibot_grad_ratio는 양수여야 한다: {ratio}")
        if not 0.0 < lam_min <= lam_max:
            raise ValueError(f"lambda 범위가 잘못됨: [{lam_min}, {lam_max}]")
        self.ratio, self.momentum = ratio, momentum
        self.lam_min, self.lam_max = lam_min, lam_max
        self.lam = min(max(init_lambda, lam_min), lam_max)
        self.initialized = False

    def observe(self, g_ce_norm: float, g_ibot_norm: float) -> float:
        if not (g_ibot_norm > 0.0) or not math.isfinite(g_ibot_norm) or not math.isfinite(g_ce_norm):
            return self.lam
        target = self.ratio * g_ce_norm / g_ibot_norm
        lam = target if not self.initialized else self.momentum * self.lam + (1.0 - self.momentum) * target
        self.lam = min(max(lam, self.lam_min), self.lam_max)
        self.initialized = True
        return self.lam


def sinkhorn_log(scores: torch.Tensor, n_iters: int) -> torch.Tensor:
    """exp(scores)에 Sinkhorn-Knopp를 적용한 이중확률행렬의 log. 로그 영역이라 overflow가 없다.

    마지막이 열 정규화라 열 합은 정확히 1, 행 합은 반복이 늘수록 1에 수렴한다(SwAV도 3회).
    """
    log_p = scores
    for _ in range(n_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
    return log_p


def gram_sinkhorn_topk(gram: torch.Tensor, n_iters: int = 3, top_k: int = 5) -> tuple[torch.Tensor, dict[str, float]]:
    """gram -> exp -> Sinkhorn-Knopp -> SVD 상위 top_k 특이성분 제거. 결과 행렬을 돌려준다.

    상위 성분 제거는 좌특이벡터 U_k(no_grad)로의 투영을 빼는 방식이다:
    U_k(U_k^T P) = sum_{i<k} sigma_i u_i v_i^T 로 값은 동일하면서 SVD backward의
    특이값 축퇴 불안정을 피한다.
    """
    p = sinkhorn_log(gram, n_iters).exp()
    with torch.no_grad():
        u_full, s_full, _ = torch.linalg.svd(p.detach().double(), full_matrices=False)
        u_k = u_full[:, :top_k].to(p.dtype)
    out = p - u_k @ (u_k.t() @ p)
    with torch.no_grad():
        aux = {
            "gram_sigma1": float(s_full[0]),
            "gram_topk_energy": float((s_full[:top_k] ** 2).sum() / (s_full**2).sum().clamp_min(1e-12)),
            "gram_out_diag": float(out.diag().mean()),
            "gram_out_offdiag": float((out.sum() - out.diag().sum()) / (out.numel() - out.shape[0])),
        }
    return out, aux
