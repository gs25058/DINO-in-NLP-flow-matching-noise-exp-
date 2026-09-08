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


class NbrQueue:
    """r9 L_nbr용 FIFO 큐: teacher anchor 뷰의 pooled 임베딩 + p_t의 top-k + 데이터셋 인덱스.

    loss 계산 *이후*에 push한다(현 배치가 자기 이웃이 되는 것 방지). 전부 no_grad,
    체크포인트에 포함하지 않는다(재시작 시 다시 채워진다).
    """

    def __init__(self, queue_size: int, topk: int, dim: int, device):
        self.size, self.topk = queue_size, topk
        self.z = torch.zeros(queue_size, dim, device=device)
        self.val = torch.zeros(queue_size, topk, device=device)
        self.idx = torch.zeros(queue_size, topk, dtype=torch.long, device=device)
        self.ds_idx = torch.full((queue_size,), -1, dtype=torch.long, device=device)
        self.ptr, self.filled = 0, 0

    @torch.no_grad()
    def push(self, z_t: torch.Tensor, p_t: torch.Tensor, ds_idx: torch.Tensor) -> None:
        val, idx = p_t.topk(self.topk, dim=-1)
        n = z_t.size(0)
        pos = (torch.arange(n, device=z_t.device) + self.ptr) % self.size
        self.z[pos], self.val[pos], self.idx[pos] = z_t.detach(), val, idx
        self.ds_idx[pos] = ds_idx
        self.ptr = (self.ptr + n) % self.size
        self.filled = min(self.filled + n, self.size)

    def view(self):
        """채워진 부분만 반환 (z, val, idx, ds_idx)."""
        n = self.filled
        return self.z[:n], self.val[:n], self.idx[:n], self.ds_idx[:n]


@torch.no_grad()
def sparse_proto_affinity(val_b, idx_b, val_q, idx_q) -> torch.Tensor:
    """W_ij = Σ_k p_i[k]·p_j[k], top-k 인덱스 교집합 위에서만 (희소 내적).

    val_*/idx_*: [N, K] (top-k 값/인덱스). 반환 [B, Q].
    """
    match = idx_b[:, None, :, None] == idx_q[None, :, None, :]      # [B,Q,K,K]
    prod = val_b[:, None, :, None] * val_q[None, :, None, :]
    return (prod * match).sum(dim=(-1, -2))


def nbr_loss(z_students: list[torch.Tensor], queue: NbrQueue, ds_idx: torch.Tensor,
             p_t: torch.Tensor, tau_w: float, max_nbr: int, topk: int,
             tau_w_rel: float | None = None):
    """L_nbr = mean_{valid i} Σ_j W_ij (1 − cos(z_s(x_i^v), sg(z_t(x_j))))  — 끌어당김 전용.

    이웃 선별은 두 방식 중 하나다:
      tau_w_rel is None : W > tau_w (절대 임계, 원 명세)
      tau_w_rel 지정     : W > tau_w_rel * (행별 최대 W)  (스케일 불변)
    절대 임계를 쓸 수 없어 상대 방식을 추가했다 - p_t가 거의 균일해(H≈8.97 vs ln8192=9.01)
    W의 절대 스케일이 ~1e-7이고 학습 상태에 따라 요동치므로, 절대 임계는 어떤 값을 잡아도
    "이웃이 잡히는가"가 tau_w 튜닝 문제로 환원된다.

    반환 (loss, aux, W). W는 행 정규화된 이웃 가중치 [B,Q] (koleo_nbr_overlap 진단용).
    유효 이웃이 하나도 없으면 loss=0(그래디언트 없음)과 진단만 돌려준다.
    """
    z_q, val_q, idx_q, ds_q = queue.view()
    zero = z_students[0].sum() * 0.0  # dtype/device 유지용 0
    aux = {"n_nbr_mean": 0.0, "nbr_valid_frac": 0.0, "nbr_w_max_mean": 0.0, "nbr_w_raw_max": 0.0}
    if z_q.size(0) == 0:
        return zero, aux, None

    with torch.no_grad():
        val_b, idx_b = p_t.topk(topk, dim=-1)
        W = sparse_proto_affinity(val_b, idx_b, val_q, idx_q)          # [B,Q]
        # 같은 문장을 먼저 지운다: 상대 임계의 기준이 되는 행 최대값이 자기 쌍이면 안 된다
        W = W * (ds_idx[:, None] != ds_q[None, :])
        if tau_w_rel is None:
            W = W * (W > tau_w)
        else:
            W = W * (W > tau_w_rel * W.max(dim=1, keepdim=True).values)
        if W.size(1) > max_nbr:                                        # 행별 top-m만 유지
            keep_val, keep_idx = W.topk(max_nbr, dim=1)
            W = torch.zeros_like(W).scatter_(1, keep_idx, keep_val)
        row_sum = W.sum(dim=1, keepdim=True)
        valid = (row_sum.squeeze(1) > 0)
        W = torch.where(row_sum > 0, W / row_sum.clamp_min(1e-12), torch.zeros_like(W))
        n_valid = int(valid.sum())
        aux = {
            "nbr_w_raw_max": float(sparse_proto_affinity(val_b, idx_b, val_q, idx_q).max()),
            "n_nbr_mean": float((W > 0).sum(dim=1)[valid].float().mean()) if n_valid else 0.0,
            "nbr_valid_frac": float(valid.float().mean()),
            "nbr_w_max_mean": float(W[valid].max(dim=1).values.mean()) if n_valid else 0.0,
        }
    if n_valid == 0:
        return zero, aux, W

    per_view = []
    for z_s in z_students:
        cos = z_s @ z_q.t()                       # 양쪽 L2 정규화 -> 내적 = cos, 큐는 stop-grad
        per_view.append(((W * (1.0 - cos)).sum(dim=1)[valid]).mean())
    return torch.stack(per_view).mean(), aux, W


@torch.no_grad()
def koleo_nbr_overlap(z_batch: torch.Tensor, ds_idx: torch.Tensor, queue: NbrQueue,
                      W_rows: torch.Tensor | None) -> float:
    """KoLeo가 미는 배치 내 최근접 이웃이 L_nbr의 이웃 집합에도 들어있는 비율.

    두 항이 같은 쌍을 밀고 당기는지 보는 위험 신호(명세 §로깅). 배치 내 최근접 이웃의
    데이터셋 인덱스가 큐에 아예 없으면 분모에서 제외한다.
    """
    if W_rows is None or queue.filled == 0:
        return float("nan")
    _, _, _, ds_q = queue.view()
    sim = z_batch @ z_batch.t()
    sim.fill_diagonal_(-2.0)
    nn_pos = sim.argmax(dim=1)                    # 배치 내 최근접 이웃 위치
    nn_ds = ds_idx[nn_pos]                        # 그 문장의 데이터셋 인덱스
    num = den = 0
    for i in range(z_batch.size(0)):
        in_queue = (ds_q == nn_ds[i])
        if not bool(in_queue.any()):
            continue                              # 큐에 없으면 분모 제외
        den += 1
        if bool((W_rows[i][in_queue] > 0).any()):
            num += 1
    return num / den if den else float("nan")


@torch.no_grad()
def nbr_cos_overlap(z_batch: torch.Tensor, queue: NbrQueue, W_rows: torch.Tensor | None):
    """프로토타입 이웃 집합이 pooled 코사인 최근접 집합과 얼마나 겹치는가.

    행마다 m = (프로토타입 이웃 수)로 두고, 큐에서 코사인 상위 m개와의 교집합 비율을 잰다.
    1.0이면 두 이웃 정의가 일치, 무작위면 m/Q 수준이다(무작위 기준선도 함께 반환).

    r9에서 KoLeo는 꺼져 있어(koleo_lambda=0) 원 명세의 koleo_nbr_overlap이 정의되지 않으므로,
    "프로토타입 이웃 = 임베딩 이웃인가"를 직접 재는 지표로 대체했다 - P-N3의 판정불가 조건
    ("프로토타입 이웃 ≠ STS 의미 이웃")을 뒷받침하거나 반박하는 근거가 된다.
    """
    if W_rows is None or queue.filled == 0:
        return float("nan"), float("nan")
    z_q = queue.view()[0]
    cos = z_batch @ z_q.t()                       # [B, Q] (양쪽 L2 정규화)
    counts = (W_rows > 0).sum(dim=1)
    num = den = 0
    for i in range(z_batch.size(0)):
        m = int(counts[i])
        if m == 0:
            continue
        top_cos = set(cos[i].topk(m).indices.tolist())
        proto = set((W_rows[i] > 0).nonzero(as_tuple=True)[0].tolist())
        num += len(top_cos & proto)
        den += m
    if den == 0:
        return float("nan"), float("nan")
    rand_baseline = float(counts[counts > 0].float().mean()) / queue.filled
    return num / den, rand_baseline
