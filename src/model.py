"""DINO-NLP 모델: ModernBERT backbone + DINO head, teacher EMA.

METHOD.md §4.1: mean pooling(Final Embedding, 평가용) / DINO head(logits, loss 전용).
CLAUDE.md 구현 원칙 3: 노이즈는 inputs_embeds로 주입, backbone forward 코드는 수정하지 않는다.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

# ModernBERT 계열 dropout 필드. 없는 필드는 조용히 건너뛴다(다른 backbone 호환).
_DROPOUT_FIELDS = (
    "attention_dropout",
    "mlp_dropout",
    "embedding_dropout",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "classifier_dropout",
)


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """기존 저장소 이식: mask.float() 가중합 / clamp(min=1e-9) 분모."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class DINOHead(nn.Module):
    """식 (8)(9): bottleneck MLP -> L2 정규화 -> weight-norm expanding -> logits.

    hidden/bottleneck 차원은 문서에 근거가 없어 가볍게 신규 결정(METHOD.md §4.1):
    2-layer MLP bottleneck=256, logit_dim=8192(Table 1)만 고정.
    """

    def __init__(self, in_dim: int, bottleneck_dim: int, logit_dim: int):
        super().__init__()
        self.dims = (in_dim, bottleneck_dim, logit_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, bottleneck_dim),
        )
        self.expand = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, logit_dim, bias=False)
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.mlp(pooled)
        x = F.normalize(x, p=2, dim=-1)
        return self.expand(x)


class DinoTextModel(nn.Module):
    """backbone + DINO head. forward(inputs_embeds, attention_mask) -> (embedding, logits)."""

    def __init__(self, backbone_name: str, bottleneck_dim: int, logit_dim: int,
                 attn_implementation: str | None = None):
        super().__init__()
        config = AutoConfig.from_pretrained(backbone_name)
        for field in _DROPOUT_FIELDS:
            if hasattr(config, field):
                setattr(config, field, 0.0)
        # attn_implementation: "sdpa"를 주면 PyTorch의 scaled_dot_product_attention을 쓴다.
        # None(기본)이면 transformers의 기본 선택에 맡긴다 - 기존 run과 동일 경로.
        kwargs = {"config": config}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        self.backbone = AutoModel.from_pretrained(backbone_name, **kwargs)
        self.head = DINOHead(config.hidden_size, bottleneck_dim, logit_dim)

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
                embed_push: torch.Tensor | None = None, pooling: str = "last",
                return_all_hidden: bool = False):
        """embed_push: teacher 쪽 mean-pooled embedding에 더하는 uniformity push 벡터
        (centering-uniform-push 브랜치, embed_uniform_push_lr 실험용). None이면(기본) 기존과
        완전히 동일 - 기존 config 재현성 유지.

        pooling: "last"(기본, 기존과 동일) | "first_last" | "cls" - 뒤의 둘은 평가 전용
        (evaluate.py 소급 재채점, scripts/compare_simcse.py). "first_last"는 첫 층(embedding
        출력)과 마지막 층 hidden state를 평균한 뒤 mean pooling(BERT-flow류 anisotropy 완화
        트릭). "cls"는 마지막 층의 [CLS] 토큰만 쓴다 - SimCSE 비지도판 공식 평가 방식
        (cls_before_pooler; MLP pooler는 우리가 backbone만 싣기 때문에 자연히 제외된다).
        학습 루프는 이 인자를 넘기지 않으므로 항상 "last" - 재현성 유지."""
        if pooling == "first_last":
            out = self.backbone(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True
            )
            hidden = out.last_hidden_state
            pool_input = (out.hidden_states[0] + out.hidden_states[-1]) / 2
        elif pooling in ("last", "cls"):
            out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                output_hidden_states=return_all_hidden)
            hidden = out.last_hidden_state
            pool_input = hidden
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        # cls는 mask 평균 대신 첫 토큰만 취한다(그 외 경로는 기존과 완전히 동일).
        pooled = pool_input[:, 0] if pooling == "cls" else masked_mean_pool(pool_input, attention_mask)
        if embed_push is not None:
            pooled = pooled + embed_push
        embedding = F.normalize(pooled, p=2, dim=-1)  # Final Embedding (평가/rank 지표)
        logits = self.head(pooled)                     # DINO loss 전용
        # pooled(정규화 전)는 r10 BYOL식 predictor 입력용으로 추가한 4번째 반환값이다.
        # 기존 호출부는 앞 3개만 언패킹하므로 동작은 그대로다.
        if return_all_hidden:
            # R15 토큰 latent 목표용: (임베딩 출력, 1층, ..., 마지막 층). 기본 호출은 4-tuple 그대로다.
            return embedding, logits, hidden, pooled, out.hidden_states
        return embedding, logits, hidden, pooled       # hidden: velocity head(§4.2, R4)용 토큰별 출력(항상 last)


class EMATeacher:
    """teacher = student의 deepcopy, 매 optimizer step EMA 갱신, stop-grad."""

    def __init__(self, student: nn.Module, momentum: float):
        self.momentum = momentum
        self.model = copy.deepcopy(student)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        for t_p, s_p in zip(self.model.parameters(), student.parameters()):
            t_p.data.mul_(self.momentum).add_(s_p.data, alpha=1.0 - self.momentum)

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            return self.model(*args, **kwargs)


class IBOTTokenHeads:
    """R18 iBOT식 토큰 경로: 마스크 위치의 마지막 층 hidden [N_mask, D] -> 프로토타입 로짓 [N_mask, K].

    pooled(문장) 경로는 건드리지 않는다 - 토큰 로짓은 forward가 이미 돌려주는 hidden에서 따로 만든다.

    mode="shared"(기본): 문장 DINO head를 그대로 쓴다. teacher 쪽은 teacher.model.head(이미 EMA 대상).
    mode="separate"(R18b, DINOv2 방식): 별도 DINOHead 인스턴스와 그 EMA 사본을 둔다. student 모듈의
    submodule로 넣지 않는 이유는 checkpoint state_dict를 기존 로더(strict load)와 호환되게 두기 위해서다 -
    train.py가 별도 키로 저장하고 optimizer에는 extra_heads로 넣는다.
    """

    def __init__(self, student: DinoTextModel, teacher: EMATeacher, mode: str = "shared"):
        if mode not in ("shared", "separate"):
            raise ValueError(f"unknown ibot_head: {mode!r} (shared|separate)")
        self.mode = mode
        self._student, self._teacher = student, teacher
        self.head: DINOHead | None = None
        self.teacher_head: EMATeacher | None = None
        if mode == "separate":
            p = next(student.head.parameters())
            self.head = DINOHead(*student.head.dims).to(device=p.device, dtype=p.dtype)
            self.teacher_head = EMATeacher(self.head, momentum=teacher.momentum)

    def trainable_modules(self) -> list[nn.Module]:
        """optimizer에 추가로 넣을 모듈. shared면 문장 head가 이미 들어가 있으므로 없다."""
        return [self.head] if self.head is not None else []

    def student_logits(self, hidden: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        head = self.head if self.head is not None else self._student.head
        return head(hidden[token_mask.bool()])

    @torch.no_grad()
    def teacher_logits(self, hidden: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        head = self.teacher_head.model if self.teacher_head is not None else self._teacher.model.head
        return head(hidden[token_mask.bool()])

    def update(self) -> None:
        """teacher EMA 직후에 부른다. separate head의 EMA를 문장 teacher와 같은 momentum으로 갱신."""
        if self.teacher_head is not None:
            self.teacher_head.momentum = self._teacher.momentum
            self.teacher_head.update(self.head)
