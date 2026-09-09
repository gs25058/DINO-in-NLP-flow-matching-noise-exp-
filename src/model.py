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

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """head 전체를 최초 생성 시와 같은 분포로 재초기화한다(R9 주기적 head 리셋).

        expand는 weight_norm parametrization이 걸려 있어 Linear.reset_parameters()를
        그대로 부를 수 없다(계산된 .weight는 leaf가 아니라 in-place 초기화가 원본
        original0/original1에 반영되지 않는다). 그래서 같은 인자로 새 DINOHead를 만들어
        state_dict를 그대로 싣는다 - "새로 만든 모델의 head"와 분포가 같음이 구성상 보장된다.
        """
        fresh = DINOHead(*self.dims).to(next(self.parameters()).device)
        self.load_state_dict(fresh.state_dict())

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.mlp(pooled)
        x = F.normalize(x, p=2, dim=-1)
        return self.expand(x)


class DinoTextModel(nn.Module):
    """backbone + DINO head. forward(inputs_embeds, attention_mask) -> (embedding, logits)."""

    def __init__(self, backbone_name: str, bottleneck_dim: int, logit_dim: int):
        super().__init__()
        config = AutoConfig.from_pretrained(backbone_name)
        for field in _DROPOUT_FIELDS:
            if hasattr(config, field):
                setattr(config, field, 0.0)
        self.backbone = AutoModel.from_pretrained(backbone_name, config=config)
        self.head = DINOHead(config.hidden_size, bottleneck_dim, logit_dim)

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
                embed_push: torch.Tensor | None = None, pooling: str = "last"):
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
            hidden = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
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
