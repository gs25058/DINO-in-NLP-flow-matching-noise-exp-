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
                embed_push: torch.Tensor | None = None):
        """embed_push: teacher 쪽 mean-pooled embedding에 더하는 uniformity push 벡터
        (centering-uniform-push 브랜치, embed_uniform_push_lr 실험용). None이면(기본) 기존과
        완전히 동일 - 기존 config 재현성 유지."""
        hidden = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        pooled = masked_mean_pool(hidden, attention_mask)
        if embed_push is not None:
            pooled = pooled + embed_push
        embedding = F.normalize(pooled, p=2, dim=-1)  # Final Embedding (평가/rank 지표)
        logits = self.head(pooled)                     # DINO loss 전용
        return embedding, logits, hidden                # hidden: velocity head(§4.2, R4)용 토큰별 출력


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
