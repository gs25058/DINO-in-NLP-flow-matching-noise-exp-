"""BERT → masked mean → DINO head; a frozen copy tracks one EMA per step."""

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel


def masked_mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@dataclass
class HeadOutput:
    logits: torch.Tensor
    bottleneck: torch.Tensor


class DINOHead(nn.Module):
    """Normalized MLP bottleneck, optionally expanded to learned prototypes."""

    def __init__(self, in_dim, bottleneck_dim, logit_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, bottleneck_dim)
        )
        self.expand = (
            nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, logit_dim, bias=False))
            if logit_dim is not None
            else None
        )

    def forward(self, pooled):
        bottleneck = F.normalize(self.mlp(pooled), p=2, dim=-1)
        # With no prototypes the normalized bottleneck itself supplies DINO logits.
        logits = self.expand(bottleneck) if self.expand is not None else bottleneck
        return HeadOutput(logits, bottleneck)


@dataclass
class ModelOutput:
    embedding: torch.Tensor
    logits: torch.Tensor
    pooled: torch.Tensor
    bottleneck: torch.Tensor


class DinoTextModel(nn.Module):
    """The sentence embedding is mean+L2; the head predicts teacher prototypes."""

    def __init__(self, backbone_name, bottleneck_dim, logit_dim, pooling_policy="all"):
        super().__init__()
        if pooling_policy not in ("all", "content"):
            raise ValueError("pooling_policy must be all or content")
        self.pooling_policy = pooling_policy
        config = AutoConfig.from_pretrained(backbone_name)
        for field in (
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "classifier_dropout",
        ):
            if hasattr(config, field):
                setattr(config, field, 0.0)
        self.backbone = AutoModel.from_pretrained(backbone_name, config=config)
        self.head = DINOHead(config.hidden_size, bottleneck_dim, logit_dim)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def forward(self, inputs_embeds, attention_mask, pooling_mask=None, token_type_ids=None) -> ModelOutput:
        if getattr(self, "pooling_policy", "all") == "content" and pooling_mask is None:
            raise ValueError("content pooling requires an explicit content mask")
        if attention_mask.ndim == 3:
            # Transformers accepts precomputed 4D additive masks. A 3D bool
            # mask would otherwise be mistaken for a padding mask in newer BERT.
            allowed = attention_mask.bool().unsqueeze(1)
            attention_mask = torch.zeros_like(allowed, dtype=inputs_embeds.dtype).masked_fill(
                ~allowed, torch.finfo(inputs_embeds.dtype).min)
        extra = {} if token_type_ids is None else {"token_type_ids": token_type_ids}
        hidden = self.backbone(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **extra
        ).last_hidden_state
        pooled = masked_mean_pool(hidden, attention_mask if pooling_mask is None else pooling_mask)
        # Keep embedding normalization before the head: backward accumulation
        # order matters when CE and covariance both reach the pooled vector.
        embedding = F.normalize(pooled, p=2, dim=-1)
        head = self.head(pooled)
        return ModelOutput(embedding, head.logits, pooled, head.bottleneck)


class EMATeacher:
    """Deepcopy once; update each distinct student parameter once after AdamW."""

    def __init__(self, student, momentum):
        self.momentum = momentum
        self.model = copy.deepcopy(student).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, student):
        for teacher, parameter in zip(self.model.parameters(), student.parameters()):
            teacher.mul_(self.momentum).add_(parameter, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
