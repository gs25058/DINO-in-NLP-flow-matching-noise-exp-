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


class DINOHead(nn.Module):
    """MLP bottleneck → L2 normalization → weight-normalized prototypes."""

    def __init__(self, in_dim, bottleneck_dim, logit_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, bottleneck_dim)
        )
        self.expand = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, logit_dim, bias=False)
        )

    def forward(self, pooled):
        return self.expand(F.normalize(self.mlp(pooled), p=2, dim=-1))


@dataclass
class ModelOutput:
    embedding: torch.Tensor
    logits: torch.Tensor
    pooled: torch.Tensor


class DinoTextModel(nn.Module):
    """The sentence embedding is mean+L2; the head predicts teacher prototypes."""

    def __init__(self, backbone_name, bottleneck_dim, logit_dim):
        super().__init__()
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

    def forward(self, inputs_embeds, attention_mask) -> ModelOutput:
        hidden = self.backbone(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        ).last_hidden_state
        pooled = masked_mean_pool(hidden, attention_mask)
        return ModelOutput(F.normalize(pooled, p=2, dim=-1), self.head(pooled), pooled)


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
