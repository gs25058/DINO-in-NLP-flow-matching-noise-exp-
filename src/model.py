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


HEAD_TYPES = ("random", "vocab", "vocab_kmeans")


class VocabHead(nn.Module):
    """MLM-shaped head: transform (dense→GELU→LN) then a decoder over K word prototypes.

    ``vocab``: transform, decoder and bias copied from the pretrained MLM head. The
    decoder is an untied clone, so the backbone's input word embeddings never share
    storage or gradients with it. ``vocab_kmeans``: BERT-style random transform and
    a decoder initialized from L2-normalized k-means centers of the word embeddings.
    The bottleneck is the L2-normalized transform output (diagnostic only).
    """

    def __init__(self, hidden_size, transform, decoder_weight, decoder_bias):
        super().__init__()
        self.transform = transform
        self.decoder = nn.Linear(hidden_size, decoder_weight.shape[0], bias=True)
        with torch.no_grad():
            self.decoder.weight.copy_(decoder_weight)
            self.decoder.bias.copy_(decoder_bias)

    @property
    def expand(self):
        # Evaluation treats the prototype layer as the head-logit space.
        return self.decoder

    def forward(self, pooled):
        hidden = self.transform(pooled)
        return HeadOutput(self.decoder(hidden), F.normalize(hidden, p=2, dim=-1))


def mlm_transform(hidden_size, eps):
    return nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size, eps=eps))


def build_vocab_head(backbone_name, head_type, kmeans_path=None, init_seed=0):
    """Built under a forked CPU RNG so head construction never shifts the noise stream."""
    from transformers import BertForMaskedLM

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        mlm = BertForMaskedLM.from_pretrained(backbone_name)
        predictions = mlm.cls.predictions
        hidden = mlm.config.hidden_size
        transform = mlm_transform(hidden, mlm.config.layer_norm_eps)
        if head_type == "vocab":
            source = predictions.transform
            if mlm.config.hidden_act != "gelu":
                raise ValueError("vocab head expects BERT's exact GELU transform")
            with torch.no_grad():
                transform[0].weight.copy_(source.dense.weight)
                transform[0].bias.copy_(source.dense.bias)
                transform[2].weight.copy_(source.LayerNorm.weight)
                transform[2].bias.copy_(source.LayerNorm.bias)
            weight = predictions.decoder.weight.detach().clone()
            bias = predictions.bias.detach().clone()
        elif head_type == "vocab_kmeans":
            if kmeans_path is None:
                raise ValueError("vocab_kmeans requires model.head.kmeans_path")
            with torch.no_grad():
                nn.init.normal_(transform[0].weight, std=mlm.config.initializer_range)
                nn.init.zeros_(transform[0].bias)
            centers = torch.load(kmeans_path, map_location="cpu", weights_only=True)["centers"]
            weight = F.normalize(centers.float(), p=2, dim=-1)
            bias = torch.zeros(weight.shape[0])
        else:
            raise ValueError(f"unknown vocab head type: {head_type}")
        head = VocabHead(hidden, transform, weight, bias)
        del mlm
    return head


@dataclass
class ModelOutput:
    embedding: torch.Tensor
    logits: torch.Tensor
    pooled: torch.Tensor
    bottleneck: torch.Tensor


class DinoTextModel(nn.Module):
    """The sentence embedding is mean+L2; the head predicts teacher prototypes."""

    def __init__(self, backbone_name, bottleneck_dim, logit_dim, pooling_policy="all",
                 head_type=None, head_freeze=False, kmeans_path=None, head_init_seed=0):
        super().__init__()
        if head_type not in (None, *HEAD_TYPES):
            raise ValueError(f"head_type must be one of {HEAD_TYPES}")
        if head_freeze and head_type in (None, "random"):
            raise ValueError("head_freeze is defined only for word-prototype heads")
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
        if head_type in (None, "random"):
            self.head = DINOHead(config.hidden_size, bottleneck_dim, logit_dim)
            return
        self.head = build_vocab_head(backbone_name, head_type, kmeans_path, head_init_seed)
        if self.head.decoder.out_features != logit_dim or bottleneck_dim != config.hidden_size:
            raise ValueError("vocab head: logit_dim must equal K and bottleneck_dim the hidden size")
        embeddings = self.backbone.get_input_embeddings().weight
        if self.head.decoder.weight.data_ptr() == embeddings.data_ptr():
            raise RuntimeError("decoder must not share storage with the input word embeddings")
        if head_freeze:
            self.head.requires_grad_(False)

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
            if not parameter.requires_grad:
                # A frozen student parameter is its own EMA; skip the rounding drift.
                continue
            teacher.mul_(self.momentum).add_(parameter, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
