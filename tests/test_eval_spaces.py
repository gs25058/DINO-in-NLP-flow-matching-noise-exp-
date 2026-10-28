import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.evaluate import (
    EVAL_SPACES,
    _stsb_metrics,
    effective_rank_metrics,
    encode_sentences,
)
from src.model import DINOHead, ModelOutput


class Model(torch.nn.Module):
    def __init__(self, dim=8, bottleneck=4, logit_dim=16):
        super().__init__()
        self.embedding = torch.nn.Embedding(20, dim)
        self.head = DINOHead(dim, bottleneck, logit_dim)
        self.n_forward = 0

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask):
        self.n_forward += 1
        pooled = inputs_embeds.mean(dim=1)
        return ModelOutput(F.normalize(pooled, p=2, dim=-1), self.head(pooled), pooled)


class Enc(dict):
    def to(self, device):
        return self


class Tokenizer:
    def __call__(self, batch, **kwargs):
        ids = torch.arange(len(batch) * 3).reshape(len(batch), 3) % 20
        return Enc(input_ids=ids, attention_mask=torch.ones_like(ids))


SENTENCES = [f"s{i}" for i in range(7)]


def test_embedding_path_remains_exact_and_restores_training_mode():
    model, tok = Model(), Tokenizer()
    single = encode_sentences(model, tok, SENTENCES, "cpu", batch_size=3)
    multiple = encode_sentences(model, tok, SENTENCES, "cpu", batch_size=3, spaces=EVAL_SPACES)
    assert torch.equal(single["embedding"], multiple["embedding"])
    assert model.training


def test_all_spaces_share_one_backbone_pass_and_are_normalized():
    model, tok = Model(), Tokenizer()
    out = encode_sentences(model, tok, SENTENCES, "cpu", batch_size=7, spaces=EVAL_SPACES)
    assert model.n_forward == 1
    assert [tuple(out[k].shape) for k in EVAL_SPACES] == [(7, 8), (7, 4), (7, 16)]
    for vectors in out.values():
        torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(7), atol=1e-5, rtol=1e-5)
    enc = tok(SENTENCES)
    pooled = model.embedding(enc["input_ids"]).mean(1)
    assert torch.equal(out["bottleneck"], F.normalize(model.head.mlp(pooled), p=2, dim=-1))
    assert torch.equal(out["head"], F.normalize(model.head(pooled), p=2, dim=-1))


def test_unknown_space_raises_before_a_forward_and_restores_mode_on_error():
    model = Model()
    with pytest.raises(ValueError, match="unknown eval space"):
        encode_sentences(model, Tokenizer(), SENTENCES, "cpu", spaces=("raw",))
    assert model.n_forward == 0 and model.training

    def invalid(*args, **kwargs):
        raise RuntimeError("tokenizer failed")

    with pytest.raises(RuntimeError, match="tokenizer failed"):
        encode_sentences(model, invalid, SENTENCES, "cpu")
    assert model.training


def test_alignment_spearman_and_uniformity_keep_their_definitions():
    torch.manual_seed(0)
    e1 = F.normalize(torch.randn(6, 5), dim=-1)
    scores = np.array([0.9, 0.85, 0.2, 0.1, 0.95, 0.3])
    noise = torch.randn(6, 5)
    noise = F.normalize(noise - (noise * e1).sum(-1, keepdim=True) * e1, dim=-1)
    e2 = F.normalize(e1 + noise * torch.tensor(1 - scores, dtype=torch.float32)[:, None], dim=-1)
    metrics = _stsb_metrics(e1, e2, scores)
    assert metrics.spearman == pytest.approx(1.0, abs=1e-9) and metrics.uniformity < 0
    identical = _stsb_metrics(e1, e1.clone(), scores)
    assert (
        identical.alignment == pytest.approx(0.0, abs=1e-6)
        and metrics.alignment > identical.alignment
    )


def test_effective_rank_uses_centered_squared_singular_values():
    model, tok = Model(), Tokenizer()
    embeddings = encode_sentences(model, tok, SENTENCES, "cpu")["embedding"]
    singular = torch.linalg.svdvals((embeddings - embeddings.mean(0, keepdim=True)).double())
    p = singular.square() / singular.square().sum()
    rank, max_sv = effective_rank_metrics(model, tok, SENTENCES, "cpu")
    assert rank == torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()
    assert max_sv == p[0].item()
