"""표현 공간별(Final Embedding / DINO head bottleneck / head logits) STS-B 평가 테스트.

핵심은 두 가지다: (1) 기본 경로가 기존과 bit-identical일 것, (2) 여러 공간을 한꺼번에
뽑아도 backbone forward는 한 번만 돌 것(평가 시간이 공간 수에 비례해 늘면 안 된다).
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.evaluate import EVAL_SPACES, _project_to_space, _stsb_triplet, embed_sentences
from src.model import DINOHead


class _StubModel(torch.nn.Module):
    """DinoTextModel의 4-tuple forward 계약만 흉내내는 최소 스텁. forward 호출 수를 센다."""

    def __init__(self, dim=8, bottleneck=4, logit_dim=16):
        super().__init__()
        self._embed = torch.nn.Embedding(20, dim)
        self.head = DINOHead(dim, bottleneck, logit_dim)
        self.n_forward = 0

    def get_input_embeddings(self):
        return self._embed

    def forward(self, inputs_embeds, attention_mask, pooling="last"):
        self.n_forward += 1
        pooled = inputs_embeds.mean(dim=1)
        return F.normalize(pooled, p=2, dim=-1), self.head(pooled), inputs_embeds, pooled


class _StubTokenizer:
    def __call__(self, batch, **kwargs):
        ids = torch.arange(len(batch) * 3).reshape(len(batch), 3) % 20
        return _Enc({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


class _Enc(dict):
    def to(self, device):
        return self


_SENTS = [f"s{i}" for i in range(7)]


def test_default_path_is_bit_identical_to_embedding_space():
    m, tok = _StubModel(), _StubTokenizer()
    legacy = embed_sentences(m, tok, _SENTS, "cpu", batch_size=3)
    multi = embed_sentences(m, tok, _SENTS, "cpu", batch_size=3, spaces=("embedding", "head"))
    assert isinstance(legacy, torch.Tensor)          # spaces=None이면 예전처럼 텐서 하나
    assert isinstance(multi, dict)
    assert torch.equal(legacy, multi["embedding"])   # allclose가 아니라 완전 일치여야 한다


def test_multi_space_shares_one_backbone_forward():
    m, tok = _StubModel(), _StubTokenizer()
    embed_sentences(m, tok, _SENTS, "cpu", batch_size=7, spaces=EVAL_SPACES)
    assert m.n_forward == 1                          # 공간 3개여도 forward는 배치당 한 번


def test_space_shapes_and_normalization():
    m, tok = _StubModel(dim=8, bottleneck=4, logit_dim=16), _StubTokenizer()
    out = embed_sentences(m, tok, _SENTS, "cpu", batch_size=4, spaces=EVAL_SPACES)
    assert out["embedding"].shape == (7, 8)
    assert out["bottleneck"].shape == (7, 4)
    assert out["head"].shape == (7, 16)
    for space, vecs in out.items():                  # 공간 간 비교가 되려면 전부 단위 벡터여야
        assert torch.allclose(vecs.norm(dim=-1), torch.ones(7), atol=1e-5), space


def test_bottleneck_matches_head_internal_representation():
    """bottleneck은 head 내부에서 prototype과 실제로 내적되는 벡터와 같아야 한다."""
    m = _StubModel()
    pooled = torch.randn(3, 8)
    got = _project_to_space(m, torch.zeros(3, 8), pooled, "bottleneck")
    assert torch.allclose(got, F.normalize(m.head.mlp(pooled), p=2, dim=-1), atol=1e-6)


def test_unknown_space_raises():
    with pytest.raises(ValueError, match="unknown eval space"):
        _project_to_space(_StubModel(), torch.zeros(2, 8), torch.zeros(2, 8), "pooled_raw")


def test_stsb_triplet_alignment_and_spearman():
    torch.manual_seed(0)
    e1 = F.normalize(torch.randn(6, 5), dim=-1)
    scores = np.array([0.9, 0.85, 0.2, 0.1, 0.95, 0.3])
    # e1에 직교하는 성분만 점수에 반비례해 섞으면 cos = 1/sqrt(1+k^2)로 k에 대해 단조 감소하므로,
    # cos 순위가 scores 순위와 정확히 일치한다(무작위 방향이면 e1과의 내적 탓에 순위가 흔들린다).
    noise = torch.randn(6, 5)
    noise = F.normalize(noise - (noise * e1).sum(-1, keepdim=True) * e1, dim=-1)
    e2 = F.normalize(e1 + noise * torch.tensor(1.0 - scores, dtype=torch.float32)[:, None], dim=-1)
    sp, align, unif = _stsb_triplet(e1, e2, scores, align_seed=0)
    assert sp == pytest.approx(1.0, abs=1e-9)        # 점수가 높을수록 cos도 높음
    assert unif < 0.0                                # log E[exp(-2 d^2)] 는 음수

    same_sp, same_align, _ = _stsb_triplet(e1, e1.clone(), scores, align_seed=0)
    assert same_align == pytest.approx(0.0, abs=1e-6)  # 동일 벡터 -> alignment 0
    assert align > same_align                          # 멀어진 쌍은 alignment가 더 큼
