import torch
import torch.nn.functional as F

from src.evaluate import postprocess_embeddings


def test_postprocess_none_is_identity():
    torch.manual_seed(0)
    embeds = F.normalize(torch.randn(20, 16), p=2, dim=-1)
    out = postprocess_embeddings(embeds, method="none")
    assert torch.equal(out, embeds)


def test_postprocess_preserves_unit_norm():
    torch.manual_seed(1)
    embeds = F.normalize(torch.randn(30, 16) * 3.0 + 5.0, p=2, dim=-1)  # 비등방적(anisotropic) 분포
    for method in ("center", "center_pc1", "center_pc2"):
        out = postprocess_embeddings(embeds, method=method)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"{method} broke unit norm"


def test_postprocess_uses_separate_fit_set():
    """fit_embeds가 주어지면 그 집합의 평균/주성분으로 embeds를 변환해야 한다
    (평가 시 쓰는 문장 풀 자체에서 통계를 내는 것이 목적 - embeds 자신과 fit 집합이 달라도 됨)."""
    torch.manual_seed(2)
    fit = F.normalize(torch.randn(50, 8) + 2.0, p=2, dim=-1)
    query = F.normalize(torch.randn(5, 8), p=2, dim=-1)

    out_with_fit = postprocess_embeddings(query, fit_embeds=fit, method="center")
    out_self_fit = postprocess_embeddings(query, method="center")  # fit_embeds=None -> query 자신으로 fit

    assert not torch.allclose(out_with_fit, out_self_fit)
    assert torch.allclose(out_with_fit.norm(dim=-1), torch.ones(5), atol=1e-5)
