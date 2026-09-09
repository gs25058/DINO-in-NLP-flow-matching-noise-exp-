"""DinoTextModel.forward의 pooling 분기 테스트.

"cls"는 SimCSE 비지도판 공식 평가 방식과 맞추기 위해 추가한 평가 전용 경로다
(scripts/compare_simcse.py). 기존 "last"/"first_last" 동작이 변하지 않는 것이 핵심.
"""
import pytest
import torch
import torch.nn.functional as F

from src.model import DinoTextModel, masked_mean_pool


class _TinyBackbone(torch.nn.Module):
    """hidden state를 결정적으로 만들어 주는 최소 백본 스텁 (HF 다운로드 없이 검증)."""

    class _Cfg:
        hidden_size = 4

    def __init__(self):
        super().__init__()
        self.config = self._Cfg()
        self._embed = torch.nn.Embedding(10, 4)

    def get_input_embeddings(self):
        return self._embed

    def forward(self, inputs_embeds, attention_mask, output_hidden_states=False):
        last = inputs_embeds * 2.0
        out = type("O", (), {})()
        out.last_hidden_state = last
        if output_hidden_states:
            out.hidden_states = [inputs_embeds, last]
        return out


def _model():
    m = DinoTextModel.__new__(DinoTextModel)
    torch.nn.Module.__init__(m)
    m.backbone = _TinyBackbone()
    m.head = torch.nn.Linear(4, 8)
    return m.eval()


def _inputs():
    torch.manual_seed(0)
    embeds = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    return embeds, mask


def test_cls_pooling_uses_only_first_token():
    m = _model()
    embeds, mask = _inputs()
    with torch.no_grad():
        emb, *_ = m(inputs_embeds=embeds, attention_mask=mask, pooling="cls")
    expected = F.normalize((embeds * 2.0)[:, 0], p=2, dim=-1)
    assert torch.allclose(emb, expected, atol=1e-6)


def test_cls_pooling_differs_from_mean_and_ignores_padding_length():
    m = _model()
    embeds, mask = _inputs()
    with torch.no_grad():
        cls, *_ = m(inputs_embeds=embeds, attention_mask=mask, pooling="cls")
        mean, *_ = m(inputs_embeds=embeds, attention_mask=mask, pooling="last")
    assert not torch.allclose(cls, mean)

    # mask를 바꿔도 cls 결과는 그대로여야 한다(첫 토큰만 보므로)
    mask2 = torch.ones_like(mask)
    with torch.no_grad():
        cls2, *_ = m(inputs_embeds=embeds, attention_mask=mask2, pooling="cls")
        mean2, *_ = m(inputs_embeds=embeds, attention_mask=mask2, pooling="last")
    assert torch.allclose(cls, cls2, atol=1e-6)
    assert not torch.allclose(mean, mean2)      # mean은 mask에 반응해야 정상


def test_existing_pooling_paths_unchanged():
    """cls 추가가 기존 두 경로의 결과를 바꾸지 않았는지 (수식과 직접 대조)."""
    m = _model()
    embeds, mask = _inputs()
    with torch.no_grad():
        last, *_ = m(inputs_embeds=embeds, attention_mask=mask, pooling="last")
        first_last, *_ = m(inputs_embeds=embeds, attention_mask=mask, pooling="first_last")
    assert torch.allclose(last, F.normalize(masked_mean_pool(embeds * 2.0, mask), p=2, dim=-1), atol=1e-6)
    assert torch.allclose(
        first_last, F.normalize(masked_mean_pool((embeds + embeds * 2.0) / 2, mask), p=2, dim=-1), atol=1e-6
    )
    # 기본값(인자 미지정)은 여전히 "last"
    with torch.no_grad():
        default, *_ = m(inputs_embeds=embeds, attention_mask=mask)
    assert torch.allclose(default, last, atol=1e-6)


def test_unknown_pooling_still_raises():
    m = _model()
    embeds, mask = _inputs()
    with pytest.raises(ValueError, match="unknown pooling"):
        m(inputs_embeds=embeds, attention_mask=mask, pooling="max")
