"""R15 토큰 수준 masked latent 예측 (data2vec/iBOT식) 테스트."""
import math

import pytest
import torch
import torch.nn.functional as F

from src.augment import FlowNoiseAug
from src.loss import TokenLatentPredictor, token_latent_loss, token_latent_targets
from src.model import DinoTextModel

D = 4
MASK = torch.full((D,), 9.0)


def _aug(**kw):
    base = dict(mode="anchor", num_student_views=3, t_lo=0.35, t_start=0.5, t_max=0.5,
                warmup_steps=1, delta_t=0.1)
    base.update(kw)
    return FlowNoiseAug(torch.zeros(D), torch.ones(D), **base)


def _batch(B=6, L=24):
    torch.manual_seed(1)
    emb = torch.randn(B, L, D)
    special = torch.zeros(B, L, dtype=torch.bool)
    special[:, 0] = True
    attn = torch.ones(B, L, dtype=torch.long)
    for b in range(B):                          # 문장마다 길이를 다르게: [CLS] 토큰들 [SEP] pad...
        n = 6 + 3 * b
        special[b, n + 1:] = True
        special[b, n + 1] = True
        attn[b, n + 2:] = 0
    return emb, special, attn


# ------------------------------------------------------------------ 기본 경로 불변
def test_disabled_is_bit_identical_to_default():
    emb, sp, attn = _batch()
    torch.manual_seed(0); base = _aug()(emb, sp, step=10, attention_mask=attn)
    torch.manual_seed(0); off = _aug(token_latent_views=0, mask_embed=MASK)(emb, sp, step=10, attention_mask=attn)
    assert base.student_token_masks is None and off.student_token_masks is None
    for a, b in zip(base.student_embeds, off.student_embeds):
        assert torch.equal(a, b)


def test_train_gates_masking_on_lambda():
    """token_latent_views 기본값이 1이라, lambda=0에서 켜지면 기존 run과 달라진다."""
    from src.train import token_latent_views
    assert token_latent_views({"loss": {}, "augment": {}}) == 0
    assert token_latent_views({"loss": {"token_latent_lambda": 0.0}, "augment": {"token_latent_views": 2}}) == 0
    assert token_latent_views({"loss": {"token_latent_lambda": 1.0}, "augment": {}}) == 1
    assert token_latent_views({"loss": {"token_latent_lambda": 1.0}, "augment": {"token_latent_views": 2}}) == 2


# ------------------------------------------------------------------ 마스크 선택
def test_mask_never_touches_special_or_padding():
    emb, sp, attn = _batch()
    v = _aug(token_latent_views=1, mask_embed=MASK, mask_ratio=0.3)(emb, sp, step=10, attention_mask=attn)
    m = v.student_token_masks[0]
    assert not (m & sp).any()
    assert not (m & ~attn.bool()).any()


def test_mask_ratio_count_is_exact():
    emb, sp, attn = _batch()
    ratio = 0.15
    v = _aug(token_latent_views=1, mask_embed=MASK, mask_ratio=ratio)(emb, sp, step=10, attention_mask=attn)
    m = v.student_token_masks[0]
    valid = (~sp) & attn.bool()
    for b in range(emb.shape[0]):
        n = int(valid[b].sum())
        expected = max(1, math.floor(n * ratio + 0.5))
        assert int(m[b].sum()) == expected, (b, n)


def test_only_first_token_latent_views_are_masked():
    emb, sp, attn = _batch()
    v = _aug(token_latent_views=2, mask_embed=MASK)(emb, sp, step=10, attention_mask=attn)
    assert v.student_token_masks[0] is not None and v.student_token_masks[1] is not None
    assert v.student_token_masks[2] is None


def test_mask_then_noise_orders():
    emb, sp, attn = _batch()
    # t=0이면 노이즈가 없으므로 "마스킹 후 노이즈"도 마스크 위치가 정확히 [MASK]여야 한다
    v0 = _aug(token_latent_views=1, mask_embed=MASK, t_lo=0.0, t_start=0.0, t_max=0.0)(
        emb, sp, step=10, attention_mask=attn)
    m0 = v0.student_token_masks[0]
    assert torch.allclose(v0.student_embeds[0][m0], MASK.expand(int(m0.sum()), D))
    # t>0, mask_then_noise=True: 마스크 위치도 노이즈를 받아 [MASK]와 달라진다
    v1 = _aug(token_latent_views=1, mask_embed=MASK, mask_then_noise=True)(emb, sp, step=10, attention_mask=attn)
    m1 = v1.student_token_masks[0]
    assert not torch.allclose(v1.student_embeds[0][m1], MASK.expand(int(m1.sum()), D))
    # mask_then_noise=False: 노이즈 뒤에 덮어써서 마스크 위치가 정확히 [MASK]
    v2 = _aug(token_latent_views=1, mask_embed=MASK, mask_then_noise=False)(emb, sp, step=10, attention_mask=attn)
    m2 = v2.student_token_masks[0]
    assert torch.equal(v2.student_embeds[0][m2], MASK.expand(int(m2.sum()), D))


def test_requires_mask_embed():
    with pytest.raises(AssertionError, match="mask_embed"):
        _aug(token_latent_views=1)


# ------------------------------------------------------------------ 목표 / 손실 / 예측기
def test_targets_have_no_grad_and_are_token_normalized():
    # 시드 고정 + 스케일 확대: 6개 층 평균은 분산이 작아져 layer_norm eps(1e-5)의 상대 영향이 커지므로
    # 무작위 입력에서는 분산 검사가 흔들린다(0.995까지 관측). 정확성은 아래 수식 대조가 보장한다.
    g = torch.Generator().manual_seed(0)
    hs = tuple((torch.randn(2, 5, D, generator=g) * 5.0).requires_grad_() for _ in range(13))
    y = token_latent_targets(hs, top_k=6)
    assert not y.requires_grad
    assert torch.allclose(y.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
    pop_var = ((y - y.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)   # layer_norm은 모분산 기준
    assert torch.allclose(pop_var, torch.ones(2, 5), atol=1e-3)   # layer_norm eps(1e-5) 때문에 약간 1 미만
    manual = F.layer_norm(torch.stack(hs[-6:]).mean(0), (D,))
    assert torch.allclose(y, manual.detach())


def test_targets_reject_bad_top_k():
    hs = tuple(torch.randn(1, 2, D) for _ in range(13))
    with pytest.raises(ValueError, match="top_k"):
        token_latent_targets(hs, top_k=13)


def test_loss_only_counts_masked_positions():
    torch.manual_seed(0)
    pred, tgt = torch.randn(2, 6, D), torch.randn(2, 6, D)
    mask = torch.zeros(2, 6, dtype=torch.bool); mask[0, 1] = mask[1, 3] = True
    l1, _ = token_latent_loss(pred, tgt, mask)
    pred2 = pred.clone(); pred2[~mask] = 100.0            # 마스크 밖을 망가뜨려도
    l2, _ = token_latent_loss(pred2, tgt, mask)
    assert torch.allclose(l1, l2)                          # 손실은 그대로
    exp = F.smooth_l1_loss(pred[mask], tgt[mask], beta=1.0)
    assert torch.allclose(l1, exp)


def test_loss_with_no_masked_positions_is_zero_and_differentiable():
    pred = torch.randn(1, 3, D, requires_grad=True)
    l, _ = token_latent_loss(pred, torch.randn(1, 3, D), torch.zeros(1, 3, dtype=torch.bool))
    assert float(l) == 0.0
    l.backward()


def test_predictor_shape():
    assert TokenLatentPredictor(D)(torch.randn(3, 7, D)).shape == (3, 7, D)


# ------------------------------------------------------------------ 모델 hidden 반환
class _Backbone(torch.nn.Module):
    class _Cfg:
        hidden_size = D

    def __init__(self):
        super().__init__()
        self.config = self._Cfg()
        self._embed = torch.nn.Embedding(10, D)

    def get_input_embeddings(self):
        return self._embed

    def forward(self, inputs_embeds, attention_mask, output_hidden_states=False):
        out = type("O", (), {})()
        out.last_hidden_state = inputs_embeds * 2.0
        out.hidden_states = (inputs_embeds, inputs_embeds * 2.0) if output_hidden_states else None
        return out


def _model():
    m = DinoTextModel.__new__(DinoTextModel)
    torch.nn.Module.__init__(m)
    m.backbone = _Backbone()
    m.head = torch.nn.Linear(D, 8)
    return m


def test_model_default_returns_four_and_all_hidden_returns_five():
    m = _model()
    x, a = torch.randn(2, 5, D), torch.ones(2, 5, dtype=torch.long)
    assert len(m(inputs_embeds=x, attention_mask=a)) == 4
    out = m(inputs_embeds=x, attention_mask=a, return_all_hidden=True)
    assert len(out) == 5 and len(out[4]) == 2
    assert torch.equal(out[0], m(inputs_embeds=x, attention_mask=a)[0])   # embedding은 동일
