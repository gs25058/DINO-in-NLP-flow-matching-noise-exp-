"""r9 L_nbr(프로토타입 이웃 끌어당김) 단위 테스트.

핵심은 (1) 희소 내적이 dense p_i·p_j와 top-k 절단 범위 안에서 일치하는지,
(2) 큐/가중치 규칙(같은 문장 마스킹, tau_w, top-m, 행 정규화)이 명세대로인지,
(3) 유효 이웃이 없을 때 NaN·gradient 오류가 없는지.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from src.loss import NbrQueue, koleo_nbr_overlap, nbr_loss, sparse_proto_affinity
from src.schedules import nbr_ramp


def _probs(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.softmax(torch.randn(n, d, generator=g) * 3.0, dim=-1)


def test_sparse_affinity_matches_dense_within_topk_support():
    """top-k 인덱스 밖을 0으로 자른 뒤의 dense 내적과 정확히 일치해야 한다."""
    B, Q, D, K = 4, 6, 32, 8
    p_b, p_q = _probs(B, D, 0), _probs(Q, D, 1)
    val_b, idx_b = p_b.topk(K, dim=-1)
    val_q, idx_q = p_q.topk(K, dim=-1)

    got = sparse_proto_affinity(val_b, idx_b, val_q, idx_q)

    # dense 기준: 각 행을 top-k 밖 0으로 절단한 벡터끼리의 내적
    tb = torch.zeros_like(p_b).scatter_(1, idx_b, val_b)
    tq = torch.zeros_like(p_q).scatter_(1, idx_q, val_q)
    assert torch.allclose(got, tb @ tq.t(), atol=1e-6)


def test_sparse_affinity_upper_bounded_by_full_dot():
    """절단했으므로 전체 내적보다 클 수 없다."""
    B, Q, D, K = 3, 5, 40, 4
    p_b, p_q = _probs(B, D, 2), _probs(Q, D, 3)
    val_b, idx_b = p_b.topk(K, dim=-1)
    val_q, idx_q = p_q.topk(K, dim=-1)
    got = sparse_proto_affinity(val_b, idx_b, val_q, idx_q)
    assert (got <= (p_b @ p_q.t()) + 1e-6).all()


def test_queue_fifo_and_partial_fill():
    q = NbrQueue(queue_size=4, topk=2, dim=3, device="cpu")
    assert q.filled == 0 and q.view()[0].numel() == 0

    z = F.normalize(torch.randn(3, 3), dim=-1)
    q.push(z, _probs(3, 10, 4), torch.tensor([10, 11, 12]))
    assert q.filled == 3                      # 덜 찼으면 있는 만큼만
    assert q.view()[0].shape == (3, 3)

    q.push(F.normalize(torch.randn(3, 3), dim=-1), _probs(3, 10, 5), torch.tensor([13, 14, 15]))
    assert q.filled == 4                      # 상한
    assert set(q.ds_idx.tolist()) == {13, 14, 15, 12}   # 가장 오래된 10, 11이 밀려남


def test_same_sentence_is_masked_out():
    """큐에 같은 데이터셋 인덱스가 있으면 이웃에서 제외된다."""
    torch.manual_seed(0)
    D = 16
    p = _probs(1, D, 6).repeat(3, 1)          # 전부 동일한 p_t -> W가 최대가 되는 상황
    z = F.normalize(torch.randn(3, 4), dim=-1)
    q = NbrQueue(queue_size=8, topk=4, dim=4, device="cpu")
    q.push(z, p, torch.tensor([100, 101, 102]))

    # 마스킹은 쌍 단위다: 자기 문장만 빠지고 나머지 2개는 이웃으로 남는다
    ds = torch.tensor([100, 101, 102])
    _, aux, W = nbr_loss([z], q, ds, p, tau_w=0.0, max_nbr=8, topk=4)
    for i in range(3):
        same = (q.ds_idx[:q.filled] == ds[i])
        assert float(W[i][same].sum()) == 0.0        # 자기 문장 가중치 0
    assert aux["n_nbr_mean"] == 2.0                  # 3개 중 자기 제외
    assert aux["nbr_valid_frac"] == 1.0

    # 큐와 겹치는 문장이 하나도 없으면 3개 전부 이웃
    _, aux2, _ = nbr_loss([z], q, torch.tensor([200, 201, 202]), p,
                          tau_w=0.0, max_nbr=8, topk=4)
    assert aux2["n_nbr_mean"] == 3.0


def test_row_normalized_and_topm_capped():
    torch.manual_seed(1)
    D, Q, M = 16, 10, 3
    p_q = _probs(Q, D, 7)
    z_q = F.normalize(torch.randn(Q, 5), dim=-1)
    q = NbrQueue(queue_size=Q, topk=4, dim=5, device="cpu")
    q.push(z_q, p_q, torch.arange(Q))

    p_b = _probs(2, D, 8)
    z_b = F.normalize(torch.randn(2, 5), dim=-1)
    _, aux, W = nbr_loss([z_b], q, torch.tensor([900, 901]), p_b,
                         tau_w=0.0, max_nbr=M, topk=4)
    nz = (W > 0).sum(dim=1)
    assert (nz <= M).all()                                   # top-m 상한
    rows = W.sum(dim=1)
    for r, n in zip(rows, nz):
        if n > 0:
            assert math.isclose(float(r), 1.0, rel_tol=1e-5)  # 행 정규화


def test_tau_w_threshold_drops_weak_pairs():
    torch.manual_seed(2)
    D = 24
    p_q, p_b = _probs(6, D, 9), _probs(3, D, 10)
    z_q, z_b = F.normalize(torch.randn(6, 5), dim=-1), F.normalize(torch.randn(3, 5), dim=-1)
    q = NbrQueue(queue_size=6, topk=4, dim=5, device="cpu")
    q.push(z_q, p_q, torch.arange(6))

    _, lo, _ = nbr_loss([z_b], q, torch.tensor([7, 8, 9]), p_b, tau_w=0.0, max_nbr=6, topk=4)
    _, hi, _ = nbr_loss([z_b], q, torch.tensor([7, 8, 9]), p_b, tau_w=0.99, max_nbr=6, topk=4)
    assert hi["n_nbr_mean"] <= lo["n_nbr_mean"]
    assert hi["nbr_valid_frac"] <= lo["nbr_valid_frac"]


def test_empty_queue_returns_zero_without_grad_error():
    q = NbrQueue(queue_size=4, topk=2, dim=3, device="cpu")
    z = F.normalize(torch.randn(2, 3, requires_grad=True), dim=-1)
    loss, aux, W = nbr_loss([z], q, torch.tensor([1, 2]), _probs(2, 8, 11),
                            tau_w=0.1, max_nbr=4, topk=2)
    assert float(loss) == 0.0 and W is None
    loss.backward()          # 그래디언트 경로가 끊기지 않아야 한다(0이어도)


def test_loss_is_attraction_only_and_pulls_toward_neighbor():
    """이웃 방향으로 갈수록 손실이 작아져야 한다(끌어당김 전용, 반발 없음)."""
    D = 8
    p = _probs(1, D, 12).repeat(2, 1)
    z_q = F.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)
    q = NbrQueue(queue_size=4, topk=3, dim=3, device="cpu")
    q.push(z_q, p[:1], torch.tensor([500]))

    near = F.normalize(torch.tensor([[0.9, 0.1, 0.0]]), dim=-1)
    far = F.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=-1)
    l_near, _, _ = nbr_loss([near], q, torch.tensor([1]), p[:1], 0.0, 4, 3)
    l_far, _, _ = nbr_loss([far], q, torch.tensor([1]), p[:1], 0.0, 4, 3)
    assert float(l_near) < float(l_far)
    assert float(l_near) >= 0.0          # 1-cos >= 0, 끌어당김만


def test_nbr_ramp_shape():
    assert nbr_ramp(0, 300, 200) == 0.0
    assert nbr_ramp(299, 300, 200) == 0.0
    assert nbr_ramp(300, 300, 200) == 0.0            # cosine ease 시작점
    assert math.isclose(nbr_ramp(500, 300, 200), 1.0, abs_tol=1e-9)
    assert nbr_ramp(9999, 300, 200) == 1.0
    prev = -1.0
    for s in range(300, 501, 10):                    # 단조 증가
        v = nbr_ramp(s, 300, 200)
        assert v >= prev - 1e-12
        prev = v


def test_koleo_overlap_range_and_queue_absent_excluded():
    torch.manual_seed(3)
    q = NbrQueue(queue_size=8, topk=3, dim=4, device="cpu")
    z_q = F.normalize(torch.randn(4, 4), dim=-1)
    q.push(z_q, _probs(4, 12, 13), torch.tensor([1, 2, 3, 4]))

    z_b = F.normalize(torch.randn(4, 4), dim=-1)
    _, _, W = nbr_loss([z_b], q, torch.tensor([1, 2, 3, 4]), _probs(4, 12, 14), 0.0, 8, 3)
    ov = koleo_nbr_overlap(z_b, torch.tensor([1, 2, 3, 4]), q, W)
    assert math.isnan(ov) or 0.0 <= ov <= 1.0

    # 배치의 최근접 이웃이 큐에 전혀 없는 인덱스면 분모에서 빠져 nan
    ov2 = koleo_nbr_overlap(z_b, torch.tensor([90, 91, 92, 93]), q, W)
    assert math.isnan(ov2)


def test_relative_threshold_is_scale_invariant():
    """tau_w_rel은 행 최대값 대비이므로 W 전체에 상수를 곱해도 같은 이웃을 고른다.

    절대 임계(tau_w)를 못 쓰는 이유가 바로 이 스케일 의존성이었다(p_t가 거의 균일해
    W~1e-7이고 학습 중 변동한다).
    """
    torch.manual_seed(4)
    D, Q = 32, 12
    p_q, p_b = _probs(Q, D, 20), _probs(3, D, 21)
    z_q, z_b = F.normalize(torch.randn(Q, 6), dim=-1), F.normalize(torch.randn(3, 6), dim=-1)
    q = NbrQueue(queue_size=Q, topk=6, dim=6, device="cpu")
    q.push(z_q, p_q, torch.arange(Q))

    _, a1, W1 = nbr_loss([z_b], q, torch.tensor([50, 51, 52]), p_b,
                         tau_w=0.0, max_nbr=8, topk=6, tau_w_rel=0.5)
    # p를 상수배해도(=W가 c^2배) 상대 임계면 선택이 동일해야 한다
    q2 = NbrQueue(queue_size=Q, topk=6, dim=6, device="cpu")
    q2.push(z_q, p_q * 1e-4, torch.arange(Q))
    _, a2, W2 = nbr_loss([z_b], q2, torch.tensor([50, 51, 52]), p_b * 1e-4,
                         tau_w=0.0, max_nbr=8, topk=6, tau_w_rel=0.5)
    assert a1["n_nbr_mean"] == a2["n_nbr_mean"]
    assert torch.allclose((W1 > 0).float(), (W2 > 0).float())


def test_lower_tau_w_rel_gives_more_neighbors():
    torch.manual_seed(5)
    D, Q = 32, 20
    p_q, p_b = _probs(Q, D, 22), _probs(4, D, 23)
    z_q, z_b = F.normalize(torch.randn(Q, 6), dim=-1), F.normalize(torch.randn(4, 6), dim=-1)
    q = NbrQueue(queue_size=Q, topk=6, dim=6, device="cpu")
    q.push(z_q, p_q, torch.arange(Q))
    counts = []
    for r in (0.9, 0.5, 0.1):
        _, aux, _ = nbr_loss([z_b], q, torch.arange(100, 104), p_b,
                             tau_w=0.0, max_nbr=16, topk=6, tau_w_rel=r)
        counts.append(aux["n_nbr_mean"])
    assert counts[0] <= counts[1] <= counts[2]


def test_self_pair_excluded_before_relative_threshold():
    """상대 임계의 기준(행 최대값)이 자기 쌍이 되면 안 된다."""
    D = 16
    p = _probs(1, D, 24).repeat(2, 1)
    z = F.normalize(torch.randn(2, 5), dim=-1)
    q = NbrQueue(queue_size=4, topk=4, dim=5, device="cpu")
    q.push(z, p, torch.tensor([700, 701]))
    # 배치가 큐의 700과 같은 문장 -> 자기 쌍이 최대여도 그것이 기준이 되면 안 되고,
    # 남은 701이 정상적으로 이웃으로 잡혀야 한다
    _, aux, W = nbr_loss([z[:1]], q, torch.tensor([700]), p[:1],
                         tau_w=0.0, max_nbr=4, topk=4, tau_w_rel=0.5)
    assert float(W[0][q.ds_idx[:q.filled] == 700].sum()) == 0.0
    assert aux["n_nbr_mean"] == 1.0


def test_nbr_cos_overlap_perfect_and_random_baseline():
    """프로토타입 이웃이 코사인 최근접과 완전히 일치하면 1.0이어야 한다."""
    from src.loss import nbr_cos_overlap
    D = 16
    # 큐 4개 중 앞 2개를 배치와 같은 방향으로 둔다 -> 코사인 최근접 = 그 2개
    z_q = F.normalize(torch.tensor([[1., 0., 0.], [0.99, 0.14, 0.],
                                    [0., 1., 0.], [0., 0., 1.]]), dim=-1)
    p_q = _probs(4, D, 30)
    q = NbrQueue(queue_size=4, topk=4, dim=3, device="cpu")
    q.push(z_q, p_q, torch.arange(4))

    z_b = F.normalize(torch.tensor([[1., 0.05, 0.]]), dim=-1)
    W = torch.tensor([[0.5, 0.5, 0.0, 0.0]])          # 프로토타입 이웃 = 큐 0,1
    ov, rand = nbr_cos_overlap(z_b, q, W)
    assert ov == 1.0                                   # 코사인 상위 2개와 완전 일치
    assert math.isclose(rand, 2 / 4)                   # m/Q

    W2 = torch.tensor([[0.0, 0.0, 0.5, 0.5]])         # 반대로 먼 것들을 이웃으로
    ov2, _ = nbr_cos_overlap(z_b, q, W2)
    assert ov2 == 0.0


def test_nbr_cos_overlap_handles_empty_and_none():
    from src.loss import nbr_cos_overlap
    q = NbrQueue(queue_size=4, topk=2, dim=3, device="cpu")
    ov, rand = nbr_cos_overlap(F.normalize(torch.randn(2, 3), dim=-1), q, None)
    assert math.isnan(ov) and math.isnan(rand)
