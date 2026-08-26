import torch

from src.diagnostics import active_prototype_count, tbin_index


def test_tbin_index_boundaries_and_classification():
    assert tbin_index(0.0) == 0
    assert tbin_index(0.05) == 0
    assert tbin_index(0.1) == 1
    assert tbin_index(0.25) == 2
    assert tbin_index(0.39) == 3
    assert tbin_index(0.4) == 4
    assert tbin_index(0.45) == 4
    assert tbin_index(0.5) == 4       # t_max 자체는 마지막 bin
    assert tbin_index(0.9) == 4       # t_max 초과도 마지막 bin으로 클램프


def test_active_prototype_count():
    n = 8192
    p = torch.zeros(n)
    p[0] = 0.5
    p[1] = 0.4
    p[2] = 0.05
    p[3:] = 0.05 / (n - 3)
    # 상위 2개(0.5+0.4=0.9)만으로 threshold 0.9에 정확히 도달
    assert active_prototype_count(p, threshold=0.9) == 2

    uniform = torch.full((n,), 1.0 / n)
    # 균등분포면 0.9 도달까지 거의 전체(90%)가 필요
    assert active_prototype_count(uniform, threshold=0.9) > n * 0.85
