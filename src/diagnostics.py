"""Part B 진단 로깅 헬퍼. train.py가 diag_* config 플래그로만 사용한다 (기본 off,
플래그가 없으면 기존 학습 동작·결과에 전혀 영향 없음).
"""
import torch

TBIN_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)  # 5 bins: [0,.1) [.1,.2) [.2,.3) [.3,.4) [.4,.5]


def tbin_index(t: float, n_bins: int = 5, t_max: float = 0.5) -> int:
    """t를 [0, t_max]를 n_bins개 균등 구간으로 나눈 bin index로 분류. t_max 이상은 마지막 bin."""
    if t >= t_max:
        return n_bins - 1
    bin_width = t_max / n_bins
    idx = int(t / bin_width)
    return min(max(idx, 0), n_bins - 1)


def active_prototype_count(p_bar_t: torch.Tensor, threshold: float = 0.9) -> int:
    """p_bar_t(마지널 사용 분포) 내림차순 누적합이 threshold를 처음 넘는 데 필요한 최소 프로토타입 개수."""
    sorted_p, _ = torch.sort(p_bar_t, descending=True)
    cumsum = torch.cumsum(sorted_p, dim=0)
    count = int((cumsum < threshold).sum().item()) + 1
    return min(count, p_bar_t.shape[0])
