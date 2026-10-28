"""배치 인덱스 공급기 테스트 (R14 예산 스케일링의 전제)."""
import random

import pytest

from src.train import BatchIndexSampler


def test_sample_mode_is_bit_identical_to_legacy_random_sample():
    """기존 루프는 random.sample(sentences, B)였다. 인덱스로 바꿔도 같은 위치가 나와야 한다."""
    pop = [f"s{i}" for i in range(5000)]
    random.seed(7)
    legacy = [random.sample(pop, 32) for _ in range(40)]
    random.seed(7)
    s = BatchIndexSampler(len(pop), 32, "sample")
    new = [[pop[i] for i in s.next()] for _ in range(40)]
    assert legacy == new


def test_epoch_mode_has_no_duplicates_within_an_epoch():
    rng = random.Random(0)
    s = BatchIndexSampler(1000, 32, "epoch", rng=rng)
    seen = []
    for _ in range(1000 // 32):
        seen.extend(s.next())
    assert len(seen) == len(set(seen)) == 31 * 32
    assert s.epoch == 0


def test_epoch_mode_reshuffles_and_counts_epochs_on_wrap():
    rng = random.Random(0)
    s = BatchIndexSampler(100, 32, "epoch", rng=rng)
    first = [s.next() for _ in range(3)]          # 96개 사용, 4개 남음
    assert s.epoch == 0
    nxt = s.next()                                # 남은 4개로는 배치를 못 채워 재셔플
    assert s.epoch == 1
    assert len(nxt) == 32 and len(set(nxt)) == 32
    assert nxt != first[0]                         # 순열이 새로 섞였다


def test_epoch_mode_is_deterministic_given_seed():
    a = BatchIndexSampler(500, 16, "epoch", rng=random.Random(3))
    b = BatchIndexSampler(500, 16, "epoch", rng=random.Random(3))
    assert [a.next() for _ in range(40)] == [b.next() for _ in range(40)]


def test_r14_budgets_fit_in_one_wiki1m_epoch():
    """스펙 전제: 7700x32, 15600x32 모두 wiki1m 한 바퀴 안이라 중복이 없어야 한다."""
    n = 985_723
    for steps in (7700, 15600):
        assert steps * 32 <= n, steps


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="unknown data_order"):
        BatchIndexSampler(100, 8, "shuffle")
    with pytest.raises(ValueError, match="batch_size"):
        BatchIndexSampler(10, 32, "epoch")
