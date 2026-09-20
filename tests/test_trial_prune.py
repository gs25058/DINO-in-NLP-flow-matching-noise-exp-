"""체크포인트 정리 규칙 검증 (scripts/r18_trial_sts7.py).

대여 서버라 trial 체크포인트(약 860MB)를 계속 쌓아둘 수 없다. 지우는 코드이므로 "남길 것을
실수로 지우지 않는다"를 테스트로 못 박는다.
"""
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("trial_sts7", ROOT / "scripts" / "r18_trial_sts7.py")
trial_sts7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trial_sts7)

ARGS = argparse.Namespace(min_7task=0.65, keep_top=5, keep_margin=0.005)
BEST_STS = 0.7500


def reasons(sts, seven, rank_sts, rank_7):
    return trial_sts7.keep_reasons({"sts_b_dev": sts, "avg_7task": seven}, 1,
                                   {1: rank_sts}, {1: rank_7}, BEST_STS, ARGS)


def test_high_7task_is_kept_even_when_sts_is_mediocre():
    assert reasons(sts=0.70, seven=0.66, rank_sts=30, rank_7=30)      # 사용자 요구의 핵심 조건


def test_top_ranks_are_kept_even_when_7task_is_low():
    assert reasons(sts=0.74, seven=0.60, rank_sts=3, rank_7=30)       # STS-B 상위
    assert reasons(sts=0.70, seven=0.63, rank_sts=30, rank_7=4)       # 7-task 상위


def test_near_best_sts_is_kept():
    assert reasons(sts=BEST_STS - 0.004, seven=0.60, rank_sts=9, rank_7=9)


def test_mediocre_trial_is_removable():
    assert reasons(sts=0.70, seven=0.60, rank_sts=20, rank_7=20) == []


def test_boundaries_are_inclusive():
    assert reasons(sts=0.60, seven=0.65, rank_sts=20, rank_7=20)              # 임계값 정확히 = 보존
    assert reasons(sts=BEST_STS - 0.005, seven=0.60, rank_sts=20, rank_7=20)  # margin 경계 = 보존
    assert reasons(sts=0.60, seven=0.6499, rank_sts=6, rank_7=6) == []        # 둘 다 한 끗 차이 = 삭제


def test_prune_only_touches_evaluated_trials_of_this_study(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(trial_sts7, "ROOT", tmp_path)
    ckpt = tmp_path / "checkpoints"
    names = ["mystudy_t1", "mystudy_t2", "mystudy_t3", "r18_bert_ctrl_s42", "mystudy_t9_running"]
    for n in names:
        (ckpt / n).mkdir(parents=True)
        (ckpt / n / "last.pt").write_bytes(b"x" * 1024)

    rows = [(1, {"sts_b_dev": 0.7500, "avg_7task": 0.60}),    # 1위 -> 보존
            (2, {"sts_b_dev": 0.7000, "avg_7task": 0.66}),    # 7-task 높음 -> 보존
            (3, {"sts_b_dev": 0.6800, "avg_7task": 0.60})]    # 근거 없음 -> 삭제
    by_sts, by_7 = {1: 1, 2: 20, 3: 30}, {1: 20, 2: 1, 3: 30}
    args = argparse.Namespace(study="mystudy", prune=True, dry_run=False,
                              min_7task=0.65, keep_top=5, keep_margin=0.005)
    trial_sts7.prune(args, {}, rows, by_sts, by_7, BEST_STS)

    assert (ckpt / "mystudy_t1").exists() and (ckpt / "mystudy_t2").exists()
    assert not (ckpt / "mystudy_t3").exists()                  # 지워진 것은 이것 하나뿐
    assert (ckpt / "r18_bert_ctrl_s42").exists()               # 다른 run은 건드리지 않는다
    assert (ckpt / "mystudy_t9_running").exists()              # 평가 안 된 trial도 건드리지 않는다
    assert "삭제" in capsys.readouterr().out


def test_dry_run_deletes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(trial_sts7, "ROOT", tmp_path)
    (tmp_path / "checkpoints" / "mystudy_t3").mkdir(parents=True)
    (tmp_path / "checkpoints" / "mystudy_t3" / "last.pt").write_bytes(b"x" * 1024)
    args = argparse.Namespace(study="mystudy", prune=True, dry_run=True,
                              min_7task=0.65, keep_top=5, keep_margin=0.005)
    trial_sts7.prune(args, {}, [(3, {"sts_b_dev": 0.60, "avg_7task": 0.60})], {3: 30}, {3: 30}, BEST_STS, )
    assert (tmp_path / "checkpoints" / "mystudy_t3" / "last.pt").exists()
    assert "dry-run" in capsys.readouterr().out


def test_prune_tb_removes_tensorboard_dirs_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(trial_sts7, "ROOT", tmp_path)
    for base in ("checkpoints", "results/tensorboard"):
        for n in ("mystudy_t3", "mystudy_t1"):
            d = tmp_path / base / n
            d.mkdir(parents=True)
            (d / "f").write_bytes(b"x")
    rows = [(1, {"sts_b_dev": 0.75, "avg_7task": 0.70}),     # 보존
            (3, {"sts_b_dev": 0.60, "avg_7task": 0.60})]     # 삭제 대상
    by_sts, by_7 = {1: 1, 3: 30}, {1: 1, 3: 30}
    base_args = dict(study="mystudy", prune=True, dry_run=False,
                     min_7task=0.68, keep_top=5, keep_margin=0.005)

    # --prune-tb 없이: 체크포인트만 사라지고 TensorBoard run은 남는다
    trial_sts7.prune(argparse.Namespace(**base_args, prune_tb=False), {}, rows, by_sts, by_7, 0.75)
    assert not (tmp_path / "checkpoints" / "mystudy_t3").exists()
    assert (tmp_path / "results/tensorboard" / "mystudy_t3").exists()

    # --prune-tb: 남아 있던 TensorBoard run도 사라진다
    trial_sts7.prune(argparse.Namespace(**base_args, prune_tb=True), {}, rows, by_sts, by_7, 0.75)
    assert not (tmp_path / "results/tensorboard" / "mystudy_t3").exists()
    assert (tmp_path / "results/tensorboard" / "mystudy_t1").exists()   # 보존 대상은 그대로
    assert (tmp_path / "checkpoints" / "mystudy_t1").exists()
