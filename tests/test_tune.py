from pathlib import Path

from scripts.tune import parse_final_sts, set_by_path


def test_set_by_path_nested():
    cfg = {"loss": {"uniform_push_lr": 0.0, "teacher_temp": 0.082}, "train": {"lr": 5.4e-4}}
    set_by_path(cfg, "loss.uniform_push_lr", 1.23)
    assert cfg["loss"]["uniform_push_lr"] == 1.23
    assert cfg["loss"]["teacher_temp"] == 0.082  # 다른 키는 영향 없음

    set_by_path(cfg, "train.lr", 1e-3)
    assert cfg["train"]["lr"] == 1e-3


def test_parse_final_sts_returns_last_eval_value(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "[step 0] EVAL sts_b_dev=0.5566 eff_rank=147.81 max_sv_ratio=0.1190\n"
        "[step 250] EVAL sts_b_dev=0.6753 eff_rank=164.16 max_sv_ratio=0.0842\n"
        "[step 500] EVAL sts_b_dev=0.6806 eff_rank=130.53 max_sv_ratio=0.1118\n",
        encoding="utf-8",
    )
    assert parse_final_sts(log) == 0.6806


def test_parse_final_sts_handles_negative_and_missing(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text("[step 0] EVAL sts_b_dev=-0.0120 eff_rank=1.00 max_sv_ratio=1.0000\n", encoding="utf-8")
    assert parse_final_sts(log) == -0.0120

    missing = tmp_path / "does_not_exist.log"
    assert parse_final_sts(missing) is None

    empty = tmp_path / "empty.log"
    empty.write_text("no eval lines here\n", encoding="utf-8")
    assert parse_final_sts(empty) is None


def test_parse_final_sts_min_step_rejects_incomplete_run(tmp_path: Path):
    # 중도 사망한 run: step 0 EVAL만 남아있다 (cov_iso_full study에서 40 trial 중 11개가 이랬다).
    # min_step 없이는 사전학습 초기값(0.5931)이 정상 점수로 보고돼 Optuna를 오염시킨다.
    crashed = tmp_path / "crashed.log"
    crashed.write_text(
        "[step 0] EVAL sts_b_dev=0.5931 eff_rank=219.68 max_sv_ratio=0.0496\n"
        "[step 10] loss=9.01 H_pt=8.95\n"
        "[step 20] loss=9.00 H_pt=8.94\n",
        encoding="utf-8",
    )
    assert parse_final_sts(crashed) == 0.5931  # min_step 기본값 0이면 기존 동작 유지
    assert parse_final_sts(crashed, min_step=749) is None

    # 완주한 run은 같은 min_step에서 정상적으로 값을 돌려준다
    finished = tmp_path / "finished.log"
    finished.write_text(
        "[step 0] EVAL sts_b_dev=0.5931 eff_rank=219.68 max_sv_ratio=0.0496\n"
        "[step 500] EVAL sts_b_dev=0.6900 eff_rank=250.00 max_sv_ratio=0.0200\n"
        "[step 749] EVAL sts_b_dev=0.7030 eff_rank=267.28 max_sv_ratio=0.0155\n",
        encoding="utf-8",
    )
    assert parse_final_sts(finished, min_step=749) == 0.7030
