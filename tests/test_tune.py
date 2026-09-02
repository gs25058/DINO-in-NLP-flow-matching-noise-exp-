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
