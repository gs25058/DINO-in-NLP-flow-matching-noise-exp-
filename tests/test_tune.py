from pathlib import Path

from scripts.tune import parse_final_sts, set_by_path


def test_collapsed_score_uses_last_valid_eval_not_incomplete_crash(tmp_path):
    from scripts.tune import parse_collapsed_sts
    log = tmp_path / "train.log"
    log.write_text("[step 0] EVAL sts_b_dev=0.59\n[step 250] EVAL sts_b_dev=0.70\n"
                   "[step 500] EVAL sts_b_dev=0.10\n"
                   "[step 500] COLLAPSE ABORT eff_rank=1 last_ok_eval_step=250\n")
    assert parse_collapsed_sts(log) == 0.70
    log.write_text("[step 0] EVAL sts_b_dev=0.59\ncrashed\n")
    assert parse_collapsed_sts(log) is None


def test_trial_reuses_interpreter_keeps_requested_checkpoint_and_cleans_config(tmp_path, monkeypatch):
    import subprocess
    import sys
    import optuna
    import yaml
    from scripts import tune
    monkeypatch.setattr(tune, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    pool = tune.GpuPool([1])
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        path = Path(command[-1])
        captured["path"] = path
        cfg = yaml.safe_load(path.read_text())
        captured["cfg"] = cfg
        log_dir = tmp_path / "results" / "logs" / cfg["run_name"]
        log_dir.mkdir(parents=True)
        (log_dir / "train.log").write_text("[step 9] EVAL sts_b_dev=0.71\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(tune.subprocess, "run", fake_run)
    score = tune.run_trial(optuna.trial.FixedTrial({"loss.uniform_push_lr": 0.5}),
                           {"train": {}, "loss": {}}, "uniform_push", 10, 42, pool, 15,
                           "test", save_checkpoints=True)
    assert score == 0.71
    assert captured["command"][0] == sys.executable
    assert captured["command"].count(sys.executable) == 2
    assert captured["cfg"]["train"]["save_checkpoint"] is True
    assert "--timeout" in captured["command"]
    assert not captured["path"].exists()
    assert pool.acquire() == 1


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
