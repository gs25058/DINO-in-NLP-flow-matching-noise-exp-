"""Exercise the controller without GPUs, training, downloads, or actual test data."""

import signal
from pathlib import Path

import pytest

from scripts import run_coviso_optuna as pipeline
from src.evaluate import STS_SUITE
from src.provenance import append_record, prepare, read_json, write_json
from src.train import load_config


@pytest.mark.parametrize("improves", [False, True])
def test_control_search_confirmation_selection_then_test(tmp_path, monkeypatch, improves):
    workspace = tmp_path / "workspace"
    campaign = tmp_path / "campaign"
    manifest = prepare(campaign, pipeline.ROOT, ["auto", "auto"], tmp_path / "report.md")
    manifest.update(
        workspace=str(workspace),
        trials=4,
        import_baseline=None,
        optuna_spec_sha256=manifest["spec_sha256"],
    )
    write_json(campaign / "source_manifest.json", manifest)
    calls = []

    def execute(self, gpu, label, command, **kwargs):
        calls.append(label)
        if "-m" in command:
            cfg = load_config(command[command.index("--config") + 1])
            log = campaign / "results/logs" / cfg["run_name"]
            log.mkdir(parents=True)
            score = 0.73 + cfg["seed"] * 0.00001 + (0.01 if improves and "_t" in label else 0)
            metrics = {
                "status": "complete",
                "completed_steps": 4000,
                "best_sts": score,
                "best_loop_step": 2750,
                "last_sts": score - 0.001,
                "last_loop_step": 3999,
            }
            write_json(log / "completion.json", metrics)
            write_json(
                log / "provenance.json",
                {"seed": cfg["seed"], "initial_student_sha256": f"same_{cfg['seed']}"},
            )
            for step, value in ((0, 0.6), (2750, score), (3999, score - 0.001)):
                append_record(
                    log / "evaluations.jsonl",
                    {
                        "loop_step": step,
                        "sts_b_dev": value,
                        "alignment": 0.2,
                        "uniformity": -2.0,
                        "eff_rank": 100.0,
                    },
                )
            append_record(
                log / "components.jsonl",
                dict(
                    loop_step=3999,
                    L_DINO=1.0,
                    L_mean=0.1,
                    L_cov=0.2,
                    weighted_mean=0.1,
                    weighted_cov=0.2,
                    grad_dino_backbone=1.0,
                    grad_mean_backbone=0.1,
                    grad_cov_backbone=0.2,
                ),
            )
        else:
            assert (campaign / "selection.json").exists(), "No STS7 before frozen selection"
            run = command[command.index("--runs") + 1]
            out = Path(command[command.index("--out") + 1])
            write_json(
                out,
                {"runs": {run: {**{task: 0.5 for task in STS_SUITE}, "avg_7task": 0.5}}},
            )

    monkeypatch.setattr(pipeline.Children, "run", execute)
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        pipeline.run(campaign, manifest)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    assert read_json(campaign / "state.json")["stage"] == "complete"
    selection = read_json(campaign / "selection.json")
    assert selection["candidate"] == (1 if improves else 0)
    assert selection["test_scores_seen"] is False
    assert sum(name.startswith("train_t") and name.endswith("_s42") for name in calls) == 3
    assert calls.count("train_control_s44") == 1 and "train_t0_s44" not in calls
    assert len(list((campaign / "results/analysis").glob("sts7_*.json"))) == (12 if improves else 6)
    selected = load_config(workspace / "configs/coviso_relax/selected_joint.yaml")
    assert (
        selected["experiment"]["selected_from_dev_only"] and selected["train"]["max_steps"] == 4000
    )
    assert read_json(campaign / "initialization_audit.json")["same_initialization_per_seed"]


def test_parallel_full_ema_adopts_active_trial_and_spends_only_remaining_budget(tmp_path):
    import threading
    import optuna
    from scripts.parallel_full_ema import parallel_search

    study = optuna.create_study(storage=f"sqlite:///{tmp_path / 'study.db'}", direction="maximize")
    old = study.ask()
    old.set_user_attr('run', {'preserved': True})
    study.tell(old, 0.7)
    active_trial = study.ask()
    active_trial.suggest_float('x', 0.0, 1.0)
    lock, barrier = threading.Lock(), threading.Barrier(2)
    counts = {'active': 0, 'peak': 0}
    calls, completed, stopped = [], [], []

    def prepare(trial):
        if trial.number == 1:
            return dict(study.trials[1].params)
        return {'x': trial.suggest_float('x', 0.0, 1.0)}

    def execute(number, gpu, cfg):
        with lock:
            calls.append(number)
            counts['active'] += 1
            counts['peak'] = max(counts['peak'], counts['active'])
        try:
            if number in (1, 2):
                barrier.wait(timeout=10)
            return {'metrics': {'best_sts': 0.7 + number / 1000}, 'config': cfg}
        finally:
            with lock:
                counts['active'] -= 1

    parallel_search(study, 6, ['auto', 'auto'], prepare, execute, completed.append,
                    lambda: stopped.append(True))
    assert sorted(calls) == [1, 2, 3, 4, 5]
    assert len(study.trials) == 6 and all(t.state.name == 'COMPLETE' for t in study.trials)
    assert study.trials[0].user_attrs['run'] == {'preserved': True}
    assert counts['peak'] == 2 and len(completed) == 5 and not stopped
    assert study.trials[1].params == active_trial.params


def test_parallel_full_ema_failure_stops_workers_before_waiting(tmp_path):
    import threading
    import optuna
    from scripts.parallel_full_ema import parallel_search

    study = optuna.create_study(storage=f"sqlite:///{tmp_path / 'study.db'}")
    entered, stopping = threading.Event(), threading.Event()

    def execute(number, gpu, cfg):
        if number == 0:
            assert entered.wait(10)
            raise RuntimeError('worker failed')
        entered.set()
        assert stopping.wait(10)
        raise RuntimeError('stopped peer')

    with pytest.raises(RuntimeError, match='worker failed'):
        parallel_search(study, 20, ['auto', 'auto'], lambda trial: {}, execute,
                        lambda run: None, stopping.set)
    assert stopping.is_set() and len(study.trials) == 2
