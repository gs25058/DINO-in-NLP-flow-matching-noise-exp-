import copy
import json
from pathlib import Path

import optuna
import pytest
import yaml
from optuna.trial import TrialState

from scripts.run_coviso_optuna import (
    attempted_count,
    complete_record,
    import_control,
    judgment,
    read_records,
    top_trials,
)
from src.joint_search import (
    SPACE,
    baseline_parameters,
    config_from_parameters,
    suggest_config,
)
from src.provenance import sha256_file
from src.train import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_is_inside_space_and_roundtrips_without_changing_method():
    base = load_config(ROOT / "configs/coviso_relax/base.yaml")
    before = copy.deepcopy(base)
    params = baseline_parameters(base)
    result = config_from_parameters(base, params)
    assert len(params) == 19
    assert base == before
    for key in ("model", "data", "eval"):
        assert result[key] == base[key]
    for section in ("train", "loss", "augment"):
        for key, value in base[section].items():
            assert (
                result[section][key] == pytest.approx(value)
                if isinstance(value, float)
                else result[section][key] == value
            )
    assert (
        result["loss"]["cov_iso_mean_lambda"]
        == result["loss"]["cov_iso_cov_lambda"]
        == 5.874093568592943
    )


def test_samples_change_only_declared_knobs_and_preserve_noise_constraints():
    base = load_config(ROOT / "configs/coviso_relax/base.yaml")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    for _ in range(12):
        trial = study.ask()
        cfg = suggest_config(trial, base)
        assert (
            0 <= cfg["augment"]["t_lo"] < cfg["augment"]["t_start"] <= cfg["augment"]["t_max"] < 1
        )
        assert cfg["train"]["momentum_end"] > cfg["train"]["momentum_start"]
        assert cfg["loss"]["warmup_teacher_temp"] <= cfg["loss"]["teacher_temp"]
        for section in ("train", "loss", "augment"):
            for key in cfg[section]:
                if f"{section}.{key}" not in SPACE and key not in ("t_start", "t_max"):
                    assert cfg[section][key] == base[section][key]
        assert cfg["model"] == base["model"]
        assert cfg["data"] == base["data"]
        study.tell(trial, 0.0)
    params = baseline_parameters(base)
    params["loss.cov_iso_mean_lambda"] = 0.05
    params["loss.cov_iso_cov_lambda"] = 8.0
    params["augment.t_lo"] = 0.6
    params["_t_span"] = 0.6
    cfg = config_from_parameters(base, params)
    assert cfg["augment"]["t_max"] == 0.99
    assert cfg["loss"]["cov_iso_mean_lambda"] != cfg["loss"]["cov_iso_cov_lambda"]
    params["train.lr"] = float("nan")
    with pytest.raises(ValueError, match="outside"):
        config_from_parameters(base, params)


def test_budget_counts_pruned_failed_running_and_imported_but_not_waiting():
    study = optuna.create_study(direction="maximize")
    study.add_trial(
        optuna.trial.create_trial(value=0.73, params={}, distributions={}, user_attrs={"run": {}})
    )
    study.enqueue_trial({})
    assert attempted_count(study) == 1
    trial = study.ask()
    assert attempted_count(study) == 2
    study.tell(trial, state=TrialState.PRUNED)
    trial = study.ask()
    study.tell(trial, state=TrialState.FAIL)
    trial = study.ask()
    assert attempted_count(study) == 4
    assert [t.number for t in top_trials(study)] == [0]


def test_pruner_waits_for_late_steps_and_enough_full_controls():
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8, n_warmup_steps=2000, interval_steps=500, n_min_trials=4
    )
    study = optuna.create_study(direction="maximize", pruner=pruner)
    for _ in range(8):
        study.add_trial(
            optuna.trial.create_trial(
                value=0.75, intermediate_values={1750: 0.73, 2000: 0.74, 2500: 0.75}
            )
        )
    trial = study.ask()
    trial.report(0.1, 1750)
    assert not trial.should_prune()
    trial.report(0.2, 2000)
    assert trial.should_prune()


def test_partial_completion_and_partial_json_lines_are_not_accepted(tmp_path):
    path = tmp_path / "completion.json"
    path.write_text(json.dumps({"status": "smoke_complete", "completed_steps": 60}))
    with pytest.raises(RuntimeError, match="4000-step"):
        complete_record(path)
    path.write_text('{"loop_step": 250}\n{"loop_')
    assert read_records(path) == [{"loop_step": 250}]


def test_joint_verdict_requires_mean_and_paired_support():
    def runs(vals):
        return {s: {"metrics": {"best_sts": v, "last_sts": v}} for s, v in zip((42, 43, 44), vals)}

    base = runs([0.73, 0.74, 0.735])
    assert judgment(base, runs([0.734, 0.744, 0.739]))["verdict"] == "성립"
    assert judgment(base, runs([0.729, 0.739, 0.734]))["verdict"] == "기각"
    assert judgment(base, runs([0.731, 0.741, 0.736]))["verdict"] == "판정불가"


def test_import_requires_matching_source_config_data_and_complete_runs(tmp_path):
    base = load_config(ROOT / "configs/coviso_relax/base.yaml")
    previous, campaign = tmp_path / "old", tmp_path / "new"
    (previous / "source/src").mkdir(parents=True)
    (previous / "configs").mkdir()
    (campaign / "configs").mkdir(parents=True)
    source = previous / "source/src/train.py"
    source.write_text("immutable source\n")
    manifest = {
        "workspace": str(tmp_path),
        "source_files": {"src/train.py": sha256_file(source)},
    }
    (previous / "source_manifest.json").write_text(
        json.dumps({"source_files": manifest["source_files"], "git": {"commit": "verified"}})
    )
    for key in ("sentences_path", "rank_eval_path", "embed_stats_path"):
        data = tmp_path / base["data"][key]
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text("fixed data")
    for seed in (42, 43):
        cfg = copy.deepcopy(base)
        cfg.update(seed=seed, run_name=f"control_s{seed}")
        for key in ("sentences_path", "rank_eval_path", "embed_stats_path"):
            cfg["data"][key] = str(tmp_path / cfg["data"][key])
        (previous / "configs" / f"control_s{seed}.yaml").write_text(yaml.safe_dump(cfg))
        logs = previous / "results/logs" / cfg["run_name"]
        checkpoints = previous / "checkpoints" / cfg["run_name"]
        logs.mkdir(parents=True)
        checkpoints.mkdir(parents=True)
        for which in ("best", "last"):
            (checkpoints / f"{which}.pt").write_text("checkpoint fixture")
        (logs / "completion.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "completed_steps": 4000,
                    "last_loop_step": 3999,
                    "best_sts": 0.74,
                    "last_sts": 0.73,
                }
            )
        )
        (logs / "provenance.json").write_text(
            json.dumps(
                {
                    "initial_student_sha256": f"init_{seed}",
                    "data": {
                        key: {
                            "path": cfg["data"][key],
                            "sha256": sha256_file(Path(cfg["data"][key])),
                        }
                        for key in (
                            "sentences_path",
                            "rank_eval_path",
                            "embed_stats_path",
                        )
                    },
                }
            )
        )
    runs = import_control(campaign, manifest, previous, base)
    assert set(runs) == {42, 43}
    assert (campaign / "checkpoints/control_s42").is_symlink()
    assert (previous / "checkpoints/control_s42/best.pt").read_text() == "checkpoint fixture"
    original_digest = manifest["source_files"]["src/train.py"]
    manifest["source_files"]["src/train.py"] = "different"
    with pytest.raises(RuntimeError, match="source mismatch"):
        import_control(campaign, manifest, previous, base)
    manifest["source_files"]["src/train.py"] = original_digest
    (tmp_path / base["data"]["sentences_path"]).write_text("changed data")
    with pytest.raises(RuntimeError, match="data changed"):
        import_control(campaign, manifest, previous, base)
