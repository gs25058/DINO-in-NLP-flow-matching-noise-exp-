"""Bounded joint Optuna search, independent seed confirmation, then frozen STS7.

The sampler runs in one controller thread; up to two guarded GPU children train.
Training uses the existing src.train loop without search-specific modifications.
"""

import argparse
import copy
import math
import os
import queue
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

import optuna
import yaml
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_optuna import export_selected_config, judgment, render  # noqa: E402
from src.joint_search import baseline_parameters, distributions, suggest_config  # noqa: E402
from src.provenance import (  # noqa: E402
    prepare,
    read_json,
    read_records,
    sha256_file,  # noqa: E402
    verify_source,
    write_json,
)
from src.train import load_config  # noqa: E402


def complete_record(path):
    if not path.exists():
        raise RuntimeError(f"missing completion: {path}")
    row = read_json(path)
    if (
        row.get("status") != "complete"
        or row.get("completed_steps") != 4000
        or row.get("last_loop_step") != 3999
        or not all(math.isfinite(row.get(k, float("nan"))) for k in ("best_sts", "last_sts"))
    ):
        raise RuntimeError(f"not a completed 4000-step run: {path}")
    return row


def attempted_count(study):
    return sum(t.state != TrialState.WAITING for t in study.get_trials(deepcopy=False))


def import_control(campaign, manifest, previous, base):
    """Reuse measured controls only after checking actual sources, config and data."""
    old = read_json(previous / "source_manifest.json")
    checked = []
    for relative, digest in old["source_files"].items():
        if relative.startswith("src/") or relative in ("scripts/eval_sts7.py",):
            if sha256_file(previous / "source" / relative) != digest:
                raise RuntimeError(f"old source changed: {relative}")
            if manifest["source_files"].get(relative) != digest:
                raise RuntimeError(f"control training/evaluation source mismatch: {relative}")
            checked.append(relative)
    result = {}
    for seed in (42, 43):
        cfg_path = previous / "configs" / f"control_s{seed}.yaml"
        cfg = load_config(cfg_path)
        for section in ("model", "train", "data", "eval", "augment", "loss"):
            expected = copy.deepcopy(base[section])
            if section == "data":
                for key in ("sentences_path", "rank_eval_path", "embed_stats_path"):
                    expected[key] = str(Path(manifest["workspace"]) / expected[key])
            if cfg[section] != expected:
                raise RuntimeError(f"control config differs: seed={seed}, section={section}")
        logs = previous / "results/logs" / cfg["run_name"]
        metrics = complete_record(logs / "completion.json")
        provenance = read_json(logs / "provenance.json")
        for item in provenance["data"].values():
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise RuntimeError("control data changed since its measured run")
        checkpoint = previous / "checkpoints" / cfg["run_name"]
        if not all((checkpoint / f"{which}.pt").is_file() for which in ("best", "last")):
            raise RuntimeError("control best/last checkpoints missing")
        # Link, never move or rewrite, the original logs/checkpoints.
        for target, original in (
            (campaign / "results/logs" / cfg["run_name"], logs),
            (campaign / "checkpoints" / cfg["run_name"], checkpoint),
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(original, target_is_directory=True)
        saved = campaign / "configs" / f"import_control_s{seed}.yaml"
        if not saved.exists():
            saved.write_text(yaml.safe_dump(cfg, sort_keys=False))
        result[seed] = {
            "run_name": cfg["run_name"],
            "config": str(saved),
            "metrics": metrics,
            "original_campaign": str(previous),
            "original_config": str(cfg_path),
            "original_source_commit": old["git"]["commit"],
            "imported": True,
            "initial_student_sha256": provenance["initial_student_sha256"],
        }
    write_json(
        campaign / "imported_control.json",
        {
            "verified_source_files": checked,
            "runs": result,
            "verified_data_sha256": True,
        },
    )
    return result


class Children:
    """Own only our GPU guards; terminate guards so they clean their child groups."""

    def __init__(self, campaign, env):
        self.campaign, self.env = campaign, env
        self.lock = threading.Lock()
        self.active = set()
        self.stopping = threading.Event()

    def stop(self, *_args):
        self.stopping.set()
        with self.lock:
            for process in self.active:
                if process.poll() is None:
                    process.terminate()

    def run(
        self,
        gpu,
        label,
        command,
        trial=None,
        evaluation_path=None,
        expected_initialization=None,
    ):
        if self.stopping.is_set():
            raise RuntimeError("controller was stopped")
        campaign = self.campaign
        guarded = [
            sys.executable,
            str(campaign / "source/scripts/gpu_guard.py"),
            "--gpu",
            gpu,
            "--wait",
            "--timeout",
            "7200",
            "--",
            *command,
        ]
        log = campaign / "commands" / f"{label}.log"
        if log.exists():
            raise RuntimeError(f"preserving existing attempt: {log}")
        write_json(
            campaign / "commands" / f"{label}.json",
            {
                "argv": guarded,
                "shell_display": shlex.join(guarded),
                "cwd": str(campaign / "source"),
                "gpu": gpu,
            },
        )
        start, seen, best, pruned = time.time(), set(), -1.0, False
        process = None
        try:
            with log.open("w") as stream:
                process = subprocess.Popen(
                    guarded,
                    cwd=campaign / "source",
                    env=self.env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                with self.lock:
                    self.active.add(process)
                    if self.stopping.is_set():
                        process.terminate()
                while process.poll() is None:
                    if expected_initialization is not None:
                        provenance = evaluation_path.parent / "provenance.json"
                        if provenance.exists():
                            if (
                                read_json(provenance)["initial_student_sha256"]
                                != expected_initialization
                            ):
                                raise RuntimeError("initialization differs from imported control")
                            expected_initialization = None
                    if trial is not None:
                        for row in read_records(evaluation_path):
                            step = row["loop_step"]
                            if step in seen:
                                continue
                            seen.add(step)
                            best = max(best, row["sts_b_dev"])
                            trial.report(best, step)
                            # Never prune at the final evaluation: it is a full trial.
                            if step < 3999 and trial.should_prune():
                                pruned = True
                                process.terminate()
                                break
                    if pruned:
                        break
                    time.sleep(1)
                process.wait(timeout=40)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=40)
                with self.lock:
                    self.active.discard(process)
            write_json(
                campaign / "status" / f"{label}.json",
                {
                    "returncode": None if process is None else process.returncode,
                    "elapsed_seconds": time.time() - start,
                    "pruned": pruned,
                    "reported_steps": sorted(seen),
                    "log": str(log),
                },
            )
        if pruned:
            raise optuna.TrialPruned(f"median pruning; last reported loop step {max(seen)}")
        if process.returncode:
            raise RuntimeError(f"{label} failed ({process.returncode}); see {log}")


def save_run_config(campaign, manifest, cfg, label, seed):
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    cfg["run_name"] = f"r18_bert_joint_optuna_{campaign.name}_{label}_s{seed}"
    for key in ("sentences_path", "rank_eval_path", "embed_stats_path"):
        cfg["data"][key] = str(Path(manifest["workspace"]) / cfg["data"][key])
    cfg["experiment"] = {
        "source_manifest": str(campaign / "source_manifest.json"),
        "protocol": "joint_optuna",
        "optuna_spec_sha256": manifest["optuna_spec_sha256"],
        "source_trial": label,
        "fresh_pretrained_start": True,
    }
    path = campaign / "configs" / f"{label}_s{seed}.yaml"
    if path.exists() and yaml.safe_load(path.read_text()) != cfg:
        raise RuntimeError(f"config changed: {path}")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path, cfg


def train(campaign, manifest, children, gpu, cfg, label, seed, trial=None):
    path, cfg = save_run_config(campaign, manifest, cfg, label, seed)
    logdir = campaign / "results/logs" / cfg["run_name"]
    completion = logdir / "completion.json"
    if completion.exists():
        metrics = complete_record(completion)
    else:
        verify_source(campaign, manifest)
        control_path = campaign / "control_runs.json"
        controls = read_json(control_path) if control_path.exists() else {}
        expected = controls.get(str(seed), {}).get("initial_student_sha256")
        children.run(
            gpu,
            f"train_{label}_s{seed}",
            [
                sys.executable,
                "-m",
                "src.train",
                "--config",
                str(path),
                "--device",
                "cuda",
            ],
            trial=trial,
            evaluation_path=logdir / "evaluations.jsonl",
            expected_initialization=expected,
        )
        metrics = complete_record(completion)
    return {
        "run_name": cfg["run_name"],
        "config": str(path),
        "metrics": metrics,
        "imported": False,
        "initial_student_sha256": read_json(logdir / "provenance.json")["initial_student_sha256"],
    }


def top_trials(study, n=3):
    return sorted(
        (
            t
            for t in study.get_trials(deepcopy=False)
            if t.state == TrialState.COMPLETE and "run" in t.user_attrs
        ),
        key=lambda t: (-t.value, t.number),
    )[:n]


def parallel_jobs(jobs, gpus, children, execute, on_result):
    available = queue.Queue()
    for gpu in gpus:
        available.put(gpu)

    def worker(job):
        gpu = available.get()
        try:
            return job, execute(gpu, job)
        except BaseException:
            children.stop()
            raise
        finally:
            available.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in as_completed([pool.submit(worker, job) for job in jobs]):
            on_result(*future.result())


def search(campaign, manifest, study, base, children):
    """Serial ask/tell, at most two concurrent GPU children, fixed attempt budget."""
    failures = 0
    with ThreadPoolExecutor(max_workers=len(manifest["gpus"])) as pool:
        active, available = {}, list(manifest["gpus"])
        while active or attempted_count(study) < manifest["trials"]:
            if children.stopping.is_set():
                raise RuntimeError("controller stopped; partial runs preserved")
            while available and attempted_count(study) < manifest["trials"]:
                gpu = available.pop(0)
                trial = study.ask()
                cfg = suggest_config(trial, base)
                future = pool.submit(
                    train,
                    campaign,
                    manifest,
                    children,
                    gpu,
                    cfg,
                    f"t{trial.number}",
                    42,
                    trial,
                )
                active[future] = (trial, gpu)
                print(f"[joint] started trial={trial.number} gpu={gpu}", flush=True)
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                trial, gpu = active.pop(future)
                available.append(gpu)
                try:
                    result = future.result()
                except optuna.TrialPruned:
                    study.tell(trial, state=TrialState.PRUNED)
                    failures = 0
                except Exception as exc:
                    trial.set_user_attr("error", str(exc))
                    study.tell(trial, state=TrialState.FAIL)
                    failures += 1
                    if failures >= 3:
                        children.stop()
                        raise RuntimeError(
                            "three consecutive run errors; stopping to preserve evidence"
                        ) from exc
                else:
                    trial.set_user_attr("run", result)
                    study.tell(trial, result["metrics"]["best_sts"])
                    failures = 0
                print(
                    f"[joint] trial={trial.number} state={study.trials[trial.number].state.name}",
                    flush=True,
                )
                render(campaign, manifest, study, "optimization")


def confirm_candidates(campaign, manifest, study, base, children, controls, top):
    """Confirm the fixed shortlist; trial0 shares the three control runs."""
    confirmations = {"control": dict(controls)}
    for t in top:
        confirmations[f"t{t.number}"] = (
            dict(controls) if t.number == 0 else {42: t.user_attrs["run"]}
        )
    jobs = [
        (label, seed)
        for label, seeds in confirmations.items()
        if label != "t0"
        for seed in (43, 44)
        if seed not in seeds
    ]

    def confirm(gpu, job):
        label, seed = job
        parent = (
            base if label == "control" else load_config(Path(confirmations[label][42]["config"]))
        )
        return train(campaign, manifest, children, gpu, parent, label, seed)

    def confirmed(job, result):
        label, seed = job
        confirmations[label][seed] = result
        if label == "control" and "t0" in confirmations:
            confirmations["t0"][seed] = result
        write_json(campaign / "confirmations.json", confirmations)
        render(campaign, manifest, study, "seed_confirmation", confirmations)

    parallel_jobs(jobs, manifest["gpus"], children, confirm, confirmed)
    return confirmations


def run(campaign, manifest):
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HOME=os.environ.get("HF_HOME", "/src/gs25058/scratch/.hf_home"),
        PYTHONPATH=str(campaign / "source"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    children = Children(campaign, env)
    signal.signal(signal.SIGTERM, children.stop)
    signal.signal(signal.SIGINT, children.stop)
    sampler = optuna.samplers.TPESampler(
        seed=42, n_startup_trials=8, multivariate=True, constant_liar=True
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8, n_warmup_steps=2000, interval_steps=500, n_min_trials=4
    )
    study = optuna.create_study(
        storage=f"sqlite:///{campaign / 'study.db'}",
        study_name=campaign.name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    base = load_config(campaign / "source/configs/coviso_relax/base.yaml")
    controls, confirmations, stage = {}, {}, "preflight"
    try:
        # The source, sampler state and partial output of a killed optimization are
        # preserved. Refuse a silent restart with a new sampler/RNG sequence.
        if study.trials:
            raise RuntimeError("existing study preserved; automatic sampler restart is disabled")
        if manifest.get("import_baseline"):
            controls = import_control(campaign, manifest, Path(manifest["import_baseline"]), base)
        else:
            stage = "control_training"
            parallel_jobs(
                (42, 43),
                manifest["gpus"],
                children,
                lambda gpu, seed: train(campaign, manifest, children, gpu, base, "control", seed),
                lambda seed, result: controls.update({seed: result}),
            )
        write_json(campaign / "control_runs.json", controls)
        baseline_params = baseline_parameters(base)
        study.add_trial(
            optuna.trial.create_trial(
                value=controls[42]["metrics"]["best_sts"],
                params=baseline_params,
                distributions=distributions(),
                user_attrs={
                    "run": controls[42],
                    "imported_control": controls[42]["imported"],
                },
                intermediate_values={
                    row["loop_step"]: max(
                        r["sts_b_dev"]
                        for r in read_records(
                            campaign
                            / "results/logs"
                            / controls[42]["run_name"]
                            / "evaluations.jsonl"
                        )
                        if r["loop_step"] <= row["loop_step"]
                    )
                    for row in read_records(
                        campaign / "results/logs" / controls[42]["run_name"] / "evaluations.jsonl"
                    )
                },
            )
        )
        anchor = dict(
            baseline_params,
            **{"loss.cov_iso_mean_lambda": 1.0, "loss.cov_iso_cov_lambda": 1.0},
        )
        study.enqueue_trial(anchor, user_attrs={"anchor": "both_weights_1_rest_R18"})
        stage = "optimization"
        render(campaign, manifest, study, stage)
        search(campaign, manifest, study, base, children)
        top = top_trials(study)
        if len(top) < 3:
            raise RuntimeError("fewer than three full-budget trials; insufficient candidates")
        write_json(
            campaign / "shortlist.json",
            {"trials": [t.number for t in top], "test_scores_seen": False},
        )
        stage = "seed_confirmation"
        confirmations = confirm_candidates(campaign, manifest, study, base, children, controls, top)
        per_seed = {}
        for path in (campaign / "results/logs").glob("*/provenance.json"):
            row = read_json(path)
            old_hash = per_seed.setdefault(row["seed"], row["initial_student_sha256"])
            if old_hash != row["initial_student_sha256"]:
                raise RuntimeError("initialization differs within the same seed")
        write_json(
            campaign / "initialization_audit.json",
            {"same_initialization_per_seed": True, "sha256": per_seed},
        )
        candidate = min(
            top,
            key=lambda t: (
                -statistics.mean(
                    confirmations[f"t{t.number}"][s]["metrics"]["best_sts"] for s in (42, 43, 44)
                ),
                t.number,
            ),
        ).number
        selected = confirmations[f"t{candidate}"]
        selection = {
            "candidate": candidate,
            "judgment": judgment(confirmations["control"], selected),
            "test_scores_seen": False,
            "selection_frozen_unix": time.time(),
            "shortlist": [t.number for t in top],
            "params": study.trials[candidate].params,
        }
        write_json(campaign / "selection.json", selection)
        destination = Path(manifest["workspace"]) / "configs/coviso_relax/selected_joint.yaml"
        export_selected_config(campaign, base, campaign / "selected_joint.yaml")
        export_selected_config(campaign, base, destination)
        stage = "test_evaluation"
        render(campaign, manifest, study, stage, confirmations)
        evaluation_groups = {"control": confirmations["control"]}
        if candidate != 0:
            evaluation_groups[f"t{candidate}"] = selected
        jobs = [
            (label, seed, which)
            for label in evaluation_groups
            for seed in (42, 43, 44)
            for which in ("best", "last")
        ]

        def evaluate(gpu, job):
            label, seed, which = job
            record = evaluation_groups[label][seed]
            out = campaign / "results/analysis" / f"sts7_{label}_s{seed}_{which}.json"
            children.run(
                gpu,
                f"eval_{label}_s{seed}_{which}",
                [
                    sys.executable,
                    "scripts/eval_sts7.py",
                    "--runs",
                    record["run_name"],
                    "--which",
                    which,
                    "--device",
                    "cuda",
                    "--tf32",
                    "--out",
                    str(out),
                ],
            )
            data = read_json(out)
            if record["run_name"] not in data["runs"]:
                raise RuntimeError("evaluation did not return the requested run")

        parallel_jobs(
            jobs,
            manifest["gpus"],
            children,
            evaluate,
            lambda *_: render(campaign, manifest, study, stage, confirmations),
        )
        render(campaign, manifest, study, "complete", confirmations)
    except BaseException as exc:
        children.stop()
        render(campaign, manifest, study, "failed", confirmations, f"{stage}: {exc}")
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--import-baseline", type=Path)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    campaign = args.campaign.resolve()
    if args.execute:
        manifest = read_json(campaign / "source_manifest.json")
        verify_source(campaign, manifest)
        if sha256_file(campaign / "OPTUNA_SPEC.md") != manifest["optuna_spec_sha256"]:
            raise RuntimeError("frozen Optuna specification changed")
        run(campaign, manifest)
    else:
        gpus = args.gpus.split(",") if args.gpus else []
        if not 1 <= len(gpus) <= 2 or (len(gpus) != len(set(gpus)) and gpus != ["auto", "auto"]):
            ap.error("supply one/two distinct physical indices, or auto / auto,auto")
        if "auto" in gpus and any(g != "auto" for g in gpus):
            ap.error("do not mix automatic and explicit GPU selection")
        if args.trials < 4 or args.trials > 40:
            ap.error("trial budget must be between 4 and 40, including the imported control")
        manifest = prepare(campaign, ROOT, gpus, ROOT / "docs/experiments/coviso_relax/report.md")
        manifest.update(
            trials=args.trials,
            import_baseline=str(args.import_baseline.resolve()) if args.import_baseline else None,
            optuna_spec_sha256=sha256_file(campaign / "OPTUNA_SPEC.md"),
        )
        write_json(campaign / "source_manifest.json", manifest)
        if not args.prepare_only:
            os.execv(
                sys.executable,
                [
                    sys.executable,
                    str(campaign / "source/scripts/run_coviso_optuna.py"),
                    "--campaign",
                    str(campaign),
                    "--execute",
                ],
            )


if __name__ == "__main__":
    main()
