"""Continue an existing full-EMA search on two GPUs without restarting its active run.

The original training snapshot stays immutable. Only this controller changes:
Optuna ask/tell stays on one thread, with up to two GPU jobs in flight.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import optuna
from optuna.trial import TrialState


def parallel_search(study, budget, gpus, prepare, execute, completed, stop):
    """Adopt RUNNING trials first; allocate only the unspent total trial budget."""
    previous = study.get_trials(deepcopy=False)
    if len(previous) > budget or any(t.state == TrialState.FAIL for t in previous):
        raise RuntimeError("unexpected existing trial budget/state; preserve and inspect")
    adopted = [optuna.Trial(study, t._trial_id) for t in previous if t.state == TrialState.RUNNING]
    if len(adopted) > len(gpus):
        raise RuntimeError("more active trials than GPU slots")
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        active, available = {}, list(gpus)
        try:
            while active or adopted or len(study.trials) < budget:
                while available and (adopted or len(study.trials) < budget):
                    gpu = available.pop(0)
                    trial = adopted.pop(0) if adopted else study.ask()
                    cfg = prepare(trial)
                    active[pool.submit(execute, trial.number, gpu, cfg)] = (trial, gpu)
                done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    trial, gpu = active.pop(future)
                    run = future.result()
                    trial.set_user_attr("run", run)
                    study.tell(trial, run["metrics"]["best_sts"])
                    completed(run)
                    available.append(gpu)
        except BaseException:
            stop()
            raise


def process_state(pid):
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return result.stdout.strip()


def run(campaign, old_controller, record):
    # Load every training/evaluation component from the ORIGINAL frozen source.
    sys.path.insert(0, str(campaign / "source"))
    from scripts import run_bottleneck as pipeline
    from src.provenance import read_json, verify_source, write_json

    manifest = read_json(campaign / "source_manifest.json")
    verify_source(campaign, manifest)
    if manifest["protocol"] != "full_ema_joint" or manifest["trials_per_arm"] != 20:
        raise RuntimeError("this continuation is registered only for the existing 20-trial full-EMA search")
    if read_json(campaign / "state.json")["stage"] != "search":
        raise RuntimeError("original controller must still be in search")
    command = Path(f"/proc/{old_controller}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if str(campaign / "source/scripts/run_bottleneck.py") not in command:
        raise RuntimeError("old PID does not match this campaign controller")
    if Path(f"/proc/{old_controller}").stat().st_uid != os.getuid():
        raise RuntimeError("controller owner differs")
    env = dict(os.environ, PYTHONPATH=str(campaign / "source"), PYTHONDONTWRITEBYTECODE="1")
    children = pipeline.Children(campaign, env)
    parked, registered, guards = False, False, []

    def stop(*_):
        children.stop()
        for pid in guards:
            if process_state(pid) and not process_state(pid).startswith("Z"):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def interrupt(*_):
        if registered:
            stop()
        else:
            children.stop()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        # Stop only scheduling threads; its independently running GPU child continues.
        os.kill(old_controller, signal.SIGSTOP)
        parked = True
        for _ in range(100):
            if process_state(old_controller).startswith("T"):
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("old controller did not park")
        study = optuna.load_study(
            study_name=f"{campaign.name}_bottleneck", storage=f"sqlite:///{campaign / 'bottleneck.db'}",
            sampler=optuna.samplers.TPESampler(seed=20261110, n_startup_trials=5,
                                              multivariate=True, constant_liar=True),
            pruner=optuna.pruners.NopPruner(),
        )
        running = [t.number for t in study.trials if t.state == TrialState.RUNNING]
        guard_output = subprocess.check_output(["ps", "-ww", "--ppid", str(old_controller), "-o", "pid=,args="], text=True)
        for line in guard_output.splitlines():
            pid, args = line.strip().split(None, 1)
            if str(campaign / "source/scripts/gpu_guard.py") not in args:
                raise RuntimeError("unexpected child of the parked controller")
            guards.append(int(pid))
        if len(running) > 1 or len(guards) > 1 or (guards and not running):
            raise RuntimeError("unexpected old controller transition boundary")
        for number in running:
            if not (campaign / f"configs/search_bottleneck_t{number}_s42.yaml").exists():
                raise RuntimeError("active trial config not yet recorded")
        registration = {
            "created_unix": time.time(), "user_request": "Prefer two GPUs",
            "old_controller": old_controller, "adopted_guards": guards, "adopted_trials": running,
            "completed_before": [t.number for t in study.trials if t.state == TrialState.COMPLETE],
            "total_search_budget": 20, "max_parallel_trials": 2,
            "sampler": {"seed": 20261110, "n_startup_trials": 5, "multivariate": True, "constant_liar": True},
            "sampler_note": "Previous sampler RNG was not persisted. New parallel continuation seed explicitly registered; existing trials retained unchanged.",
            "training_source_commit": manifest["git"]["commit"],
            "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "training_evaluation_selection_unchanged": True,
        }
        write_json(record, registration)
        registered = True
        calibration = read_json(campaign / "calibration.json")
        base = pipeline.load_config(campaign / "source/configs/full_ema_joint/base.yaml")
        runs = [t.user_attrs["run"] for t in study.trials if t.state == TrialState.COMPLETE]
        adopted = set(running)
        retired = False

        def retire():
            nonlocal retired
            if not retired:
                os.kill(old_controller, signal.SIGKILL)
                retired = True

        if not adopted:
            retire()

        def prepare(trial):
            if trial.number in adopted:
                return pipeline.load_config(campaign / f"configs/search_bottleneck_t{trial.number}_s42.yaml")
            return pipeline.trial_config(trial, base, "bottleneck", calibration)

        def execute(number, gpu, cfg):
            if number in adopted:
                while any(process_state(pid) and not process_state(pid).startswith("Z") for pid in guards):
                    if children.stopping.wait(1):
                        raise RuntimeError("continuation stopped")
                log = campaign / "results/logs" / cfg["run_name"]
                pipeline.complete_record(log / "completion.json")
                # Parent was parked, so it could not collect the already-exited guard.
                retire()
                write_json(campaign / "status" / f"train_search_bottleneck_t{number}_s42.json", {
                    "returncode": 0, "adopted_by_parallel_controller": True,
                    "completion_validated": True, "pruned": False,
                    "elapsed_seconds": time.time() - (campaign / "commands" / f"train_search_bottleneck_t{number}_s42.json").stat().st_mtime,
                    "registration": str(record),
                })
            return pipeline.execute_train(campaign, manifest, children, gpu, cfg,
                                          f"search_bottleneck_t{number}", 42)

        def completed(run):
            runs.append(run)
            pipeline.audit_runs(campaign, runs, calibration)
            pipeline.set_stage(campaign, "search", completed_by_arm={"bottleneck": len(runs)},
                               completed_pairs=len(runs), parallel_trials=2,
                               continuation_registration=str(record))

        pipeline.set_stage(campaign, "search", completed_by_arm={"bottleneck": len(runs)},
                           completed_pairs=len(runs), parallel_trials=2, continuation_registration=str(record))
        parallel_search(study, 20, ["auto", "auto"], prepare, execute, completed, stop)
        pipeline.set_stage(campaign, "confirmation")
        groups, confirmed = pipeline.confirm(campaign, manifest, {"bottleneck": study}, children, calibration)
        groups["baseline"] = read_json(campaign / "fixed_baseline.json")["runs"]
        write_json(campaign / "final_runs.json", groups)
        pipeline.audit_runs(campaign, runs + confirmed, calibration)
        pipeline.set_stage(campaign, "test_evaluation")
        pipeline.evaluate(campaign, manifest, groups, children)
        pipeline.set_stage(campaign, "complete")
    except BaseException as exc:
        if parked and not registered:
            children.stop()
            os.kill(old_controller, signal.SIGCONT)
        elif registered:
            stop()
            # Never let the old fixed-length scheduler spend this budget again.
            if process_state(old_controller) and not process_state(old_controller).startswith("Z"):
                os.kill(old_controller, signal.SIGKILL)
            pipeline.set_stage(campaign, "failed", error=str(exc), continuation_registration=str(record))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--old-controller", type=int, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    if args.record.exists():
        parser.error("preserving existing continuation registration")
    run(args.campaign.resolve(), args.old_controller, args.record.resolve())


if __name__ == "__main__":
    main()
