"""Bounded joint Optuna search, independent seed confirmation, then frozen STS7.

The sampler runs in one controller thread; up to two guarded GPU children train.
Training uses the existing src.train loop without search-specific modifications.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time

import optuna
from optuna.trial import TrialState
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_coviso_relax import prepare, verify_source, read_json, write_json  # noqa: E402
from src.joint_search import SPACE, baseline_parameters, distributions, suggest_config  # noqa: E402
from src.provenance import sha256_file  # noqa: E402
from src.train import load_config  # noqa: E402


def complete_record(path):
    if not path.exists():
        raise RuntimeError(f"missing completion: {path}")
    row = read_json(path)
    if (row.get("status") != "complete" or row.get("completed_steps") != 4000
            or row.get("last_loop_step") != 3999
            or not all(math.isfinite(row.get(k, float("nan"))) for k in ("best_sts", "last_sts"))):
        raise RuntimeError(f"not a completed 4000-step run: {path}")
    return row


def read_records(path):
    if not path.exists():
        return []
    # The writer can be in the middle of its last line while we poll it.
    lines = path.read_text().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n")]


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
        for section in ("model", "train", "data", "eval", "logging", "augment", "loss", "collapse_guard"):
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
        for target, original in ((campaign / "results/logs" / cfg["run_name"], logs),
                                 (campaign / "checkpoints" / cfg["run_name"], checkpoint)):
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(original, target_is_directory=True)
        saved = campaign / "configs" / f"import_control_s{seed}.yaml"
        if not saved.exists():
            saved.write_text(yaml.safe_dump(cfg, sort_keys=False))
        result[seed] = {"run_name": cfg["run_name"], "config": str(saved), "metrics": metrics,
                        "original_campaign": str(previous), "original_config": str(cfg_path),
                        "original_source_commit": old["git"]["commit"], "imported": True,
                        "initial_student_sha256": provenance["initial_student_sha256"]}
    write_json(campaign / "imported_control.json", {"verified_source_files": checked,
                                                     "runs": result, "verified_data_sha256": True})
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

    def run(self, gpu, label, command, trial=None, evaluation_path=None, expected_initialization=None):
        if self.stopping.is_set():
            raise RuntimeError("controller was stopped")
        campaign = self.campaign
        guarded = [sys.executable, str(campaign / "source/scripts/gpu_guard.py"),
                   "--gpu", gpu, "--wait", "--timeout", "7200", "--", *command]
        log = campaign / "commands" / f"{label}.log"
        if log.exists():
            raise RuntimeError(f"preserving existing attempt: {log}")
        write_json(campaign / "commands" / f"{label}.json",
                   {"argv": guarded, "shell_display": shlex.join(guarded),
                    "cwd": str(campaign / "source"), "gpu": gpu})
        start, seen, best, pruned = time.time(), set(), -1., False
        process = None
        try:
            with log.open("w") as stream:
                process = subprocess.Popen(guarded, cwd=campaign / "source", env=self.env,
                                           stdout=stream, stderr=subprocess.STDOUT)
                with self.lock:
                    self.active.add(process)
                    if self.stopping.is_set():
                        process.terminate()
                while process.poll() is None:
                    if expected_initialization is not None:
                        provenance = evaluation_path.parent / "provenance.json"
                        if provenance.exists():
                            if read_json(provenance)["initial_student_sha256"] != expected_initialization:
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
            write_json(campaign / "status" / f"{label}.json",
                       {"returncode": None if process is None else process.returncode,
                        "elapsed_seconds": time.time() - start, "pruned": pruned,
                        "reported_steps": sorted(seen), "log": str(log)})
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
    cfg["experiment"] = {"source_manifest": str(campaign / "source_manifest.json"),
                         "protocol": "joint_optuna", "optuna_spec_sha256": manifest["optuna_spec_sha256"],
                         "source_trial": label, "fresh_pretrained_start": True}
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
        imported = read_json(campaign / "imported_control.json")["runs"]
        expected = imported.get(str(seed), {}).get("initial_student_sha256")
        children.run(gpu, f"train_{label}_s{seed}",
                     [sys.executable, "-m", "src.train", "--config", str(path), "--device", "cuda"],
                     trial=trial, evaluation_path=logdir / "evaluations.jsonl", expected_initialization=expected)
        metrics = complete_record(completion)
    return {"run_name": cfg["run_name"], "config": str(path), "metrics": metrics, "imported": False}


def trial_runs(study):
    return {t.number: t.user_attrs["run"] for t in study.get_trials(deepcopy=False)
            if t.state == TrialState.COMPLETE and "run" in t.user_attrs}


def top_trials(study, n=3):
    return sorted((t for t in study.get_trials(deepcopy=False)
                   if t.state == TrialState.COMPLETE and "run" in t.user_attrs),
                  key=lambda t: (-t.value, t.number))[:n]


def judgment(control, candidate):
    deltas = [candidate[s]["metrics"]["best_sts"] - control[s]["metrics"]["best_sts"] for s in (42, 43, 44)]
    mean = statistics.mean(deltas)
    status = "성립" if mean >= .003 and sum(d > 0 for d in deltas) >= 2 else (
        "기각" if mean <= 0 and sum(d <= 0 for d in deltas) >= 2 else "판정불가")
    return {"verdict": status, "paired_best_deltas": deltas, "paired_best_mean": mean,
            "paired_last_mean": statistics.mean(candidate[s]["metrics"]["last_sts"] -
                                                control[s]["metrics"]["last_sts"] for s in (42, 43, 44))}


def render(campaign, manifest, study, stage, confirmations=None, error=None):
    trials = study.get_trials(deepcopy=False)
    counts = {state.name: sum(t.state == state for t in trials) for state in TrialState}
    lines = ["# 공동 Optuna 최적화 — 실행 보고서", "", f"상태: **{stage}**. `{campaign}`.",
             "", "사용자 지시로 수동 성분 매트릭스를 공동 성능 최적화로 대체했다. 사전 등록: OPTUNA_SPEC.md.",
             "여러 축을 함께 바꾸므로 특정 정규화 항의 인과 효과를 이 결과로 판정하지 않는다.", "",
             f"실행 소스: `{manifest['git']['commit']}`, branch `{manifest['git']['branch']}`.",
             f"기준선 원격 소스: `{manifest['baseline']['upstream_commit']}`.",
             f"실제 R18 대조군 λ₀={manifest['baseline']['lambda_0']} (재구성 기준선 아님).",
             "실제 명령·exit code: campaign의 commands/ 및 status/. 실행 소스와 diff: source_manifest.json, source_diff.patch, source/.",
             "기존 control 42/43은 핵심 소스·resolved config·데이터 해시·4000-step 완료를 확인하고 재사용했다.",
             f"trial 상태: `{json.dumps(counts)}`. 최대 예산 `{manifest['trials']}` (pruned/failed/기준선 포함).",
             "", "## seed42 탐색 결과 (완주 trial만 후보 자격)", "",
             "| trial | state | dev best | dev last | best step |", "|---|---|---|---|---|"]
    for t in trials:
        m = t.user_attrs.get("run", {}).get("metrics", {})
        lines.append(f"| {t.number} | {t.state.name} | {m.get('best_sts', '')} | {m.get('last_sts', '')} | {m.get('best_loop_step', '')} |")
    lines += ["", "trial별 정확한 19개 값은 trials.json, resolved config는 configs/ 및 각 run의 resolved.yaml에 기록한다.",
              "진단은 각 로그의 components.jsonl·gradient_diagnostics.jsonl·evaluations.jsonl에 있다."]
    if confirmations:
        lines += ["", "## 추가 시드 검증", "", "| 설정 | seed | best dev | last dev | alignment last | uniformity last | effective rank last |",
                  "|---|---|---|---|---|---|---|"]
        for label, seeds in sorted(confirmations.items()):
            for seed, run in sorted(seeds.items()):
                m = run["metrics"]
                rows = read_records(campaign / "results/logs" / run["run_name"] / "evaluations.jsonl")
                last = rows[-1]
                lines.append(f"| {label} | {seed} | {m['best_sts']:.6f} | {m['last_sts']:.6f} | {last['alignment']:.5f} | {last['uniformity']:.5f} | {last['eff_rank']:.2f} |")
        lines += ["", "| 설정 | n | best mean ± sample SD | last mean ± sample SD |", "|---|---|---|---|"]
        for label, seeds in sorted(confirmations.items()):
            vals = [[run["metrics"][key] for run in seeds.values()] for key in ("best_sts", "last_sts")]
            summaries = [f"{statistics.mean(v):.6f} ± {statistics.stdev(v) if len(v)>1 else 0:.6f}" for v in vals]
            lines.append(f"| {label} | {len(seeds)} | {summaries[0]} | {summaries[1]} |")
    selection_path = campaign / "selection.json"
    if selection_path.exists():
        selection = read_json(selection_path)
        lines += ["", f"최종 후보: trial `{selection['candidate']}`. 판정: **{selection['judgment']['verdict']}**.",
                  f"paired best 평균 Δ={selection['judgment']['paired_best_mean']:.6f}.",
                  "선택은 3시드 dev만 사용했고 selection.json을 동결한 뒤 test 평가를 시작했다.",
                  "재사용 기준선: configs/coviso_relax/control.yaml. 선택 조합: configs/coviso_relax/selected_joint.yaml."]
    evaluations = sorted((campaign / "results/analysis").glob("sts7_*.json")) if (campaign / "results/analysis").exists() else []
    if evaluations:
        scores = {}
        lines += ["", "## STS7 test (선택 후 측정, ×100)", "", "| 설정/seed/checkpoint | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | 평균 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for path in evaluations:
            data = read_json(path)
            row = next(iter(data["runs"].values()))
            label = path.stem.removeprefix("sts7_")
            lines.append("| " + label + " | " + " | ".join(f"{row[k]*100:.3f}" for k in
                ("STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICK-R", "avg_7task")) + " |")
            group = label.rsplit("_s", 1)[0] + "_" + label.rsplit("_", 1)[1]
            scores.setdefault(group, []).append(row["avg_7task"] * 100)
        for group, vals in sorted(scores.items()):
            lines.append(f"\n{group}: n={len(vals)}, 평균 {statistics.mean(vals):.3f}, sample SD {statistics.stdev(vals) if len(vals)>1 else 0:.3f}.")
    if confirmations:
        lines += ["", "## 마지막 step 손실과 backbone gradient norm", "",
                  "gradient는 각 가중치를 적용한 항을 독립 미분한 값이며 벡터 방향이 달라 norm의 합이 총 norm은 아니다.", "",
                  "| 설정/seed | DINO | mean raw | cov raw | mean 기여 | cov 기여 | g DINO | g mean | g cov |",
                  "|---|---|---|---|---|---|---|---|---|"]
        diagnostics = {}
        for label, seeds in sorted(confirmations.items()):
            for seed, run in sorted(seeds.items()):
                logdir = campaign / "results/logs" / run["run_name"]
                records = read_records(logdir / "components.jsonl") + read_records(logdir / "gradient_diagnostics.jsonl")
                rows = sorted((r for r in records if "grad_dino_backbone" in r), key=lambda r:r["loop_step"])
                if not rows:
                    continue
                last = rows[-1]
                diagnostics[f"{label}_s{seed}"] = {"first": rows[0], "last": last, "sampled_gradient_steps": rows}
                lines.append(f"| {label}/s{seed} | " + " | ".join(f"{last[k]:.6g}" for k in
                    ("L_DINO", "L_mean", "L_cov", "weighted_mean", "weighted_cov", "grad_dino_backbone", "grad_mean_backbone", "grad_cov_backbone")) + " |")
        write_json(campaign / "diagnostic_summary.json", diagnostics)
    lines += ["", "## 해석과 남은 제약", "",
              "R18 과거 dev 약0.7465, 7-task 약66.09는 참고 기록이다. 현재 환경의 대조군을 비교 기준으로 사용한다.",
              "노이즈 강도·최적화·EMA가 함께 바뀌므로 어느 단일 성분 완화의 인과 효과인지는 미확인이다.",
              "alignment/uniformity/rank는 진단 지표다. 손실의 절대 크기만으로 지배 항을 판단하지 않는다.",
              "미완주 trial 및 미평가 test 결과를 추정치로 채우지 않는다."]
    if error:
        lines += ["", f"실행 오류: `{error}`"]
    text = "\n".join(lines) + "\n"
    (campaign / "report.md").write_text(text)
    Path(manifest["report"]).write_text(text)
    write_json(campaign / "trials.json", [{"number": t.number, "state": t.state.name, "value": t.value,
                                            "params": t.params, "user_attrs": t.user_attrs,
                                            "intermediate_values": t.intermediate_values} for t in trials])
    write_json(campaign / "state.json", {"stage": stage, "trial_counts": counts, "error": error})


def run(campaign, manifest):
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled",
               HF_HOME=os.environ.get("HF_HOME", "/src/gs25058/scratch/.hf_home"),
               PYTHONPATH=str(campaign / "source"), PYTHONDONTWRITEBYTECODE="1")
    children = Children(campaign, env)
    signal.signal(signal.SIGTERM, children.stop)
    signal.signal(signal.SIGINT, children.stop)
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=8, multivariate=True, constant_liar=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2000, interval_steps=500, n_min_trials=4)
    study = optuna.create_study(storage=f"sqlite:///{campaign / 'study.db'}", study_name=campaign.name,
                                direction="maximize", sampler=sampler, pruner=pruner, load_if_exists=True)
    base = load_config(campaign / "source/configs/coviso_relax/base.yaml")
    controls, confirmations, stage = {}, {}, "preflight"
    try:
        controls = import_control(campaign, manifest, Path(manifest["import_baseline"]), base)
        # The source, sampler state and partial output of a killed optimization are
        # preserved. Refuse a silent restart with a new sampler/RNG sequence.
        if study.trials:
            raise RuntimeError("existing study preserved; automatic sampler restart is disabled")
        baseline_params = baseline_parameters(base)
        study.add_trial(optuna.trial.create_trial(value=controls[42]["metrics"]["best_sts"],
            params=baseline_params, distributions=distributions(),
            user_attrs={"run": controls[42], "imported_control": True},
            intermediate_values={row["loop_step"]: max(r["sts_b_dev"] for r in
                read_records(campaign / "results/logs" / controls[42]["run_name"] / "evaluations.jsonl")
                if r["loop_step"] <= row["loop_step"]) for row in
                read_records(campaign / "results/logs" / controls[42]["run_name"] / "evaluations.jsonl")}))
        anchor = dict(baseline_params, **{"loss.cov_iso_mean_lambda": 1., "loss.cov_iso_cov_lambda": 1.})
        study.enqueue_trial(anchor, user_attrs={"anchor": "both_weights_1_rest_R18"})
        stage = "optimization"
        render(campaign, manifest, study, stage)
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
                    future = pool.submit(train, campaign, manifest, children, gpu, cfg, f"t{trial.number}", 42, trial)
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
                            raise RuntimeError("three consecutive run errors; stopping to preserve evidence") from exc
                    else:
                        trial.set_user_attr("run", result)
                        study.tell(trial, result["metrics"]["best_sts"])
                        failures = 0
                    print(f"[joint] trial={trial.number} state={study.trials[trial.number].state.name}", flush=True)
                    render(campaign, manifest, study, stage)
        top = top_trials(study)
        if len(top) < 3:
            raise RuntimeError("fewer than three full-budget trials; insufficient candidates")
        write_json(campaign / "shortlist.json", {"trials": [t.number for t in top], "test_scores_seen": False})
        stage = "seed_confirmation"
        confirmations = {"control": dict(controls)}
        for t in top:
            confirmations[f"t{t.number}"] = dict(controls) if t.number == 0 else {42: t.user_attrs["run"]}
        jobs = [(label, seed) for label, seeds in confirmations.items() if label != "t0"
                for seed in (43, 44) if seed not in seeds]
        with ThreadPoolExecutor(max_workers=len(manifest["gpus"])) as pool:
            active, available = {}, list(manifest["gpus"])
            while jobs or active:
                while available and jobs:
                    label, seed = jobs.pop(0)
                    gpu = available.pop(0)
                    parent = base if label in ("control", "t0") else load_config(Path(confirmations[label][42]["config"]))
                    future = pool.submit(train, campaign, manifest, children, gpu, parent, label, seed)
                    active[future] = (label, seed, gpu)
                done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    label, seed, gpu = active.pop(future)
                    confirmations[label][seed] = future.result()
                    available.append(gpu)
                    # t0 and control are the same setting; avoid duplicate seed44.
                    if label == "control" and "t0" in confirmations:
                        confirmations["t0"][seed] = confirmations[label][seed]
                        jobs = [(a,s) for a,s in jobs if not (a == "t0" and s == seed)]
                    write_json(campaign / "confirmations.json", confirmations)
                    render(campaign, manifest, study, stage, confirmations)
        per_seed = {}
        for path in (campaign / "results/logs").glob("*/provenance.json"):
            row = read_json(path)
            old_hash = per_seed.setdefault(row["seed"], row["initial_student_sha256"])
            if old_hash != row["initial_student_sha256"]:
                raise RuntimeError("initialization differs within the same seed")
        write_json(campaign / "initialization_audit.json", {"same_initialization_per_seed": True, "sha256": per_seed})
        candidate = min(top, key=lambda t: (-statistics.mean(confirmations[f"t{t.number}"][s]["metrics"]["best_sts"]
                                                              for s in (42, 43, 44)), t.number)).number
        selected = confirmations[f"t{candidate}"]
        selection = {"candidate": candidate, "judgment": judgment(confirmations["control"], selected),
                     "test_scores_seen": False, "selection_frozen_unix": time.time(),
                     "shortlist": [t.number for t in top], "params": study.trials[candidate].params}
        write_json(campaign / "selection.json", selection)
        chosen = load_config(Path(selected[42]["config"]))
        chosen["run_name"] = "r18_bert_joint_optuna_selected_s42"
        chosen["experiment"] = {"selection_campaign": str(campaign), "selected_trial": candidate,
                               "selected_from_dev_only": True, "recommended": selection["judgment"]["verdict"] == "성립"}
        for key in ("sentences_path", "rank_eval_path", "embed_stats_path"):
            chosen["data"][key] = base["data"][key]
        chosen_text = yaml.safe_dump(chosen, sort_keys=False)
        (campaign / "selected_joint.yaml").write_text(chosen_text)
        destination = Path(manifest["workspace"]) / "configs/coviso_relax/selected_joint.yaml"
        if destination.exists():
            raise RuntimeError("selected config already exists; preserving it")
        destination.write_text(chosen_text)
        stage = "test_evaluation"
        render(campaign, manifest, study, stage, confirmations)
        evaluation_groups = {"control": confirmations["control"]}
        if candidate != 0:
            evaluation_groups[f"t{candidate}"] = selected
        jobs = [(label, seed, which) for label in evaluation_groups for seed in (42,43,44) for which in ("best","last")]
        def evaluate(gpu, job):
            label, seed, which = job
            record = evaluation_groups[label][seed]
            out = campaign / "results/analysis" / f"sts7_{label}_s{seed}_{which}.json"
            children.run(gpu, f"eval_{label}_s{seed}_{which}", [sys.executable, "scripts/eval_sts7.py",
                "--config", record["config"], "--runs", record["run_name"], "--which", which,
                "--pooling", "last", "--entity", "student", "--device", "cuda", "--tf32", "--out", str(out)])
            data = read_json(out)
            if record["run_name"] not in data["runs"]:
                raise RuntimeError("evaluation did not return the requested run")
        with ThreadPoolExecutor(max_workers=len(manifest["gpus"])) as pool:
            active, available = {}, list(manifest["gpus"])
            while jobs or active:
                while available and jobs:
                    gpu = available.pop(0)
                    active[pool.submit(evaluate, gpu, jobs.pop(0))] = gpu
                done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    available.append(active.pop(future))
                    future.result()
                    render(campaign, manifest, study, stage, confirmations)
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
        if args.import_baseline is None:
            ap.error("--import-baseline is required and must contain completed matching controls")
        manifest = prepare(campaign, ROOT, gpus, ROOT / "docs/experiments/coviso_relax/report.md")
        import shutil
        shutil.copy2(ROOT / "docs/experiments/coviso_relax/OPTUNA_SPEC.md", campaign / "OPTUNA_SPEC.md")
        manifest.update(trials=args.trials, import_baseline=str(args.import_baseline.resolve()),
                        optuna_spec_sha256=sha256_file(campaign / "OPTUNA_SPEC.md"))
        write_json(campaign / "source_manifest.json", manifest)
        if not args.prepare_only:
            os.execv(sys.executable, [sys.executable, str(campaign / "source/scripts/run_coviso_optuna.py"),
                                     "--campaign", str(campaign), "--execute"])


if __name__ == "__main__":
    main()
