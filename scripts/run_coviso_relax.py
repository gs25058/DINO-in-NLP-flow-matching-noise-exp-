"""Fixed covariance-component experiment, conditional decay, confirmation, then STS7.

Run with nohup on the host through the configured Python. Every GPU child passes
gpu_guard.py. Source/config snapshots are frozen before launching any training.
No Optuna, no resume from training weights, no test-score-driven decisions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.provenance import git_info, sha256_file  # noqa: E402
from src.train import load_config  # noqa: E402

FIXED = ("control", "both1", "cov1", "mean1")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def verdict(control, candidate, seeds):
    deltas = [candidate[s]["best_sts"] - control[s]["best_sts"] for s in seeds]
    last = statistics.mean(candidate[s]["last_sts"] - control[s]["last_sts"] for s in seeds)
    needed = 2
    mean = statistics.mean(deltas)
    if mean >= 0.003 and sum(d > 0 for d in deltas) >= needed and last >= -0.002:
        status = "성립"
    elif mean <= 0 and sum(d <= 0 for d in deltas) >= needed:
        status = "기각"
    else:
        status = "판정불가"
    return {"verdict": status, "paired_best_deltas": deltas, "paired_best_mean": mean,
            "paired_last_mean": last, "n_seeds": len(seeds)}


def fixed_decision(metrics):
    seeds = (42, 43)
    candidates = sorted(set(metrics) - {"control"})
    candidate = min(candidates, key=lambda a: (-statistics.mean(metrics[a][s]["best_sts"] for s in seeds), a))
    judgments = {arm: verdict(metrics["control"], metrics[arm], seeds) for arm in candidates}
    terms = []
    if judgments[candidate]["verdict"] == "성립":
        for term, single in (("mean", "mean1"), ("cov", "cov1")):
            if candidate in (single, "both1") and judgments[single]["verdict"] == "성립":
                terms.append(term)
    return {"constant_candidate": candidate, "supported_decay_terms": terms, "fixed_verdicts": judgments}


def prepare(campaign, workspace, gpus, report):
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / "source"
    source.mkdir()
    for directory in ("src", "scripts", "configs", "tests"):
        shutil.copytree(workspace / directory, source / directory, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for name in ("pyproject.toml", "uv.lock", "METHOD.md", "CLAUDE.md", "GPU_POLICY.md"):
        shutil.copy2(workspace / name, source / name)
    for name in ("SPEC.md", "baseline_provenance.json", "validation.json"):
        shutil.copy2(workspace / "docs/experiments/coviso_relax" / name, campaign / name)
    files = {str(path.relative_to(source)): sha256_file(path) for path in source.rglob("*") if path.is_file()}
    manifest = {"git": git_info(workspace), "workspace": str(workspace), "campaign": str(campaign),
                "gpus": gpus, "report": str(report), "created_unix": time.time(), "source_files": files,
                "spec_sha256": sha256_file(campaign / "SPEC.md"),
                "baseline": read_json(campaign / "baseline_provenance.json")}
    write_json(campaign / "source_manifest.json", manifest)
    (campaign / "source_diff.patch").write_text(manifest["git"]["diff"] + "\n")
    for name in ("results", "checkpoints"):
        target = campaign / name
        target.mkdir()
        (source / name).symlink_to(target, target_is_directory=True)
    for relative in files:
        (source / relative).chmod(0o444)
    (campaign / "configs").mkdir()
    (campaign / "commands").mkdir()
    (campaign / "status").mkdir()
    write_json(campaign / "state.json", {"stage": "prepared", "metrics": {}, "commands": []})
    return manifest


def verify_source(campaign, manifest):
    for relative, expected in manifest["source_files"].items():
        if sha256_file(campaign / "source" / relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")


def make_config(campaign, manifest, arm, seed, parent=None, decay=None):
    cfg = copy.deepcopy(parent) if parent else load_config(campaign / "source/configs/coviso_relax" / f"{arm}.yaml")
    cfg["seed"] = seed
    cfg["run_name"] = f"{campaign.name}_{arm}_s{seed}"
    for key in ("sentences_path", "rank_eval_path", "embed_stats_path"):
        path = Path(cfg["data"][key])
        if not path.is_absolute():
            cfg["data"][key] = str(Path(manifest["workspace"]) / path)
    cfg["experiment"] = {"source_manifest": str(campaign / "source_manifest.json"), "arm": arm,
                         "fresh_pretrained_start": True, "fixed_spec_sha256": manifest["spec_sha256"]}
    if decay:
        cfg["loss"][f"cov_iso_{decay}_schedule"] = {"shape": "cosine", "start_frac": 0.5,
                                                       "end_frac": 1.0, "min_ratio": 0.1}
    path = campaign / "configs" / f"{arm}_s{seed}.yaml"
    if path.exists() and yaml.safe_load(path.read_text()) != cfg:
        raise RuntimeError(f"config already exists with different contents: {path}")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path, cfg


def execute_child(campaign, gpu, label, command, env):
    log_path = campaign / "commands" / f"{label}.log"
    command_path = campaign / "commands" / f"{label}.json"
    guarded = [sys.executable, str(campaign / "source/scripts/gpu_guard.py"), "--gpu", gpu,
               "--wait", "--timeout", "7200", "--", *command]
    if log_path.exists():
        raise RuntimeError(f"refusing to overwrite prior attempt: {log_path}")
    write_json(command_path, {"argv": guarded, "shell_display": shlex.join(guarded), "cwd": str(campaign / "source")})
    start = time.time()
    with log_path.open("w") as stream:
        result = subprocess.run(guarded, cwd=campaign / "source", env=env, stdout=stream, stderr=subprocess.STDOUT)
    write_json(campaign / "status" / f"{label}.json", {"returncode": result.returncode, "elapsed_seconds": time.time() - start,
                                                       "command_file": str(command_path), "log": str(log_path)})
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}); see {log_path}")


def train_one(campaign, manifest, gpu, arm, seed, env, parent=None, decay=None, smoke=False):
    path, cfg = make_config(campaign, manifest, arm, seed, parent, decay)
    run = cfg["run_name"]
    completion = campaign / "results/logs" / run / "completion.json"
    if completion.exists():
        completed = read_json(completion)
        expected = "smoke_complete" if smoke else "complete"
        if completed["status"] == expected:
            return arm, seed, completed
        raise RuntimeError(f"non-complete prior run preserved: {run}")
    verify_source(campaign, manifest)
    command = [sys.executable, "-m", "src.train", "--config", str(path), "--device", "cuda"]
    if smoke:
        command += ["--stop-after-steps", "60"]
    execute_child(campaign, gpu, f"train_{arm}_s{seed}", command, env)
    completed = read_json(completion)
    if completed["status"] != ("smoke_complete" if smoke else "complete"):
        raise RuntimeError(f"run did not finish its requested budget: {run}")
    return arm, seed, completed


def parallel_jobs(jobs, gpus, function, on_result):
    pending = queue.Queue()
    stopped = threading.Event()
    for job in jobs:
        pending.put(job)
    def lane(gpu):
        results = []
        while not stopped.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return results
            try:
                result = function(gpu, job)
            except Exception:
                stopped.set()
                raise
            results.append(result)
            on_result(result)
        return results
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in as_completed([pool.submit(lane, gpu) for gpu in gpus]):
            future.result()


def metrics_from_disk(campaign):
    result = {}
    for path in (campaign / "configs").glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        complete = campaign / "results/logs" / cfg["run_name"] / "completion.json"
        if complete.exists() and read_json(complete)["status"] == "complete":
            result.setdefault(cfg["experiment"]["arm"], {})[cfg["seed"]] = read_json(complete)
    return result


def assert_initializations(campaign):
    per_seed = {}
    for path in (campaign / "results/logs").glob("*/provenance.json"):
        data = read_json(path)
        expected = per_seed.setdefault(data["seed"], data["initial_student_sha256"])
        if expected != data["initial_student_sha256"]:
            raise RuntimeError(f"different pretrained initialization for seed {data['seed']}")
    write_json(campaign / "initialization_audit.json", {"same_initialization_per_seed": True, "sha256": per_seed})


def render_report(campaign, manifest, stage, error=None):
    metrics = metrics_from_disk(campaign)
    lines = ["# 공분산 정규화 완화·성분 분리 — 실행 보고서", "",
             f"상태: **{stage}**. campaign: `{campaign}`.",
             "사전 등록은 SPEC.md이며, test 점수는 설정/후보 선택에 사용하지 않는다.", "",
             "## 기준선과 실행 소스", "",
             f"실제 R18 원격 기준선 `{manifest['baseline']['upstream_commit']}`; λ₀={manifest['baseline']['lambda_0']}.",
             f"작업 시작 commit `{manifest['git']['commit']}`, branch `{manifest['git']['branch']}`.",
             "실행한 전체 소스·변경 내역·파일 해시는 campaign의 source/, source_manifest.json, source_diff.patch에 보존한다.",
             "R18 과거 STS-B dev 0.7465 / STS7 66.09는 참고값이며, 현 A6000 대조군과 직접 동등하다고 가정하지 않는다.", "",
             "## 시드별 dev 및 진단 (실측)", "",
             "| arm | seed | best dev | best loop step | last dev | alignment last | uniformity last | rank last |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, seeds in sorted(metrics.items()):
        for seed, row in sorted(seeds.items()):
            cfg = yaml.safe_load((campaign / "configs" / f"{arm}_s{seed}.yaml").read_text())
            records = [json.loads(line) for line in (campaign / "results/logs" / cfg["run_name"] / "evaluations.jsonl").read_text().splitlines()]
            last = records[-1]
            lines.append(f"| {arm} | {seed} | {row['best_sts']:.6f} | {row['best_loop_step']} | {row['last_sts']:.6f} | "
                         f"{last['alignment']:.6f} | {last['uniformity']:.6f} | {last['eff_rank']:.2f} |")
    lines += ["", "## 평균과 시드 변동", "", "표준편차는 표본 SD이며 신뢰구간이 아니다.", "",
              "| arm | n | best mean ± SD | last mean ± SD |", "|---|---|---|---|"]
    for arm, seeds in sorted(metrics.items()):
        values = list(seeds.values())
        def fmt(key):
            points = [r[key] for r in values]
            return f"{statistics.mean(points):.6f} ± {statistics.stdev(points) if len(points)>1 else 0:.6f}"
        lines.append(f"| {arm} | {len(values)} | {fmt('best_sts')} | {fmt('last_sts')} |")
    lines += ["", "## 손실 성분과 backbone gradient", "",
              "가중 기여는 loss 절대값이며 영향력은 별도 gradient 진단과 함께 본다.", "",
              "| arm | seed | L_DINO last | L_mean last | L_cov last | weighted mean | weighted cov | g_mean/g_DINO | g_cov/g_DINO |",
              "|---|---|---|---|---|---|---|---|---|"]
    for arm, seeds in sorted(metrics.items()):
        for seed in sorted(seeds):
            cfg = yaml.safe_load((campaign / "configs" / f"{arm}_s{seed}.yaml").read_text())
            path = campaign / "results/logs" / cfg["run_name"] / "components.jsonl"
            row = json.loads(path.read_text().splitlines()[-1])
            denominator = row.get("grad_dino_backbone", 0)
            ratio = lambda key: f"{row.get(key, 0)/denominator:.4f}" if denominator else "미측정"
            lines.append(f"| {arm} | {seed} | {row['L_DINO']:.6f} | {row['L_mean']:.6f} | {row['L_cov']:.6f} | "
                         f"{row['weighted_mean']:.6f} | {row['weighted_cov']:.6f} | {ratio('grad_mean_backbone')} | {ratio('grad_cov_backbone')} |")
    lines += ["", "시간에 따른 변화는 각 run의 components.jsonl / gradient_diagnostics.jsonl / evaluations.jsonl에 보존한다."]
    decision_path = campaign / "fixed_decision.json"
    if decision_path.exists():
        decision = read_json(decision_path)
        lines += ["", "## 고정 가중치 가설 판정", ""]
        for arm, judgment in decision["fixed_verdicts"].items():
            lines.append(f"- {arm}: **{judgment['verdict']}**, paired best Δ={judgment['paired_best_mean']:+.6f}, last Δ={judgment['paired_last_mean']:+.6f}.")
        lines.append(f"- 감쇠 근거 확보 항: {decision['supported_decay_terms'] or '없음; 감쇠 미실행'}.")
    else:
        lines += ["", "가설 판정: **판정불가(고정 매트릭스 미완료)**."]
    comparisons_path = campaign / "decay_comparisons.json"
    if comparisons_path.exists() and read_json(comparisons_path):
        lines += ["", "## 감쇠와 대응 상수 대조군", ""]
        for arm, judgment in read_json(comparisons_path).items():
            lines.append(f"- {arm}: **{judgment['verdict']}**, paired best Δ={judgment['paired_best_mean']:+.6f}, last Δ={judgment['paired_last_mean']:+.6f}.")
    selection_path = campaign / "selection.json"
    if selection_path.exists():
        selection = read_json(selection_path)
        lines += ["", "## 동결된 최종 후보와 후속 실험", "",
                  f"후보 `{selection['candidate']}`. 3시드 판정: **{selection['judgment']['verdict']}**.",
                  f"정확한 선택 config: `{campaign / 'selected_relaxation.yaml'}`.",
                  "저장소에서 재사용할 경로: `configs/coviso_relax/selected_relaxation.yaml` (선택 판정/출처 포함).",
                  "두 번째 직접 자기증류 실험의 기본 대조군은 `configs/coviso_relax/control.yaml`이다.",
                  "완화가 성립한 경우에만 위 선택 config를 후보로 재사용한다. 모든 새 arm은 pretrained에서 독립 시작한다."]
    lines += ["", "## STS7 (설정 동결 후 test)", "", "| arm | seed | checkpoint | STS7 (%) |", "|---|---|---|---|"]
    seven_groups = {}
    for path in sorted((campaign / "results/analysis").glob("sts7_*.json")):
        parts = path.stem.removeprefix("sts7_").rsplit("_", 2)
        arm, seed, which = parts
        scores = read_json(path)["runs"]
        average = next(iter(scores.values()))["avg_7task"] * 100
        seven_groups.setdefault((arm, which), []).append(average)
        lines.append(f"| {arm} | {seed} | {which} | {average:.4f} |")
    if not seven_groups:
        lines.append("| 미실행 | — | — | — |")
    for (arm, which), scores in sorted(seven_groups.items()):
        lines.append(f"| {arm} | mean ± SD, n={len(scores)} | {which} | {statistics.mean(scores):.4f} ± {statistics.stdev(scores) if len(scores)>1 else 0:.4f} |")
    lines += ["", "## 실제 명령과 완료 상태", ""]
    for path in sorted((campaign / "commands").glob("*.json")):
        record = read_json(path)
        status = campaign / "status" / path.name
        code = read_json(status)["returncode"] if status.exists() else "실행 중/미완료"
        lines += [f"- `{path.stem}` exit={code}", f"  `{record['shell_display']}`"]
    if error:
        lines += ["", f"실행 차단/실패: `{error}`. 미실행 점수는 추정하지 않았다."]
    if (campaign / "validation.json").exists():
        validation = read_json(campaign / "validation.json")
        lines += ["", "## 구현 검증", "",
                  f"변경 전 {validation['before_tests']}개, 변경 후 {validation['after_tests']}개 CPU 테스트 통과.",
                  "기존 수치·기울기·EMA 순서와 횟수, 가중치 0, teacher/EMA detach, 감쇠 경계, 단일 학습 루프와 best/last 저장을 검사했다.",
                  f"별도 GPU 스모크: `{validation['smoke_campaign']}`. control/cov1 60 step 정상 종료, 동일 초기 student 해시 확인.",
                  "스모크는 4000-step 스케줄을 유지했으며 본 실험 표에는 포함하지 않았다."]
    lines += ["", "남은 교락: 현재 서버/라이브러리와 과거 R18 환경 차이, 2~3시드의 표본 한계.",
              "공통 초기값·학습 스케줄·데이터·평가를 고정한 범위에서만 가중치 효과를 해석한다.",
              "uniformity와 rank는 진단 지표이며 SimCSE 수치를 목표로 강제하지 않았다."]
    text = "\n".join(lines) + "\n"
    (campaign / "report.md").write_text(text)
    Path(manifest["report"]).write_text(text)


def run(campaign, manifest, smoke=False):
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled",
               HF_HOME=os.environ.get("HF_HOME", str(Path.home() / "scratch/.hf_home")),
               PYTHONPATH=str(campaign / "source"), PYTHONDONTWRITEBYTECODE="1")
    gpus = manifest["gpus"]
    if not 1 <= len(gpus) <= 2 or len(set(gpus)) != len(gpus):
        raise ValueError("supply one or two distinct GPUs; account-wide policy is enforced per child")
    stage = "smoke" if smoke else "fixed_training"
    try:
        jobs = [(arm, seed) for arm in (("control", "cov1") if smoke else FIXED) for seed in ((42,) if smoke else (42, 43))]
        def update(_result):
            print(f"[coviso_relax] completed {_result[0]} seed={_result[1]}", flush=True)
        parallel_jobs(jobs, gpus, lambda gpu, job: train_one(campaign, manifest, gpu, *job, env, smoke=smoke), update)
        assert_initializations(campaign)
        if smoke:
            write_json(campaign / "state.json", {"stage": "smoke_complete"})
            render_report(campaign, manifest, "smoke_complete")
            return
        metrics = metrics_from_disk(campaign)
        decision = fixed_decision(metrics)
        write_json(campaign / "fixed_decision.json", decision)
        render_report(campaign, manifest, "fixed_complete")
        parent = yaml.safe_load((campaign / "configs" / f"{decision['constant_candidate']}_s42.yaml").read_text())
        stage = "conditional_decay"
        decay_jobs = [(f"{decision['constant_candidate']}_decay_{term}", seed, term)
                      for term in decision["supported_decay_terms"] for seed in (42, 43)]
        parallel_jobs(decay_jobs, gpus, lambda gpu, job: train_one(campaign, manifest, gpu, job[0], job[1], env,
                                                                 parent=parent, decay=job[2]), update)
        metrics = metrics_from_disk(campaign)
        comparisons = {arm: verdict(metrics[decision["constant_candidate"]], metrics[arm], (42,43))
                       for arm in metrics if "_decay_" in arm}
        write_json(campaign / "decay_comparisons.json", comparisons)
        candidates = sorted(set(metrics) - {"control"})
        candidate = min(candidates, key=lambda arm: (-statistics.mean(metrics[arm][s]["best_sts"] for s in (42,43)), arm))
        write_json(campaign / "dev_candidate.json", {"candidate": candidate, "test_scores_seen": False})
        stage = "seed44_confirmation"
        parents = {arm: yaml.safe_load((campaign / "configs" / f"{arm}_s42.yaml").read_text()) for arm in ("control", candidate)}
        parallel_jobs([(arm, 44) for arm in parents], gpus,
                      lambda gpu, job: train_one(campaign, manifest, gpu, *job, env, parent=parents[job[0]]), update)
        assert_initializations(campaign)
        metrics = metrics_from_disk(campaign)
        selection = {"candidate": candidate, "judgment": verdict(metrics["control"], metrics[candidate], (42,43,44)),
                     "test_scores_seen": False, "selection_frozen_unix": time.time(), "fixed_decision": decision}
        write_json(campaign / "selection.json", selection)
        chosen = copy.deepcopy(parents[candidate])
        chosen["run_name"] = "coviso_relax_selected_for_next_experiment"
        chosen["seed"] = 42
        chosen["experiment"].pop("source_manifest", None)
        chosen["experiment"]["selected_arm"] = candidate
        (campaign / "selected_relaxation.yaml").write_text(yaml.safe_dump(chosen, sort_keys=False))
        reusable = {"extends": "base.yaml", "run_name": "coviso_relax_selected_relaxation", "loss": {},
                    "experiment": {"selected_arm": candidate, "selected_from_dev_only": True,
                                   "selection_campaign": str(campaign),
                                   "recommended": selection["judgment"]["verdict"] == "성립"}}
        for key in ("cov_iso_mean_lambda", "cov_iso_cov_lambda", "cov_iso_mean_schedule", "cov_iso_cov_schedule"):
            if key in chosen["loss"]:
                reusable["loss"][key] = chosen["loss"][key]
        destination = Path(manifest["workspace"]) / "configs/coviso_relax/selected_relaxation.yaml"
        if destination.exists():
            raise RuntimeError("prior selected config exists; preserve it before starting another campaign")
        destination.write_text(yaml.safe_dump(reusable, sort_keys=False))
        stage = "test_evaluation"
        render_report(campaign, manifest, stage)
        eval_jobs = [(arm, seed, which) for arm, seeds in sorted(metrics.items()) for seed in sorted(seeds) for which in ("best", "last")]
        def evaluate(gpu, job):
            arm, seed, which = job
            out = campaign / "results/analysis" / f"sts7_{arm}_s{seed}_{which}.json"
            if out.exists():
                return job
            cfg_path = campaign / "configs" / f"{arm}_s{seed}.yaml"
            cfg = yaml.safe_load(cfg_path.read_text())
            command = [sys.executable, "scripts/eval_sts7.py", "--config", str(cfg_path), "--runs", cfg["run_name"],
                       "--which", which, "--pooling", "last", "--entity", "student", "--device", "cuda", "--tf32", "--out", str(out)]
            execute_child(campaign, gpu, f"eval_{arm}_s{seed}_{which}", command, env)
            return job
        parallel_jobs(eval_jobs, gpus, evaluate, lambda result: print(f"[coviso_relax] STS7 {result}", flush=True))
        write_json(campaign / "state.json", {"stage": "complete", "candidate": candidate})
        render_report(campaign, manifest, "complete")
    except Exception as exc:
        write_json(campaign / "state.json", {"stage": "failed", "failed_stage": stage, "error": str(exc)})
        render_report(campaign, manifest, "failed", str(exc))
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True, type=Path)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--execute", action="store_true", help="execute an already frozen campaign")
    ap.add_argument("--smoke", action="store_true", help="two 60-step runs, preserving 4000-step schedules")
    args = ap.parse_args()
    campaign = args.campaign.resolve()
    if args.execute:
        manifest = read_json(campaign / "source_manifest.json")
        verify_source(campaign, manifest)
        run(campaign, manifest, args.smoke)
    else:
        manifest = prepare(campaign, ROOT, args.gpus.split(",") if args.gpus else [],
                           ROOT / "docs/experiments/coviso_relax/report.md")
        print(f"[coviso_relax] frozen source: {campaign}", flush=True)
        if not args.prepare_only:
            command = [sys.executable, str(campaign / "source/scripts/run_coviso_relax.py"),
                       "--campaign", str(campaign), "--execute"]
            if args.smoke:
                command.append("--smoke")
            os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
