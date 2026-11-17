"""Word-prototype DINO (mean pooling fixed): Phase 0 → smokes → 3 independent Optuna studies
→ top-3 confirmation → frozen selection → STS7 (raw + postprocessed), C0 = t13 reused.

See docs/experiments/vocab_proto/OPTUNA_SPEC.md. At most two guarded GPU children.
"""

import argparse
import copy
import hashlib
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import optuna
import yaml
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_coviso_optuna import Children, complete_record, parallel_jobs  # noqa: E402
from src import vocab_proto as vp  # noqa: E402
from src.provenance import prepare, read_json, read_records, sha256_file, verify_source, write_json  # noqa: E402
from src.train import load_config  # noqa: E402

CONDITIONS = tuple(vp.CONDITIONS)
SEARCH_SEED = 42
FINAL_SEEDS = (42, 43, 44)  # C0 (t13) was confirmed on exactly these seeds
TRIALS = 40
SAMPLER_SEED = 20261117
KMEANS = "data/vocab_kmeans_bert_base_uncased_k8192_seed0.pt"
C0_CAMPAIGN = "results/experiments/coviso_joint_optuna_20261028"
C0_RUN = "r18_bert_joint_optuna_coviso_joint_optuna_20261028_t13_s{seed}"
DATA_KEYS = ("sentences_path", "rank_eval_path", "embed_stats_path")
lock = threading.Lock()


def run_name(campaign, label, seed):
    return f"vproto_{campaign.name}_{label}_s{seed}"


def train(campaign, manifest, children, gpu, cfg, label, seed, smoke=False, trial=None, report="best"):
    cfg = copy.deepcopy(cfg)
    cfg.update(seed=seed, run_name=run_name(campaign, label, seed))
    for key in DATA_KEYS:
        cfg["data"][key] = str(Path(manifest["data_root"]) / cfg["data"][key])
    cfg["experiment"] = {"source_manifest": str(campaign / "source_manifest.json"), "protocol": "vocab_proto",
                         "label": label, "spec_sha256": manifest["spec_sha256"], "fresh_pretrained_start": True}
    path = campaign / "configs" / f"{label}_s{seed}.yaml"
    if path.exists() and yaml.safe_load(path.read_text()) != cfg:
        raise RuntimeError(f"preserving changed config: {path}")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log = campaign / "results/logs" / cfg["run_name"]
    if not (log / "completion.json").exists():
        verify_source(campaign, manifest)
        command = [sys.executable, "-m", "src.train", "--config", str(path), "--device", "cuda"]
        if smoke:
            command += ["--stop-after-steps", "60"]
        children.run(gpu, f"train_{label}_s{seed}", command, trial=trial,
                     evaluation_path=log / "evaluations.jsonl", report=report)
    metrics = read_json(log / "completion.json") if smoke else complete_record(log / "completion.json")
    if smoke and (metrics["status"] != "smoke_complete" or metrics["completed_steps"] != 60):
        raise RuntimeError("60-step smoke incomplete")
    provenance = read_json(log / "provenance.json")
    return {"run_name": cfg["run_name"], "config": str(path), "label": label, "seed": seed,
            "condition": label.split("_")[1], "metrics": metrics,
            "initial_backbone_sha256": provenance["initial_backbone_sha256"],
            "initial_student_sha256": provenance["initial_student_sha256"],
            "first_views": read_json(log / "first_views.json"),
            "augment_sha256": hashlib.sha256(yaml.safe_dump(cfg["augment"], sort_keys=True).encode()).hexdigest(),
            "batch_order": read_json(log / "batch_order.json")["sha256"],
            "data": {k: v["sha256"] for k, v in provenance["data"].items()}}


def audit(campaign, records):
    """Same backbone init, data order and data per seed; same full init per (condition, seed);
    same first noisy views per (seed, noise configuration) across C1–C3."""
    backbone, order, full, views, data = {}, {}, {}, {}, None
    with lock:
        for r in records:
            s = r["seed"]
            if backbone.setdefault(s, r["initial_backbone_sha256"]) != r["initial_backbone_sha256"]:
                raise RuntimeError("unequal pretrained backbone within seed")
            if order.setdefault(s, r["batch_order"]) != r["batch_order"]:
                raise RuntimeError("unequal data order within seed")
            key = f"{r['condition']}_{s}"
            if full.setdefault(key, r["initial_student_sha256"]) != r["initial_student_sha256"]:
                raise RuntimeError("unequal initialization within condition and seed")
            group = f"{s}_{r['augment_sha256']}"
            if views.setdefault(group, r["first_views"]) != r["first_views"]:
                raise RuntimeError("unequal first noisy views within seed and noise configuration")
            data = data or r["data"]
            if r["data"] != data:
                raise RuntimeError("training data changed between runs")
        write_json(campaign / "comparability_audit.json", {
            "runs": len(records), "backbone_sha256": backbone, "batch_order_sha256": order,
            "full_initial_sha256": full, "first_view_groups": len(views), "data_sha256": data})


def stage(campaign, name, **extra):
    write_json(campaign / "state.json", {"stage": name, "updated_unix": time.time(), **extra})
    from scripts.report_vocab_proto import render
    render(campaign)


def phase0(campaign, manifest):
    out = campaign / "phase0.json"
    log = campaign / "commands" / "phase0.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(campaign / "source"))
    with log.open("w") as stream:
        code = subprocess.call([sys.executable, "scripts/vocab_phase0.py", "--kmeans", manifest["kmeans_path"],
                                "--out", str(out)], cwd=campaign / "source", env=env,
                               stdout=stream, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f"Phase 0 consistency check failed (exit {code}); see {out}")


def search(campaign, manifest, children, base, records):
    storage = f"sqlite:///{campaign / 'studies.db'}"
    studies = {}
    for c in CONDITIONS:
        studies[c] = optuna.create_study(
            study_name=f"{campaign.name}_{c}", storage=storage, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED, n_startup_trials=8, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1500, interval_steps=250,
                                               n_min_trials=4))
        for point in vp.anchors(base, c):
            studies[c].enqueue_trial(point)
    found = {c: {} for c in CONDITIONS}
    attempted = {c: 0 for c in CONDITIONS}
    running, failures = {}, []
    gpus = list(manifest["gpus"])

    def attempt(gpu, c, n, trial, cfg):
        try:
            return train(campaign, manifest, children, gpu, cfg, f"search_{c}_t{n}", SEARCH_SEED, trial=trial)
        except optuna.TrialPruned:
            return "pruned"

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = {}
        while True:
            while gpus:
                idle = [c for c in CONDITIONS if c not in running and attempted[c] < TRIALS]
                if not idle:
                    break
                c = min(idle, key=lambda k: (attempted[k], CONDITIONS.index(k)))
                n = attempted[c]
                attempted[c] += 1
                trial = studies[c].ask()
                params = {k: trial.suggest_float(k, lo, hi, log=log) for k, (lo, hi, log) in vp.space(c).items()}
                cfg = vp.configure(base, c, params, kmeans_path=manifest["kmeans_path"])
                gpu = gpus.pop()
                running[c] = (n, trial, params)
                futures[pool.submit(attempt, gpu, c, n, trial, cfg)] = (c, gpu)
            if not futures:
                break
            stage(campaign, "search", attempted=attempted)
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                c, gpu = futures.pop(future)
                gpus.append(gpu)
                n, trial, params = running.pop(c)
                try:
                    result = future.result()
                except Exception as exc:  # a failed run (e.g. nonfinite loss) stays in the budget
                    studies[c].tell(trial, state=TrialState.FAIL)
                    found[c][str(n)] = {"params": params, "state": "FAIL", "error": str(exc)}
                    failures.append(str(exc))
                    if children.stopping.is_set() or len(failures) >= 6:
                        raise
                    continue
                if result == "pruned":
                    studies[c].tell(trial, state=TrialState.PRUNED)
                    found[c][str(n)] = {"params": params, "state": "PRUNED"}
                else:
                    studies[c].tell(trial, result["metrics"]["best_sts"])
                    found[c][str(n)] = {**result, "params": params, "state": "COMPLETE"}
                    records.append(result)
                    audit(campaign, records)
                write_json(campaign / "search.json", found)
    return found


def confirm(campaign, manifest, children, base, records, found):
    shortlist = {c: sorted((n for n, r in found[c].items() if r["state"] == "COMPLETE"),
                           key=lambda n: (-found[c][n]["metrics"]["best_sts"], int(n)))[:3] for c in CONDITIONS}
    write_json(campaign / "shortlist.json", {"trials": shortlist, "metric": "seed-42 dev best", "test_seen": False})
    confirmed = {c: {n: {"42": found[c][n]} for n in shortlist[c]} for c in CONDITIONS}
    jobs = [(c, n, s) for s in FINAL_SEEDS[1:] for c in CONDITIONS for n in shortlist[c]]

    def execute(gpu, job):
        c, n, s = job
        cfg = vp.configure(base, c, found[c][n]["params"], kmeans_path=manifest["kmeans_path"])
        return train(campaign, manifest, children, gpu, cfg, f"confirm_{c}_t{n}", s)

    def saved(job, r):
        c, n, s = job
        confirmed[c][n][str(s)] = r
        records.append(r)
        audit(campaign, records)
        write_json(campaign / "confirmations.json", confirmed)
        stage(campaign, "confirmation")

    parallel_jobs(jobs, manifest["gpus"], children, execute, saved)
    selected = {c: min(shortlist[c], key=lambda n: (-statistics.mean(
        confirmed[c][n][str(s)]["metrics"]["best_sts"] for s in FINAL_SEEDS), int(n))) for c in CONDITIONS}
    groups = {c: {str(s): confirmed[c][selected[c]][str(s)] for s in FINAL_SEEDS} for c in CONDITIONS}
    write_json(campaign / "selection.json", {
        "selected_trials": selected, "selected_params": {c: found[c][n]["params"] for c, n in selected.items()},
        "selection_metric": "three-seed (42/43/44) mean of 768d STS-B dev best",
        "primary_test": "STS7 raw, dev-best checkpoint (as C0's 67.097)",
        "selection_frozen_unix": time.time(), "test_scores_seen": False})
    write_json(campaign / "final_runs.json", groups)
    return groups


def evaluate(campaign, manifest, children, groups):
    if not (campaign / "selection.json").exists():
        raise RuntimeError("test requires frozen selection")
    c0_dir = Path(manifest["data_root"]) / C0_CAMPAIGN / "checkpoints"
    jobs = [(c, s, w) for c in ("C0", *CONDITIONS) for s in FINAL_SEEDS for w in ("best", "last")]

    def execute(gpu, job):
        c, s, w = job
        out = campaign / "results/analysis" / f"{c}_s{s}_{w}.json"
        if out.exists():
            return
        name = C0_RUN.format(seed=s) if c == "C0" else groups[c][str(s)]["run_name"]
        command = [sys.executable, "scripts/eval_sts7.py", "--runs", name, "--which", w, "--device", "cuda",
                   "--tf32", "--diagnose-bottleneck", "--out", str(out),
                   "--postprocess-corpus", str(Path(manifest["data_root"]) / "data/sentences.jsonl")]
        if c == "C0":
            command += ["--checkpoint-dir", str(c0_dir)]
        children.run(gpu, f"eval_{c}_s{s}_{w}", command)
        result = read_json(out)
        result["experiment"] = {"condition": c, "seed": s, "which": w, "selection_frozen_before_test": True}
        write_json(out, result)

    parallel_jobs(jobs, manifest["gpus"], children, execute, lambda *_: stage(campaign, "test_evaluation"))


def run(campaign, manifest):
    verify_source(campaign, manifest)
    base = load_config(campaign / "source/configs/vocab_proto/base.yaml")
    env = dict(os.environ, PYTHONPATH=str(campaign / "source"), PYTHONDONTWRITEBYTECODE="1")
    children = Children(campaign, env)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True
        children.stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    records = []
    try:
        stage(campaign, "phase0")
        phase0(campaign, manifest)
        stage(campaign, "smoke")
        smokes = {}
        parallel_jobs(list(CONDITIONS), manifest["gpus"], children,
                      lambda gpu, c: train(campaign, manifest, children, gpu,
                                           vp.configure(base, c, vp.anchors(base, c)[0], manifest["kmeans_path"]),
                                           f"smoke_{c}", SEARCH_SEED, smoke=True),
                      lambda c, r: smokes.update({c: r}))
        audit(campaign, list(smokes.values()))
        write_json(campaign / "smoke.json", smokes)
        found = search(campaign, manifest, children, base, records)
        stage(campaign, "confirmation")
        groups = confirm(campaign, manifest, children, base, records, found)
        stage(campaign, "test_evaluation")
        evaluate(campaign, manifest, children, groups)
        stage(campaign, "complete")
    except BaseException as exc:
        children.stop()
        stage(campaign, "stopped_by_user" if stopped else "failed", error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.execute:
        run(campaign, read_json(campaign / "source_manifest.json"))
        return
    data_root = (ROOT / "data").resolve().parent
    kmeans = data_root / KMEANS
    meta = read_json(kmeans.with_suffix(".json"))
    if sha256_file(kmeans) != meta["sha256"]:
        raise RuntimeError("k-means cache hash mismatch")
    for run_seed in FINAL_SEEDS:
        for which in ("best", "last"):
            if not (data_root / C0_CAMPAIGN / "checkpoints" / C0_RUN.format(seed=run_seed) / f"{which}.pt").exists():
                raise RuntimeError("C0 (t13) checkpoint missing")
    manifest = prepare(campaign, ROOT, ["auto", "auto"], ROOT / "docs/experiments/vocab_proto/report.md",
                       experiment_dir="vocab_proto")
    manifest.update(protocol="vocab_proto", data_root=str(data_root), kmeans_path=str(kmeans),
                    kmeans_sha256=meta["sha256"], launch_argv=sys.argv)
    write_json(campaign / "source_manifest.json", manifest)
    os.execv(sys.executable, [sys.executable, str(campaign / "source/scripts/run_vocab_proto.py"),
                              "--campaign", str(campaign), "--execute"])


if __name__ == "__main__":
    main()
