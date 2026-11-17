"""Phase B of the late-drop study: tune the schedules of the two causal factors (sharpening,
teacher EMA tracking) on C1/C2 with a late-dev objective, confirm on fresh seeds, then test the
last checkpoint against C0 (t13) retrained on the same seeds. See docs/experiments/late_drop/OPTUNA_SPEC.md.
"""

import argparse
import os
import signal
import statistics
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import optuna
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_vocab_proto as vproto  # noqa: E402
from scripts.run_coviso_optuna import Children, parallel_jobs  # noqa: E402
from src import late_schedule as ls  # noqa: E402
from src.provenance import prepare, read_json, read_records, verify_source, write_json  # noqa: E402
from src.train import load_config  # noqa: E402

VOCAB = "results/experiments/vocab_proto_20261116"
LATE_DROP = "results/experiments/late_drop_20261117"
SEARCH_SEED = 42
SEEDS = (101, 102, 103)
TRIALS = 40
SAMPLER_SEED = 20261118
LATE = (3500, 3750, 3999)
# Anchor index → earlier run that it must reproduce exactly (seed 42, every logged step).
REPRODUCE = {
    "C2": {0: (LATE_DROP, "iv_C2_t9_sharpen_off"), 1: (LATE_DROP, "iv_C2_t9_teacher_freeze"),
           2: (VOCAB, "search_C2_t3")},
    "C1": {0: (LATE_DROP, "iv_C1_t16_sharpen_off"), 1: (LATE_DROP, "iv_C1_t16_teacher_freeze"),
           2: (VOCAB, "search_C1_t29")},
}


def late_dev(campaign, record):
    rows = {r["loop_step"]: r["sts_b_dev"] for r in read_records(campaign / "results/logs" / record["run_name"] / "evaluations.jsonl")}
    return statistics.mean(rows[s] for s in LATE)


def check_reproduction(campaign, manifest, condition, index, record):
    if index not in REPRODUCE[condition]:
        return None
    source, label = REPRODUCE[condition][index]
    root = Path(manifest["data_root"]) / source
    reference = root / "results/logs" / vproto.run_name(root, label, SEARCH_SEED)
    mine = campaign / "results/logs" / record["run_name"]
    for name in ("components.jsonl", "evaluations.jsonl"):
        a, b = read_records(reference / name), read_records(mine / name)
        # Only vocab diagnostics keys may differ in naming between campaigns; compare the shared keys.
        if len(a) != len(b) or any({k: x[k] for k in x if k in y} != {k: y[k] for k in x if k in y} for x, y in zip(a, b)):
            raise RuntimeError(f"anchor {condition}#{index} does not reproduce {label}: {name}")
    return f"{source}/{label}"


def audit(campaign, records):
    vproto.audit(campaign, [r for r in records if r["condition"] != "C0"])
    vocab = {r["seed"]: r for r in records if r["condition"] != "C0"}
    for r in records:
        other = vocab.get(r["seed"])
        if r["condition"] == "C0" and other is not None:
            if (r["initial_backbone_sha256"], r["batch_order"]) != (other["initial_backbone_sha256"], other["batch_order"]):
                raise RuntimeError("C0 backbone initialization or data order differs within seed")


def search(campaign, manifest, children, base, previous, records):
    storage = f"sqlite:///{campaign / 'studies.db'}"
    studies, found = {}, {c: {} for c in ls.CONDITIONS}
    for c in ls.CONDITIONS:
        studies[c] = optuna.create_study(
            study_name=f"{campaign.name}_{c}", storage=storage, direction="maximize",
            # constant_liar: concurrent asks within a condition do not collapse onto one point.
            sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED, n_startup_trials=8, multivariate=True,
                                               constant_liar=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2000, interval_steps=250,
                                               n_min_trials=4))
        for point in ls.anchors(previous, c):
            studies[c].enqueue_trial(point)
    attempted, failures = {c: 0 for c in ls.CONDITIONS}, []
    gpus = list(manifest["gpus"])
    per_condition = max(1, len(gpus) // len(ls.CONDITIONS))
    reproduced = {}

    def attempt(gpu, c, n, trial, cfg):
        try:
            return vproto.train(campaign, manifest, children, gpu, cfg, f"search_{c}_t{n}", SEARCH_SEED,
                                trial=trial, report="current")
        except optuna.TrialPruned:
            return "pruned"

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = {}
        while True:
            while gpus:
                busy = {c: sum(f[0] == c for f in futures.values()) for c in ls.CONDITIONS}
                idle = [c for c in ls.CONDITIONS if busy[c] < per_condition and attempted[c] < TRIALS]
                if not idle:
                    break
                c = min(idle, key=lambda k: (attempted[k], ls.CONDITIONS.index(k)))
                n = attempted[c]
                attempted[c] += 1
                trial = studies[c].ask()
                params = {k: trial.suggest_float(k, lo, hi, log=log) for k, (lo, hi, log) in ls.space(c).items()}
                cfg = ls.configure(base, c, params, kmeans_path=manifest["kmeans_path"])
                gpu = gpus.pop()
                futures[pool.submit(attempt, gpu, c, n, trial, cfg)] = (c, gpu, n, trial, params)
            if not futures:
                break
            write_json(campaign / "state.json", {"stage": "search", "attempted": attempted, "updated_unix": time.time()})
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                c, gpu, n, trial, params = futures.pop(future)
                gpus.append(gpu)
                try:
                    result = future.result()
                except Exception as exc:
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
                    same = check_reproduction(campaign, manifest, c, n, result)
                    if same:
                        reproduced[f"{c}_t{n}"] = same
                        write_json(campaign / "reproduction_audit.json", reproduced)
                    value = late_dev(campaign, result)
                    studies[c].tell(trial, value)
                    found[c][str(n)] = {**result, "params": params, "state": "COMPLETE", "late_dev": value}
                    records.append(result)
                    audit(campaign, records)
                write_json(campaign / "search.json", found)
                render(campaign)
    return found


def confirm(campaign, manifest, children, base, records, found):
    shortlist = {c: sorted((n for n, r in found[c].items() if r["state"] == "COMPLETE"),
                           key=lambda n: (-found[c][n]["late_dev"], int(n)))[:3] for c in ls.CONDITIONS}
    write_json(campaign / "shortlist.json", {"trials": shortlist, "metric": "seed-42 late dev", "test_seen": False})
    confirmed = {c: {n: {} for n in shortlist[c]} for c in ls.CONDITIONS}
    confirmed["C0"] = {"t13": {}}
    jobs = [(c, n, s) for s in SEEDS for c in ls.CONDITIONS for n in shortlist[c]] + [("C0", "t13", s) for s in SEEDS]

    def execute(gpu, job):
        c, n, s = job
        cfg = base if c == "C0" else ls.configure(base, c, found[c][n]["params"], kmeans_path=manifest["kmeans_path"])
        return vproto.train(campaign, manifest, children, gpu, cfg, f"confirm_{c}_t{n}".replace("_tt13", "_t13"), s)

    def saved(job, r):
        c, n, s = job
        confirmed[c][n][str(s)] = {**r, "late_dev": late_dev(campaign, r)}
        records.append(r)
        audit(campaign, records)
        write_json(campaign / "confirmations.json", confirmed)
        write_json(campaign / "state.json", {"stage": "confirmation", "updated_unix": time.time()})
        render(campaign)

    parallel_jobs(jobs, manifest["gpus"], children, execute, saved)
    selected = {c: min(shortlist[c], key=lambda n: (-statistics.mean(
        confirmed[c][n][str(s)]["late_dev"] for s in SEEDS), int(n))) for c in ls.CONDITIONS}
    groups = {c: confirmed[c][selected[c]] for c in ls.CONDITIONS}
    groups["C0"] = confirmed["C0"]["t13"]
    write_json(campaign / "selection.json", {
        "selected_trials": selected, "selected_params": {c: found[c][n]["params"] for c, n in selected.items()},
        "selection_metric": "three-seed (101/102/103) mean of late dev (steps 3500/3750/3999)",
        "primary_test": "STS7 raw, last checkpoint", "selection_frozen_unix": time.time(), "test_scores_seen": False})
    return groups, selected


def ablate(campaign, manifest, children, base, records, found, selected, groups):
    """Selected configuration with the schedule axes neutral: the schedules' own effect."""
    for c in ls.CONDITIONS:
        groups[f"{c}_off"] = {}
    jobs = [(c, s) for s in SEEDS for c in ls.CONDITIONS]

    def execute(gpu, job):
        c, s = job
        params = {**found[c][selected[c]]["params"], **ls.NEUTRAL}
        return vproto.train(campaign, manifest, children, gpu, ls.configure(base, c, params, manifest["kmeans_path"]),
                            f"off_{c}_t{selected[c]}", s)

    def saved(job, r):
        c, s = job
        groups[f"{c}_off"][str(s)] = {**r, "late_dev": late_dev(campaign, r)}
        records.append(r)
        audit(campaign, records)
        write_json(campaign / "final_runs.json", groups)
        render(campaign)

    parallel_jobs(jobs, manifest["gpus"], children, execute, saved)
    write_json(campaign / "final_runs.json", groups)


def evaluate(campaign, manifest, children, groups):
    if not (campaign / "selection.json").exists():
        raise RuntimeError("test requires frozen selection")
    jobs = [(g, s, w) for g in groups for s in SEEDS for w in ("best", "last")]

    def execute(gpu, job):
        g, s, w = job
        out = campaign / "results/analysis" / f"{g}_s{s}_{w}.json"
        if out.exists():
            return
        children.run(gpu, f"eval_{g}_s{s}_{w}", [
            sys.executable, "scripts/eval_sts7.py", "--runs", groups[g][str(s)]["run_name"], "--which", w,
            "--device", "cuda", "--tf32", "--diagnose-bottleneck", "--out", str(out),
            "--postprocess-corpus", str(Path(manifest["data_root"]) / "data/sentences.jsonl")])
        result = read_json(out)
        result["experiment"] = {"group": g, "seed": s, "which": w, "selection_frozen_before_test": True}
        write_json(out, result)

    parallel_jobs(jobs, manifest["gpus"], children, execute, lambda *_: render(campaign))


def verdict(diffs):
    if len(diffs) != 3:
        return "판정불가(미실행)"
    mean = statistics.mean(diffs)
    if mean >= 0.003 and sum(d > 0 for d in diffs) >= 2:
        return "성립"
    if mean <= 0 and sum(d <= 0 for d in diffs) >= 2:
        return "기각"
    return "판정불가"


def render(campaign):
    state = read_json(campaign / "state.json")
    lines = ["# 후반 하락: 원인 요인 스케줄 튜닝 (Phase B)", "", f"Stage: {state['stage']}", "",
             "C1/C2 각각 vocab_proto 공간 + 스케줄 4축(sharpening 완화 u·시작·길이, teacher 동결 시점), "
             "late dev 목적함수, seed 42, 조건당 40 attempted. 확인과 test는 seed 101/102/103, C0 t13도 같은 seed로 재학습.", ""]
    if (campaign / "reproduction_audit.json").exists():
        lines.append("재현 검증(anchor가 이전 run과 모든 기록 일치): " + ", ".join(
            f"{k} = {v}" for k, v in read_json(campaign / "reproduction_audit.json").items()) + ".")
        lines.append("")
    if (campaign / "search.json").exists():
        found = read_json(campaign / "search.json")
        lines += ["| 조건 | attempted | complete | pruned | failed | best trial | late dev | u | start | len | freeze |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, rows in found.items():
            states = [r["state"] for r in rows.values()]
            done = {n: r for n, r in rows.items() if r["state"] == "COMPLETE"}
            cells = ["—"] * 6
            if done:
                n, r = max(done.items(), key=lambda kv: (kv[1]["late_dev"], -int(kv[0])))
                p = r["params"]
                cells = [n, f"{r['late_dev']*100:.2f}", f"{p['_late_relax']:.2f}", f"{p['_late_start_frac']:.2f}",
                         f"{p['_late_len_frac']:.2f}", f"{p['train.teacher_freeze_frac']:.2f}"]
            lines.append(f"| {c} | {len(rows)} | {states.count('COMPLETE')} | {states.count('PRUNED')} | "
                         f"{states.count('FAIL')} | " + " | ".join(cells) + " |")
        lines.append("")
    if (campaign / "selection.json").exists():
        sel = read_json(campaign / "selection.json")
        lines.append("선택(test 전 동결): " + ", ".join(
            f"{c} trial {n} (u {sel['selected_params'][c]['_late_relax']:.2f}, start {sel['selected_params'][c]['_late_start_frac']:.2f}, "
            f"len {sel['selected_params'][c]['_late_len_frac']:.2f}, freeze {sel['selected_params'][c]['train.teacher_freeze_frac']:.2f})"
            for c, n in sel["selected_trials"].items()) + ".")
        lines.append("")
    scores = {}
    for file in sorted((campaign / "results/analysis").glob("*.json")):
        doc = read_json(file)
        ex = doc.get("experiment")
        if ex:
            run = next(iter(doc["runs"]))
            scores[(ex["group"], ex["seed"], ex["which"])] = {
                "raw": doc["runs"][run]["avg_7task"],
                "pc2": doc.get("postprocessed", {}).get(run, {}).get("center_pc2", {}).get("avg_7task")}
    final = read_json(campaign / "final_runs.json") if (campaign / "final_runs.json").exists() else {}
    if scores or final:
        lines += ["## STS7 test (×100, seed 101/102/103)", "",
                  "| 그룹 | late dev 평균 | raw last 평균 | raw last 시드별 | raw best 평균 | best − last | center_pc2 last |",
                  "|---|---|---|---|---|---|---|"]
        for g in ("C0", *ls.CONDITIONS, *(f"{c}_off" for c in ls.CONDITIONS)):
            dev = [final.get(g, {}).get(str(s), {}).get("late_dev") for s in SEEDS]
            last = [scores.get((g, s, "last"), {}).get("raw") for s in SEEDS]
            best = [scores.get((g, s, "best"), {}).get("raw") for s in SEEDS]
            pc2 = [scores.get((g, s, "last"), {}).get("pc2") for s in SEEDS]
            m = lambda xs: f"{statistics.mean(xs)*100:.2f}" if xs and None not in xs else "—"
            gap = m([b - l for b, l in zip(best, last)]) if None not in best + last else "—"
            lines.append(f"| {g} | {m(dev)} | {m(last)} | {' / '.join('—' if x is None else f'{x*100:.2f}' for x in last)} | "
                         f"{m(best)} | {gap} | {m(pc2)} |")
        lines += ["", "## 사전 등록 판정", ""]

        def diffs(a, b, which="last"):
            pairs = [(scores.get((a, s, which), {}).get("raw"), scores.get((b, s, which), {}).get("raw")) for s in SEEDS]
            return [] if any(None in p for p in pairs) else [x - y for x, y in pairs]

        for c in ls.CONDITIONS:
            d = diffs(c, "C0")
            lines.append(f"- **R1 {c} vs C0 (STS7 raw last):** **{verdict(d)}**; Δ {', '.join(f'{x*100:+.2f}' for x in d) or '미실행'}.")
        for c in ls.CONDITIONS:
            last = [scores.get((c, s, "last"), {}).get("raw") for s in SEEDS]
            best = [scores.get((c, s, "best"), {}).get("raw") for s in SEEDS]
            if None in last + best:
                continue
            gaps = [b - l for b, l in zip(best, last)]
            v = ("성립" if statistics.mean(gaps) <= 0.005 and max(gaps) <= 0.01 else
                 "기각" if statistics.mean(gaps) > 0.01 else "판정불가")
            lines.append(f"- **R2 {c} 후반 하락 해소 (best − last ≤ 0.5pp, 시드별 ≤ 1.0pp):** **{v}**; "
                         f"{', '.join(f'{g*100:+.2f}' for g in gaps)}.")
        for c in ls.CONDITIONS:
            d = diffs(c, f"{c}_off")
            lines.append(f"- **R3 {c} 스케줄 자체 효과 (같은 설정, 스케줄 끔 대비, STS7 raw last):** **{verdict(d)}**; "
                         f"Δ {', '.join(f'{x*100:+.2f}' for x in d) or '미실행'}.")
        lines.append("")
    lines += ["미실행 단계는 추정하지 않는다. 3시드로 통계적 유의성을 주장하지 않는다."]
    if state.get("error"):
        lines.append("Error: " + state["error"])
    text = "\n".join(lines) + "\n"
    (campaign / "report.md").write_text(text)
    Path(read_json(campaign / "source_manifest.json")["report"]).write_text(text)


def run(campaign, manifest):
    verify_source(campaign, manifest)
    base = load_config(campaign / "source/configs/vocab_proto/base.yaml")
    previous = read_json(Path(manifest["data_root"]) / VOCAB / "search.json")
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
        write_json(campaign / "state.json", {"stage": "search", "updated_unix": time.time()})
        found = search(campaign, manifest, children, base, previous, records)
        write_json(campaign / "state.json", {"stage": "confirmation", "updated_unix": time.time()})
        groups, selected = confirm(campaign, manifest, children, base, records, found)
        write_json(campaign / "state.json", {"stage": "ablation", "updated_unix": time.time()})
        ablate(campaign, manifest, children, base, records, found, selected, groups)
        write_json(campaign / "state.json", {"stage": "test_evaluation", "updated_unix": time.time()})
        evaluate(campaign, manifest, children, groups)
        write_json(campaign / "state.json", {"stage": "complete", "updated_unix": time.time()})
        render(campaign)
    except BaseException as exc:
        children.stop()
        write_json(campaign / "state.json", {"stage": "stopped_by_user" if stopped else "failed",
                                             "error": str(exc), "updated_unix": time.time()})
        render(campaign)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--slots", type=int, default=2,
                        help="concurrent guarded GPU children; >2 only when the user lifted the account limit")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.execute:
        run(campaign, read_json(campaign / "source_manifest.json"))
        return
    data_root = (ROOT / "data").resolve().parent
    vocab_manifest = read_json(data_root / VOCAB / "source_manifest.json")
    if read_json(data_root / LATE_DROP / "state.json")["stage"] != "complete":
        raise RuntimeError("Phase A must be complete")
    if args.slots > 2 and int(os.environ.get("FLOWDINO_GPU_MAX_PROCESSES", "2")) < args.slots:
        raise RuntimeError("more than 2 slots requires FLOWDINO_GPU_MAX_PROCESSES to be raised as well")
    manifest = prepare(campaign, ROOT, ["auto"] * args.slots, ROOT / "docs/experiments/late_drop/report_phase_b.md",
                       experiment_dir="late_drop")
    manifest.update(protocol="late_schedule", data_root=str(data_root), kmeans_path=vocab_manifest["kmeans_path"],
                    launch_argv=sys.argv)
    write_json(campaign / "source_manifest.json", manifest)
    os.execv(sys.executable, [sys.executable, str(campaign / "source/scripts/run_late_schedule.py"),
                              "--campaign", str(campaign), "--execute"])


if __name__ == "__main__":
    main()
