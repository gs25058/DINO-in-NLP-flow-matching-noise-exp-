"""Phase A of the late-drop study: single-factor interventions from step 1000 on dropping configs.

Each run reuses a vocab_proto configuration and seed exactly and switches one factor at step 1000
(src/intervention.py). Rows before step 1000 must equal the original run bit for bit.
See docs/experiments/late_drop/OPTUNA_SPEC.md.
"""

import argparse
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_vocab_proto as vproto  # noqa: E402
from scripts.run_coviso_optuna import Children, parallel_jobs  # noqa: E402
from src import vocab_proto as vp  # noqa: E402
from src.provenance import prepare, read_json, read_records, verify_source, write_json  # noqa: E402
from src.train import load_config  # noqa: E402

SOURCE = "results/experiments/vocab_proto_20261116"
FROM_STEP = 1000
# (condition, trial, seed, original label): the two configurations with the largest late drop.
TARGETS = (("C2", "9", 42, "search_C2_t9"), ("C2", "9", 43, "confirm_C2_t9"), ("C1", "16", 42, "search_C1_t16"))
C0_NOISE = {"t_lo": 0.4664484163749474, "t_max": 0.901954744114874}  # t13 range
KINDS = ("sharpen_off", "center_freeze", "iso_off", "teacher_freeze", "noise", "lr_zero")
LATE = (3500, 3750, 3999)


def late(rows):
    by = {r["loop_step"]: r["sts_b_dev"] for r in rows}
    return statistics.mean(by[s] for s in LATE)


def summarize(campaign, manifest, results):
    source = Path(manifest["data_root"]) / SOURCE
    out = {}
    for (cond, n, seed, original), runs in results.items():
        orig_log = source / "results/logs" / vproto.run_name(source, original, seed)
        orig_ev = read_records(orig_log / "evaluations.jsonl")
        orig = {"best": max(r["sts_b_dev"] for r in orig_ev), "late": late(orig_ev),
                "at_switch": next(r["sts_b_dev"] for r in orig_ev if r["loop_step"] == FROM_STEP)}
        rows = {}
        for kind, record in runs.items():
            ev = read_records(campaign / "results/logs" / record["run_name"] / "evaluations.jsonl")
            value = late(ev)
            gap = orig["best"] - orig["late"]
            rows[kind] = {"late": value, "last": ev[-1]["sts_b_dev"], "delta_late": value - orig["late"],
                          "recovery": (value - orig["late"]) / gap if gap > 0 else None,
                          "alignment_last": ev[-1]["alignment"], "uniformity_last": ev[-1]["uniformity"]}
        out[f"{cond}_t{n}_s{seed}"] = {"original": orig, "interventions": rows}
    write_json(campaign / "interventions.json", out)
    render(campaign, out)


def render(campaign, out):
    lines = ["# 후반 하락 원인: 단일 요인 개입 (Phase A)", "",
             f"step {FROM_STEP}부터 한 요인만 바꾼다. late = step 3500/3750/3999 dev 평균(×100). "
             "회복률 = (개입 late − 원래 late) / (원래 best − 원래 late).", ""]
    for key, row in out.items():
        o = row["original"]
        lines += [f"## {key}: 원래 best {o['best']*100:.2f}, step {FROM_STEP} {o['at_switch']*100:.2f}, late {o['late']*100:.2f}", "",
                  "| 개입 | late | Δlate | 회복률 | last align | last unif |", "|---|---|---|---|---|---|"]
        for kind, r in sorted(row["interventions"].items(), key=lambda kv: -kv[1]["late"]):
            rec = "—" if r["recovery"] is None else f"{r['recovery']:+.2f}"
            lines.append(f"| {kind} | {r['late']*100:.2f} | {r['delta_late']*100:+.2f} | {rec} | "
                         f"{r['alignment_last']:.3f} | {r['uniformity_last']:.2f} |")
        lines.append("")
    if out:
        lines += ["## 사전 등록 판정 (lr_zero는 조기 종료 기준선이라 판정 대상이 아니다)", ""]
        for kind in KINDS:
            if kind == "lr_zero":
                continue
            recs = [row["interventions"].get(kind, {}).get("recovery") for row in out.values()]
            if len(recs) != len(TARGETS) or None in recs:
                continue
            verdict = ("원인" if all(r >= 0.5 for r in recs) else "부분 원인" if statistics.mean(recs) >= 0.25
                       else "악화" if statistics.mean(recs) <= -0.25 else "원인 아님")
            lines.append(f"- {kind}: **{verdict}** (회복률 {', '.join(f'{r:+.2f}' for r in recs)})")
    (campaign / "report.md").write_text("\n".join(lines) + "\n")
    Path(read_json(campaign / "source_manifest.json")["report"]).write_text("\n".join(lines) + "\n")


def check_prefix(campaign, manifest, record, original, seed):
    """Before the switch step the intervened run must reproduce the original logs exactly."""
    source = Path(manifest["data_root"]) / SOURCE / "results/logs" / vproto.run_name(
        Path(manifest["data_root"]) / SOURCE, original, seed)
    mine = campaign / "results/logs" / record["run_name"]
    for name in ("components.jsonl", "evaluations.jsonl"):
        a = [r for r in read_records(source / name) if r["loop_step"] < FROM_STEP]
        b = [r for r in read_records(mine / name) if r["loop_step"] < FROM_STEP]
        if not a or a != b:
            raise RuntimeError(f"pre-switch {name} differs from the original run: {record['run_name']}")


def run(campaign, manifest):
    verify_source(campaign, manifest)
    base = load_config(campaign / "source/configs/vocab_proto/base.yaml")
    search = read_json(Path(manifest["data_root"]) / SOURCE / "search.json")
    env = dict(os.environ, PYTHONPATH=str(campaign / "source"), PYTHONDONTWRITEBYTECODE="1")
    children = Children(campaign, env)
    signal.signal(signal.SIGTERM, lambda *_: children.stop())
    results = {target: {} for target in TARGETS}
    jobs = [(target, kind) for kind in KINDS for target in TARGETS]

    def execute(gpu, job):
        (cond, n, seed, original), kind = job
        cfg = vp.configure(base, cond, search[cond][n]["params"], kmeans_path=manifest["kmeans_path"])
        cfg["intervention"] = {"kind": kind, "from_step": FROM_STEP, **(C0_NOISE if kind == "noise" else {})}
        record = vproto.train(campaign, manifest, children, gpu, cfg, f"iv_{cond}_t{n}_{kind}", seed)
        check_prefix(campaign, manifest, record, original, seed)
        return record

    def saved(job, record):
        target, kind = job
        results[target][kind] = record
        write_json(campaign / "state.json", {"stage": "interventions", "updated_unix": time.time(),
                                             "done": sum(len(v) for v in results.values()), "total": len(jobs)})
        summarize(campaign, manifest, results)

    try:
        write_json(campaign / "state.json", {"stage": "interventions", "updated_unix": time.time()})
        parallel_jobs(jobs, manifest["gpus"], children, execute, saved)
        write_json(campaign / "state.json", {"stage": "complete", "updated_unix": time.time()})
    except BaseException as exc:
        children.stop()
        write_json(campaign / "state.json", {"stage": "failed", "error": str(exc), "updated_unix": time.time()})
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
    source_manifest = read_json(data_root / SOURCE / "source_manifest.json")
    manifest = prepare(campaign, ROOT, ["auto", "auto"], ROOT / "docs/experiments/late_drop/report.md",
                       experiment_dir="late_drop")
    manifest.update(protocol="late_drop", data_root=str(data_root), kmeans_path=source_manifest["kmeans_path"],
                    source_campaign=str(data_root / SOURCE), launch_argv=sys.argv)
    write_json(campaign / "source_manifest.json", manifest)
    os.execv(sys.executable, [sys.executable, str(campaign / "source/scripts/run_late_drop.py"),
                              "--campaign", str(campaign), "--execute"])


if __name__ == "__main__":
    main()
