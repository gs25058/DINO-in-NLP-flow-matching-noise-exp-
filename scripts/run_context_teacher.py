"""DINO-style local→global context: teacher reads B in a 5..7-sentence window, student solo B.

Fixed hyperparameters (context_distill combination 1); arms differ only in
``context.teacher_mode``. 60-step smokes, then A/B/C × seeds 101..103 paired,
then frozen-before-test STS7 of best/last checkpoints.
"""
import argparse
import os
from pathlib import Path
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_bottleneck as p
from src.provenance import prepare, read_json, sha256_file, verify_source, write_json

ARMS = {"A": "none", "B": "real", "C": "unrelated"}
SEEDS = (101, 102, 103)


def audit(campaign, records, calibration):
    p.audit_runs(campaign, records, calibration)
    student, teacher = {}, {}
    for r in records:
        info = read_json(campaign / "results/logs" / r["run_name"] / "context_first_views.json")
        key = (r["seed"], r["label"].startswith("smoke"))
        common = {k: info[k] for k in ("center_noise_sha256", "center_token_sha256", "third_padded_length")}
        if student.setdefault(key, common) != common:
            raise RuntimeError("student B tokens/noise differ across arms")
        if r["label"].split("_")[-1] in ("B", "C"):
            layout = {k: info[k] for k in ("teacher_tokens", "teacher_padded_length")}
            if teacher.setdefault(key, layout) != layout:
                raise RuntimeError("teacher context lengths differ between real and unrelated arms")
    write_json(campaign / "context_comparability_audit.json",
               {"same_student_B_tokens_noise": True, "same_BC_teacher_lengths": True, "runs": len(records)})


def render(campaign):
    state = read_json(campaign / "state.json")
    lines = ["# Context-reading teacher (local→global)", "", f"Stage: {state['stage']}",
             "Teacher: clean B inside a 5..7-sentence same-paragraph window, B tokens pooled.",
             "Student: three noisy solo-B views (weights 1, 1, β); identical in all arms.",
             "A: teacher solo B. B: real window. C: other-document context, exact per-side token lengths.",
             "Fixed hyperparameters: context_distill combination 1. Selection: none (fixed config).", ""]
    runs = read_json(campaign / "final_runs.json") if (campaign / "final_runs.json").exists() else {}
    if runs:
        lines += ["| arm | seed | dev best | dev last | best step | wall s | teacher nonpad tokens |", "|---|---|---|---|---|---|---|"]
        for arm, seeds in runs.items():
            for seed, r in sorted(seeds.items()):
                m = r["metrics"]
                lines.append(f"| {arm} | {seed} | {m['best_sts']*100:.2f} | {m['last_sts']*100:.2f} | "
                             f"{m.get('best_loop_step','')} | {m.get('wall_seconds',0):.0f} | {m.get('teacher_nonpad_tokens','')} |")
        for arm, seeds in runs.items():
            if len(seeds) == 3:
                v = [r["metrics"]["best_sts"] * 100 for r in seeds.values()]
                lines.append(f"\n{arm} dev best: {statistics.mean(v):.2f} ± {statistics.stdev(v):.2f} (sample SD).")
    scores = {}
    for file in sorted((campaign / "results/analysis").glob("*.json")):
        doc = read_json(file)
        if "experiment" in doc:
            ex = doc["experiment"]
            scores.setdefault((ex["group"], ex["which"]), {})[str(ex["seed"])] = next(iter(doc["runs"].values()))["avg_7task"]
    if scores:
        lines += ["", "| arm | checkpoint | STS7 per seed 101/102/103 | mean |", "|---|---|---|---|"]
        for (arm, which), v in sorted(scores.items()):
            vals = [v.get(str(s)) for s in SEEDS]
            shown = " / ".join(f"{x*100:.2f}" if x is not None else "—" for x in vals)
            mean = f"{statistics.mean(x*100 for x in vals):.2f}" if None not in vals else ""
            lines.append(f"| {arm} | {which} | {shown} | {mean} |")
        for control in ("C", "A"):
            b, o = scores.get(("B", "best"), {}), scores.get((control, "best"), {})
            if len(b) == len(o) == 3:
                d = [b[str(s)] - o[str(s)] for s in SEEDS]
                verdict = ("성립" if statistics.mean(d) >= .003 and sum(x > 0 for x in d) >= 2 else
                           "기각" if statistics.mean(d) <= 0 and sum(x <= 0 for x in d) >= 2 else "판정불가")
                lines.append(f"\nB vs {control} (STS7, best): {verdict}, paired Δ={statistics.mean(d)*100:+.2f} pp, "
                             f"per seed {', '.join(f'{x*100:+.2f}' for x in d)}.")
    lines += ["", "Missing results are unexecuted, not estimated. No statistical significance is claimed."]
    if state.get("error"):
        lines.append("Error: " + state["error"])
    text = "\n".join(lines) + "\n"
    (campaign / "report.md").write_text(text)
    Path(read_json(campaign / "source_manifest.json")["report"]).write_text(text)


def stage(campaign, name, **extra):
    write_json(campaign / "state.json", {"stage": name, "updated_unix": time.time(), **extra})
    render(campaign)


def run(campaign, manifest):
    verify_source(campaign, manifest)
    configs = {arm: p.load_config(campaign / f"source/configs/context_teacher/{arm}.yaml") for arm in ARMS}
    env = dict(os.environ, PYTHONPATH=str(campaign / "source"), PYTHONDONTWRITEBYTECODE="1")
    children = p.Children(campaign, env)
    stopped = False
    p.render, p.set_stage = render, stage

    def stop(*_):
        nonlocal stopped
        stopped = True
        children.stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    calibration = {}
    write_json(campaign / "calibration.json", calibration)
    records = []

    def train(gpu, arm, label, seed, smoke=False):
        return p.execute_train(campaign, manifest, children, gpu, configs[arm], label, seed, smoke)

    try:
        stage(campaign, "smoke")
        smokes = {}
        p.parallel_jobs(list(ARMS), manifest["gpus"], children,
                        lambda gpu, arm: train(gpu, arm, f"smoke_{arm}", 42, smoke=True),
                        lambda arm, r: smokes.update({arm: r}))
        audit(campaign, list(smokes.values()), calibration)
        write_json(campaign / "smoke.json", smokes)
        groups = {arm: {} for arm in ARMS}
        stage(campaign, "paired_runs")
        # Seed-major order: each finished seed is a complete A/B/C pair early on.
        jobs = [(arm, seed) for seed in SEEDS for arm in ARMS]

        def saved(job, r):
            arm, seed = job
            groups[arm][str(seed)] = r
            records.append(r)
            write_json(campaign / "final_runs.json", groups)
            audit(campaign, records, calibration)
            render(campaign)

        p.parallel_jobs(jobs, manifest["gpus"], children,
                        lambda gpu, job: train(gpu, job[0], f"run_{job[0]}", job[1]), saved)
        write_json(campaign / "selection.json", {"fixed_configuration": True, "selection_frozen_unix": time.time(),
                                                  "test_scores_seen": False, "arms": ARMS})
        stage(campaign, "test_evaluation")
        p.evaluate(campaign, manifest, groups, children)
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
    c = args.campaign.resolve()
    if args.execute:
        run(c, read_json(c / "source_manifest.json"))
        return
    base = p.load_config(ROOT / "configs/context_teacher/base.yaml")
    data = ROOT / base["data"]["context_manifest_path"]
    cache = read_json(data)
    for name, h in cache["files"].items():
        if sha256_file(data.parent / name) != h:
            raise RuntimeError("prepared cache hash mismatch")
    if sha256_file(data.parent / "embed_stats_bert_base_uncased.pt") != cache["statistics"]["file_sha256"]:
        raise RuntimeError("statistics changed")
    manifest = prepare(c, ROOT, ["auto", "auto"], ROOT / "docs/experiments/context_teacher/report.md",
                       experiment_dir="context_teacher")
    manifest.update(protocol="context_teacher", reference_groups=[], launch_argv=sys.argv,
                    data_manifest_sha256=sha256_file(data), data_counts=cache["counts"])
    write_json(c / "source_manifest.json", manifest)
    os.execv(sys.executable, [sys.executable, str(c / "source/scripts/run_context_teacher.py"),
                              "--campaign", str(c), "--execute"])


if __name__ == "__main__":
    main()
