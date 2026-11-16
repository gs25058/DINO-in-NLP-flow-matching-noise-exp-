"""DINO-style local→global context: teacher reads B in a 5..7-sentence window, student solo B.

Arms differ only in ``context.teacher_mode``. Two designs, both starting with
60-step smokes and ending with frozen-before-test STS7 of best/last checkpoints:

- fixed: context_distill combination 1 for every arm, A/B/C × seeds 101..103.
- tuned: an independent, equal-budget Optuna study per arm (lockstep), top-3
  fresh-seed confirmation per arm, and C run at B's selected parameters.
"""
import argparse
import copy
import os
from pathlib import Path
import signal
import statistics
import sys
import math
import time

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_bottleneck as p
from src import joint_search
from src.provenance import prepare, read_json, read_records, sha256_file, verify_source, write_json

ARMS = {"A": "none", "B": "real", "C": "unrelated"}
SEEDS = (101, 102, 103)
TRIALS = 12
SAMPLER_SEED = 20261115
SPACE = {"backbone_lr": (3.4382477416615245e-5, 1.3752990966646098e-4, True),
         "head_lr": (.00029124167925665, .0011649667170266, True),
         "beta": (.25, 2., True)}


def configuration(base, params):
    cfg = copy.deepcopy(base)
    cfg["train"].update(lr=params["backbone_lr"], head_lr=params["head_lr"])
    cfg["context"]["beta"] = params["beta"]
    cfg["loss"]["view_weights"] = [1., 1., params["beta"]]
    return cfg


def anchor(base):
    return {"backbone_lr": base["train"]["lr"], "head_lr": base["train"]["head_lr"], "beta": base["context"]["beta"]}


# design=full: B and C only, every major loss/optimizer/EMA/noise axis plus β.
# The coviso_relax joint space, except the teacher temperature reaches 0.03.
FULL_ARMS = ("B", "C")
FULL_TRIALS = 40
FULL_SPACE = {**joint_search.SPACE, "loss.teacher_temp": (0.03, 0.18, False), "context.beta": (0.25, 2.0, True)}
LATE_STEPS = (3500, 3750, 3999)


def full_configuration(base, params):
    """Joint-search parameterization, bounds checked against FULL_SPACE."""
    if set(params) != set(FULL_SPACE):
        raise ValueError("full search parameters must match the declared space")
    cfg = copy.deepcopy(base)
    for key, value in params.items():
        lo, hi, _ = FULL_SPACE[key]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"parameter outside declared search space: {key}")
        if not key.startswith("_"):
            section, name = key.split(".")
            cfg[section][name] = float(value)
    aug = cfg["augment"]
    aug["t_max"] = min(0.99, aug["t_lo"] + params["_t_span"])
    aug["t_start"] = aug["t_lo"] + (aug["t_max"] - aug["t_lo"]) * params["_t_start_frac"]
    cfg["loss"]["view_weights"] = [1., 1., cfg["context"]["beta"]]
    return cfg


def full_anchors(base):
    """Measured base point, and the same point with strong DINO-style sharpening."""
    point = joint_search.baseline_parameters(base)
    point["context.beta"] = float(base["context"]["beta"])
    return [point, {**point, "loss.teacher_temp": 0.05}]


def late_dev(campaign, record):
    """Collapse-aware objective: mean dev Spearman of the last three evaluations."""
    rows = {r["loop_step"]: r["sts_b_dev"] for r in read_records(campaign / "results/logs" / record["run_name"] / "evaluations.jsonl")}
    if not all(step in rows for step in LATE_STEPS):
        raise RuntimeError(f"missing late evaluations for {record['run_name']}")
    return statistics.mean(rows[step] for step in LATE_STEPS)


def audit(campaign, records, calibration):
    p.audit_runs(campaign, records, calibration)
    student, teacher = {}, {}
    for r in records:
        info = read_json(campaign / "results/logs" / r["run_name"] / "context_first_views.json")
        key = (r["seed"], r["label"].startswith("smoke"))
        common = {k: info[k] for k in ("center_noise_sha256", "center_token_sha256", "third_padded_length")}
        if student.setdefault(key, common) != common:
            raise RuntimeError("student B tokens/noise differ across arms")
        if r["label"].split("_")[1] in ("B", "C"):
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
             ""]
    manifest = read_json(campaign / "source_manifest.json")
    design = manifest.get("design", "fixed")
    search = read_json(campaign / "search.json") if (campaign / "search.json").exists() else {}
    if design == "full":
        lines += [f"Design: full. B and C only; independent lockstep TPE over {len(FULL_SPACE)} axes "
                  f"({FULL_TRIALS} attempted trials each, seed 42, median pruning on current dev from step 2000). "
                  "Objective and selection: late dev = mean of steps 3500/3750/3999. Primary test checkpoint: last.", "",
                  "| trial | B late dev | B best | C late dev | C best |", "|---|---|---|---|---|"]
        for n in range(FULL_TRIALS):
            cells = []
            for arm in FULL_ARMS:
                r = search.get(arm, {}).get(str(n))
                cells += (["pruned", ""] if r and r["pruned"] else
                          [f"{r['late_dev']*100:.2f}", f"{r['metrics']['best_sts']*100:.2f}"] if r else ["", ""])
            if any(cells):
                lines.append(f"| {n} | " + " | ".join(cells) + " |")
        for arm, rows in search.items():
            done = {n: r for n, r in rows.items() if not r["pruned"]}
            if done:
                n, r = max(done.items(), key=lambda kv: kv[1]["late_dev"])
                temps = r["params"]["loss.teacher_temp"], r["params"]["loss.student_temp"]
                lines.append(f"\n{arm} search best: trial {n}, late dev {r['late_dev']*100:.2f} "
                             f"(teacher/student temp {temps[0]:.3f}/{temps[1]:.3f}); "
                             f"{len(rows) - len(done)} of {len(rows)} pruned.")
        lines.append("")
        search = {}
    elif design == "tuned":
        lines += [f"Design: tuned. Independent equal-budget Optuna per arm ({TRIALS} trials, seed 42, lockstep), "
                  "top-3 confirmed on seeds 101..103, selection by fresh-seed dev mean; C also run at B's parameters.", ""]
    else:
        lines += ["Design: fixed. context_distill combination 1 for every arm; no selection.", ""]
    if search:
        lines += ["| trial | A dev best | B dev best | C dev best |", "|---|---|---|---|"]
        for n in range(TRIALS):
            cells = [f"{search[a][str(n)]['metrics']['best_sts']*100:.2f}" if str(n) in search.get(a, {}) else "" for a in ARMS]
            if any(cells):
                lines.append(f"| {n} | " + " | ".join(cells) + " |")
        for arm, rows in search.items():
            if rows:
                n, r = max(rows.items(), key=lambda kv: kv[1]["metrics"]["best_sts"])
                params = ", ".join(f"{k}={v:.4g}" for k, v in r["params"].items())
                lines.append(f"\n{arm} search best: trial {n}, {r['metrics']['best_sts']*100:.2f} ({params}).")
        lines.append("")
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
        which = "last" if design == "full" else "best"
        for control in ("C", "C_matched_B", "A"):
            b, o = scores.get(("B", which), {}), scores.get((control, which), {})
            if len(b) == len(o) == 3:
                d = [b[str(s)] - o[str(s)] for s in SEEDS]
                verdict = ("성립" if statistics.mean(d) >= .003 and sum(x > 0 for x in d) >= 2 else
                           "기각" if statistics.mean(d) <= 0 and sum(x <= 0 for x in d) >= 2 else "판정불가")
                lines.append(f"\nB vs {control} (STS7, {which}): {verdict}, paired Δ={statistics.mean(d)*100:+.2f} pp, "
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

    def train(gpu, arm, label, seed, smoke=False, params=None):
        cfg = configs[arm] if params is None else configuration(configs[arm], params)
        return p.execute_train(campaign, manifest, children, gpu, cfg, label, seed, smoke)

    try:
        stage(campaign, "smoke")
        smokes = {}
        design = manifest.get("design", "fixed")
        p.parallel_jobs(list(FULL_ARMS if design == "full" else ARMS), manifest["gpus"], children,
                        lambda gpu, arm: train(gpu, arm, f"smoke_{arm}", 42, smoke=True),
                        lambda arm, r: smokes.update({arm: r}))
        audit(campaign, list(smokes.values()), calibration)
        write_json(campaign / "smoke.json", smokes)
        if design == "full":
            groups = full(campaign, manifest, children, records, calibration, configs)
        elif design == "tuned":
            groups = tuned(campaign, manifest, children, train, records, calibration, configs)
        else:
            groups = fixed(campaign, manifest, children, train, records, calibration)
        stage(campaign, "test_evaluation")
        p.evaluate(campaign, manifest, groups, children)
        stage(campaign, "complete")
    except BaseException as exc:
        children.stop()
        stage(campaign, "stopped_by_user" if stopped else "failed", error=str(exc))
        raise


def fixed(campaign, manifest, children, train, records, calibration):
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
    return groups


def tuned(campaign, manifest, children, train, records, calibration, configs):
    """Equal budgets: trial n of every arm is asked only after trial n-1 of every arm is told."""
    storage = f"sqlite:///{campaign / 'studies.db'}"
    studies = {}
    for arm in ARMS:
        studies[arm] = optuna.create_study(
            study_name=f"{campaign.name}_{arm}", storage=storage, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED, n_startup_trials=4, multivariate=True),
            pruner=optuna.pruners.NopPruner())
        studies[arm].enqueue_trial(anchor(configs[arm]))
    search = {arm: {} for arm in ARMS}
    for n in range(TRIALS):
        stage(campaign, "search", trial=n, completed_trials=n)
        trials = {arm: studies[arm].ask() for arm in ARMS}
        params = {arm: {k: trials[arm].suggest_float(k, lo, hi, log=log) for k, (lo, hi, log) in SPACE.items()}
                  for arm in ARMS}

        def done(arm, r):
            search[arm][str(n)] = {**r, "params": params[arm]}
            records.append(r)
            write_json(campaign / "search.json", search)
            audit(campaign, records, calibration)

        p.parallel_jobs(list(ARMS), manifest["gpus"], children,
                        lambda gpu, arm: train(gpu, arm, f"search_{arm}_t{n}", 42, params=params[arm]), done)
        for arm in ARMS:
            studies[arm].tell(trials[arm], search[arm][str(n)]["metrics"]["best_sts"])
        render(campaign)
    shortlist = {arm: sorted(range(TRIALS), key=lambda n: (-search[arm][str(n)]["metrics"]["best_sts"], n))[:3]
                 for arm in ARMS}
    write_json(campaign / "shortlist.json", {"trials": shortlist, "seeds": SEEDS, "test_seen": False})
    confirmed = {arm: {str(n): {} for n in shortlist[arm]} for arm in ARMS}
    stage(campaign, "confirmation")

    def saved(job, r):
        arm, n, seed = job
        confirmed[arm][str(n)][str(seed)] = r
        records.append(r)
        write_json(campaign / "confirmations.json", confirmed)
        audit(campaign, records, calibration)
        render(campaign)

    jobs = [(arm, n, seed) for seed in SEEDS for arm in ARMS for n in shortlist[arm]]
    p.parallel_jobs(jobs, manifest["gpus"], children,
                    lambda gpu, job: train(gpu, job[0], f"confirm_{job[0]}_t{job[1]}", job[2],
                                           params=search[job[0]][str(job[1])]["params"]), saved)
    selected = {arm: min(shortlist[arm], key=lambda n: (-statistics.mean(
        r["metrics"]["best_sts"] for r in confirmed[arm][str(n)].values()), n)) for arm in ARMS}
    groups = {arm: confirmed[arm][str(n)] for arm, n in selected.items()}
    bparams = search["B"][str(selected["B"])]["params"]
    stage(campaign, "matched_unrelated_control")
    groups["C_matched_B"] = {}

    def matched(seed, r):
        groups["C_matched_B"][str(seed)] = r
        records.append(r)
    p.parallel_jobs(list(SEEDS), manifest["gpus"], children,
                    lambda gpu, seed: train(gpu, "C", "matched_C_B", seed, params=bparams),
                    lambda seed, r: matched(seed, r))
    audit(campaign, records, calibration)
    write_json(campaign / "selection.json", {
        "selected_trials": selected, "selected_params": {arm: search[arm][str(n)]["params"] for arm, n in selected.items()},
        "C_matched_B_params": bparams, "selection_metric": "fresh three-seed solo 768d STS-B dev best mean",
        "selection_frozen_unix": time.time(), "test_scores_seen": False})
    write_json(campaign / "final_runs.json", groups)
    return groups


def full(campaign, manifest, children, records, calibration, configs):
    """B and C: lockstep independent TPE over FULL_SPACE, late-dev objective, current-value pruning."""
    storage = f"sqlite:///{campaign / 'studies.db'}"
    studies, search = {}, {arm: {} for arm in FULL_ARMS}
    for arm in FULL_ARMS:
        studies[arm] = optuna.create_study(
            study_name=f"{campaign.name}_{arm}", storage=storage, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED, n_startup_trials=8, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2000, interval_steps=500,
                                               n_min_trials=4))
        for point in full_anchors(configs[arm]):
            studies[arm].enqueue_trial(point)

    def train(gpu, arm, label, seed, params, trial=None):
        return p.execute_train(campaign, manifest, children, gpu, full_configuration(configs[arm], params),
                               label, seed, trial=trial, report="current")

    for n in range(FULL_TRIALS):
        stage(campaign, "search", trial=n, completed_trials=n)
        trials = {arm: studies[arm].ask() for arm in FULL_ARMS}
        params = {arm: {k: trials[arm].suggest_float(k, lo, hi, log=log) for k, (lo, hi, log) in FULL_SPACE.items()}
                  for arm in FULL_ARMS}
        outcome = {}

        def attempt(gpu, arm):
            try:
                return train(gpu, arm, f"search_{arm}_t{n}", 42, params[arm], trials[arm])
            except optuna.TrialPruned:
                return None

        p.parallel_jobs(list(FULL_ARMS), manifest["gpus"], children, attempt,
                        lambda arm, r: outcome.update({arm: r}))
        for arm in FULL_ARMS:
            r = outcome[arm]
            if r is None:
                studies[arm].tell(trials[arm], state=optuna.trial.TrialState.PRUNED)
                search[arm][str(n)] = {"params": params[arm], "pruned": True}
                continue
            value = late_dev(campaign, r)
            records.append(r)
            studies[arm].tell(trials[arm], value)
            search[arm][str(n)] = {**r, "params": params[arm], "pruned": False, "late_dev": value}
        write_json(campaign / "search.json", search)
        audit(campaign, records, calibration)
        render(campaign)
    complete = {arm: [n for n, r in search[arm].items() if not r["pruned"]] for arm in FULL_ARMS}
    shortlist = {arm: sorted(complete[arm], key=lambda n: (-search[arm][n]["late_dev"], int(n)))[:3] for arm in FULL_ARMS}
    write_json(campaign / "shortlist.json", {"trials": shortlist, "seeds": SEEDS, "test_seen": False,
                                              "metric": "late dev (steps 3500/3750/3999 mean)"})
    confirmed = {arm: {n: {} for n in shortlist[arm]} for arm in FULL_ARMS}
    stage(campaign, "confirmation")

    def saved(job, r):
        arm, n, seed = job
        confirmed[arm][n][str(seed)] = {**r, "late_dev": late_dev(campaign, r)}
        records.append(r)
        write_json(campaign / "confirmations.json", confirmed)
        audit(campaign, records, calibration)
        render(campaign)

    jobs = [(arm, n, seed) for seed in SEEDS for arm in FULL_ARMS for n in shortlist[arm]]
    p.parallel_jobs(jobs, manifest["gpus"], children,
                    lambda gpu, job: train(gpu, job[0], f"confirm_{job[0]}_t{job[1]}", job[2],
                                           search[job[0]][job[1]]["params"]), saved)
    selected = {arm: min(shortlist[arm], key=lambda n: (-statistics.mean(
        r["late_dev"] for r in confirmed[arm][n].values()), int(n))) for arm in FULL_ARMS}
    groups = {arm: confirmed[arm][n] for arm, n in selected.items()}
    bparams = search["B"][selected["B"]]["params"]
    stage(campaign, "matched_unrelated_control")
    groups["C_matched_B"] = {}

    def matched(seed, r):
        groups["C_matched_B"][str(seed)] = {**r, "late_dev": late_dev(campaign, r)}
        records.append(r)

    p.parallel_jobs(list(SEEDS), manifest["gpus"], children,
                    lambda gpu, seed: train(gpu, "C", "matched_C_B", seed, bparams), matched)
    audit(campaign, records, calibration)
    write_json(campaign / "selection.json", {
        "selected_trials": selected, "selected_params": {arm: search[arm][n]["params"] for arm, n in selected.items()},
        "C_matched_B_params": bparams, "selection_metric": "fresh three-seed late dev mean (steps 3500/3750/3999)",
        "primary_test_checkpoint": "last", "selection_frozen_unix": time.time(), "test_scores_seen": False})
    write_json(campaign / "final_runs.json", groups)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--design", choices=("fixed", "tuned", "full"), default="fixed")
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
    manifest.update(protocol="context_teacher", design=args.design, reference_groups=[], launch_argv=sys.argv,
                    data_manifest_sha256=sha256_file(data), data_counts=cache["counts"])
    write_json(c / "source_manifest.json", manifest)
    os.execv(sys.executable, [sys.executable, str(c / "source/scripts/run_context_teacher.py"),
                              "--campaign", str(c), "--execute"])


if __name__ == "__main__":
    main()
