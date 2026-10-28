"""Analyze measured per-seed trajectories from an explicit CSV or JSONL file.

Required fields: model, seed, optimizer_steps, sts_b_dev, alignment, uniformity.
Only common measured steps are averaged; no interpolation or replay merging.
Input files and existing output directories are never overwritten.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ("sts_b_dev", "alignment", "uniformity")


def load_runs(path):
    with path.open() as stream:
        records = list(csv.DictReader(stream)) if path.suffix == ".csv" else [json.loads(line) for line in stream if line.strip()]
    runs = {}
    for record in records:
        model, seed = record["model"], int(record["seed"])
        step = int(record["optimizer_steps"])
        if not model or step < 0 or float(record["optimizer_steps"]) != step:
            raise ValueError("model and nonnegative integer optimizer_steps are required")
        row = {"model": model, "seed": seed, "optimizer_steps": step,
               **{key: float(record[key]) for key in METRICS}}
        if not all(math.isfinite(row[key]) for key in METRICS):
            raise ValueError(f"nonfinite metric: {model} seed={seed} step={step}")
        rows = runs.setdefault((model, seed), [])
        if rows and step <= rows[-1]["optimizer_steps"]:
            raise ValueError(f"duplicate/non-increasing steps: {model} seed={seed}")
        rows.append(row)
    if not runs:
        raise ValueError("no measured trajectories")
    return runs


def aggregate(runs, model):
    seeds = sorted(seed for name, seed in runs if name == model)
    per_seed = {seed: {r["optimizer_steps"]: r for r in runs[(model, seed)]} for seed in seeds}
    common = sorted(set.intersection(*(set(rows) for rows in per_seed.values())))
    if not common:
        raise ValueError(f"no common measured steps for {model}")
    result = []
    for step in common:
        row = {"model": model, "optimizer_steps": step, "n_seeds": len(seeds)}
        for key in METRICS:
            values = [per_seed[seed][step][key] for seed in seeds]
            row.update({key: statistics.mean(values), key + "_min": min(values), key + "_max": max(values)})
        result.append(row)
    return result


def analyze(data, out):
    runs = load_runs(data)
    means = {model: aggregate(runs, model) for model in sorted({model for model, _ in runs})}
    summary = {model: {"n_seeds": rows[0]["n_seeds"], "n_mean_evaluations": len(rows),
                       "initial": rows[0], "final": rows[-1], "peak_sts": max(rows, key=lambda r: r["sts_b_dev"]),
                       "per_seed_peak_sts": {str(seed): max(points, key=lambda r: r["sts_b_dev"])
                                             for (name, seed), points in runs.items() if name == model}}
               for model, rows in means.items()}
    # A fresh directory keeps prior plots and raw measurements intact.
    out.mkdir(parents=True, exist_ok=False)
    provenance = {"input": str(data.resolve()), "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                  "aggregation": "all available seeds per model, common measured steps only",
                  "warning": "input may contain partial runs; observed peaks are not confirmed optima",
                  "series": summary}
    (out / "summary.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (out / "mean_trajectories.csv").open("w", newline="") as stream:
        rows = [row for points in means.values() for row in points]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    phase, phase_ax = plt.subplots(figsize=(6.8, 5.2))
    metrics, axes = plt.subplots(3, 1, figsize=(7.2, 8), sharex=True)
    colors = plt.get_cmap("tab10")
    for index, (model, rows) in enumerate(means.items()):
        color = colors(index % 10)
        label = f"{model} (n={rows[0]['n_seeds']})"
        for (name, _), points in runs.items():
            if name == model:
                phase_ax.plot([r["uniformity"] for r in points], [r["alignment"] for r in points],
                              color=color, alpha=0.2, linewidth=0.7)
        phase_ax.plot([r["uniformity"] for r in rows], [r["alignment"] for r in rows], color=color, label=label)
        for ax, key in zip(axes, METRICS):
            steps = [r["optimizer_steps"] for r in rows]
            ax.plot(steps, [r[key] for r in rows], color=color, label=label)
            ax.fill_between(steps, [r[key + "_min"] for r in rows], [r[key + "_max"] for r in rows], color=color, alpha=0.12)
            ax.set_ylabel(key)
            ax.grid(alpha=0.2)
    phase_ax.set(xlabel="Uniformity (lower is better)", ylabel="Alignment (lower is better)")
    phase_ax.legend(frameon=False)
    phase_ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Completed training iterations (optimizer_steps)")
    metrics.text(0.5, 0.01, "Shading: seed min–max; not a confidence interval.", ha="center", fontsize=8)
    for fig, name in ((phase, "alignment_uniformity"), (metrics, "metrics_vs_step")):
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        for extension in ("png", "pdf", "svg"):
            fig.savefig(out / f"{name}.{extension}", dpi=200)
        plt.close(fig)
    return provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="new output directory")
    args = ap.parse_args()
    analyze(args.data, args.out)
    print(f"Saved measured trajectories to {args.out}")


if __name__ == "__main__":
    main()
