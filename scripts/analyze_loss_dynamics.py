"""Read completed runs, derive loss dynamics, and export figures; never trains.

Uses the existing environment's matplotlib installation with uv run --no-sync.
Optional checkpoint inspection loads weights on CPU only. No source snapshots,
training logs, model buffers, or checkpoints are changed.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.provenance import read_json, read_records, write_json
from src.train import load_config

SOURCES = {
    "baseline": ("bneck_dino256_20261105", "baseline"),
    "direct256": ("bneck_dino256_20261105", "bottleneck"),
    "cos_iso": ("bneck_iso_20261104", "bottleneck"),
    "cos_only": ("bneck_iso_20261104", "iso_off"),
    "iso_only": ("bneck_iso_20261104", "aux_off"),
}
COLORS = {"baseline": "#2874A6", "direct256": "#D76A26", "cos_iso": "#27956B", "cos_only": "#9564B5", "iso_only": "#777777"}


def moments(runs):
    """Aggregate each observed step over seeds; SD is between seeds, not steps."""
    result = []
    for step in sorted({r["loop_step"] for run in runs for r in run}):
        rows = [r for run in runs for r in run if r["loop_step"] == step]
        keys = set.intersection(*(set(r) for r in rows)) - {"loop_step"}
        row = {"step": step, "n": len(rows)}
        for key in keys:
            values = [r[key] for r in rows]
            if all(isinstance(v, (float, int)) for v in values):
                row[key] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
        result.append(row)
    return result


def interference(rows, seed):
    """Reconstruct the sum of pairwise gradient dots from actual pre-clip norms.

    Only valid for direct256: its trainable parameters are backbone + MLP and
    its objective has exactly the three measured weighted loss components.
    This does not recover individual pairwise angles or AdamW update vectors.
    """
    out = []
    for r in rows:
        norms = [math.hypot(r[f"grad_{x}_backbone"], r[f"grad_{x}_mlp"]) for x in ("dino", "mean", "cov")]
        squared = sum(n * n for n in norms)
        pair_norms = sum(norms[i] * norms[j] for i, j in ((0, 1), (0, 2), (1, 2)))
        ratio = r["grad_norm_total"] / math.sqrt(squared)
        if r["loop_step"] == 0:
            assert abs(ratio - 1) < 1e-5, "unaccounted parameter group or loss"
        if pair_norms == 0:
            continue
        dot_sum = (r["grad_norm_total"] ** 2 - squared) / 2
        beta = dot_sum / pair_norms
        assert -1.0001 <= beta <= 1.0001
        out.append({"loop_step": r["loop_step"], "seed": int(seed), "weighted_mean_pair_cos": beta,
                    "actual_over_orthogonal": ratio, "sum_pair_dot": dot_sum})
    return out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect():
    groups, csv_rows, interactions = {}, {"components": [], "gradients": [], "evaluations": []}, []
    for label, (campaign, group) in SOURCES.items():
        c = ROOT / "results/experiments" / campaign
        runs = read_json(c / "final_runs.json")[group]
        streams = {k: [] for k in csv_rows}
        for seed, run in runs.items():
            d = c / "results/logs" / run["run_name"]
            cfg = load_config(run["config"])
            k = cfg["model"]["head"]["logit_dim"] or cfg["model"]["head"]["bottleneck_dim"]
            components = read_records(d / "components.jsonl")
            for row in components:
                row.update(target_diversity=row["H_p_bar_t"] - row["H_pt"],
                           target_kl_uniform=math.log(k) - row["H_pt"],
                           normalized_target_entropy=row["H_pt"] / math.log(k))
            gradients = sorted([r for r in components + read_records(d / "gradient_diagnostics.jsonl") if "grad_dino_backbone" in r], key=lambda r: r["loop_step"])
            data = {"components": components, "gradients": gradients, "evaluations": read_records(d / "evaluations.jsonl")}
            for kind, rows in data.items():
                streams[kind].append(rows)
                csv_rows[kind] += [{"group": label, "seed": seed, "class_count": k, **r} for r in rows]
            if label == "direct256":
                interactions += interference(gradients, seed)
        groups[label] = {kind: moments(rows) for kind, rows in streams.items()}
    for kind, rows in csv_rows.items():
        assert rows, kind
    return groups, csv_rows, interactions


def trials():
    c = ROOT / "results/experiments/bneck_dino256_20261105"
    rows = []
    for path in sorted((c / "configs").glob("search_bottleneck_t*_s42.yaml")):
        cfg = load_config(path)
        d = c / "results/logs" / cfg["run_name"]
        ev, loss = read_records(d / "evaluations.jsonl"), read_records(d / "components.jsonl")
        if not ev or not (d / "completion.json").exists():
            continue
        last = loss[-1]
        t = cfg["loss"]["student_temp"]
        rows.append({"trial": int(path.stem.split("_t")[1].split("_")[0]),
                     "lambda_iso": cfg["loss"]["cov_iso_mean_lambda"], "student_temp": t,
                     "teacher_temp": cfg["loss"]["teacher_temp"],
                     "teacher_student_ratio": cfg["loss"]["teacher_temp"] / t,
                     "student_max_probability": 1 / (1 + 255 * math.exp(-math.sqrt(256 / 255) / t)),
                     "dev_best": max(r["sts_b_dev"] for r in ev), "dev_last": ev[-1]["sts_b_dev"],
                     "bneck_last": ev[-1]["sts_b_dev_bottleneck"],
                     "target_entropy_last": last["H_pt"], "target_batch_entropy_last": last["H_p_bar_t"],
                     "kl_last": last["KL_pt_ps"], "gradient_norm_last": last["grad_norm_total"],
                     "logged_clipping_fraction": sum(r["grad_norm_total"] > cfg["train"]["grad_clip"] for r in loss) / len(loss),
                     "logged_steps": len(loss)})
    return sorted(rows, key=lambda r: r["trial"])


def checkpoint_drift():
    import gc
    import torch

    torch.set_num_threads(4)
    result = []
    keys = ("backbone.embeddings.word_embeddings.weight", "head.mlp.0.weight", "head.mlp.2.weight")
    for group in ("baseline", "direct256"):
        campaign, arm = SOURCES[group]
        c = ROOT / "results/experiments" / campaign
        for seed, run in read_json(c / "final_runs.json")[arm].items():
            checkpoint = torch.load(c / "checkpoints" / run["run_name"] / "last.pt", map_location="cpu", weights_only=True)
            student, teacher = checkpoint["state_dict"], checkpoint["teacher_state_dict"]
            result.append({"group": group, "seed": seed,
                           "center_saved": "dino_state_dict" in checkpoint,
                           "relative_student_teacher_difference": {
                               k: ((student[k] - teacher[k]).norm() / teacher[k].norm()).item()
                               for k in keys}})
            del checkpoint, student, teacher
            gc.collect()
    return result


def collapse_stationary_example():
    import torch
    from src.loss import CovIsoPenalty, DINOLoss

    x = torch.ones(32, 256, dtype=torch.float64, requires_grad=True)
    b = torch.nn.functional.normalize(x, dim=-1)
    penalty = CovIsoPenalty(256, 0.8195105695006707).double()
    objective = DINOLoss(256, 0.92).double()
    with torch.no_grad():
        penalty.mu_ema.copy_(b[0])
        penalty.m2_ema.copy_(torch.outer(b[0], b[0]))
        objective.center.copy_(b[0])
    terms = penalty(b)
    dino, _ = objective(b.detach(), [b], 0.1, 0.11, update_center=False)
    return {name: {"loss": term.item(), "gradient_before_l2_norm": torch.autograd.grad(term, x, retain_graph=True)[0].norm().item()}
            for name, term in (("dino", dino), ("mean", terms.mean), ("cov", terms.cov))}


def plots(out, groups, interactions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "savefig.bbox": "tight"})

    def line(ax, stream, key, label, color, scale=1, marker=None):
        values = [r for r in stream if key in r]
        x = [r["step"] for r in values]
        y = np.array([r[key]["mean"] * scale for r in values])
        sd = np.array([r[key]["sd"] * scale for r in values])
        ax.plot(x, y, label=label, color=color, marker=marker, markersize=4, linewidth=1.7)
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.12)

    def finish(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    panels = [("evaluations", "sts_b_dev", "768d STS-B dev", 100),
              ("evaluations", "sts_b_dev_bottleneck", "256d STS-B dev", 100),
              ("evaluations", "eff_rank", "768d centered effective rank", 1),
              ("evaluations", "effective_rank_bottleneck", "256d centered effective rank", 1),
              ("evaluations", "mean_norm_bottleneck", "256d clean centroid norm", 1),
              ("components", "target_diversity", "Teacher batch distribution diversity (nats)", 1)]
    for ax, (kind, key, title, scale) in zip(axes.flat, panels):
        for group in ("baseline", "direct256", "cos_iso"):
            line(ax, groups[group][kind], key, group, COLORS[group], scale)
        ax.set_title(title)
        ax.axvline(56, color="grey", ls=":", lw=1)
        ax.set_xlabel("loop step")
        ax.legend(fontsize=8)
    finish(fig, "overview")

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    direct = groups["direct256"]
    for key, label, color in (("KL_pt_ps", "DINO KL(t||s)", "#2874A6"), ("weighted_mean", "weighted mean", "#D76A26"), ("weighted_cov", "weighted cov", "#27956B")):
        line(axes[0, 0], direct["components"], key, label, color)
    axes[0, 0].set_title("Direct256: CE minus teacher entropy, and penalties")
    axes[0, 0].set_yscale("log")
    for ax, scope in ((axes[0, 1], "backbone"), (axes[1, 0], "mlp")):
        for term, color in (("dino", "#2874A6"), ("mean", "#D76A26"), ("cov", "#27956B")):
            line(ax, direct["gradients"], f"grad_{term}_{scope}", term, color, marker="o")
        ax.set_title(f"Weighted gradient norms: {scope}")
        ax.set_yscale("log")
    for group in ("baseline", "direct256"):
        line(axes[1, 1], groups[group]["components"], "normalized_target_entropy", group, COLORS[group])
    axes[1, 1].set_title("Teacher conditional entropy / log(K)")
    per_seed = [[r for r in interactions if r["seed"] == seed] for seed in (101, 102, 103)]
    line(axes[2, 0], moments(per_seed), "weighted_mean_pair_cos", "net interference", "#9564B5", marker="o")
    axes[2, 0].axhline(0, color="black", linewidth=0.8)
    axes[2, 0].set_title("Norm-derived weighted mean pair cosine")
    x = np.arange(4000)
    lr = np.where(x < 206, x / 206, 0.5 * (1 + np.cos(np.pi * (x - 206) / (4000 - 206))))
    axes[2, 1].plot(x, lr, label="LR / peak LR", color="#2874A6")
    temperature = np.array([r["teacher_temp"]["mean"] for r in direct["components"]])
    steps = [r["step"] for r in direct["components"]]
    axes[2, 1].plot(steps, temperature / 0.10791228321287379, label="teacher T / student T", color="#D76A26")
    axes[2, 1].set_title("Schedules (LR warmup 206, temperature warmup 315)")
    for ax in axes.flat:
        ax.axvline(56, color="grey", ls=":", linewidth=1)
        ax.set_xlabel("loop step")
        ax.legend(fontsize=8)
    finish(fig, "direct_gradients")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    c = ROOT / "results/experiments/bneck_dino256_20261105"
    for n, color in ((15, "#D76A26"), (1, "#9564B5"), (2, "#2874A6"), (3, "#777777")):
        cfg = load_config(c / "configs" / f"search_bottleneck_t{n}_s42.yaml")
        d = c / "results/logs" / cfg["run_name"]
        ev, loss = read_records(d / "evaluations.jsonl"), read_records(d / "components.jsonl")
        axes[0].plot([r["loop_step"] for r in ev], [r["sts_b_dev"] * 100 for r in ev], label=f"trial {n}", color=color)
        axes[1].plot([r["loop_step"] for r in loss], [r["H_pt"] / math.log(256) for r in loss], label=f"trial {n}", color=color)
        axes[2].plot([r["loop_step"] for r in loss], [r["grad_norm_total"] for r in loss], label=f"trial {n}", color=color, alpha=0.8)
    axes[2].axhline(5.590297628724676, color="black", ls="--", label="clip threshold")
    axes[2].set_yscale("log")
    for ax, title in zip(axes, ("768d STS-B dev", "Teacher entropy / log(256)", "Pre-clip total gradient norm")):
        ax.set_title(title)
        ax.set_xlabel("loop step")
        ax.legend(fontsize=8)
    finish(fig, "sharp_trials")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-drift", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    groups, rows, interactions = collect()
    for kind, values in rows.items():
        write_csv(args.out / f"{kind}.csv", values)
    write_csv(args.out / "gradient_interference.csv", interactions)
    trial_rows = trials()
    write_csv(args.out / "search_trials.csv", trial_rows)
    summary = {"groups": groups, "gradient_interference": moments([[r for r in interactions if r["seed"] == seed] for seed in (101, 102, 103)]),
               "search_trials": trial_rows, "uniform_collapse_stationary_example": collapse_stationary_example(),
               "checkpoint_drift": checkpoint_drift() if args.checkpoint_drift else [],
               "scope": "read-only offline log/checkpoint analysis; no training or new STS7 selection"}
    write_json(args.out / "summary.json", summary)
    plots(args.out, groups, interactions)
    print(args.out / "summary.json")


if __name__ == "__main__":
    main()
