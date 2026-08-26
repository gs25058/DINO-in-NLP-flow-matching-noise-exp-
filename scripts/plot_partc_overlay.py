"""Part C 전용: r2_diag vs r5d_diag 진단 지표 겹쳐그리기 (results/analysis/partc_overlay.png)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_logs import LOG_DIR, _series, parse_log

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "analysis"


def main():
    r2 = parse_log(LOG_DIR / "r2_diag" / "train.log")
    r5d = parse_log(LOG_DIR / "r5d_diag" / "train.log")

    fig, axes = plt.subplots(6, 1, figsize=(9, 20), sharex=True)

    def plot_pair(ax, rows_key, field, ylabel, log_y=False):
        for name, d, color in [("r2_diag", r2, "tab:red"), ("r5d_diag", r5d, "tab:blue")]:
            xs, ys = _series(d[rows_key], field)
            if xs:
                ax.plot(xs, ys, "o-", label=name, color=color, markersize=3)
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=8)

    plot_pair(axes[0], "eval", "sts_b_dev", "STS-B dev (student)")
    plot_pair(axes[1], "eval", "eff_rank", "effective rank")
    plot_pair(axes[2], "train", "diag_grad_norm_backbone", "grad_norm: backbone", log_y=True)
    plot_pair(axes[3], "train", "diag_active_prototypes", "active prototypes (of 8192)")
    plot_pair(axes[4], "eval", "teacher_sts_b_dev", "STS-B dev (teacher)")
    plot_pair(axes[5], "eval", "drift_cosine", "adjacent-eval drift cosine")
    axes[-1].set_xlabel("step")
    fig.suptitle("Part C: r2 (collapsing) vs r5d (stable) diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "partc_overlay.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    # grad norm 세 그룹 비교(각 run별로 backbone/bottleneck/prototype 한 그림에)
    fig2, axes2 = plt.subplots(2, 1, figsize=(9, 8), sharex=True, sharey=True)
    for ax, (name, d) in zip(axes2, [("r2_diag", r2), ("r5d_diag", r5d)]):
        for field, color in [("diag_grad_norm_backbone", "tab:blue"),
                              ("diag_grad_norm_bottleneck", "tab:orange"),
                              ("diag_grad_norm_prototype", "tab:green")]:
            xs, ys = _series(d["train"], field)
            ax.plot(xs, ys, label=field.replace("diag_grad_norm_", ""), color=color)
        ax.set_yscale("log")
        ax.set_ylabel(f"{name}\ngrad norm (log)")
        ax.legend(fontsize=8)
    axes2[-1].set_xlabel("step")
    fig2.tight_layout()
    out_path2 = OUT_DIR / "partc_grad_norms.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"wrote {out_path2}")


if __name__ == "__main__":
    main()
