"""koleo_lambda 감쇠 스케줄 실험(P-KS1~3) 겹쳐그리기.

상수 λ(대조군, 기존 run) vs koleomin0.25(부분 감쇠) vs koleomin0(완전 소거) 3조건 x 시드 2개를
같은 그림에 겹쳐 그려 results/analysis/koleo_schedule/overlay.png로 저장한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_logs import LOG_DIR, _series, parse_log

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "analysis" / "koleo_schedule"

CONDITIONS = [
    ("constant λ (baseline)", "tab:red", [
        "r6_bert_koleo_lrsplit_optuna_best_t27_s42",
        "r6_bert_koleo_lrsplit_optuna_best_t27_s43",
    ]),
    ("koleomin0.25 (λ/4 floor)", "tab:blue", [
        "r6_bert_koleo_lrsplit_optuna_best_t27_koleomin0.25_s42",
        "r6_bert_koleo_lrsplit_optuna_best_t27_koleomin0.25_s43",
    ]),
    ("koleomin0 (full decay to 0)", "tab:green", [
        "r6_bert_koleo_lrsplit_optuna_best_t27_koleomin0_s42",
        "r6_bert_koleo_lrsplit_optuna_best_t27_koleomin0_s43",
    ]),
]


def main():
    data = {name: [parse_log(LOG_DIR / run / "train.log") for run in runs] for name, _, runs in CONDITIONS}

    fig, axes = plt.subplots(6, 1, figsize=(9, 22), sharex=True)

    def plot_field(ax, rows_key, field, ylabel):
        for name, color, _ in CONDITIONS:
            for i, d in enumerate(data[name]):
                xs, ys = _series(d[rows_key], field)
                if xs:
                    ax.plot(xs, ys, "o-" if i == 0 else "s--", color=color, markersize=3, alpha=0.9 if i == 0 else 0.6,
                             label=name if i == 0 else None)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.axvline(300, color="gray", linestyle=":", linewidth=1)
        ax.axvline(900, color="gray", linestyle=":", linewidth=1)

    plot_field(axes[0], "eval", "sts_b_dev", "STS-B dev")
    plot_field(axes[1], "eval", "alignment", "alignment (↓ better)")
    plot_field(axes[2], "eval", "uniformity", "uniformity (more − better)")
    plot_field(axes[3], "eval", "eff_rank", "effective rank")
    plot_field(axes[4], "train", "koleo_lambda", "koleo_lambda (applied)")
    plot_field(axes[5], "train", "L_koleo", "L_koleo")
    axes[-1].set_xlabel("step (dotted lines: hold end=300, decay end=900)")
    fig.suptitle("KoLeo λ decay schedule: constant vs koleomin0.25 vs koleomin0")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "overlay.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    # summary table: peak STS, final STS, final alignment/uniformity/eff_rank per run
    print()
    print(f"{'run':<70} {'peak_sts':>9} {'peak_step':>10} {'final_sts':>10} {'final_align':>12} {'final_unif':>11} {'final_effrank':>14}")
    for name, _, runs in CONDITIONS:
        for run in runs:
            d = parse_log(LOG_DIR / run / "train.log")
            steps, sts = _series(d["eval"], "sts_b_dev")
            if not steps:
                continue
            peak_i = max(range(len(sts)), key=lambda i: sts[i])
            _, align = _series(d["eval"], "alignment")
            _, unif = _series(d["eval"], "uniformity")
            _, effr = _series(d["eval"], "eff_rank")
            print(f"{run:<70} {sts[peak_i]:>9.4f} {steps[peak_i]:>10d} {sts[-1]:>10.4f} {align[-1]:>12.4f} {unif[-1]:>11.4f} {effr[-1]:>14.2f}")


if __name__ == "__main__":
    main()
