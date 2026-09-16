"""R14 예산 스케일링 분석: step별 곡선을 하네스 SimCSE 재현 곡선과 겹쳐 그리고, 표본 효율 교차점을 찾는다.

산출물 (results/analysis/r14_r17/):
  r14_curves.png        STS-B dev / alignment / uniformity / eff_rank vs step (우리 arm별 시드 평균+범위, SimCSE)
  r14_sample_eff.png    STS-B dev vs 본 문장 수(step x batch) - 배치 크기가 달라(우리 32, SimCSE 64) 문장 기준이 공정
  r14_table.md          step 500/1500/4000/7700/15600 지점 표 + 교차점 + 사전 등록 판정 재료

주의: 두 방법은 각자의 기본 pooling으로 평가된 값이다(우리 mean, SimCSE cls = 원 논문 방식).
실행: uv run python scripts/plot_r14_vs_simcse.py
"""
import csv
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path("/src/gs25058/noise_experiment/noise_experiment/flowdino")
OUT = ROOT / "results" / "analysis" / "r14_r17"
SIMCSE_CSV = MAIN / "results" / "analysis" / "simcse_compare" / "curves_eval.csv"
EVAL = re.compile(r"\[step (\d+)\] EVAL sts_b_dev=([-\d.]+) eff_rank=([-\d.]+) max_sv_ratio=[-\d.]+ "
                  r"alignment=([-\d.]+) uniformity=([-\d.]+)")
METRICS = ["sts", "eff_rank", "alignment", "uniformity"]
ARMS = [  # (run, label, batch, color)
    ("r14_bert_champ_7700", "ours 7700 (mom 0.9985)", 32, "#1f77b4"),
    ("r14_bert_champ_7700_mom999", "ours 7700 (mom 0.999)", 32, "#2ca02c"),
    ("r14_bert_champ_15600", "ours 15600", 32, "#9467bd"),
]
CHAMP = ("r12_bert_noise_optuna_t35", "champion 1500", 32, "#7f7f7f")
CHECKPOINTS = [500, 1500, 4000, 7700, 15600]


def load_run(run: str, roots=(ROOT, MAIN)) -> dict[int, dict[str, float]]:
    """시드별 EVAL을 모아 step -> {metric: (mean, min, max, n)}."""
    per_step: dict[int, dict[str, list[float]]] = {}
    for seed in (42, 43):
        for r in roots:
            f = r / "results" / "logs" / f"{run}_s{seed}" / "train.log"
            if f.exists():
                for m in EVAL.findall(f.read_text(errors="ignore")):
                    step = int(m[0])
                    d = per_step.setdefault(step, {k: [] for k in METRICS})
                    for k, v in zip(METRICS, m[1:]):
                        d[k].append(float(v))
                break
    return {s: {k: (statistics.mean(v), min(v), max(v), len(v)) for k, v in d.items() if v}
            for s, d in sorted(per_step.items())}


def load_simcse() -> dict[int, dict[str, float]]:
    out = {}
    with open(SIMCSE_CSV) as f:
        for row in csv.DictReader(f):
            if row["run"] == "simcse_repro_1epoch_s42":
                out[int(row["step"])] = {"sts": float(row["sts_b_dev"]), "eff_rank": float(row["eff_rank"]),
                                         "alignment": float(row["alignment"]), "uniformity": float(row["uniformity"])}
    return dict(sorted(out.items()))


def nearest(curve: dict, step: int, tol: int = 250):
    if not curve:
        return None
    s = min(curve, key=lambda x: abs(x - step))
    return (s, curve[s]) if abs(s - step) <= tol else None


def crossover(ours: dict, simcse: dict, batch_ours: int, batch_simcse: int = 64):
    """본 문장 수 기준으로 SimCSE STS가 우리 STS를 처음 넘는 지점(선형 보간)."""
    xs_o = [(s * batch_ours, v["sts"][0]) for s, v in ours.items()]
    xs_s = [(s * batch_simcse, v["sts"]) for s, v in simcse.items()]
    if not xs_o or not xs_s:
        return None

    def interp(pts, x):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / max(x1 - x0, 1)
        return None

    lo, hi = max(xs_o[0][0], xs_s[0][0]), min(xs_o[-1][0], xs_s[-1][0])
    prev = None
    for x in range(int(lo), int(hi) + 1, 3200):
        yo, ys = interp(xs_o, x), interp(xs_s, x)
        if yo is None or ys is None:
            continue
        sign = ys - yo
        if prev is not None and (prev[1] <= 0 < sign or prev[1] >= 0 > sign):
            return x, yo, ys, "SimCSE가 앞서기 시작" if sign > 0 else "우리가 앞서기 시작"
        prev = (x, sign)
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    simcse = load_simcse()
    curves = {run: load_run(run) for run, *_ in ARMS + [CHAMP]}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = {"sts": "STS-B dev Spearman", "alignment": "alignment (lower better)",
              "uniformity": "uniformity (lower better)", "eff_rank": "effective rank"}
    for ax, k in zip(axes.flat, ["sts", "alignment", "uniformity", "eff_rank"]):
        for run, label, _, color in ARMS + [CHAMP]:
            c = curves[run]
            if not c:
                continue
            xs = list(c)
            ax.plot(xs, [c[s][k][0] for s in xs], color=color, label=label, lw=1.6)
            ax.fill_between(xs, [c[s][k][1] for s in xs], [c[s][k][2] for s in xs], color=color, alpha=0.15)
        ax.plot(list(simcse), [v[k] for v in simcse.values()], color="#d62728", ls="--", lw=1.6,
                label="SimCSE repro (cls)")
        ax.set_title(titles[k]); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("R14 budget scaling vs harness SimCSE reproduction (band = seed min-max)")
    fig.tight_layout(); fig.savefig(OUT / "r14_curves.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for run, label, batch, color in ARMS:
        c = curves[run]
        if c:
            ax.plot([s * batch / 1e3 for s in c], [v["sts"][0] for v in c.values()], color=color, label=label)
    ax.plot([s * 64 / 1e3 for s in simcse], [v["sts"] for v in simcse.values()], color="#d62728", ls="--",
            label="SimCSE repro")
    ax.set_xlabel("sentences seen (thousands)"); ax.set_ylabel("STS-B dev"); ax.grid(alpha=0.3)
    ax.set_title("Sample efficiency (x = step x batch)"); ax.legend(fontsize=8)
    cross = crossover(curves[ARMS[0][0]], simcse, 32)
    if cross:
        ax.axvline(cross[0] / 1e3, color="k", ls=":", lw=1)
        ax.annotate(f"crossover ~{cross[0] / 1e3:.0f}k", (cross[0] / 1e3, cross[1]), fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "r14_sample_eff.png", dpi=130); plt.close(fig)

    lines = ["# R14 체크포인트 표 (자동 생성: scripts/plot_r14_vs_simcse.py)", "",
             "값은 시드 평균 (최소~최대). SimCSE는 시드 1개, cls pooling. 우리는 mean pooling.", ""]
    for k in ["sts", "alignment", "uniformity", "eff_rank"]:
        lines += [f"## {titles[k]}", "", "| step | " + " | ".join(l for _, l, _, _ in ARMS) + " | SimCSE repro |",
                  "|---|" + "---|" * (len(ARMS) + 1)]
        for st in CHECKPOINTS:
            cells = []
            for run, *_ in ARMS:
                hit = nearest(curves[run], st)
                cells.append(f"{hit[1][k][0]:.4f} ({hit[1][k][1]:.4f}~{hit[1][k][2]:.4f})" if hit else "-")
            sh = nearest(simcse, st)
            cells.append(f"{sh[1][k]:.4f}" if sh else "-")
            lines.append(f"| {st} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## 표본 효율 교차점 (7700 arm vs SimCSE, 본 문장 수 기준)", "",
              f"{cross[3]}: 약 {cross[0]:,}문장 (우리 {cross[1]:.4f}, SimCSE {cross[2]:.4f})" if cross else "교차 없음(또는 데이터 부족)", ""]
    (OUT / "r14_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[plot_r14] 저장: {OUT}/r14_curves.png, r14_sample_eff.png, r14_table.md")


if __name__ == "__main__":
    main()
