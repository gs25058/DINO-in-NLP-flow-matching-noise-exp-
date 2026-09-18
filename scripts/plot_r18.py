"""R18 분석: arm별 학습 곡선을 대조군(+있으면 R14/R15)과 겹쳐 그리고 사전 등록 판정 재료를 표로 만든다.

산출물 (results/analysis/r18/):
  r18_curves.png   STS-B dev / alignment / uniformity / eff_rank vs step (arm별 시드 평균 + 범위)
  r18_token.png    토큰 경로 진단: L_ibot / KL_tok / H_p_bar_tok / ibot_lambda vs step
  r18_table.md     step 1000/2000/3000/3999 지점 표 + P-18a/b/c 판정 재료

R14/R15 로그는 이 서버에 없을 수 있다(이전 서버 results/logs 미이관). FLOWDINO_MAIN으로 다른 저장소를
가리키면 거기서도 찾고, 없으면 해당 arm을 조용히 건너뛴다.

실행: uv run python scripts/plot_r18.py
"""
import os
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path(os.environ.get("FLOWDINO_MAIN", ROOT))
OUT = ROOT / "results" / "analysis" / "r18"

EVAL = re.compile(r"\[step (\d+)\] EVAL sts_b_dev=([-\d.]+) eff_rank=([-\d.]+) max_sv_ratio=[-\d.]+ "
                  r"alignment=([-\d.]+) uniformity=([-\d.]+)")
EVAL_METRICS = ["sts", "eff_rank", "alignment", "uniformity"]
TRAIN_KEYS = ["L_ibot", "KL_tok", "H_p_bar_tok", "ibot_lambda"]

ARMS = [  # (run, label, color)
    ("r18_bert_ctrl", "ctrl (iBOT off, 4000)", "#7f7f7f"),
    ("r18a_lam0.3", "a: lam 0.3 shared", "#1f77b4"),
    ("r18a_lam1.0", "a: lam 1.0 shared", "#17becf"),
    ("r18a_ratio0.5", "a: ratio 0.5 shared", "#2ca02c"),
    ("r18b_sep_ratio0.5", "b: ratio 0.5 separate", "#d62728"),
    ("r18a_ratio0.5_mask0.3", "a: ratio 0.5 mask 0.30", "#9467bd"),
    ("r14_bert_champ_7700", "R14 (prev server)", "#8c564b"),
    ("r15_bert_tok_lam1.0", "R15 (prev server)", "#e377c2"),
]
CHECKPOINTS = [1000, 2000, 3000, 3999]
SEEDS = (42, 43)


def _logs(run: str) -> list[Path]:
    out = []
    for seed in SEEDS:
        for r in (ROOT, MAIN):
            f = r / "results" / "logs" / f"{run}_s{seed}" / "train.log"
            if f.exists():
                out.append(f)
                break
    return out


def load_eval(run: str) -> dict[int, dict[str, tuple[float, float, float, int]]]:
    """시드별 EVAL -> step -> {metric: (mean, min, max, n)}."""
    per_step: dict[int, dict[str, list[float]]] = {}
    for f in _logs(run):
        for m in EVAL.findall(f.read_text(errors="ignore")):
            d = per_step.setdefault(int(m[0]), {k: [] for k in EVAL_METRICS})
            for k, v in zip(EVAL_METRICS, m[1:]):
                d[k].append(float(v))
    return {s: {k: (statistics.mean(v), min(v), max(v), len(v)) for k, v in d.items() if v}
            for s, d in sorted(per_step.items())}


def load_train(run: str) -> dict[str, list[tuple[int, float]]]:
    """토큰 경로 로깅(시드 42 한 줄만 - 진단용이라 평균낼 필요가 없다)."""
    files = _logs(run)
    series: dict[str, list[tuple[int, float]]] = {k: [] for k in TRAIN_KEYS}
    if not files:
        return series
    for line in files[0].read_text(errors="ignore").splitlines():
        m = re.match(r"\[step (\d+)\] loss", line)
        if not m:
            continue
        step, d = int(m.group(1)), dict(re.findall(r"(\w+)=(-?[\d.]+)", line))
        for k in TRAIN_KEYS:
            if k in d:
                series[k].append((step, float(d[k])))
    return series


def _plot(ax, steps, mean, lo, hi, label, color):
    ax.plot(steps, mean, label=label, color=color, lw=1.6)
    ax.fill_between(steps, lo, hi, color=color, alpha=0.15, lw=0)


def curves(data: dict[str, dict]) -> None:
    titles = {"sts": "STS-B dev Spearman", "alignment": "alignment (lower is better)",
              "uniformity": "uniformity (lower is better)", "eff_rank": "effective rank"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric in zip(axes.flat, ["sts", "alignment", "uniformity", "eff_rank"]):
        for run, label, color in ARMS:
            d = data.get(run)
            if not d:
                continue
            steps = [s for s in d if metric in d[s]]
            _plot(ax, steps, [d[s][metric][0] for s in steps], [d[s][metric][1] for s in steps],
                  [d[s][metric][2] for s in steps], label, color)
        ax.set_title(titles[metric])
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    if data.get("r18a_lam0.3"):
        axes.flat[3].axhline(300, ls=":", c="k", lw=1)          # P-18a 기준선(step 2000에 >= 300)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("R18 iBOT token CE - per arm (seed mean, band = seed range)")
    fig.tight_layout()
    fig.savefig(OUT / "r18_curves.png", dpi=150)
    plt.close(fig)


def token_diagnostics() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, key in zip(axes.flat, TRAIN_KEYS):
        for run, label, color in ARMS:
            pts = load_train(run).get(key) or []
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=label, color=color, lw=1.2)
        ax.set_title(key)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("R18 token-path diagnostics (seed 42)")
    fig.tight_layout()
    fig.savefig(OUT / "r18_token.png", dpi=150)
    plt.close(fig)


def table(data: dict[str, dict]) -> None:
    lines = ["# R18 지점 표 (시드 평균, 괄호 = 시드 수)", ""]
    for metric in ["sts", "alignment", "uniformity", "eff_rank"]:
        lines += [f"## {metric}", "",
                  "| run | " + " | ".join(f"step {c}" for c in CHECKPOINTS) + " |",
                  "|---|" + "---|" * len(CHECKPOINTS)]
        for run, label, _ in ARMS:
            d = data.get(run)
            if not d:
                continue
            cells = []
            for c in CHECKPOINTS:
                near = min((s for s in d if metric in d[s]), key=lambda s: abs(s - c), default=None)
                cells.append("–" if near is None or abs(near - c) > 250
                             else f"{d[near][metric][0]:.4f} ({d[near][metric][3]})")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## 사전 등록 판정 재료", ""]
    for run, label, _ in ARMS:
        d = data.get(run)
        if not d or run.startswith(("r14", "r15", "r18_bert_ctrl")):
            continue
        near2000 = min((s for s in d if "eff_rank" in d[s] and abs(s - 2000) <= 250), key=lambda s: abs(s - 2000), default=None)
        rank2000 = d[near2000]["eff_rank"][0] if near2000 is not None else None
        best = max((d[s]["sts"][0] for s in d if "sts" in d[s]), default=None)
        final = d[max(d)]["sts"][0] if d else None
        p18a = "성립" if rank2000 and rank2000 >= 300 else "기각" if rank2000 else "판정불가"
        lines.append(f"- **{label}**: P-18a(step 2000 eff_rank >= 300) {p18a} "
                     f"(eff_rank {rank2000:.1f})" if rank2000 else f"- **{label}**: P-18a 판정불가")
        if best is not None:
            lines.append(f"  - STS 최고 {best:.4f} (챔피언 0.7333), 마감 {final:.4f} (R14 0.7087)")
    (OUT / "r18_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {run: load_eval(run) for run, _, _ in ARMS}
    found = [run for run, d in data.items() if d]
    print(f"[plot_r18] 로그를 찾은 run: {', '.join(found) if found else '없음'}")
    curves(data)
    token_diagnostics()
    table(data)
    print(f"[plot_r18] 저장: {OUT}/r18_curves.png, r18_token.png, r18_table.md")


if __name__ == "__main__":
    main()
