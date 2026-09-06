"""R7 cov_iso 실험(P-I1~I5) 겹쳐그리기 - BERT (2026-09-03 백본 합의 이후 버전).

r5d_bert(정규화 없음) / r6_bert_koleo_a(쌍별 반발) / r7_coviso_bert_lam{0.5,1,2}(분포
통계량 제약)를 같은 그림에 겹쳐 results/analysis/coviso/overlay.png로 저장한다.

r5d_bert/r6_bert_koleo 로그는 results/logs_archive/ 아래에 보존돼 있어 그쪽을 함께
뒤진다(재실행 금지 - 기존 결과 재사용). r5d_bert_combined는 alignment/uniformity
로깅이 추가되기 전 run이라 해당 패널에서는 자동으로 빠진다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_logs import ROOT, _series, parse_log

LOG_DIRS = [ROOT / "results" / "logs", ROOT / "results" / "logs_archive"]
OUT_DIR = ROOT / "results" / "analysis" / "coviso"

CONDITIONS = [
    ("r5d_bert (no regularizer)", "tab:gray", ["r5d_bert_combined", "r5d_bert_combined_seed43"]),
    ("r6_bert_koleo_a (pairwise)", "tab:red", ["r6_bert_koleo_a_s42", "r6_bert_koleo_a_s43"]),
    ("r7 cov_iso λ=0.5", "tab:blue", ["r7_coviso_bert_lam0.5_s42", "r7_coviso_bert_lam0.5_s43"]),
    ("r7 cov_iso λ=1", "tab:green", ["r7_coviso_bert_lam1_s42", "r7_coviso_bert_lam1_s43"]),
    ("r7 cov_iso λ=2", "tab:purple", ["r7_coviso_bert_lam2_s42", "r7_coviso_bert_lam2_s43"]),
]


def find_log(run: str) -> Path | None:
    for d in LOG_DIRS:
        p = d / run / "train.log"
        if p.exists():
            return p
    return None


def main():
    data = {}
    for name, _, runs in CONDITIONS:
        parsed = []
        for run in runs:
            p = find_log(run)
            if p is None:
                print(f"[warn] missing log: {run}")
                continue
            parsed.append(parse_log(p))
        data[name] = parsed

    panels = [
        ("eval", "sts_b_dev", "STS-B dev (raw)"),
        ("eval", "alignment", "alignment (↓ better)"),
        ("eval", "uniformity", "uniformity (more − better)"),
        ("eval", "eff_rank", "effective rank"),
        ("train", "L_iso_mean", "L_iso_mean (centering 몫)"),
        ("train", "L_iso_cov", "L_iso_cov (스펙트럼 평탄화 몫)"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 4 * len(panels)), sharex=True)
    for ax, (rows_key, field, ylabel) in zip(axes, panels):
        for name, color, _ in CONDITIONS:
            for i, d in enumerate(data[name]):
                xs, ys = _series(d[rows_key], field)
                if xs:
                    ax.plot(xs, ys, "o-" if i == 0 else "s--", color=color, markersize=3,
                             alpha=0.9 if i == 0 else 0.6, label=name if i == 0 else None)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle("R7 cov_iso vs KoLeo vs no-regularizer (BERT, 1500 step)")
    fig.tight_layout(rect=(0, 0, 1, 0.99))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "overlay.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    header = f"{'run':<28} {'peak_sts':>9} {'peak_step':>10} {'final_sts':>10} {'final_align':>12} {'final_unif':>11} {'final_effrank':>14}"
    print()
    print(header)
    for name, _, runs in CONDITIONS:
        for run in runs:
            p = find_log(run)
            if p is None:
                continue
            d = parse_log(p)
            steps, sts = _series(d["eval"], "sts_b_dev")
            if not steps:
                continue
            pi = max(range(len(sts)), key=lambda i: sts[i])
            _, align = _series(d["eval"], "alignment")
            _, unif = _series(d["eval"], "uniformity")
            _, effr = _series(d["eval"], "eff_rank")
            a = f"{align[-1]:>12.4f}" if align else f"{'-':>12}"
            u = f"{unif[-1]:>11.4f}" if unif else f"{'-':>11}"
            e = f"{effr[-1]:>14.2f}" if effr else f"{'-':>14}"
            print(f"{run:<28} {sts[pi]:>9.4f} {steps[pi]:>10d} {sts[-1]:>10.4f}{a}{u}{e}")


if __name__ == "__main__":
    main()
