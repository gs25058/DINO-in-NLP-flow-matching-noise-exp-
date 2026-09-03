"""results/logs/**/train.log (신규, run별 폴더) + results/logs/*.log (레거시, flat)을
파싱해 학습 지표 그래프(png)를 results/plots/에 저장한다.

읽는 로그 라인 형식 (src/train.py logger 출력, 필드는 run/config마다 다를 수 있음):
  [step N] key=value key=value ...
  [step N] EVAL key=value key=value ...
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "results" / "logs"
OUT_DIR = ROOT / "results" / "plots"

LINE_RE = re.compile(r"^\[step (\d+)\](?: (EVAL))? (.+)$")
KV_RE = re.compile(r"(\w+)=([-+]?[\d.]+(?:[eE][-+]?\d+)?)")

# 학습 로그 패널 순서(있는 필드만 그린다). eval 필드(sts_b_dev/eff_rank/max_sv_ratio)는 별도 패널.
TRAIN_PANEL_ORDER = [
    "loss", "H_pt", "KL_pt_ps", "batch_KL", "H_p_bar_t", "L_vel",
    "grad_norm", "teacher_temp", "teacher_momentum",
]


def parse_log(path: Path) -> dict:
    train_rows, eval_rows = [], []
    for line in path.read_text(errors="ignore").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        step = int(m.group(1))
        kv = {k: float(v) for k, v in KV_RE.findall(m.group(3))}
        (eval_rows if m.group(2) == "EVAL" else train_rows).append((step, kv))
    return {"train": train_rows, "eval": eval_rows}


def _series(rows, key):
    xs, ys = [], []
    for step, kv in rows:
        if key in kv:
            xs.append(step)
            ys.append(kv[key])
    return xs, ys


def plot_run(name: str, d: dict, out_dir: Path) -> Path | None:
    train_rows, eval_rows = d["train"], d["eval"]
    if not train_rows:
        return None  # step 로그가 아직 없는 런(진행 중이거나 실패)은 건너뜀

    present_keys = [k for k in TRAIN_PANEL_ORDER if any(k in kv for _, kv in train_rows)]
    n_panels = len(present_keys) + (1 if eval_rows else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.2 * n_panels), sharex=True)
    axes = [axes] if n_panels == 1 else list(axes)
    fig.suptitle(name)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    i = 0
    for key in present_keys:
        xs, ys = _series(train_rows, key)
        axes[i].plot(xs, ys, color=colors[i % len(colors)])
        axes[i].set_ylabel(key)
        i += 1

    if eval_rows:
        xs, sts = _series(eval_rows, "sts_b_dev")
        axes[i].plot(xs, sts, "o-", color="tab:cyan", label="sts_b_dev")
        axes[i].set_ylabel("sts_b_dev")
        xs2, eff = _series(eval_rows, "eff_rank")
        if eff:
            ax2 = axes[i].twinx()
            ax2.plot(xs2, eff, "s--", color="tab:brown", label="eff_rank")
            ax2.set_ylabel("eff_rank")
            lines1, labels1 = axes[i].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[i].legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    axes[-1].set_xlabel("step")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_align_uniform(name: str, d: dict, out_dir: Path) -> Path | None:
    """SimCSE 논문 Fig.2 스타일: uniformity(x)-alignment(y) 궤적. 색 그라데이션 + 마지막 화살표로
    학습 진행 방향을 표시한다 (둘 다 낮을수록 좋음, 왼쪽 아래로 향하는 게 이상적)."""
    xs_u, uniformity = _series(d["eval"], "uniformity")
    xs_a, alignment = _series(d["eval"], "alignment")
    if not uniformity or not alignment:
        return None
    n = min(len(uniformity), len(alignment))
    uniformity, alignment = uniformity[:n], alignment[:n]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(uniformity, alignment, "-", color="gray", alpha=0.5, zorder=2)
    sc = ax.scatter(uniformity, alignment, c=range(n), cmap="viridis", s=40, zorder=3)
    if n >= 2:
        ax.annotate(
            "", xy=(uniformity[-1], alignment[-1]), xytext=(uniformity[-2], alignment[-2]),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5), zorder=4,
        )
    ax.set_xlabel("uniformity (lower = better)")
    ax.set_ylabel("alignment (lower = better)")
    ax.set_title(f"{name}\nalign-uniform trajectory (SimCSE Fig.2 style)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("eval index, normalized (0=start, 1=end; arrow marks direction)")
    fig.tight_layout()
    out_path = out_dir / f"align_uniform_{name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_align_uniform_comparison(all_data: dict, out_dir: Path) -> Path | None:
    """SimCSE 논문 Fig.3 스타일: 여러 run의 최종 alignment/uniformity를 한 판에 비교."""
    points = {}
    for name, d in all_data.items():
        _, uniformity = _series(d["eval"], "uniformity")
        _, alignment = _series(d["eval"], "alignment")
        _, sts = _series(d["eval"], "sts_b_dev")
        if uniformity and alignment:
            points[name] = (uniformity[-1], alignment[-1], sts[-1] if sts else None)
    if not points:
        return None

    fig, ax = plt.subplots(figsize=(9, 8))
    for name, (u, a, s) in sorted(points.items()):
        ax.scatter(u, a, s=50)
        label = name if s is None else f"{name} ({s:.2f})"
        ax.annotate(label, (u, a), fontsize=6)
    ax.set_xlabel("uniformity (final, lower = better)")
    ax.set_ylabel("alignment (final, lower = better)")
    ax.set_title("Final alignment vs uniformity across runs (SimCSE Fig.3 style, label = STS-B dev)")
    fig.tight_layout()
    out_path = out_dir / "align_uniform_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_comparison(all_data: dict, out_dir: Path, out_name: str = "comparison_sts_b_eff_rank.png",
                     title: str = "Cross-run comparison: STS-B dev / effective rank") -> Path | None:
    complete = {name: d for name, d in all_data.items() if d["eval"]}
    if not complete:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for name, d in sorted(complete.items()):
        xs, sts = _series(d["eval"], "sts_b_dev")
        axes[0].plot(xs, sts, "o-", label=name, markersize=3)
        xs2, eff = _series(d["eval"], "eff_rank")
        axes[1].plot(xs2, eff, "o-", label=name, markersize=3)
    axes[0].set_ylabel("sts_b_dev (Spearman)")
    axes[1].set_ylabel("effective rank")
    axes[1].set_xlabel("step")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].set_title(title)
    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def discover_logs() -> dict:
    """run_name -> log path. 신규(폴더별 train.log) 우선, 레거시(flat *.log)는 보완."""
    found = {}
    for p in sorted(LOG_DIR.glob("*/train.log")):
        found[p.parent.name] = p
    for p in sorted(LOG_DIR.glob("*.log")):
        found.setdefault(p.stem, p)
    return found


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logs = discover_logs()
    if not logs:
        print(f"[plot_logs] {LOG_DIR}에 로그 파일이 없습니다.")
        return

    all_data = {}
    written = []
    for name, path in logs.items():
        d = parse_log(path)
        all_data[name] = d
        out_path = plot_run(name, d, OUT_DIR)
        if out_path:
            written.append(out_path)
        else:
            print(f"[plot_logs] {name}: step 로그 없음 (진행 중이거나 실패한 런) - 스킵")

        align_path = plot_align_uniform(name, d, OUT_DIR)
        if align_path:
            written.append(align_path)

    cmp_path = plot_comparison(all_data, OUT_DIR)
    if cmp_path:
        written.append(cmp_path)

    align_cmp_path = plot_align_uniform_comparison(all_data, OUT_DIR)
    if align_cmp_path:
        written.append(align_cmp_path)

    bert_data = {name: d for name, d in all_data.items() if "bert" in name}
    bert_cmp_path = plot_comparison(
        bert_data, OUT_DIR, out_name="comparison_bert.png",
        title="BERT backbone runs: STS-B dev / effective rank",
    )
    if bert_cmp_path:
        written.append(bert_cmp_path)

    print(f"[plot_logs] {len(written)}개 그래프 저장 완료 -> {OUT_DIR}")
    for p in written:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
