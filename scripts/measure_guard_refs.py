"""붕괴 감시(collapse_guard) 기준값을 base run 자신의 plateau에서 잰다.

R13에서 base를 챔피언으로 바꾸고도 r8의 기준값을 재사용했다가, 챔피언 자신의 batch_KL(0.0300)이 r8 기준
중단선(0.0310) 아래라 모든 trial이 step 300에서 붕괴로 중단된 사고가 있었다. batch_KL은 t 범위 같은 증강
설정에 따라 절반까지 달라지므로 기준값은 계보를 넘어 재사용할 수 없다 - 반드시 쓰려는 base에서 잰다.

규칙 (train.py의 판정식과 짝):
  h_p_bar_t_ref : plateau 평균              -> 중단선 ref - 0.30
  eff_rank_ref  : plateau 평가값 평균        -> 중단선 ref x 0.7
  batch_kl_ref  : min(plateau 평균, 2 x 감시구간 최소 / 1.47)
                  -> 중단선 ref x 0.5가 감시 구간에서 관측된 최소값보다 1.47배 이상 아래에 오게 한다.
                     batch_KL은 개입이 켜진 run에서 평균보다 한참 낮게 내려가 두 번 오발동한 지표다.
plateau = max_steps의 45%~끝 (1500-step 기준 675~1499와 같은 비율), 감시 구간 = start_frac(기본 0.2)~끝.

실행: uv run python scripts/measure_guard_refs.py --run r14_bert_champ_7700 --max-steps 7700
"""
import argparse
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARGIN = 1.47


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--plateau-frac", type=float, default=0.45)
    ap.add_argument("--start-frac", type=float, default=0.2)
    args = ap.parse_args()

    p0, g0 = int(args.plateau_frac * args.max_steps), int(args.start_frac * args.max_steps)
    hbar, bkl, bkl_guard, rank = [], [], [], []
    last = {}
    for s in args.seeds:
        f = ROOT / "results" / "logs" / f"{args.run}_s{s}" / "train.log"
        for line in f.read_text(errors="ignore").splitlines():
            m = re.match(r"\[step (\d+)\] (loss|EVAL)", line)
            if not m:
                continue
            step, kind = int(m.group(1)), m.group(2)
            last[s] = max(last.get(s, 0), step)
            d = dict(re.findall(r"(\w+)=(-?[\d.]+)", line))
            if kind == "loss":
                if step >= p0:
                    hbar.append(float(d["H_p_bar_t"])); bkl.append(float(d["batch_KL"]))
                if step >= g0:
                    bkl_guard.append(float(d["batch_KL"]))
            elif step >= p0 and "eff_rank" in d:
                rank.append(float(d["eff_rank"]))

    if not (hbar and bkl and rank):
        raise SystemExit(f"plateau(step >= {p0}) 데이터가 부족하다: 마지막 step {last}")
    h_ref, r_ref, b_mean, b_min = statistics.mean(hbar), statistics.mean(rank), statistics.mean(bkl), min(bkl_guard)
    b_ref = min(b_mean, 2 * b_min / MARGIN)
    print(f"# {args.run} seeds={args.seeds} 마지막 step={last} plateau>={p0} 감시>={g0}")
    print(f"#   H_p_bar_t 평균 {h_ref:.4f} (n={len(hbar)}) / eff_rank 평균 {r_ref:.2f} (n={len(rank)}) / "
          f"batch_KL 평균 {b_mean:.4f}, 감시구간 최소 {b_min:.4f}")
    print(f"#   중단선: H_p_bar_t < {h_ref - 0.30:.4f}, eff_rank < {r_ref * 0.7:.2f}, batch_KL < {b_ref * 0.5:.4f} "
          f"(최소 대비 여유 {b_min / (b_ref * 0.5):.2f}배)")
    print("collapse_guard:")
    print("  enabled: true")
    print(f"  h_p_bar_t_ref: {h_ref:.4f}")
    print(f"  eff_rank_ref: {r_ref:.2f}")
    print(f"  batch_kl_ref: {b_ref:.4f}")
    print(f"  start_frac: {args.start_frac}")
    print("  patience: 3")


if __name__ == "__main__":
    main()
