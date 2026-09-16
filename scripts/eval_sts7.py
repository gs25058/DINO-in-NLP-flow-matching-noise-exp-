"""체크포인트의 SimCSE 7-task STS Spearman (STS12-16, STSBenchmark, SICK-R; test split).

학습과 같은 mean pooling("last")이 기본이다. first_last는 BERT-flow류 트릭이고 후처리 없는
보고 수치와 나란히 놓으면 안 되므로, 쓰려면 --pooling으로 명시해야 한다.

실행:
    uv run python scripts/eval_sts7.py --config configs/<cfg>.yaml --runs <run>_s42 <run>_s43
    uv run python scripts/eval_sts7.py --config ... --runs ... --pooling first_last --out results/analysis/x.json
체크포인트 위치: checkpoints/<run>/last.pt (train.py 저장 형식: {"state_dict", "teacher_state_dict", ...})
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluate import sts_suite_spearman  # noqa: E402
from src.model import DinoTextModel  # noqa: E402
from src.train import load_config  # noqa: E402

TASKS = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICK-R", "avg_7task"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="모델 차원을 읽을 config (run config면 충분)")
    ap.add_argument("--runs", nargs="+", required=True, help="checkpoints/<run>/ 이름들")
    ap.add_argument("--which", default="last", help="checkpoints/<run>/<which>.pt")
    ap.add_argument("--pooling", default="last", choices=["last", "first_last", "cls"])
    ap.add_argument("--entity", default="student", choices=["student", "teacher"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="결과 json 경로 (기본: results/analysis/sts7_<첫 run>_<pooling>.json)")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    key = "state_dict" if args.entity == "student" else "teacher_state_dict"
    rows = {}
    for run in args.runs:
        ck = ROOT / "checkpoints" / run / f"{args.which}.pt"
        if not ck.exists():
            print(f"[eval_sts7] 체크포인트 없음: {ck}")
            continue
        model = DinoTextModel(cfg["model"]["backbone"], cfg["model"]["head"]["bottleneck_dim"],
                              cfg["model"]["head"]["logit_dim"]).to(args.device).eval()
        sd = torch.load(ck, map_location=args.device, weights_only=True)
        # 체크포인트에 predictor 등 모델 밖 파라미터가 섞일 수 있어 strict=False로 싣되, 모델 쪽 누락은 막는다
        missing, _unexpected = model.load_state_dict(sd[key], strict=False)
        if missing:
            raise RuntimeError(f"{run}: 모델 파라미터 누락 {missing[:5]}")
        rows[run] = sts_suite_spearman(model, tok, args.device, pooling=args.pooling)
        del model
        torch.cuda.empty_cache()

    if not rows:
        sys.exit(1)
    print(f"\n[eval_sts7] pooling={args.pooling} entity={args.entity}")
    print(f"  {'run':<40} " + " ".join(f"{k[:8]:>8}" for k in TASKS))
    for r, s in rows.items():
        print(f"  {r:<40} " + " ".join(f"{s[k] * 100:>8.2f}" for k in TASKS))
    if len(rows) > 1:
        print(f"  {'-- 평균 --':<40} " + " ".join(
            f"{statistics.mean(s[k] for s in rows.values()) * 100:>8.2f}" for k in TASKS))
        avgs = [s["avg_7task"] * 100 for s in rows.values()]
        print(f"  avg_7task 시드 편차(최대-최소): {max(avgs) - min(avgs):.2f}")

    out = Path(args.out) if args.out else ROOT / "results" / "analysis" / f"sts7_{args.runs[0]}_{args.pooling}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pooling": args.pooling, "entity": args.entity, "runs": rows}, indent=2), encoding="utf-8")
    print(f"  저장: {out}")


if __name__ == "__main__":
    main()
