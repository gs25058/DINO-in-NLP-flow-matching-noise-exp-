"""Post-hoc STS7 for retained Optuna checkpoints, adapted from b64e9c0.

The objective remains STS-B dev. Reports rank only the evaluated subset;
missing/collapsed checkpoints are skipped. No training or checkpoint pruning.
GPU use must be wrapped in scripts/gpu_guard.py (default evaluation is CPU).
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import optuna
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_sts7 import load_eval_model, resolve_pooling  # noqa: E402
from src.evaluate import sts_suite_spearman  # noqa: E402
from src.train import load_config  # noqa: E402

EVALUATOR_FILES = [ROOT / p for p in ("scripts/eval_sts7.py", "scripts/eval_study_sts7.py",
                   "scripts/train_simcse_baseline.py", "src/evaluate.py", "src/model.py")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--storage", help="Optuna storage URL; default results/analysis/<study>.db")
    ap.add_argument("--top", type=int, default=0, help="STS-B dev top N (0=all completed)")
    ap.add_argument("--pooling", choices=["auto", "last", "cls", "first_last"], default="auto")
    ap.add_argument("--entity", choices=["student", "teacher"], default="student")
    ap.add_argument("--which", default="last")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    if args.top < 0:
        ap.error("top must be nonnegative")
    cfg = load_config(ROOT / args.config)
    context = {"study": args.study, "config": cfg, "pooling": args.pooling,
               "entity": args.entity, "which": args.which, "device": args.device,
               "evaluator_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in EVALUATOR_FILES)).hexdigest()}
    out = Path(args.out_dir) if args.out_dir else ROOT / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / f"{args.study}_sts7.json"
    previous = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache = previous.get("trials", {}) if previous.get("context") == context else {}
    study = optuna.load_study(study_name=args.study,
                             storage=args.storage or f"sqlite:///{ROOT}/results/analysis/{args.study}.db")
    done = sorted((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                   and t.value is not None and math.isfinite(t.value) and not t.user_attrs.get("collapsed")),
                  key=lambda t: -t.value)
    if args.top:
        done = done[:args.top]
    selected = {}
    for trial in done:
        key = str(trial.number)
        checkpoint = ROOT / "checkpoints" / f"{args.study}_t{trial.number}" / f"{args.which}.pt"
        if not checkpoint.exists():
            print(f"[study_sts7] skip t{trial.number}: checkpoint not retained")
            continue
        stat = checkpoint.stat()
        stamp = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sts_b_dev": trial.value}
        if key not in cache or cache[key].get("checkpoint") != stamp:
            sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if sd.get("train_status") == "collapsed":
                continue
            model, backbone, kind = load_eval_model(cfg, sd, args.entity, args.device)
            pooling = resolve_pooling(args.pooling, kind)
            tok = AutoTokenizer.from_pretrained(backbone)
            scores = sts_suite_spearman(model, tok, args.device, pooling=pooling)
            cache[key] = {"checkpoint": stamp, "sts_b_dev": trial.value, "params": trial.params,
                          "pooling": pooling, **{k: float(v) for k, v in scores.items()}}
            # Atomic incremental cache; all evaluative settings are part of its context.
            temp = cache_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps({"context": context, "trials": cache}, indent=2, ensure_ascii=False))
            temp.replace(cache_path)
            del model, sd
            torch.cuda.empty_cache()
        selected[key] = cache[key]
    if not selected:
        print("[study_sts7] no checkpoints scored; tuning requires --save-checkpoints / SAVE_CKPT=1")
        return 1
    by_sts = {k: i + 1 for i, k in enumerate(sorted(selected, key=lambda k: -selected[k]["sts_b_dev"]))}
    by_seven = {k: i + 1 for i, k in enumerate(sorted(selected, key=lambda k: -selected[k]["avg_7task"]))}
    lines = [f"# {args.study}: STS-B dev vs STS7", "",
             "순위는 평가된 체크포인트 집합 안에서 계산한다. STS7은 사후 보고 지표이며 튜닝 목적함수가 아니다.", "",
             "| trial | STS-B dev | STS7 (%) | STS-B rank | STS7 rank | pooling |",
             "|---|---|---|---|---|---|"]
    for key in sorted(selected, key=lambda k: by_seven[k]):
        row = selected[key]
        lines.append(f"| {key} | {row['sts_b_dev']:.4f} | {100 * row['avg_7task']:.2f} | "
                     f"{by_sts[key]} | {by_seven[key]} | {row['pooling']} |")
    (out / f"{args.study}_sts7.md").write_text("\n".join(lines) + "\n")
    print(f"[study_sts7] {len(selected)} checkpoints: {cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
