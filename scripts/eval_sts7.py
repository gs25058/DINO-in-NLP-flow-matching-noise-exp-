"""체크포인트의 SimCSE 7-task STS Spearman (STS12-16, STSBenchmark, SICK-R; test split).

기본 auto는 FlowDINO/학습 전 backbone에 mean("last"), SimCSE에 CLS를 사용한다.
first_last는 BERT-flow류 트릭이고 후처리 없는
보고 수치와 나란히 놓으면 안 되므로, 쓰려면 --pooling으로 명시해야 한다.

실행:
    uv run python scripts/eval_sts7.py --config configs/<cfg>.yaml --runs <run>_s42 <run>_s43
    uv run python scripts/eval_sts7.py --config ... --runs ... --pooling first_last --out results/analysis/x.json
체크포인트 위치: checkpoints/<run>/last.pt (train.py 저장 형식: {"state_dict", "teacher_state_dict", ...})
run 이름 "pretrained"는 체크포인트 없이 학습 전 backbone을 잰다(평가는 Final embedding만 쓰므로 head 초기화와 무관).
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
PRETRAINED = "pretrained"


def resolve_pooling(requested: str, kind: str) -> str:
    return ("cls" if kind == "simcse" else "last") if requested == "auto" else requested


def load_eval_model(cfg, checkpoint, entity, device):
    """Read the saved architecture, while retaining legacy and pretrained support."""
    model_cfg = checkpoint.get("model_cfg", cfg.get("model", {})) if checkpoint is not None else cfg["model"]
    backbone = model_cfg["backbone"]
    if model_cfg.get("simcse", False):
        if entity != "student":
            raise ValueError("SimCSE has no teacher checkpoint")
        from scripts.train_simcse_baseline import SimCSEModel
        model = SimCSEModel(backbone, model_cfg.get("dropout", 0.1))
        kind = "simcse"
    else:
        head = model_cfg["head"]
        model = DinoTextModel(backbone, head["bottleneck_dim"], head["logit_dim"])
        kind = "pretrained" if checkpoint is None else "flowdino"
    if checkpoint is not None:
        key = "state_dict" if entity == "student" else "teacher_state_dict"
        if key not in checkpoint:
            raise ValueError(f"checkpoint has no {key}")
        missing, unexpected = model.load_state_dict(checkpoint[key], strict=False)
        if missing:
            raise RuntimeError(f"checkpoint missing model parameters: {missing[:5]}")
        # Legacy SimCSE included an unused BERT pooler. All encoder weights must exist.
        if kind == "simcse":
            unexpected = set(unexpected) - {"backbone.pooler.dense.weight", "backbone.pooler.dense.bias"}
            if unexpected:
                raise RuntimeError(f"unexpected SimCSE parameters: {sorted(unexpected)[:5]}")
    return model.to(device).eval(), backbone, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="모델 차원을 읽을 config (run config면 충분)")
    ap.add_argument("--runs", nargs="+", required=True, help="checkpoints/<run>/ 이름들")
    ap.add_argument("--which", default="last", help="checkpoints/<run>/<which>.pt")
    ap.add_argument("--pooling", default="auto", choices=["auto", "last", "first_last", "cls"])
    ap.add_argument("--entity", default="student", choices=["student", "teacher"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tf32", action="store_true", help="match an explicitly TF32-enabled training experiment")
    ap.add_argument("--out", default=None, help="결과 json 경로 (기본: results/analysis/sts7_<첫 run>_<pooling>.json)")
    args = ap.parse_args()
    if args.tf32:
        if not args.device.startswith("cuda") or torch.cuda.get_device_capability() < (8, 0):
            ap.error("TF32 evaluation requires an Ampere-or-newer CUDA device")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg = load_config(ROOT / args.config)
    rows = {}
    metadata = {}
    tokenizers = {}
    for run in args.runs:
        ck = ROOT / "checkpoints" / run / f"{args.which}.pt"
        if run != PRETRAINED and not ck.exists():
            print(f"[eval_sts7] 체크포인트 없음: {ck}")
            continue
        sd = None if run == PRETRAINED else torch.load(ck, map_location="cpu", weights_only=True)
        model, backbone, kind = load_eval_model(cfg, sd, args.entity, args.device)
        pooling = resolve_pooling(args.pooling, kind)
        if backbone not in tokenizers:
            tokenizers[backbone] = AutoTokenizer.from_pretrained(backbone)
        rows[run] = sts_suite_spearman(model, tokenizers[backbone], args.device, pooling=pooling)
        metadata[run] = {"kind": kind, "backbone": backbone, "pooling": pooling,
                         "baseline_version": sd.get("model_cfg", {}).get("baseline_version", "legacy_unversioned") if sd else None}
        del model
        del sd
        torch.cuda.empty_cache()

    if not rows:
        sys.exit(1)
    print(f"\n[eval_sts7] pooling={args.pooling} entity={args.entity}")
    print(f"  {'run':<40} " + " ".join(f"{k[:8]:>8}" for k in TASKS))
    for r, s in rows.items():
        print(f"  {r:<40} " + " ".join(f"{s[k] * 100:>8.2f}" for k in TASKS))
    if len(rows) > 1 and len({(m["kind"], m["pooling"], m["backbone"], m["baseline_version"]) for m in metadata.values()}) == 1:
        print(f"  {'-- 평균 --':<40} " + " ".join(
            f"{statistics.mean(s[k] for s in rows.values()) * 100:>8.2f}" for k in TASKS))
        avgs = [s["avg_7task"] * 100 for s in rows.values()]
        print(f"  avg_7task 시드 편차(최대-최소): {max(avgs) - min(avgs):.2f}")

    out = Path(args.out) if args.out else ROOT / "results" / "analysis" / f"sts7_{args.runs[0]}_{args.pooling}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    poolings = {m["pooling"] for m in metadata.values()}
    actual_pooling = next(iter(poolings)) if len(poolings) == 1 else "mixed"
    out.write_text(json.dumps({"pooling": actual_pooling, "requested_pooling": args.pooling,
                               "entity": args.entity, "runs": rows, "metadata": metadata}, indent=2), encoding="utf-8")
    print(f"  저장: {out}")


if __name__ == "__main__":
    main()
