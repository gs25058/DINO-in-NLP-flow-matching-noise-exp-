"""Evaluate current FlowDINO best/last checkpoints, always masked mean + L2."""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import sts_suite_spearman  # noqa: E402
from src.model import DinoTextModel  # noqa: E402


def load_eval_model(checkpoint, device):
    cfg = checkpoint["model_cfg"]
    model = DinoTextModel(cfg["backbone"], cfg["head"]["bottleneck_dim"], cfg["head"]["logit_dim"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--which", choices=("best", "last"), default="last")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.tf32:
        if not args.device.startswith("cuda") or torch.cuda.get_device_capability() < (
            8,
            0,
        ):
            parser.error("TF32 evaluation requires an Ampere-or-newer CUDA device")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    rows, metadata = {}, {}
    for run in args.runs:
        checkpoint = torch.load(
            ROOT / "checkpoints" / run / f"{args.which}.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = load_eval_model(checkpoint, args.device)
        backbone = checkpoint["model_cfg"]["backbone"]
        tokenizer = AutoTokenizer.from_pretrained(backbone)
        rows[run] = sts_suite_spearman(model, tokenizer, args.device)
        metadata[run] = {
            "backbone": backbone,
            "pooling": "mean+l2",
            "loop_step": checkpoint["loop_step"],
        }
        print(
            f"{run}/{args.which}: " + " ".join(f"{k}={v * 100:.3f}" for k, v in rows[run].items()),
            flush=True,
        )
        del model, checkpoint
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "pooling": "mean+l2",
                "entity": "student",
                "runs": rows,
                "metadata": metadata,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
