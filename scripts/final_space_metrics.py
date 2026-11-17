"""Per-space metrics for the final Phase B runs (last checkpoints, seeds 101/102/103).

For embedding / bottleneck / head: STS7 test, STS-B dev Spearman + alignment + uniformity, and
rank-set geometry (effective rank, max SV ratio, mean norm, mean pair cosine). Read-only.
Usage: final_space_metrics.py OUT.json GROUP [GROUP ...]
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_sts7 import load_eval_model  # noqa: E402
from src.evaluate import (EVAL_SPACES, encode_sentences, representation_geometry,  # noqa: E402
                          sts_b_dev_metrics_by_space, sts_suite_by_space)
from src.train import load_sentences  # noqa: E402

CAMPAIGN = (ROOT / "data").resolve().parent / "results/experiments/late_schedule_20261117"
SEEDS = ("101", "102", "103")


def main():
    out_path, groups = Path(sys.argv[1]), sys.argv[2:]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    final = json.loads((CAMPAIGN / "final_runs.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    rank_sentences = load_sentences(ROOT / "data/rank_eval_sentences.jsonl")
    out = {}
    for group in groups:
        for seed in SEEDS:
            run = final[group][seed]["run_name"]
            ck = torch.load(CAMPAIGN / "checkpoints" / run / "last.pt", map_location="cpu", weights_only=True)
            model = load_eval_model(ck, "cuda")
            sts7 = sts_suite_by_space(model, tokenizer, "cuda", spaces=EVAL_SPACES)
            dev = sts_b_dev_metrics_by_space(model, tokenizer, "cuda", spaces=EVAL_SPACES)
            z = encode_sentences(model, tokenizer, rank_sentences, "cuda", spaces=EVAL_SPACES)
            out[f"{group}_s{seed}"] = {space: {
                "dim": z[space].shape[1], "sts7": sts7[space]["avg_7task"], "sts7_tasks": sts7[space],
                "dev_spearman": dev[space].spearman, "alignment": dev[space].alignment,
                "uniformity": dev[space].uniformity, **representation_geometry(z[space])} for space in EVAL_SPACES}
            out_path.write_text(json.dumps(out, indent=1))
            print(group, seed, {s: round(out[f"{group}_s{seed}"][s]["sts7"] * 100, 2) for s in EVAL_SPACES}, flush=True)
            del model, ck
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
