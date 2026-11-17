"""STS7 (embedding / bottleneck / head), STS-B dev alignment/uniformity and rank-set geometry per space.

Read-only on checkpoints. Embedding is the 768d mean+L2 vector (the reported model); bottleneck and
head are diagnostics (C0: 256d bottleneck / 8192 prototype logits; vocab heads: 768d transform / word logits).
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

W = Path("/src/gs25058/noise_experiment/noise_experiment/flowdino/.claude/worktrees/vocab-proto")
sys.path.insert(0, str(W))
from scripts.eval_sts7 import load_eval_model  # noqa: E402
from src.evaluate import (EVAL_SPACES, encode_sentences, representation_geometry,  # noqa: E402
                          sts_b_dev_metrics_by_space, sts_suite_by_space)
from src.train import load_sentences  # noqa: E402

E = Path("/src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments")
RUNS = {
    "C2_t35_s42": E / "late_schedule_20261117/checkpoints/vproto_late_schedule_20261117_search_C2_t35_s42",
    "C1_t30_s42": E / "late_schedule_20261117/checkpoints/vproto_late_schedule_20261117_search_C1_t30_s42",
    "C0_t13_s42": E / "coviso_joint_optuna_20261028/checkpoints/r18_bert_joint_optuna_coviso_joint_optuna_20261028_t13_s42",
}
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
rank_sentences = load_sentences(W / "data/rank_eval_sentences.jsonl")
out = {}
for name, path in RUNS.items():
    for which in ("best", "last"):
        ck = torch.load(path / f"{which}.pt", map_location="cpu", weights_only=True)
        model = load_eval_model(ck, device)
        row = {"loop_step": ck["loop_step"]}
        sts7 = sts_suite_by_space(model, tokenizer, device, spaces=EVAL_SPACES)
        dev = sts_b_dev_metrics_by_space(model, tokenizer, device, spaces=EVAL_SPACES)
        z = encode_sentences(model, tokenizer, rank_sentences, device, spaces=EVAL_SPACES)
        for space in EVAL_SPACES:
            geometry = representation_geometry(z[space])
            row[space] = {"dim": z[space].shape[1], "sts7": sts7[space], "dev_spearman": dev[space].spearman,
                          "alignment": dev[space].alignment, "uniformity": dev[space].uniformity, **geometry}
        out[f"{name}_{which}"] = row
        print(name, which, row["loop_step"], {s: round(row[s]["sts7"]["avg_7task"] * 100, 2) for s in EVAL_SPACES}, flush=True)
        del model, ck
        torch.cuda.empty_cache()
Path("/src/gs25058/.claude/jobs/0d3058a9/tmp/full_metrics.json").write_text(json.dumps(out, indent=1))
