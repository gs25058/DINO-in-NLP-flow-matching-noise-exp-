"""BERT run만 빠르게 alignment/uniformity 재계산 + 시각화 (전체 37개 분석을 기다리지 않기 위한 임시 스크립트).
scripts/analyze_checkpoints.py의 함수를 그대로 재사용한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from scripts.analyze_checkpoints import (
    CKPT_ROOT,
    analyze_one,
    analyze_pretrained_reference,
    load_checkpoint_model,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "analysis"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("sentence-transformers/stsb", split="validation")
    import numpy as np
    stsb = {"s1": list(ds["sentence1"]), "s2": list(ds["sentence2"]), "scores": np.array(ds["score"])}
    rank_eval_sentences = [
        __import__("json").loads(line)["text"]
        for line in open(Path(__file__).resolve().parent.parent / "data" / "rank_eval_sentences.jsonl", encoding="utf-8")
    ]

    bert_runs = sorted(p.name for p in CKPT_ROOT.iterdir() if (p / "last.pt").exists() and "bert" in p.name)
    print(f"BERT runs found: {bert_runs}")

    rows = []
    for name in bert_runs:
        model, backbone = load_checkpoint_model(name, device)
        row = analyze_one(name, model, backbone, device, stsb, rank_eval_sentences)
        rows.append(row)
        print(f"[analyze_bert] {name}: align={row['alignment']:.4f} unif={row['uniformity']:.4f} sts={row['sts_b_dev_spearman']:.4f}")
        del model
        torch.cuda.empty_cache()

    ref_row = analyze_pretrained_reference("bert-base-uncased", device, stsb)
    rows.append(ref_row)
    print(f"[analyze_bert] {ref_row['run']}: align={ref_row['alignment']:.4f} unif={ref_row['uniformity']:.4f} sts={ref_row['sts_b_dev_spearman']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 7))
    for row in rows:
        marker = "*" if row["run"].startswith("pretrained_") else "o"
        size = 150 if marker == "*" else 70
        ax.scatter(row["uniformity"], row["alignment"], marker=marker, s=size)
        label = f"{row['run']} ({row['sts_b_dev_spearman']:.2f})"
        ax.annotate(label, (row["uniformity"], row["alignment"]), fontsize=8)
    ax.axvline(-2.6, color="gray", linestyle="--", alpha=0.5, label="SimCSE-BERT-base unif ref (-2.6)")
    ax.set_xlabel("uniformity (final, lower = better)")
    ax.set_ylabel("alignment (final, lower = better)")
    ax.set_title("BERT-backbone runs: alignment vs uniformity (corrected)\nlabel = STS-B dev Spearman")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = OUT_DIR / "align_uniform_bert_only.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
