"""C3 control: k-means (k=8192) of BERT's input word-embedding rows → data/ cache with hashes."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.provenance import sha256_file  # noqa: E402
from src.vocab_proto import kmeans_word_embeddings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="bert-base-uncased")
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "data/vocab_kmeans_bert_base_uncased_k8192_seed0.pt")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    torch.set_num_threads(args.threads)
    weight = AutoModel.from_pretrained(args.backbone).get_input_embeddings().weight.detach()
    started = time.time()
    result = kmeans_word_embeddings(weight, args.k, args.iterations, args.seed)
    result["source"] = {
        "backbone": args.backbone,
        "matrix": "input word embeddings (= tied MLM decoder)",
        "rows": len(weight),
        "dim": weight.shape[1],
        "weight_sha256": hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
    }
    torch.save(result, args.out)
    history = json.loads(result["history"])
    meta = {
        "path": str(args.out),
        "sha256": sha256_file(args.out),
        "seconds": time.time() - started,
        "iterations_run": len(history),
        "final": history[-1],
        "cluster_sizes": torch.bincount(result["labels"], minlength=args.k).tolist(),
        **result["source"],
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "cluster_sizes"}, indent=1))


if __name__ == "__main__":
    main()
