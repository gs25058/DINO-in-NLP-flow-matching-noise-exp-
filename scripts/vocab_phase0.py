"""Phase 0 (no training, CPU): the C1/C3 heads at step 0 against the zero-training probe.

Probe reference (2026-11-16, STS-B dev, mean pooling → MLM head, centered by the dev mean, temp 1):
raw head Spearman 0.5938, mean-pooling 768d 0.5931, H 6.081 nats (ln V 10.326), KL 7.582.
Writes JSON; exits non-zero if any C1 check fails, so the campaign stops before Phase 1.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import _load_stsb_dev, encode_sentences, sts_b_dev_metrics_by_space  # noqa: E402
from src.model import DinoTextModel  # noqa: E402

PROBE = {"head_raw": 0.5938, "embedding_raw": 0.5931, "H_temp1": 6.081, "KL_temp1": 7.582}
TOLERANCE = {"head_raw": 0.001, "embedding_raw": 0.001, "H_temp1": 0.03, "KL_temp1": 0.05}  # abs, abs, rel, rel


@torch.no_grad()
def distribution_stats(logits, temperature=1.0):
    centered = (logits - logits.mean(0, keepdim=True)) / temperature
    log_p = F.log_softmax(centered.double(), dim=-1)
    p = log_p.exp()
    entropy = -(p * log_p).sum(-1).mean().item()
    total, count = 0.0, 0
    for start in range(0, len(p), 256):  # all ordered pairs i≠j, chunked
        block = p[start:start + 256]
        cross = block @ log_p.t()
        own = (block * log_p[start:start + 256]).sum(-1, keepdim=True)
        kl = own - cross
        idx = torch.arange(start, start + len(block))
        kl[torch.arange(len(block)), idx] = 0
        total += kl.sum().item()
        count += len(block) * (len(p) - 1)
    return {"H": entropy, "H_over_lnK": entropy / math.log(logits.shape[1]), "KL_pairs": total / count,
            "mean_top1": p.max(-1).values.mean().item()}


def condition(name, model, tokenizer, device):
    metrics = sts_b_dev_metrics_by_space(model, tokenizer, device)
    s1, s2, _ = _load_stsb_dev()
    with torch.no_grad():
        sentences = s1 + s2
        logits = []
        for offset in range(0, len(sentences), 64):
            enc = tokenizer(sentences[offset:offset + 64], truncation=True, max_length=128,
                            padding=True, return_tensors="pt").to(device)
            out = model(inputs_embeds=model.get_input_embeddings()(enc["input_ids"]),
                        attention_mask=enc["attention_mask"])
            logits.append(out.logits.cpu())
        logits = torch.cat(logits)
    row = {"condition": name, "K": logits.shape[1],
           "embedding_raw": metrics["embedding"].spearman, "head_raw": metrics["head"].spearman,
           "bottleneck_raw": metrics["bottleneck"].spearman,
           **{f"{k}_temp{t}": v for t in (1.0, 0.5) for k, v in distribution_stats(logits, t).items()}}
    row["decoder_shares_input_storage"] = (
        model.head.decoder.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kmeans", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    rows = {}
    torch.manual_seed(42)
    rows["C1"] = condition("C1", DinoTextModel("bert-base-uncased", 768, 30522, head_type="vocab"), tokenizer, "cpu")
    rows["C3"] = condition("C3", DinoTextModel("bert-base-uncased", 768, 8192, head_type="vocab_kmeans",
                                               kmeans_path=str(args.kmeans)), tokenizer, "cpu")
    c1 = rows["C1"]
    measured = {"head_raw": c1["head_raw"], "embedding_raw": c1["embedding_raw"],
                "H_temp1": c1["H_temp1.0"], "KL_temp1": c1["KL_pairs_temp1.0"]}
    checks = {}
    for key, reference in PROBE.items():
        gap = abs(measured[key] - reference)
        limit = TOLERANCE[key] * (reference if key.startswith(("H_", "KL_")) else 1)
        checks[key] = {"probe": reference, "measured": measured[key], "abs_gap": gap, "limit": limit, "ok": gap <= limit}
    checks["untied_decoder"] = {"ok": not c1["decoder_shares_input_storage"]}
    result = {"probe_reference": PROBE, "checks": checks, "conditions": rows,
              "passed": all(c["ok"] for c in checks.values()),
              "note": "C2 equals C1 at step 0 (same weights, frozen). Probe KL used 2000 random pairs; "
                      "here every ordered pair is averaged. Step-0 training values come from the smokes."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(checks, indent=1))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
