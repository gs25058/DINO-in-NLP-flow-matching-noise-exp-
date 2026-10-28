"""Prepare wiki1m sentences and embedding statistics in a new output directory."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import load_config  # noqa: E402

SEED = 0
STATS_N = 10_000
RANK_EVAL_N = 2_048
MAX_TOKENS = 128
MIN_CHARS = 10


def download_sentences() -> list[str]:
    path = hf_hub_download(
        repo_id="princeton-nlp/datasets-for-simcse",
        filename="wiki1m_for_simcse.txt",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return [line for line in lines if len(line) >= MIN_CHARS]


def compute_embed_stats(sentences, tokenizer, embed_layer, device, batch_size=256):
    sums = None
    sq_sums = None
    count = 0
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=MAX_TOKENS,
                padding=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            special_mask = enc["special_tokens_mask"].bool()
            pad_mask = ~enc["attention_mask"].bool()
            noise_mask = ~(special_mask | pad_mask)  # 노이즈 대상(비특수, 비패딩) 토큰만

            emb = embed_layer(input_ids)  # (B, T, D)
            selected = emb[noise_mask].double()  # (n_tok, D)

            if sums is None:
                sums = selected.sum(dim=0)
                sq_sums = (selected**2).sum(dim=0)
            else:
                sums += selected.sum(dim=0)
                sq_sums += (selected**2).sum(dim=0)
            count += selected.shape[0]

    mean = sums / count
    var = sq_sums / count - mean**2
    std = var.clamp(min=1e-12).sqrt()
    return mean.float(), std.float()


def write_jsonl(path: Path, sentences: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/coviso_relax/control.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    backbone = cfg["model"]["backbone"]
    outputs = {
        key: args.out_dir / Path(cfg["data"][key]).name
        for key in ("sentences_path", "rank_eval_path", "embed_stats_path")
    }
    if any(path.exists() for path in outputs.values()):
        parser.error("existing data preserved; use a new --out-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[prepare_data] downloading wiki1m_for_simcse.txt ...")
    sentences = download_sentences()
    print(f"[prepare_data] {len(sentences)} sentences after >= {MIN_CHARS} char filter")

    rng = random.Random(SEED)
    shuffled = sentences[:]
    rng.shuffle(shuffled)

    stats_sentences = shuffled[:STATS_N]
    rank_eval_sentences = shuffled[STATS_N : STATS_N + RANK_EVAL_N]

    sentences_path = outputs["sentences_path"]
    write_jsonl(sentences_path, sentences)
    print(f"[prepare_data] wrote {sentences_path} ({len(sentences)} sentences)")

    rank_eval_path = outputs["rank_eval_path"]
    write_jsonl(rank_eval_path, rank_eval_sentences)
    print(f"[prepare_data] wrote {rank_eval_path} ({len(rank_eval_sentences)} sentences)")

    print(f"[prepare_data] loading {backbone} for embedding stats ...")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModel.from_pretrained(backbone)
    model.eval().to(args.device)
    embed_layer = model.get_input_embeddings()

    print(f"[prepare_data] computing mu/sigma over {len(stats_sentences)} sentences ...")
    mean, std = compute_embed_stats(stats_sentences, tokenizer, embed_layer, args.device)

    stats_path = outputs["embed_stats_path"]
    torch.save({"mean": mean, "std": std}, stats_path)
    print(f"[prepare_data] wrote {stats_path} (dim={mean.shape[0]})")


if __name__ == "__main__":
    main()
