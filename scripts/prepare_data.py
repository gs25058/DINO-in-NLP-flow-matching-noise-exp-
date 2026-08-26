"""SimCSE wiki1m 문장 추출 + 토큰 임베딩 통계(mu, sigma) 사전 계산 (METHOD.md 데이터/§2.1 절).

로그인 노드(네트워크 가능)에서 실행:
    source scripts/env.sh && uv run python scripts/prepare_data.py

출력:
    data/sentences.jsonl            - 필터링된 전체 학습 문장 (10자 미만 제거)
    data/rank_eval_sentences.jsonl  - effective rank 평가용 고정 2048 문장 (학습 통계용과 분리)
    data/embed_stats.pt             - {"mean": mu[D], "std": sigma[D]} (비특수/비패딩 토큰 기준)
"""

import argparse
import json
import random
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

BACKBONE = "answerdotai/ModernBERT-base"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
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


def _stats_filename(backbone: str) -> str:
    if backbone == BACKBONE:
        return "embed_stats.pt"  # 기존 기본 백본과의 하위 호환(파일명 유지)
    slug = backbone.split("/")[-1].replace("-", "_").lower()
    return f"embed_stats_{slug}.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default=BACKBONE, help="다른 백본의 임베딩 통계만 새로 계산할 때 사용")
    parser.add_argument("--skip-sentences", action="store_true",
                         help="sentences.jsonl/rank_eval_sentences.jsonl이 이미 있으면 재사용(다른 백본 stats만 추가할 때)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sentences_path = DATA_DIR / "sentences.jsonl"
    rank_eval_path = DATA_DIR / "rank_eval_sentences.jsonl"

    if args.skip_sentences and sentences_path.exists() and rank_eval_path.exists():
        print(f"[prepare_data] --skip-sentences: {sentences_path}, {rank_eval_path} 재사용")
        stats_sentences = [json.loads(line)["text"] for line in open(sentences_path, encoding="utf-8")][:STATS_N]
    else:
        print("[prepare_data] downloading wiki1m_for_simcse.txt ...")
        sentences = download_sentences()
        print(f"[prepare_data] {len(sentences)} sentences after >= {MIN_CHARS} char filter")

        rng = random.Random(SEED)
        shuffled = sentences[:]
        rng.shuffle(shuffled)

        stats_sentences = shuffled[:STATS_N]
        rank_eval_sentences = shuffled[STATS_N : STATS_N + RANK_EVAL_N]

        write_jsonl(sentences_path, sentences)
        print(f"[prepare_data] wrote {sentences_path} ({len(sentences)} sentences)")

        write_jsonl(rank_eval_path, rank_eval_sentences)
        print(f"[prepare_data] wrote {rank_eval_path} ({len(rank_eval_sentences)} sentences)")

    print(f"[prepare_data] loading {args.backbone} for embedding stats ...")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModel.from_pretrained(args.backbone)
    model.eval().to(args.device)
    embed_layer = model.get_input_embeddings()

    print(f"[prepare_data] computing mu/sigma over {len(stats_sentences)} sentences ...")
    mean, std = compute_embed_stats(stats_sentences, tokenizer, embed_layer, args.device)

    stats_path = DATA_DIR / _stats_filename(args.backbone)
    torch.save({"mean": mean, "std": std}, stats_path)
    print(f"[prepare_data] wrote {stats_path} (dim={mean.shape[0]})")


if __name__ == "__main__":
    main()
