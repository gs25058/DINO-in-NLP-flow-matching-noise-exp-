"""Fixed STS-B dev, alignment/uniformity, centered effective rank, and STS7."""

import random
from dataclasses import dataclass

import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from torch.nn import functional as F

EVAL_SPACES = ("embedding", "bottleneck", "head")
STS_SUITE = {
    "STS12": "mteb/sts12-sts",
    "STS13": "mteb/sts13-sts",
    "STS14": "mteb/sts14-sts",
    "STS15": "mteb/sts15-sts",
    "STS16": "mteb/sts16-sts",
    "STSBenchmark": "mteb/stsbenchmark-sts",
    "SICK-R": "mteb/sickr-sts",
}
_STSB_DEV_CACHE = None


@torch.no_grad()
def encode_sentences(
    model,
    tokenizer,
    sentences,
    device,
    batch_size=64,
    max_length=128,
    spaces=("embedding",),
):
    """One backbone pass per batch; return named, normalized representation spaces."""
    unknown = set(spaces) - set(EVAL_SPACES)
    if unknown:
        raise ValueError(f"unknown eval space: {unknown}")
    was_training = model.training
    model.eval()
    chunks = {space: [] for space in spaces}
    try:
        for offset in range(0, len(sentences), batch_size):
            enc = tokenizer(
                sentences[offset : offset + batch_size],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            output = model(
                inputs_embeds=model.get_input_embeddings()(enc["input_ids"]),
                attention_mask=enc["attention_mask"],
            )
            representations = {"embedding": output.embedding}
            if "bottleneck" in spaces:
                representations["bottleneck"] = F.normalize(
                    model.head.mlp(output.pooled), p=2, dim=-1
                )
            if "head" in spaces:
                representations["head"] = F.normalize(output.logits, p=2, dim=-1)
            for space in spaces:
                chunks[space].append(representations[space].cpu())
    finally:
        model.train(was_training)
    return {space: torch.cat(values, dim=0) for space, values in chunks.items()}


def _load_stsb_dev():
    global _STSB_DEV_CACHE
    if _STSB_DEV_CACHE is None:
        ds = load_dataset("sentence-transformers/stsb", split="validation")
        _STSB_DEV_CACHE = (
            list(ds["sentence1"]),
            list(ds["sentence2"]),
            list(ds["score"]),
        )
    return _STSB_DEV_CACHE


def _uniformity(embeds, seed=0, n_pairs=500):
    rng = random.Random(seed)
    n = embeds.shape[0]
    i = [rng.randrange(n) for _ in range(n_pairs)]
    j = [rng.randrange(n) for _ in range(n_pairs)]
    distance = ((embeds[i] - embeds[j]) ** 2).sum(dim=-1)
    return torch.log(torch.exp(-2 * distance).mean().clamp_min(1e-12)).item()


@dataclass
class STSDevMetrics:
    spearman: float
    alignment: float
    uniformity: float


def _stsb_metrics(e1, e2, scores, align_seed=0):
    correlations = F.cosine_similarity(e1, e2, dim=-1).numpy()
    spearman = float(spearmanr(correlations, scores)[0])
    positive = torch.tensor(scores) >= 0.8  # Original STS score >=4/5.
    distances = ((e1 - e2) ** 2).sum(dim=-1)
    alignment = distances[positive].mean().item() if positive.any() else float("nan")
    return STSDevMetrics(
        spearman, alignment, _uniformity(torch.cat([e1, e2], dim=0), seed=align_seed)
    )


def sts_b_dev_metrics_by_space(
    model,
    tokenizer,
    device,
    spaces=EVAL_SPACES,
    batch_size=64,
    max_length=128,
    align_seed=0,
):
    s1, s2, scores = _load_stsb_dev()
    e1 = encode_sentences(model, tokenizer, s1, device, batch_size, max_length, spaces)
    e2 = encode_sentences(model, tokenizer, s2, device, batch_size, max_length, spaces)
    return {space: _stsb_metrics(e1[space], e2[space], scores, align_seed) for space in spaces}


def effective_rank_metrics(model, tokenizer, sentences, device, batch_size=64, max_length=128):
    embeddings = encode_sentences(model, tokenizer, sentences, device, batch_size, max_length)[
        "embedding"
    ]
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.double())
    probabilities = singular_values**2 / (singular_values**2).sum()
    rank = torch.exp(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()).item()
    return rank, probabilities[0].item()


def sts_suite_spearman(model, tokenizer, device, batch_size=64, max_length=128):
    """Evaluate test splits only after the dev-only candidate selection is frozen."""
    result = {}
    for task, dataset in STS_SUITE.items():
        ds = load_dataset(dataset, split="test")
        e1 = encode_sentences(
            model, tokenizer, list(ds["sentence1"]), device, batch_size, max_length
        )["embedding"]
        e2 = encode_sentences(
            model, tokenizer, list(ds["sentence2"]), device, batch_size, max_length
        )["embedding"]
        cosines = F.cosine_similarity(e1, e2, dim=-1).numpy()
        result[task] = float(spearmanr(cosines, list(ds["score"]))[0])
    result["avg_7task"] = sum(result.values()) / len(result)
    return result
