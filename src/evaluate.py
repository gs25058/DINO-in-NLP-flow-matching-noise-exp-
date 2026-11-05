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
                representations["bottleneck"] = output.bottleneck
            if "head" in spaces:
                if getattr(model.head, "expand", True) is None:
                    raise ValueError("prototype head evaluation is unavailable without expansion")
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


def representation_geometry(embeddings):
    """Centered spectrum plus uncentered direction concentration; diagnostics only."""
    mean = embeddings.mean(dim=0, keepdim=True)
    energy = torch.linalg.svdvals((embeddings - mean).double()).square()
    total = energy.sum().item()
    probabilities = energy / total if total > 0 else torch.zeros_like(energy)
    n = len(embeddings)
    return {
        "effective_rank": torch.exp(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        ).item() if total > 0 else 0.0,
        "max_sv_ratio": probabilities[0].item(),
        "mean_norm": mean.norm().item(),
        "mean_pair_cosine": (
            (embeddings.sum(dim=0).square().sum() - embeddings.square().sum()) / (n * (n - 1))
        ).item() if n > 1 else 0.0,
    }


def geometry_by_space(model, tokenizer, sentences, device, batch_size=64, max_length=128):
    embeddings = encode_sentences(
        model, tokenizer, sentences, device, batch_size, max_length,
        spaces=("embedding", "bottleneck"),
    )
    return {space: representation_geometry(z) for space, z in embeddings.items()}


def sts_suite_by_space(
    model, tokenizer, device, batch_size=64, max_length=128, spaces=("embedding", "bottleneck")
):
    """Evaluate test splits only after the dev-only candidate selection is frozen."""
    result = {space: {} for space in spaces}
    for task, dataset in STS_SUITE.items():
        ds = load_dataset(dataset, split="test")
        e1 = encode_sentences(
            model, tokenizer, list(ds["sentence1"]), device, batch_size, max_length, spaces=spaces
        )
        e2 = encode_sentences(
            model, tokenizer, list(ds["sentence2"]), device, batch_size, max_length, spaces=spaces
        )
        for space in spaces:
            cosines = F.cosine_similarity(e1[space], e2[space], dim=-1).numpy()
            result[space][task] = float(spearmanr(cosines, list(ds["score"]))[0])
    for scores in result.values():
        scores["avg_7task"] = sum(scores.values()) / len(scores)
    return result


def sts_suite_spearman(model, tokenizer, device, batch_size=64, max_length=128):
    return sts_suite_by_space(
        model, tokenizer, device, batch_size, max_length, spaces=("embedding",)
    )["embedding"]
