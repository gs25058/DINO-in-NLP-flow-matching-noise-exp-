"""평가 유틸. train.py의 주기 평가(250 step마다 STS-B dev + rank 지표)가 이 모듈을 쓴다.
최종 7-task(mteb) 평가 + 길이 probing + results/summary.md 집계는 6단계에서 이 파일에 추가한다.
"""
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import spearmanr


@torch.no_grad()
def embed_sentences(model, tokenizer, sentences, device, batch_size=64, max_length=128):
    """model: DinoTextModel (forward -> (embedding, logits, hidden)). Final Embedding만 반환."""
    was_training = model.training
    model.eval()
    embeds = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        ).to(device)
        token_embeds = model.get_input_embeddings()(enc["input_ids"])
        embedding, _, _ = model(inputs_embeds=token_embeds, attention_mask=enc["attention_mask"])
        embeds.append(embedding.cpu())
    if was_training:
        model.train()
    return torch.cat(embeds, dim=0)


def sts_b_dev_spearman(model, tokenizer, device, batch_size=64, max_length=128) -> float:
    """학습 중 빠른 추적용. STS-B dev(validation) split, Spearman."""
    ds = load_dataset("sentence-transformers/stsb", split="validation")
    s1, s2, scores = list(ds["sentence1"]), list(ds["sentence2"]), list(ds["score"])

    e1 = embed_sentences(model, tokenizer, s1, device, batch_size, max_length)
    e2 = embed_sentences(model, tokenizer, s2, device, batch_size, max_length)
    cos = F.cosine_similarity(e1, e2, dim=-1).numpy()
    corr, _ = spearmanr(cos, scores)
    return float(corr)


def effective_rank_metrics(model, tokenizer, sentences, device, batch_size=64, max_length=128):
    """METHOD.md §5: 고정 평가 문장(pooled Final Embedding)의 열 평균 센터링 -> SVD.
    p_i = sigma_i^2 / sum(sigma_j^2); Effective Rank = exp(-sum(p_i log p_i)); Max SV Ratio = p_1.
    """
    embeds = embed_sentences(model, tokenizer, sentences, device, batch_size, max_length)
    x = embeds - embeds.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(x.double())
    p = (sv**2) / (sv**2).sum()
    effective_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()
    max_sv_ratio = p[0].item()
    return effective_rank, max_sv_ratio
