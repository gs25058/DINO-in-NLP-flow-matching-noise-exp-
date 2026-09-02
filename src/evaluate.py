"""평가 유틸.
train.py의 주기 평가(250 step마다 STS-B dev + rank 지표)와, 6단계의 최종 집계
(`uv run python -m src.evaluate`: 7-task mteb 평균 + 길이 probing R² + rank 지표 ->
results/summary.md)가 이 모듈을 쓴다.
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
_STSB_DEV_CACHE = None  # (s1, s2, scores) - 프로세스당 1회만 다운로드/로드

# METHOD.md §5: SimCSE 논문과 동일한 7-task. mteb 태스크명 -> HF dataset(모두 mteb/*-sts,
# sentence1/sentence2/score 스키마, split="test")로 매핑 확인 완료 (mteb.get_task(...).metadata).
STS_SUITE = {
    "STS12": "mteb/sts12-sts",
    "STS13": "mteb/sts13-sts",
    "STS14": "mteb/sts14-sts",
    "STS15": "mteb/sts15-sts",
    "STS16": "mteb/sts16-sts",
    "STSBenchmark": "mteb/stsbenchmark-sts",
    "SICK-R": "mteb/sickr-sts",
}


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


def _load_stsb_dev():
    global _STSB_DEV_CACHE
    if _STSB_DEV_CACHE is None:
        ds = load_dataset("sentence-transformers/stsb", split="validation")
        _STSB_DEV_CACHE = (list(ds["sentence1"]), list(ds["sentence2"]), list(ds["score"]))
    return _STSB_DEV_CACHE


def _uniformity(embeds: torch.Tensor, seed: int = 0, n_pairs: int = 500) -> float:
    """Wang & Isola (2020) uniformity: log E[exp(-2||f(x)-f(y)||^2)], 고정 시드 무작위 쌍."""
    rng = random.Random(seed)
    n = embeds.shape[0]
    idx_i = [rng.randrange(n) for _ in range(n_pairs)]
    idx_j = [rng.randrange(n) for _ in range(n_pairs)]
    d2 = ((embeds[idx_i] - embeds[idx_j]) ** 2).sum(dim=-1)
    return torch.log(torch.exp(-2 * d2).mean().clamp_min(1e-12)).item()


def sts_b_dev_metrics(model, tokenizer, device, batch_size=64, max_length=128, align_seed=0):
    """학습 중 매 평가마다 호출. STS-B dev(validation) 한 번의 인코딩으로:
    - spearman: 학습 중 빠른 추적용 주 성능 지표
    - alignment/uniformity: SimCSE 논문 Fig.2/각주3과 동일 정의
      (ppos = score>=4.0[정규화 스케일 0.8] STS-B dev 쌍, pdata = STS-B dev 전체 문장 풀)
    반환: (spearman, alignment, uniformity)
    """
    s1, s2, scores = _load_stsb_dev()
    e1 = embed_sentences(model, tokenizer, s1, device, batch_size, max_length)
    e2 = embed_sentences(model, tokenizer, s2, device, batch_size, max_length)

    cos = F.cosine_similarity(e1, e2, dim=-1).numpy()
    spearman = float(spearmanr(cos, scores)[0])

    scores_t = torch.tensor(scores)
    pos_mask = scores_t >= 0.8  # sentence-transformers/stsb는 [0,5]->[0,1] 정규화 (원 기준 >=4.0)
    d2 = ((e1 - e2) ** 2).sum(dim=-1)
    alignment = d2[pos_mask].mean().item() if pos_mask.any() else float("nan")

    pool = torch.cat([e1, e2], dim=0)  # SimCSE 각주3: "all STS-B sentences"
    uniformity = _uniformity(pool, seed=align_seed)

    return spearman, alignment, uniformity


def sts_b_dev_spearman(model, tokenizer, device, batch_size=64, max_length=128) -> float:
    """학습 중 빠른 추적용. STS-B dev(validation) split, Spearman. (하위호환용 얇은 래퍼)"""
    spearman, _, _ = sts_b_dev_metrics(model, tokenizer, device, batch_size, max_length)
    return spearman


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


def _task_spearman(model, tokenizer, device, hf_path, batch_size=64, max_length=128) -> float:
    ds = load_dataset(hf_path, split="test")
    s1, s2, scores = list(ds["sentence1"]), list(ds["sentence2"]), list(ds["score"])
    e1 = embed_sentences(model, tokenizer, s1, device, batch_size, max_length)
    e2 = embed_sentences(model, tokenizer, s2, device, batch_size, max_length)
    cos = F.cosine_similarity(e1, e2, dim=-1).numpy()
    corr, _ = spearmanr(cos, scores)
    return float(corr)


def sts_suite_spearman(model, tokenizer, device, batch_size=64, max_length=128):
    """METHOD.md §5 최종 평가: SimCSE 7-task(STS12-16, STSBenchmark, SICK-R) Spearman + 평균."""
    scores = {
        name: _task_spearman(model, tokenizer, device, path, batch_size, max_length)
        for name, path in STS_SUITE.items()
    }
    scores["avg_7task"] = sum(scores.values()) / len(scores)
    return scores


def length_probing_r2(model, tokenizer, device, batch_size=64, max_length=128, seed=0) -> float:
    """METHOD.md §5 길이 probing: STS-B(test) 문장 임베딩 -> 토큰 길이 ridge 회귀, held-out R²."""
    ds = load_dataset(STS_SUITE["STSBenchmark"], split="test")
    sentences = list(ds["sentence1"]) + list(ds["sentence2"])
    lengths = [len(tokenizer.encode(s, truncation=True, max_length=max_length)) for s in sentences]
    embeds = embed_sentences(model, tokenizer, sentences, device, batch_size, max_length).numpy()

    x_train, x_test, y_train, y_test = train_test_split(embeds, lengths, test_size=0.2, random_state=seed)
    reg = Ridge(alpha=1.0)
    reg.fit(x_train, y_train)
    preds = reg.predict(x_test)
    return float(r2_score(y_test, preds))


def write_summary_md(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "run", "avg_7task", "STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICK-R",
        "length_probe_r2", "effective_rank", "max_sv_ratio",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c, "")
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", default="last", help="checkpoints/<run>/<which>.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(ROOT / "results" / "summary.md"))
    args = parser.parse_args()

    from src.model import DinoTextModel  # 지연 임포트 (train.py <-> evaluate.py 순환 방지)

    with open(ROOT / "configs" / "base.yaml", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    tokenizer = AutoTokenizer.from_pretrained(base_cfg["model"]["backbone"])
    rank_eval_sentences = [
        json.loads(line)["text"]
        for line in open(ROOT / base_cfg["data"]["rank_eval_path"], encoding="utf-8")
    ]

    ckpt_root = ROOT / "checkpoints"
    rows = []
    for run_dir in sorted(p for p in ckpt_root.iterdir() if p.is_dir()):
        ckpt_path = run_dir / f"{args.which}.pt"
        if not ckpt_path.exists():
            continue
        model = DinoTextModel(
            base_cfg["model"]["backbone"],
            base_cfg["model"]["head"]["bottleneck_dim"],
            base_cfg["model"]["head"]["logit_dim"],
        ).to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device, weights_only=True))
        model.eval()

        scores = sts_suite_spearman(model, tokenizer, args.device)
        len_r2 = length_probing_r2(model, tokenizer, args.device)
        eff_rank, max_sv = effective_rank_metrics(model, tokenizer, rank_eval_sentences, args.device)

        row = {"run": run_dir.name, **scores, "length_probe_r2": len_r2,
               "effective_rank": eff_rank, "max_sv_ratio": max_sv}
        rows.append(row)
        print(f"[evaluate] {run_dir.name}: avg_7task={scores['avg_7task']:.4f} "
              f"len_r2={len_r2:.4f} eff_rank={eff_rank:.1f} max_sv={max_sv:.4f}")

        del model
        torch.cuda.empty_cache()

    write_summary_md(rows, Path(args.out))
    print(f"[evaluate] wrote {args.out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
