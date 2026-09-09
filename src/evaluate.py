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


#: 평가 가능한 표현 공간. 셋 다 같은 backbone forward에서 파생되므로 함께 뽑으면
#: 추가 비용은 head MLP 한 번(문장당 768->768->256->8192 곱)뿐이다.
#:   "embedding"  - mean-pooled + L2 정규화. 기존 유일한 평가 대상이자 논문의 Final Embedding.
#:   "bottleneck" - DINO head MLP 출력의 L2 정규화판(256차원). head 내부에서 실제로
#:                  prototype과 내적되는 벡터로, SimCLR류의 "projection head 출력"에 해당.
#:   "head"       - head 최종 출력 logits(8192차원)를 L2 정규화한 것. prototype 유사도 프로필
#:                  자체를 문장 표현으로 본 경우.
EVAL_SPACES = ("embedding", "bottleneck", "head")


def _project_to_space(model, embedding, pooled, space: str) -> torch.Tensor:
    """forward 결과에서 표현 공간별 벡터를 만든다(모두 L2 정규화된 상태로 반환).

    embedding은 forward가 이미 정규화해 두었으므로 재정규화하지 않는다 - 재정규화는 수학적
    항등이지만 부동소수점상 bit-identical이 아니라서, 기존 run과의 수치 재현성이 깨진다."""
    if space == "embedding":
        return embedding
    if space == "bottleneck":
        return F.normalize(model.head.mlp(pooled), p=2, dim=-1)
    if space == "head":
        return F.normalize(model.head(pooled), p=2, dim=-1)
    raise ValueError(f"unknown eval space: {space}")


@torch.no_grad()
def embed_sentences(model, tokenizer, sentences, device, batch_size=64, max_length=128, pooling="last",
                    spaces=None):
    """model: DinoTextModel (forward -> (embedding, logits, hidden, pooled)).

    spaces=None(기본): 기존과 완전히 동일하게 Final Embedding 텐서 하나만 반환한다.
    spaces=("embedding", "head", ...): 같은 backbone forward를 공유해 여러 공간을 한꺼번에
    뽑아 {space: tensor} dict로 반환한다(EVAL_SPACES 참고). 평가 비용이 공간 수에 비례해
    늘지 않게 하려는 것이 이 인자의 존재 이유다.

    pooling: "last"(기본, 기존과 동일) | "first_last" | "cls" (model.py DinoTextModel.forward 참고,
    소급 재채점 scripts/rescore_checkpoints.py 전용 - 학습 경로는 이 인자를 넘기지 않는다)."""
    want = ("embedding",) if spaces is None else tuple(spaces)
    was_training = model.training
    model.eval()
    embeds = {s: [] for s in want}
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        ).to(device)
        token_embeds = model.get_input_embeddings()(enc["input_ids"])
        embedding, _logits, _hidden, pooled = model(
            inputs_embeds=token_embeds, attention_mask=enc["attention_mask"], pooling=pooling
        )
        for s in want:
            embeds[s].append(_project_to_space(model, embedding, pooled, s).cpu())
    if was_training:
        model.train()
    out = {s: torch.cat(v, dim=0) for s, v in embeds.items()}
    return out["embedding"] if spaces is None else out


def postprocess_embeddings(
    embeds: torch.Tensor, fit_embeds: torch.Tensor | None = None, method: str = "none"
) -> torch.Tensor:
    """평가 후처리(소급 재채점 전용, scripts/rescore_checkpoints.py). 학습에는 관여하지 않음.

    method: "none"(기본, 항등) | "center"(fit_embeds 평균 제거 후 재정규화) |
    "center_pc1"/"center_pc2"(center 후 fit_embeds에서 계산한 상위 K개 주성분 성분 추가 제거,
    재정규화) - SIF/whitening-lite 관례대로 평균·주성분은 fit_embeds(평가 시 쓰는 문장 풀 자체)
    에서 계산하고 외부 통계는 쓰지 않는다.
    fit_embeds 생략 시 embeds 자신으로 fit(평가 대상 집합이 곧 통계 집합인 경우)."""
    if method == "none":
        return embeds
    if fit_embeds is None:
        fit_embeds = embeds
    mean = fit_embeds.mean(dim=0, keepdim=True)
    out = embeds - mean
    if method in ("center_pc1", "center_pc2"):
        fit_centered = fit_embeds - mean
        _, _, vh = torch.linalg.svd(fit_centered.double(), full_matrices=False)
        k = 1 if method == "center_pc1" else 2
        components = vh[:k].to(out.dtype)  # [k, D], 이미 정규직교(orthonormal)
        out = out - (out @ components.T) @ components
    elif method != "center":
        raise ValueError(f"unknown postprocess method: {method}")
    return F.normalize(out, p=2, dim=-1)


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


def _stsb_triplet(e1, e2, scores, align_seed: int):
    """이미 인코딩된 STS-B dev 쌍에서 (spearman, alignment, uniformity)를 낸다.
    표현 공간이 달라져도 정의는 동일해야 하므로 계산은 여기 한 군데에만 둔다."""
    cos = F.cosine_similarity(e1, e2, dim=-1).numpy()
    spearman = float(spearmanr(cos, scores)[0])

    scores_t = torch.tensor(scores)
    pos_mask = scores_t >= 0.8  # sentence-transformers/stsb는 [0,5]->[0,1] 정규화 (원 기준 >=4.0)
    d2 = ((e1 - e2) ** 2).sum(dim=-1)
    alignment = d2[pos_mask].mean().item() if pos_mask.any() else float("nan")

    pool = torch.cat([e1, e2], dim=0)  # SimCSE 각주3: "all STS-B sentences"
    uniformity = _uniformity(pool, seed=align_seed)
    return spearman, alignment, uniformity


def sts_b_dev_metrics_by_space(model, tokenizer, device, spaces=("embedding",), batch_size=64,
                               max_length=128, align_seed=0, pooling="last"):
    """STS-B dev를 한 번만 인코딩해 여러 표현 공간의 (spearman, alignment, uniformity)를 낸다.

    반환: {space: (spearman, alignment, uniformity)}. backbone forward를 공유하므로 공간을
    추가해도 평가 시간은 거의 늘지 않는다(추가분은 head MLP뿐)."""
    s1, s2, scores = _load_stsb_dev()
    spaces = tuple(spaces)
    e1 = embed_sentences(model, tokenizer, s1, device, batch_size, max_length, pooling=pooling, spaces=spaces)
    e2 = embed_sentences(model, tokenizer, s2, device, batch_size, max_length, pooling=pooling, spaces=spaces)
    return {s: _stsb_triplet(e1[s], e2[s], scores, align_seed) for s in spaces}


def sts_b_dev_metrics(model, tokenizer, device, batch_size=64, max_length=128, align_seed=0,
                      pooling="last"):
    """학습 중 매 평가마다 호출. STS-B dev(validation) 한 번의 인코딩으로:
    - spearman: 학습 중 빠른 추적용 주 성능 지표
    - alignment/uniformity: SimCSE 논문 Fig.2/각주3과 동일 정의
      (ppos = score>=4.0[정규화 스케일 0.8] STS-B dev 쌍, pdata = STS-B dev 전체 문장 풀)
    반환: (spearman, alignment, uniformity) - Final Embedding 공간 기준.
    여러 공간이 필요하면 sts_b_dev_metrics_by_space를 쓴다.
    """
    return sts_b_dev_metrics_by_space(
        model, tokenizer, device, ("embedding",), batch_size, max_length, align_seed, pooling
    )["embedding"]


def sts_b_dev_spearman(model, tokenizer, device, batch_size=64, max_length=128) -> float:
    """학습 중 빠른 추적용. STS-B dev(validation) split, Spearman. (하위호환용 얇은 래퍼)"""
    spearman, _, _ = sts_b_dev_metrics(model, tokenizer, device, batch_size, max_length)
    return spearman


def effective_rank_metrics(model, tokenizer, sentences, device, batch_size=64, max_length=128,
                           pooling="last"):
    """METHOD.md §5: 고정 평가 문장(pooled Final Embedding)의 열 평균 센터링 -> SVD.
    p_i = sigma_i^2 / sum(sigma_j^2); Effective Rank = exp(-sum(p_i log p_i)); Max SV Ratio = p_1.
    """
    embeds = embed_sentences(model, tokenizer, sentences, device, batch_size, max_length, pooling=pooling)
    x = embeds - embeds.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(x.double())
    p = (sv**2) / (sv**2).sum()
    effective_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()
    max_sv_ratio = p[0].item()
    return effective_rank, max_sv_ratio


def _task_spearman(model, tokenizer, device, hf_path, batch_size=64, max_length=128,
                   pooling="last") -> float:
    ds = load_dataset(hf_path, split="test")
    s1, s2, scores = list(ds["sentence1"]), list(ds["sentence2"]), list(ds["score"])
    e1 = embed_sentences(model, tokenizer, s1, device, batch_size, max_length, pooling=pooling)
    e2 = embed_sentences(model, tokenizer, s2, device, batch_size, max_length, pooling=pooling)
    cos = F.cosine_similarity(e1, e2, dim=-1).numpy()
    corr, _ = spearmanr(cos, scores)
    return float(corr)


def sts_suite_spearman(model, tokenizer, device, batch_size=64, max_length=128, pooling="last"):
    """METHOD.md §5 최종 평가: SimCSE 7-task(STS12-16, STSBenchmark, SICK-R) Spearman + 평균."""
    scores = {
        name: _task_spearman(model, tokenizer, device, path, batch_size, max_length, pooling=pooling)
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
