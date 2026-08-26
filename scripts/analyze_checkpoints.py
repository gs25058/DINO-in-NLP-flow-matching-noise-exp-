"""Part A: 체크포인트 소급 분석 (학습 없음).

checkpoints/*/last.pt 전체에 대해 alignment/uniformity(Wang & Isola),
anisotropy, 학습 전 backbone 대비 linear CKA, prototype 기하,
STS-B dev cosine-vs-gold 산점도 데이터를 계산해
results/analysis/checkpoint_metrics.csv + 플롯으로 저장한다.

실행: uv run python scripts/analyze_checkpoints.py --device cuda
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from src.evaluate import embed_sentences
from src.model import DinoTextModel

ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = ROOT / "checkpoints"
OUT_DIR = ROOT / "results" / "analysis"
SEED = 0
N_ALIGN_UNIF_PAIRS = 500
N_ANISO_PAIRS = 512
N_PROTO_PAIRS = 2000
REPRESENTATIVE_RUNS = ["r2_anchor_curriculum", "r5d_combined", "r2_bert_lr3e5"]

_tokenizer_cache: dict[str, AutoTokenizer] = {}
_pretrained_ref_cache: dict[str, dict] = {}


def get_tokenizer(backbone: str) -> AutoTokenizer:
    if backbone not in _tokenizer_cache:
        _tokenizer_cache[backbone] = AutoTokenizer.from_pretrained(backbone)
    return _tokenizer_cache[backbone]


def load_base_cfg(is_bert: bool) -> dict:
    path = ROOT / "configs" / ("base_bert.yaml" if is_bert else "base.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint_model(run_name: str, device: str):
    ckpt = torch.load(CKPT_ROOT / run_name / "last.pt", map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and "model_cfg" in ckpt:
        state_dict, model_cfg = ckpt["state_dict"], ckpt["model_cfg"]
    else:
        state_dict = ckpt  # 구버전(raw state_dict) - 전부 ModernBERT run이었음
        model_cfg = load_base_cfg(is_bert=False)["model"]
    model = DinoTextModel(
        model_cfg["backbone"], model_cfg["head"]["bottleneck_dim"], model_cfg["head"]["logit_dim"]
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_cfg["backbone"]


def get_pretrained_reference(backbone: str, device: str) -> dict:
    """학습 전(step 0) 상태: Final Embedding만 필요하므로 head는 무관(랜덤 초기화 그대로)."""
    if backbone in _pretrained_ref_cache:
        return _pretrained_ref_cache[backbone]
    model = DinoTextModel(backbone, 256, 8192).to(device)
    model.eval()
    ref = {"model": model, "backbone": backbone}
    _pretrained_ref_cache[backbone] = ref
    return ref


def alignment_uniformity(e1: torch.Tensor, e2: torch.Tensor, scores: np.ndarray,
                          seed: int = SEED, n_pairs: int = N_ALIGN_UNIF_PAIRS):
    rng = random.Random(seed)
    idx = list(range(len(scores)))
    rng.shuffle(idx)
    idx = idx[:n_pairs]
    e1s, e2s, sc = e1[idx], e2[idx], scores[idx]

    d2 = ((e1s - e2s) ** 2).sum(dim=-1)
    uniformity = torch.log(torch.exp(-2 * d2).mean().clamp_min(1e-12)).item()

    pos_mask = sc >= 0.8  # sentence-transformers/stsb는 score를 [0,5]->[0,1]로 정규화 (원 기준 >=4.0)
    alignment = d2[pos_mask].mean().item() if pos_mask.sum() > 0 else float("nan")
    return alignment, uniformity, idx


def anisotropy(embeds: torch.Tensor, seed: int = SEED, n_pairs: int = N_ANISO_PAIRS) -> float:
    rng = random.Random(seed)
    n = embeds.shape[0]
    idx_i = [rng.randrange(n) for _ in range(n_pairs)]
    idx_j = [rng.randrange(n) for _ in range(n_pairs)]
    cos = F.cosine_similarity(embeds[idx_i], embeds[idx_j], dim=-1)
    return cos.mean().item()


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = (x - x.mean(dim=0, keepdim=True)).double()
    y = (y - y.mean(dim=0, keepdim=True)).double()
    hsic = torch.norm(y.T @ x, p="fro") ** 2
    denom = torch.norm(x.T @ x, p="fro") * torch.norm(y.T @ y, p="fro")
    return (hsic / denom.clamp_min(1e-12)).item()


def prototype_cosine(model: DinoTextModel, seed: int = SEED, n_pairs: int = N_PROTO_PAIRS):
    w = F.normalize(model.head.expand.weight.detach(), dim=-1)  # [logit_dim, bottleneck_dim]
    n = w.shape[0]
    rng = random.Random(seed)
    idx_i = [rng.randrange(n) for _ in range(n_pairs)]
    idx_j = [rng.randrange(n) for _ in range(n_pairs)]
    cos = (w[idx_i] * w[idx_j]).sum(dim=-1)
    return cos.mean().item(), cos.std().item()


def analyze_one(name: str, model, backbone: str, device: str, stsb, rank_eval_sentences,
                 batch_size: int = 64, max_length: int = 128) -> dict:
    tokenizer = get_tokenizer(backbone)
    s1, s2, scores = stsb["s1"], stsb["s2"], stsb["scores"]

    e1 = embed_sentences(model, tokenizer, s1, device, batch_size, max_length)
    e2 = embed_sentences(model, tokenizer, s2, device, batch_size, max_length)
    alignment, uniformity, idx = alignment_uniformity(e1, e2, scores)

    cos_all = F.cosine_similarity(e1, e2, dim=-1).numpy()
    from scipy.stats import spearmanr
    sts_spearman = float(spearmanr(cos_all, scores)[0])

    all_embeds = torch.cat([e1, e2], dim=0)
    aniso = anisotropy(all_embeds)

    rank_embeds = embed_sentences(model, tokenizer, rank_eval_sentences, device, batch_size, max_length)
    ref = get_pretrained_reference(backbone, device)
    ref_embeds = embed_sentences(ref["model"], tokenizer, rank_eval_sentences, device, batch_size, max_length)
    cka = linear_cka(rank_embeds, ref_embeds)

    proto_mean, proto_std = prototype_cosine(model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f"scatter_{name}.npz", cosine=cos_all, score=np.array(scores))

    return dict(
        run=name, backbone=backbone, alignment=alignment, uniformity=uniformity,
        anisotropy=aniso, cka_vs_pretrained=cka, proto_cos_mean=proto_mean,
        proto_cos_std=proto_std, sts_b_dev_spearman=sts_spearman,
    )


def analyze_pretrained_reference(backbone: str, device: str, stsb, batch_size=64, max_length=128) -> dict:
    tokenizer = get_tokenizer(backbone)
    ref = get_pretrained_reference(backbone, device)
    model = ref["model"]
    s1, s2, scores = stsb["s1"], stsb["s2"], stsb["scores"]
    e1 = embed_sentences(model, tokenizer, s1, device, batch_size, max_length)
    e2 = embed_sentences(model, tokenizer, s2, device, batch_size, max_length)
    alignment, uniformity, _ = alignment_uniformity(e1, e2, scores)
    all_embeds = torch.cat([e1, e2], dim=0)
    aniso = anisotropy(all_embeds)
    cos_all = F.cosine_similarity(e1, e2, dim=-1).numpy()
    from scipy.stats import spearmanr
    sts_spearman = float(spearmanr(cos_all, scores)[0])
    return dict(
        run=f"pretrained_{backbone.split('/')[-1]}", backbone=backbone, alignment=alignment,
        uniformity=uniformity, anisotropy=aniso, cka_vs_pretrained=1.0,
        proto_cos_mean=float("nan"), proto_cos_std=float("nan"), sts_b_dev_spearman=sts_spearman,
    )


def plot_scatter(name: str, out_dir: Path):
    d = np.load(OUT_DIR / f"scatter_{name}.npz")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(d["cosine"], d["score"], s=6, alpha=0.3)
    ax.set_xlabel("predicted cosine similarity")
    ax.set_ylabel("gold STS-B score")
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out_dir / f"scatter_{name}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("sentence-transformers/stsb", split="validation")
    stsb = {"s1": list(ds["sentence1"]), "s2": list(ds["sentence2"]), "scores": np.array(ds["score"])}
    rank_eval_sentences = [
        __import__("json").loads(line)["text"]
        for line in open(ROOT / "data" / "rank_eval_sentences.jsonl", encoding="utf-8")
    ]

    run_dirs = sorted(p.name for p in CKPT_ROOT.iterdir() if (p / "last.pt").exists())
    rows = []
    backbones_seen = set()
    for name in run_dirs:
        model, backbone = load_checkpoint_model(name, args.device)
        row = analyze_one(name, model, backbone, args.device, stsb, rank_eval_sentences)
        rows.append(row)
        backbones_seen.add(backbone)
        print(f"[analyze] {name} ({backbone}): align={row['alignment']:.4f} unif={row['uniformity']:.4f} "
              f"aniso={row['anisotropy']:.4f} cka={row['cka_vs_pretrained']:.4f} "
              f"proto_cos={row['proto_cos_mean']:.4f} sts={row['sts_b_dev_spearman']:.4f}")
        del model
        torch.cuda.empty_cache()

    for backbone in sorted(backbones_seen):
        row = analyze_pretrained_reference(backbone, args.device, stsb)
        rows.append(row)
        print(f"[analyze] {row['run']} (reference): align={row['alignment']:.4f} unif={row['uniformity']:.4f} "
              f"aniso={row['anisotropy']:.4f} sts={row['sts_b_dev_spearman']:.4f}")

    csv_path = OUT_DIR / "checkpoint_metrics.csv"
    cols = ["run", "backbone", "alignment", "uniformity", "anisotropy", "cka_vs_pretrained",
            "proto_cos_mean", "proto_cos_std", "sts_b_dev_spearman"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[analyze] wrote {csv_path} ({len(rows)} rows)")

    for name in REPRESENTATIVE_RUNS:
        if (OUT_DIR / f"scatter_{name}.npz").exists():
            plot_scatter(name, OUT_DIR)
            print(f"[analyze] wrote scatter plot for {name}")

    # alignment/uniformity 비교 산점도 (모든 run 한 눈에)
    fig, ax = plt.subplots(figsize=(7, 6))
    for row in rows:
        marker = "*" if row["run"].startswith("pretrained_") else ("s" if "bert" in row["run"] else "o")
        ax.scatter(row["uniformity"], row["alignment"], marker=marker, s=60)
        ax.annotate(row["run"], (row["uniformity"], row["alignment"]), fontsize=6)
    ax.set_xlabel("uniformity (log E[exp(-2||f(x)-f(y)||^2)])")
    ax.set_ylabel("alignment (E[||f(x)-f(y)||^2], score>=4.0)")
    ax.set_title("Alignment vs Uniformity (Wang & Isola 2020)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alignment_uniformity.png", dpi=150)
    plt.close(fig)
    print(f"[analyze] wrote {OUT_DIR / 'alignment_uniformity.png'}")


if __name__ == "__main__":
    main()
