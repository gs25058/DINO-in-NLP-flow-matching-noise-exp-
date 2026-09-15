"""R12-B0: 뷰 설정별 "pooled 난이도" 사전 측정 (학습 없음).

목적은 오류 2의 직접 검증이다: 토큰별 독립 가우시안 노이즈는 mean pooling에서 분산이 1/L로
소멸하므로, t를 올려 토큰 수준을 아무리 어렵게 만들어도 pooled 수준의 positive는 거의 공짜로
남는다. 반면 상관 노이즈(rho)나 span cutoff는 구조적 열화라 pooling을 견뎌야 한다.

측정:
  pooled 난이도  = mean cos(student 뷰의 pooled 임베딩, 깨끗한 뷰의 pooled 임베딩)
  토큰 난이도    = mean cos(student 뷰의 토큰 임베딩, 원본 토큰 임베딩)  (비특수 토큰만)
둘 다 1에 가까울수록 쉽다.

실행:
    source scripts/env.sh
    uv run python scripts/measure_view_hardness.py --checkpoint checkpoints/<run>/last.pt
체크포인트를 주지 않으면 사전학습 backbone 그대로 측정한다(구조적 주장 자체는 학습 상태와
무관하지만, 보고에는 r8 체크포인트 값을 쓴다).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.augment import FlowNoiseAug  # noqa: E402
from src.model import DinoTextModel, masked_mean_pool  # noqa: E402
from src.train import load_config, load_sentences  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

#: 스펙이 지정한 비교 설정. iid는 t를 올려도 pooled가 안 변해야 하고(P-V1),
#: rho/cutoff는 pooled가 유의하게 내려가야 한다.
SETTINGS = [
    ("iid t=0.425",      dict(t_lo=0.425, t_hi=0.425)),
    ("iid t=0.63",       dict(t_lo=0.63, t_hi=0.63)),
    ("rho=0.5 @t=0.425", dict(t_lo=0.425, t_hi=0.425, noise_corr_rho=0.5)),
    ("rho=1.0 @t=0.425", dict(t_lo=0.425, t_hi=0.425, noise_corr_rho=1.0)),
    ("cutoff 0.1 @t=0.425", dict(t_lo=0.425, t_hi=0.425, cutoff_span_frac=0.1)),
    ("cutoff 0.2 @t=0.425", dict(t_lo=0.425, t_hi=0.425, cutoff_span_frac=0.2)),
]


@torch.no_grad()
def measure(model, tokenizer, sentences, stats, device, t_lo, t_hi,
            noise_corr_rho=0.0, cutoff_span_frac=0.0, batch_size=64, max_length=128, seed=0):
    aug = FlowNoiseAug(stats["mean"], stats["std"], mode="anchor", num_student_views=1,
                       t_lo=t_lo, t_start=t_hi, t_max=t_hi, warmup_steps=1, delta_t=0.1,
                       noise_corr_rho=noise_corr_rho, cutoff_span_frac=cutoff_span_frac)
    pooled_cos, token_cos = [], []
    torch.manual_seed(seed)
    for i in range(0, len(sentences), batch_size):
        enc = tokenizer(sentences[i:i + batch_size], truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt", return_special_tokens_mask=True).to(device)
        attn = enc["attention_mask"]
        token_embeds = model.get_input_embeddings()(enc["input_ids"])
        special = enc["special_tokens_mask"].bool() | (~attn.bool())
        views = aug(token_embeds, special, step=10_000, attention_mask=attn)
        s_embeds = views.student_embeds[0]
        s_mask = views.student_masks[0] if views.student_masks is not None else attn

        clean, *_ = model(inputs_embeds=token_embeds, attention_mask=attn)
        noisy, *_ = model(inputs_embeds=s_embeds, attention_mask=s_mask)
        pooled_cos.append(F.cosine_similarity(noisy, clean, dim=-1).cpu())

        # 토큰 수준: 비특수·유효 위치만. cutoff drop 위치는 임베딩이 그대로라 제외한다.
        keep = (~special) & attn.bool() & s_mask.bool()
        tc = F.cosine_similarity(s_embeds, token_embeds, dim=-1)
        token_cos.append(tc[keep].cpu())
    return torch.cat(pooled_cos).mean().item(), torch.cat(token_cos).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r8_bert_coviso_sched_optuna_t50.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-sentences", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/analysis/r11_r13/view_hardness.json")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    model = DinoTextModel(cfg["model"]["backbone"], cfg["model"]["head"]["bottleneck_dim"],
                          cfg["model"]["head"]["logit_dim"]).to(args.device).eval()
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        # train.py가 저장하는 형식: {"state_dict", "teacher_state_dict", "model_cfg", ...}
        for key in ("state_dict", "student", "student_state_dict"):
            if key in sd:
                sd = sd[key]
                break
        model.load_state_dict(sd)
        print(f"체크포인트 로드: {args.checkpoint}")
    else:
        print("체크포인트 없음 - 사전학습 backbone 그대로 측정")

    stats = torch.load(ROOT / cfg["data"]["embed_stats_path"], weights_only=True)
    sentences = load_sentences(ROOT / cfg["data"]["sentences_path"])[: args.n_sentences]
    print(f"문장 {len(sentences)}개\n")

    rows = []
    print(f"{'설정':<24} {'pooled 코사인':>14} {'토큰 코사인':>13}")
    print("-" * 54)
    for name, kw in SETTINGS:
        p, t = measure(model, tokenizer, sentences, stats, args.device, **kw)
        rows.append({"setting": name, "pooled_cos": p, "token_cos": t, **kw})
        print(f"{name:<24} {p:>14.4f} {t:>13.4f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {args.out}")

    iid = {r["setting"]: r for r in rows if r["setting"].startswith("iid")}
    if len(iid) == 2:
        a, b = iid["iid t=0.425"], iid["iid t=0.63"]
        print(f"\nP-V1 점검: iid t 0.425 -> 0.63 에서 "
              f"토큰 {a['token_cos']:.4f} -> {b['token_cos']:.4f} (Δ{b['token_cos']-a['token_cos']:+.4f}), "
              f"pooled {a['pooled_cos']:.4f} -> {b['pooled_cos']:.4f} (Δ{b['pooled_cos']-a['pooled_cos']:+.4f})")
        print("  -> pooled 변화가 토큰 변화보다 훨씬 작으면 '소멸' 주장이 확인된 것이다.")


if __name__ == "__main__":
    main()
