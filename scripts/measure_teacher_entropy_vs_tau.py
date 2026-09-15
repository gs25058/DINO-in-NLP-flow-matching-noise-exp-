"""R11-A 캘리브레이션: teacher 온도 tau에 대한 H(p_t) 곡선 측정 (학습 없음).

A 매트릭스의 설계값(개루프 final tau, 폐루프 delta nats)이 서로 도달 가능한 범위인지
확인하기 위한 것이다. 200-step 드라이런에서 tau를 0.67% 내렸을 때 H_pt가 0.0020밖에
움직이지 않아, delta=0.3 nats가 tau 하한까지 가야 하는 값일 가능성이 보였다.

주의: DINOLoss의 center 버퍼는 체크포인트에 저장되지 않으므로, 수렴한 center의 근사로
배치 teacher 로짓의 평균을 쓴다. 절대 H 값보다 "tau를 얼마나 내려야 H가 얼마나 내려가는가"
라는 기울기가 목적이므로 이 근사로 충분하다.

실행:
    uv run python scripts/measure_teacher_entropy_vs_tau.py --checkpoint <ckpt>
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import DinoTextModel  # noqa: E402
from src.train import load_config, load_sentences  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r8_bert_coviso_sched_optuna_t50.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-sentences", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    model = DinoTextModel(cfg["model"]["backbone"], cfg["model"]["head"]["bottleneck_dim"],
                          cfg["model"]["head"]["logit_dim"]).to(args.device).eval()
    sd = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(sd["teacher_state_dict"] if "teacher_state_dict" in sd else sd["state_dict"])

    sents = load_sentences(ROOT / cfg["data"]["sentences_path"])[: args.n_sentences]
    logits = []
    for i in range(0, len(sents), 64):
        enc = tok(sents[i:i + 64], truncation=True, max_length=128, padding=True,
                  return_tensors="pt").to(args.device)
        emb = model.get_input_embeddings()(enc["input_ids"])
        _, lg, *_ = model(inputs_embeds=emb, attention_mask=enc["attention_mask"])
        logits.append(lg)
    logits = torch.cat(logits)
    center = logits.mean(dim=0, keepdim=True)      # 수렴한 center의 근사
    K = logits.shape[-1]

    base_tau = cfg["loss"]["teacher_temp"]
    print(f"  문장 {len(sents)}개, prototype {K}개, ln K = {torch.log(torch.tensor(float(K))):.4f}")
    print(f"  기준 teacher_temp = {base_tau}\n")
    print(f"  {'tau':>8} {'H(p_t)':>10} {'기준 대비 Δ':>12}")
    print("  " + "-" * 32)
    rows = []
    h_base = None
    for tau in (0.1348, 0.12, 0.105, 0.095, 0.085, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02):
        p = F.softmax((logits - center) / tau, dim=-1)
        h = -(p * (p.clamp_min(1e-12)).log()).sum(-1).mean().item()
        if h_base is None:
            h_base = h
        rows.append((tau, h, h - h_base))
        print(f"  {tau:>8.4f} {h:>10.4f} {h - h_base:>12.4f}")

    print("\n  A 매트릭스 설계값 도달 가능성:")
    for target in (0.3, 0.6):
        reach = [r for r in rows if -r[2] >= target]
        if reach:
            print(f"    delta={target} nats -> tau ≈ {reach[0][0]:.4f} 필요")
        else:
            print(f"    delta={target} nats -> 측정 범위(tau>={rows[-1][0]}) 안에서 도달 불가 "
                  f"(최대 낙폭 {-rows[-1][2]:.4f})")
    for tau_final in (0.105, 0.085):
        near = min(rows, key=lambda r: abs(r[0] - tau_final))
        print(f"    개루프 final={tau_final} -> ΔH ≈ {near[2]:.4f} nats")


if __name__ == "__main__":
    main()
