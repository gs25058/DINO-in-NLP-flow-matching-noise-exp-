"""R18 부수 관측: 토큰 수준 프로토타입이 의미 클러스터를 이루는지 정성 표로 본다.

teacher(깨끗한 입력, 학습 목표를 만드는 쪽)의 토큰 로짓을 학습 때와 같은 방식으로 centering/sharpening한 뒤
argmax 프로토타입에 토큰을 모아, 많이 쓰인 프로토타입별로 실제 토큰을 나열한다.

실행(학습이 끝난 뒤 - GPU를 학습과 나눠 쓰지 않는다):
    uv run python scripts/r18_token_prototypes.py --run r18a_lam0.3_s42 --config configs/r18a_lam0.3.yaml
산출물: results/analysis/r18/token_prototypes_<run>.md
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import DINOHead, DinoTextModel  # noqa: E402
from src.train import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="checkpoints/<run>/last.pt")
    ap.add_argument("--config", required=True, help="그 run의 config (teacher_temp, 차원, 데이터 경로)")
    ap.add_argument("--n-sentences", type=int, default=2000)
    ap.add_argument("--top-prototypes", type=int, default=20)
    ap.add_argument("--tokens-per", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    ck = torch.load(ROOT / "checkpoints" / args.run / "last.pt", map_location=args.device, weights_only=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])

    model = DinoTextModel(cfg["model"]["backbone"], cfg["model"]["head"]["bottleneck_dim"],
                          cfg["model"]["head"]["logit_dim"]).to(args.device).eval()
    missing, _ = model.load_state_dict(ck["teacher_state_dict"], strict=False)
    if missing:
        raise RuntimeError(f"teacher 파라미터 누락 {missing[:5]}")
    # separate head면 토큰 로짓은 별도 head가 만든다(문장 head가 아니다).
    head = model.head
    if "ibot_teacher_head_state_dict" in ck:
        head = DINOHead(*model.head.dims).to(args.device).eval()
        head.load_state_dict(ck["ibot_teacher_head_state_dict"])
    center = ck.get("ibot_token_center")
    if center is None:
        raise SystemExit(f"{args.run}: ibot_token_center가 없다 - iBOT run이 아니다")
    center = center.to(args.device)
    tau = cfg["loss"]["teacher_temp"]        # 학습 마지막 구간의 teacher 온도(스케줄 plateau 값)

    sentences = [json.loads(l)["text"] for l in open(ROOT / cfg["data"]["rank_eval_path"], encoding="utf-8")]
    random.Random(args.seed).shuffle(sentences)
    sentences = sentences[:args.n_sentences]

    by_proto: dict[int, Counter] = defaultdict(Counter)
    counts = Counter()
    with torch.no_grad():
        for i in range(0, len(sentences), args.batch_size):
            enc = tok(sentences[i:i + args.batch_size], truncation=True, max_length=cfg["data"]["max_tokens"],
                      padding=True, return_special_tokens_mask=True, return_tensors="pt")
            ids = enc["input_ids"].to(args.device)
            attn = enc["attention_mask"].to(args.device)
            keep = (~enc["special_tokens_mask"].bool().to(args.device)) & attn.bool()
            hidden = model(inputs_embeds=model.get_input_embeddings()(ids), attention_mask=attn)[2]
            logits = head(hidden[keep])
            proto = F.softmax((logits - center) / tau, dim=-1).argmax(dim=-1)
            for p, t in zip(proto.tolist(), ids[keep].tolist()):
                by_proto[p][tok.convert_ids_to_tokens(t)] += 1
                counts[p] += 1

    total = sum(counts.values())
    out = Path(args.out) if args.out else ROOT / "results" / "analysis" / "r18" / f"token_prototypes_{args.run}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# 토큰 프로토타입 정성 표 - {args.run}", "",
             f"teacher 토큰 분포(centering + teacher_temp {tau:.4f})의 argmax 기준. "
             f"문장 {len(sentences)}개, 비특수 토큰 {total:,}개.",
             f"사용된 프로토타입 {len(counts):,} / {cfg['model']['head']['logit_dim']:,} "
             f"({len(counts) / cfg['model']['head']['logit_dim']:.1%}), "
             f"최다 프로토타입 점유율 {counts.most_common(1)[0][1] / total:.2%}.", "",
             "| 프로토타입 | 토큰 수 | 상위 토큰 |", "|---|---|---|"]
    for p, n in counts.most_common(args.top_prototypes):
        toks = " ".join(f"{t}({c})" for t, c in by_proto[p].most_common(args.tokens_per))
        lines.append(f"| {p} | {n} | {toks} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[token_prototypes] 저장: {out}")
    print(f"  프로토타입 {len(counts)}개 사용, 최다 점유율 {counts.most_common(1)[0][1] / total:.2%}")


if __name__ == "__main__":
    main()
