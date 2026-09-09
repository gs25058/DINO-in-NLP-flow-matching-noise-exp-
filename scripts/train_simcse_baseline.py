"""SimCSE(비지도) 기준선을 우리와 동일한 예산·데이터·평가로 학습해 궤적을 비교한다.

왜 별도 스크립트인가: SimCSE는 손실(InfoNCE)도 증강(dropout 두 번 통과)도 teacher도 우리
DINO 루프와 전혀 다르다. CLAUDE.md의 "조건 간 코드 경로 차이 금지"는 *우리 실험 조건들* 사이의
규칙이므로, 외부 기준선을 그 루프에 if-분기로 밀어넣는 것이 오히려 그 원칙에 어긋난다.
대신 데이터 로딩·평가 지표·로그 형식은 전부 기존 모듈을 그대로 재사용해서 궤적이 우리 run과
직접 겹쳐 그려지도록 했다(scripts/plot_logs.py가 그대로 파싱한다).

CLAUDE.md "하지 말 것"의 dropout 증강 재실험 금지에 대한 예외: 그 조항의 취지는 "이미 수치가
있는 실험을 중복 수행하지 말라"인데, 여기서 필요한 것은 기존 기록(STS-B 70.6, 250 step 후 하락)
으로는 얻을 수 없는 *step별 궤적*이다. 사용자 승인 하에 진행.

기본값은 공식 저장소 run_unsup_example.sh를 그대로 따른다:
  batch 64, lr 3e-5, max_seq_length 32, temp 0.05, dropout 0.1, 1 epoch,
  CLS + MLP pooler(mlp_only_train - 평가 시 MLP 제외 = cls_before_pooler),
  warmup 없음 + linear decay(HF Trainer 기본), fp16,
  125 step마다 STS-B dev 평가 후 best 체크포인트 채택(load_best_model_at_end).
샘플링도 복원추출이 아니라 epoch 단위 셔플 순회다.

앞서 돌린 예산 맞춤 비교 run들은 다음 인자를 썼다(기본값이 그 뒤 원본 기준으로 바뀌었으므로 재현 시 명시 필요):
    --max-seq-length 128 --scheduler cosine --eval-steps 250 --no-fp16 --sampling with-replacement

실행(원본 재현, 1 epoch):
    uv run python scripts/train_simcse_baseline.py --device cuda --run-name simcse_repro_s42 --epochs 1
"""
import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoConfig, AutoTokenizer, BertModel,
                          get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import effective_rank_metrics, sts_b_dev_metrics  # noqa: E402
from src.train import load_config, load_sentences, setup_logger  # noqa: E402


class SimCSEModel(nn.Module):
    """BERT + SimCSE MLP pooler.

    forward()는 우리 평가 함수가 요구하는 DinoTextModel 인터페이스
    (inputs_embeds/attention_mask/pooling -> (embedding, logits, hidden, pooled))를 그대로 흉내낸다.
    평가는 MLP를 통과시키지 않는다(cls_before_pooler) - 비지도 SimCSE 공식 방식.
    """

    def __init__(self, backbone_name: str, dropout: float):
        super().__init__()
        cfg = AutoConfig.from_pretrained(backbone_name)
        cfg.hidden_dropout_prob = dropout
        cfg.attention_probs_dropout_prob = dropout
        self.backbone = BertModel.from_pretrained(backbone_name, config=cfg)
        h = self.backbone.config.hidden_size
        self.mlp = nn.Sequential(nn.Linear(h, h), nn.Tanh())  # 학습 전용 pooler

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def forward(self, inputs_embeds, attention_mask, pooling: str = "last"):
        out = self.backbone(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            output_hidden_states=(pooling == "first_last"),
        )
        hidden = out.last_hidden_state
        if pooling == "cls":
            pooled = hidden[:, 0]
        elif pooling == "first_last":
            hs = (out.hidden_states[0] + out.hidden_states[-1]) / 2
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        elif pooling == "last":
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        return F.normalize(pooled, p=2, dim=-1), pooled, hidden, pooled

    def train_embed(self, input_ids, attention_mask):
        """학습 경로: CLS -> MLP pooler (정규화는 InfoNCE에서 cosine으로 처리)."""
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.mlp(h[:, 0])


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float):
    """SimCSE 식 (2): 같은 문장의 두 dropout 뷰가 positive, 배치 내 나머지가 negative."""
    sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    loss = F.cross_entropy(sim, labels)
    with torch.no_grad():
        acc = (sim.argmax(dim=1) == labels).float().mean()
        pos = sim.diag().mean() * temperature
        neg = ((sim.sum(1) - sim.diag()) / (sim.size(0) - 1)).mean() * temperature
    return loss, {"infonce_acc": acc.item(), "pos_cos": pos.item(), "neg_cos": neg.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="simcse_baseline_s42")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 예산: --epochs를 주면 1 epoch = len(sentences)//batch step (공식 방식)
    ap.add_argument("--epochs", type=float, default=None, help="주면 max-steps보다 우선")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=64)
    # 방법 하이퍼파라미터: 공식 run_unsup_example.sh
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-seq-length", type=int, default=32)
    ap.add_argument("--warmup-frac", type=float, default=0.0)   # 공식: warmup 없음
    ap.add_argument("--scheduler", default="linear", choices=["linear", "cosine"])
    ap.add_argument("--no-fp16", action="store_true", help="공식은 fp16 사용")
    ap.add_argument("--sampling", default="epoch", choices=["epoch", "with-replacement"])
    ap.add_argument("--eval-steps", type=int, default=125)
    ap.add_argument("--eval-pooling", default="cls", help="cls(SimCSE 공식) | last | first_last")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    base_cfg = load_config(ROOT / "configs" / "base_bert.yaml")
    backbone = base_cfg["model"]["backbone"]
    max_tokens = args.max_seq_length          # 공식: 32 (우리 base_bert.yaml의 128이 아니다)
    eval_every = args.eval_steps
    log_every = base_cfg["eval"]["log_every_steps"]
    use_amp = (not args.no_fp16) and args.device.startswith("cuda")

    logger = setup_logger(args.run_name)
    # train.py와 동일한 레이아웃/태그로 남겨 우리 run과 TensorBoard에서 바로 겹쳐 볼 수 있게 한다.
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=str(ROOT / "results" / "tensorboard" / args.run_name / run_ts))
    logger.info(f"[simcse] backbone={backbone} batch={args.batch_size} lr={args.lr} "
                f"temp={args.temperature} dropout={args.dropout} max_seq_len={max_tokens} "
                f"sched={args.scheduler} fp16={use_amp} sampling={args.sampling} "
                f"eval_every={eval_every} eval_pooling={args.eval_pooling} seed={args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    sentences = load_sentences(ROOT / base_cfg["data"]["sentences_path"])
    rank_sentences = load_sentences(ROOT / base_cfg["data"]["rank_eval_path"])
    logger.info(f"[simcse] train sentences={len(sentences)} rank_eval={len(rank_sentences)}")

    steps_per_epoch = len(sentences) // args.batch_size
    max_steps = int(args.epochs * steps_per_epoch) if args.epochs else args.max_steps
    logger.info(f"[simcse] steps_per_epoch={steps_per_epoch} -> max_steps={max_steps}")

    model = SimCSEModel(backbone, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup = max(0, int(args.warmup_frac * max_steps))
    make_sched = get_linear_schedule_with_warmup if args.scheduler == "linear" else get_cosine_schedule_with_warmup
    scheduler = make_sched(optimizer, warmup, max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = {"sts": -1.0, "step": -1}

    def evaluate(step):
        model.eval()
        spearman, alignment, uniformity = sts_b_dev_metrics(
            model, tokenizer, args.device, pooling=args.eval_pooling)
        eff_rank, max_sv = effective_rank_metrics(
            model, tokenizer, rank_sentences, args.device, pooling=args.eval_pooling)
        logger.info(f"[step {step}] EVAL sts_b_dev={spearman:.4f} eff_rank={eff_rank:.2f} "
                    f"max_sv_ratio={max_sv:.4f} alignment={alignment:.4f} uniformity={uniformity:.4f}")
        for k, v in {"sts_b_dev": spearman, "eff_rank": eff_rank, "max_sv_ratio": max_sv,
                     "alignment": alignment, "uniformity": uniformity}.items():
            tb_writer.add_scalar(f"eval/{k}", v, step)
        # 공식은 load_best_model_at_end(metric=stsb_spearman) - 최종 산물이 '마지막'이 아니라 '최고'다
        if spearman > best["sts"]:
            best.update(sts=spearman, step=step)
            ckpt_dir = ROOT / "checkpoints" / args.run_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "model_cfg": {"backbone": backbone, "simcse": True,
                                      "temperature": args.temperature, "dropout": args.dropout},
                        "best_sts": spearman, "best_step": step}, ckpt_dir / "best.pt")
        model.train()

    def batches():
        """공식은 데이터셋을 셔플해 1 epoch 순회한다(복원추출이 아니다)."""
        if args.sampling == "with-replacement":
            while True:
                yield random.sample(sentences, args.batch_size)
        while True:
            order = list(range(len(sentences)))
            random.shuffle(order)
            for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
                yield [sentences[j] for j in order[i:i + args.batch_size]]

    model.train()
    evaluate(0)
    stream = batches()
    for step in range(max_steps):
        batch = next(stream)
        enc = tokenizer(batch, truncation=True, max_length=max_tokens, padding=True,
                        return_tensors="pt").to(args.device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            # 같은 배치를 두 번 통과 - dropout 마스크가 달라 서로 다른 뷰가 된다(SimCSE의 증강)
            z1 = model.train_embed(enc["input_ids"], enc["attention_mask"])
            z2 = model.train_embed(enc["input_ids"], enc["attention_mask"])
            loss, aux = info_nce(z1.float(), z2.float(), args.temperature)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e10).item()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % log_every == 0 or step == max_steps - 1:
            logger.info(f"[step {step}] loss={loss.item():.4f} infonce_acc={aux['infonce_acc']:.4f} "
                        f"pos_cos={aux['pos_cos']:.4f} neg_cos={aux['neg_cos']:.4f} "
                        f"grad_norm_total={grad_norm:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
            for k, v in {"loss": loss.item(), **aux, "grad_norm_total": grad_norm,
                         "lr": scheduler.get_last_lr()[0]}.items():
                tb_writer.add_scalar(f"train/{k}", v, step)
        if (step % eval_every == 0 and step > 0) or step == max_steps - 1:
            evaluate(step)

    ckpt_dir = ROOT / "checkpoints" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "model_cfg": {"backbone": backbone, "simcse": True,
                              "temperature": args.temperature, "dropout": args.dropout}},
               ckpt_dir / "last.pt")
    logger.info(f"[simcse] saved checkpoint -> {ckpt_dir / 'last.pt'}")
    logger.info(f"[simcse] best STS-B dev={best['sts']:.4f} @step {best['step']} "
                f"(checkpoints/{args.run_name}/best.pt) - 공식 load_best_model_at_end 대응")
    tb_writer.close()


if __name__ == "__main__":
    main()
