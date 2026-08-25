"""단일 학습 루프. 실험 조건(run)은 오직 config yaml 값만 다르다 (CLAUDE.md 구현 원칙 1).

실행:
    source scripts/env.sh
    uv run python -m src.train --config configs/r1a.yaml [--max-steps 500] [--device cuda]
"""
import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import wandb
import yaml
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.augment import FlowNoiseAug
from src.evaluate import effective_rank_metrics, sts_b_dev_spearman
from src.loss import DINOLoss, batch_kl_diagnostic, velocity_loss
from src.model import DinoTextModel, EMATeacher

ROOT = Path(__file__).resolve().parent.parent


def load_config(path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    extends = cfg.pop("extends", None)
    if extends:
        base = load_config(path.parent / extends)
        merged = dict(base)
        for key, value in cfg.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        cfg = merged
    return cfg


def load_sentences(path) -> list[str]:
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            sentences.append(json.loads(line)["text"])
    return sentences


def build_augment(cfg: dict) -> FlowNoiseAug:
    stats = torch.load(ROOT / cfg["data"]["embed_stats_path"], weights_only=True)
    a = cfg["augment"]
    warmup_steps = max(1, int(a["warmup_frac"] * cfg["train"]["max_steps"]))
    return FlowNoiseAug(
        stats["mean"], stats["std"],
        mode=a["mode"], num_student_views=a["num_student_views"],
        t_lo=a["t_lo"], t_start=a["t_start"], t_max=a["t_max"],
        warmup_steps=warmup_steps, delta_t=a["delta_t"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="config의 train.max_steps override (스모크용)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = args.device
    max_steps = cfg["train"]["max_steps"]

    os.environ.setdefault("WANDB_MODE", cfg["logging"].get("wandb_mode", "offline"))
    wandb.init(project="flowdino-text", name=cfg["run_name"], config=cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    sentences = load_sentences(ROOT / cfg["data"]["sentences_path"])
    rank_eval_sentences = load_sentences(ROOT / cfg["data"]["rank_eval_path"])

    student = DinoTextModel(
        cfg["model"]["backbone"], cfg["model"]["head"]["bottleneck_dim"], cfg["model"]["head"]["logit_dim"]
    ).to(device)
    teacher = EMATeacher(student, momentum=cfg["train"]["teacher_momentum"])

    aug = build_augment(cfg)
    dino_loss = DINOLoss(cfg["model"]["head"]["logit_dim"], cfg["loss"]["center_momentum"]).to(device)

    velocity_head = None
    if cfg["loss"]["velocity_head"]:
        hidden = student.backbone.config.hidden_size
        velocity_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        ).to(device)

    params = list(student.parameters())
    if velocity_head is not None:
        params += list(velocity_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    warmup_steps = max(1, int(cfg["train"]["warmup_frac"] * max_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    teacher_temp = cfg["loss"]["teacher_temp"]
    student_temp = cfg["loss"]["student_temp"]
    max_tokens = cfg["data"]["max_tokens"]
    batch_size = cfg["train"]["batch_size"]
    log_every = cfg["eval"]["log_every_steps"]
    eval_every = cfg["eval"]["every_steps"]
    n_pairs = cfg["eval"]["batch_kl_pairs"]

    student.train()
    for step in range(max_steps):
        batch_sentences = random.sample(sentences, batch_size)
        enc = tokenizer(
            batch_sentences, truncation=True, max_length=max_tokens, padding=True,
            return_special_tokens_mask=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        special_mask = enc["special_tokens_mask"].bool().to(device) | (~attention_mask.bool())

        token_embeds = student.get_input_embeddings()(input_ids)
        views = aug(token_embeds, special_mask, step)

        with torch.no_grad():
            _, t_logits, _ = teacher(inputs_embeds=views.teacher_embeds, attention_mask=attention_mask)

        student_logits = []
        vel_losses = []
        for k, s_embeds in enumerate(views.student_embeds):
            _, s_logits, s_hidden = student(inputs_embeds=s_embeds, attention_mask=attention_mask)
            student_logits.append(s_logits)
            if velocity_head is not None:
                v_pred = velocity_head(s_hidden)
                vel_losses.append(velocity_loss(v_pred, views.eps[k], views.x0_std, special_mask))

        loss, aux = dino_loss(t_logits, student_logits, teacher_temp, student_temp)
        total_loss = loss
        if velocity_head is not None:
            l_vel = torch.stack(vel_losses).mean()
            total_loss = total_loss + cfg["loss"]["velocity_lambda"] * l_vel
            aux["L_vel"] = l_vel.detach()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        teacher.update(student)

        if step % log_every == 0 or step == max_steps - 1:
            with torch.no_grad():
                b_kl = batch_kl_diagnostic(student_logits, student_temp, n_pairs=n_pairs)
            log = {
                "loss": total_loss.item(),
                "H_pt": aux["H_pt"].item(),
                "KL_pt_ps": aux["KL_pt_ps"].item(),
                "batch_KL": b_kl.item(),
            }
            if "L_vel" in aux:
                log["L_vel"] = aux["L_vel"].item()
            wandb.log(log, step=step)
            print(f"[step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in log.items()))

        if step % eval_every == 0 or step == max_steps - 1:
            sts = sts_b_dev_spearman(student, tokenizer, device)
            eff_rank, max_sv = effective_rank_metrics(student, tokenizer, rank_eval_sentences, device)
            student.train()
            wandb.log({"sts_b_dev_spearman": sts, "effective_rank": eff_rank, "max_sv_ratio": max_sv}, step=step)
            print(f"[step {step}] EVAL sts_b_dev={sts:.4f} eff_rank={eff_rank:.2f} max_sv_ratio={max_sv:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
