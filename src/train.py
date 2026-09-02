"""단일 학습 루프. 실험 조건(run)은 오직 config yaml 값만 다르다 (CLAUDE.md 구현 원칙 1).

실행:
    source scripts/env.sh
    uv run python -m src.train --config configs/r1a.yaml [--max-steps 500] [--device cuda]

Part B 진단 로깅(diag_*, dense_early_eval)은 전부 config의 train 블록 키로만 켜진다.
키가 없으면 기존 config를 그대로 재실행한 것과 동일하게 동작한다.
"""
import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.augment import FlowNoiseAug
from src.diagnostics import active_prototype_count, tbin_index
from src.evaluate import effective_rank_metrics, embed_sentences, sts_b_dev_metrics, sts_b_dev_spearman
from src.loss import DINOLoss, batch_kl_diagnostic, velocity_loss
from src.model import DinoTextModel, EMATeacher
from src.schedules import teacher_momentum_schedule, teacher_temp_schedule

ROOT = Path(__file__).resolve().parent.parent


def setup_logger(run_name: str) -> logging.Logger:
    """run별 폴더(results/logs/<run_name>/train.log)에 기록 + 콘솔 동시 출력.
    미처리 예외도 logger.exception()으로 이 파일에 남긴다."""
    log_dir = ROOT / "results" / "logs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("flowdino.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_dir / "train.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


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
    parser.add_argument("--max-steps", type=int, default=None, help="config의 train.max_steps override (스모크/3000-step 연장판용)")
    parser.add_argument("--seed", type=int, default=None, help="config의 seed override (매트릭스 다중 시드용)")
    parser.add_argument("--run-name-suffix", default="", help="wandb run_name에 덧붙일 접미사 (예: _seed43)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.run_name_suffix:
        cfg["run_name"] = cfg["run_name"] + args.run_name_suffix

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = args.device
    max_steps = cfg["train"]["max_steps"]

    logger = setup_logger(cfg["run_name"])
    # run_name 아래 실행 시각 하위 폴더에 기록한다: 같은 run_name을 재실행해도 TensorBoard
    # run 선택기에서 run_name이 그룹으로 묶이고, 각 실행이 시각으로 구분된 별도 run으로 보인다
    # (이전에는 같은 폴더에 이벤트 파일이 누적되어 step이 뒤섞이고 실행 시각도 알 수 없었다).
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=str(ROOT / "results" / "tensorboard" / cfg["run_name"] / run_ts))

    os.environ.setdefault("WANDB_MODE", cfg["logging"].get("wandb_mode", "offline"))
    wandb.init(project="flowdino-text", name=cfg["run_name"], config=cfg)

    try:
        _run(cfg, device, max_steps, logger, tb_writer)
    except Exception:
        logger.exception("training crashed")
        raise
    finally:
        tb_writer.close()


def _run(cfg, device, max_steps, logger, tb_writer) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    sentences = load_sentences(ROOT / cfg["data"]["sentences_path"])
    rank_eval_sentences = load_sentences(ROOT / cfg["data"]["rank_eval_path"])

    student = DinoTextModel(
        cfg["model"]["backbone"], cfg["model"]["head"]["bottleneck_dim"], cfg["model"]["head"]["logit_dim"]
    ).to(device)
    teacher = EMATeacher(student, momentum=cfg["train"]["teacher_momentum"])

    aug = build_augment(cfg)
    dino_loss = DINOLoss(
        cfg["model"]["head"]["logit_dim"], cfg["loss"]["center_momentum"],
        centering=cfg["loss"].get("centering", "ema"),
        uniform_push_lr=cfg["loss"].get("uniform_push_lr", 0.0),
    ).to(device)

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

    teacher_temp_static = cfg["loss"]["teacher_temp"]
    warmup_teacher_temp = cfg["loss"].get("warmup_teacher_temp")
    teacher_temp_warmup_frac = cfg["loss"].get("teacher_temp_warmup_frac")
    teacher_temp_warmup_steps = (
        max(1, int(teacher_temp_warmup_frac * max_steps)) if warmup_teacher_temp is not None else None
    )
    momentum_start = cfg["train"].get("momentum_start")
    momentum_end = cfg["train"].get("momentum_end")

    student_temp = cfg["loss"]["student_temp"]
    max_tokens = cfg["data"]["max_tokens"]
    batch_size = cfg["train"]["batch_size"]
    log_every = cfg["eval"]["log_every_steps"]
    eval_every = cfg["eval"]["every_steps"]
    n_pairs = cfg["eval"]["batch_kl_pairs"]

    # --- Part B 진단 플래그 (전부 기본 off) ---
    diag_grad_norms = cfg["train"].get("diag_grad_norms", False)
    diag_tbin_kl = cfg["train"].get("diag_tbin_kl", False)
    diag_teacher_eval = cfg["train"].get("diag_teacher_eval", False)
    diag_confidence = cfg["train"].get("diag_confidence", False)
    diag_drift = cfg["train"].get("diag_drift", False)
    dense_early_eval = cfg["train"].get("dense_early_eval", False)

    n_tbins = 5
    tbin_sums = [0.0] * n_tbins
    tbin_counts = [0] * n_tbins

    drift_sentences = None
    prev_drift_embeds = None
    if diag_drift:
        drift_rng = random.Random(0)
        n_sub = min(512, len(rank_eval_sentences))
        drift_idx = drift_rng.sample(range(len(rank_eval_sentences)), n_sub)
        drift_sentences = [rank_eval_sentences[i] for i in drift_idx]

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

        if warmup_teacher_temp is not None:
            teacher_temp = teacher_temp_schedule(step, warmup_teacher_temp, teacher_temp_static, teacher_temp_warmup_steps)
        else:
            teacher_temp = teacher_temp_static

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

        if diag_tbin_kl:
            for k, t_k in enumerate(views.t_students):
                kl_k = aux["kl_per_view"][k].item()
                for t_val in t_k.tolist():
                    b = tbin_index(t_val)
                    tbin_sums[b] += kl_k
                    tbin_counts[b] += 1

        optimizer.zero_grad()
        total_loss.backward()

        grad_norm_log = {}
        if diag_grad_norms:
            # max_norm=1e10 -> 사실상 clip 없이 그룹별 norm만 측정 (진단용)
            grad_norm_log["diag_grad_norm_backbone"] = torch.nn.utils.clip_grad_norm_(
                list(student.backbone.parameters()), max_norm=1e10
            ).item()
            grad_norm_log["diag_grad_norm_bottleneck"] = torch.nn.utils.clip_grad_norm_(
                list(student.head.mlp.parameters()), max_norm=1e10
            ).item()
            grad_norm_log["diag_grad_norm_prototype"] = torch.nn.utils.clip_grad_norm_(
                list(student.head.expand.parameters()), max_norm=1e10
            ).item()

        optimizer.step()
        scheduler.step()
        if momentum_start is not None:
            teacher.momentum = teacher_momentum_schedule(step, momentum_start, momentum_end, max_steps)
        teacher.update(student)

        if step % log_every == 0 or step == max_steps - 1:
            with torch.no_grad():
                b_kl = batch_kl_diagnostic(student_logits, student_temp, n_pairs=n_pairs)
            log = {
                "loss": total_loss.item(),
                "H_pt": aux["H_pt"].item(),
                "KL_pt_ps": aux["KL_pt_ps"].item(),
                "batch_KL": b_kl.item(),
                "H_p_bar_t": aux["H_p_bar_t"].item(),
            }
            if "L_vel" in aux:
                log["L_vel"] = aux["L_vel"].item()
            if "push_grad_norm" in aux:
                log["push_grad_norm"] = aux["push_grad_norm"]
            if warmup_teacher_temp is not None:
                log["teacher_temp"] = teacher_temp
            if momentum_start is not None:
                log["teacher_momentum"] = teacher.momentum
            log.update(grad_norm_log)

            if diag_tbin_kl:
                for b in range(n_tbins):
                    if tbin_counts[b] > 0:
                        log[f"diag_tbin_kl_{b}"] = tbin_sums[b] / tbin_counts[b]
                    tbin_sums[b] = 0.0
                    tbin_counts[b] = 0

            if diag_confidence:
                with torch.no_grad():
                    log["diag_confidence_max_p"] = aux["p_t"].max(dim=-1).values.mean().item()
                    log["diag_active_prototypes"] = active_prototype_count(aux["p_bar_t"])

            wandb.log(log, step=step)
            for k, v in log.items():
                tb_writer.add_scalar(f"train/{k}", v, step)
            logger.info(f"[step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in log.items()))

        do_dense_eval = dense_early_eval and step <= 300 and step % 25 == 0
        if step % eval_every == 0 or step == max_steps - 1 or do_dense_eval:
            sts, alignment, uniformity = sts_b_dev_metrics(student, tokenizer, device)
            eff_rank, max_sv = effective_rank_metrics(student, tokenizer, rank_eval_sentences, device)
            eval_log = {
                "sts_b_dev_spearman": sts, "effective_rank": eff_rank, "max_sv_ratio": max_sv,
                "alignment": alignment, "uniformity": uniformity,
            }
            eval_msg = (
                f"[step {step}] EVAL sts_b_dev={sts:.4f} eff_rank={eff_rank:.2f} max_sv_ratio={max_sv:.4f} "
                f"alignment={alignment:.4f} uniformity={uniformity:.4f}"
            )

            if diag_teacher_eval:
                teacher_sts = sts_b_dev_spearman(teacher.model, tokenizer, device)
                eval_log["teacher_sts_b_dev"] = teacher_sts
                eval_msg += f" teacher_sts_b_dev={teacher_sts:.4f}"

            if diag_drift:
                cur_drift_embeds = embed_sentences(student, tokenizer, drift_sentences, device)
                if prev_drift_embeds is not None:
                    drift_cos = F.cosine_similarity(cur_drift_embeds, prev_drift_embeds, dim=-1).mean().item()
                    eval_log["diag_drift_cosine"] = drift_cos
                    eval_msg += f" drift_cosine={drift_cos:.4f}"
                prev_drift_embeds = cur_drift_embeds

            wandb.log(eval_log, step=step)
            for k, v in eval_log.items():
                tb_writer.add_scalar(f"eval/{k}", v, step)
            logger.info(eval_msg)

    ckpt_path = ROOT / "checkpoints" / cfg["run_name"] / "last.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": student.state_dict(),
        "model_cfg": cfg["model"],
        "teacher_state_dict": teacher.model.state_dict(),
    }, ckpt_path)
    logger.info(f"[train] saved checkpoint -> {ckpt_path}")

    opt = cfg.get("_optuna")
    if opt:
        # HPARAMS 탭에서 study 전체 trial을 한 표/평행좌표로 비교할 수 있게 로깅한다.
        # `uv run tensorboard --logdir results/tensorboard`의 HPARAMS 탭이 하위 폴더를
        # 재귀적으로 스캔하므로 tune.py가 도는 study의 모든 trial이 한 곳에 모인다.
        tb_writer.add_hparams(
            {"study": opt["study_name"], "trial_number": opt["trial_number"], **opt["params"]},
            {"hparam/sts_b_dev": sts, "hparam/effective_rank": eff_rank, "hparam/uniformity": uniformity},
        )

    wandb.finish()


if __name__ == "__main__":
    main()
