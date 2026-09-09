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
from src.evaluate import (EVAL_SPACES, effective_rank_metrics, embed_sentences,
                          sts_b_dev_metrics_by_space, sts_b_dev_spearman)
from src.loss import CovIsoPenalty, DINOLoss, EmbedUniformPush, batch_kl_diagnostic, koleo_loss, velocity_loss
from src.model import DinoTextModel, EMATeacher
from src.schedules import resolve_koleo_lambda, teacher_momentum_schedule, teacher_temp_schedule

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


def _is_no_decay_param(name: str) -> bool:
    """bias 또는 정규화 레이어(LayerNorm/norm) 파라미터인지, 이름 기반으로 판정.
    BERT는 "LayerNorm.weight"/"...bias", ModernBERT는 bias 없이 "norm.weight"만 씀
    (직접 확인, 두 backbone 모두 소문자화 후 "norm"/"bias" 부분 문자열 검사로 커버)."""
    lname = name.lower()
    return "bias" in lname or "norm" in lname


def build_param_groups(
    student, velocity_head, lr: float, head_lr: float, weight_decay: float, exclude_ln_bias_wd: bool
) -> list[dict]:
    """backbone/head(+velocity_head) x decay/no-decay 4-way(또는 그 이하) param group 구성.

    head_lr==lr 이고 exclude_ln_bias_wd=False 이면(둘 다 신규 키 미지정 시 기본값) 단일
    그룹으로 접혀 기존 `AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)`와
    파라미터 집합·순서·하이퍼파라미터가 완전히 동일 - 기존 config 재현성 보존."""
    backbone_named = list(student.backbone.named_parameters())
    head_named = list(student.head.named_parameters())
    if velocity_head is not None:
        head_named += list(velocity_head.named_parameters())

    if not exclude_ln_bias_wd and head_lr == lr:
        all_params = [p for _, p in backbone_named + head_named]
        return [{"params": all_params, "lr": lr, "weight_decay": weight_decay}]

    def split(named):
        decay = [p for n, p in named if not _is_no_decay_param(n)]
        no_decay = [p for n, p in named if _is_no_decay_param(n)]
        return decay, no_decay

    bb_decay, bb_no_decay = split(backbone_named)
    hd_decay, hd_no_decay = split(head_named)

    if exclude_ln_bias_wd:
        groups = [
            {"params": bb_decay, "lr": lr, "weight_decay": weight_decay},
            {"params": bb_no_decay, "lr": lr, "weight_decay": 0.0},
            {"params": hd_decay, "lr": head_lr, "weight_decay": weight_decay},
            {"params": hd_no_decay, "lr": head_lr, "weight_decay": 0.0},
        ]
    else:
        groups = [
            {"params": bb_decay + bb_no_decay, "lr": lr, "weight_decay": weight_decay},
            {"params": hd_decay + hd_no_decay, "lr": head_lr, "weight_decay": weight_decay},
        ]
    return [g for g in groups if len(g["params"]) > 0]


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


def reinit_dino_head(student, teacher, optimizer, dino_loss, scope: str = "sync") -> None:
    """DINO head를 주기적으로 재초기화한다 (config train.head_reinit_every_steps > 0일 때만).

    scope로 "무엇까지 되돌리는가"가 갈린다. backbone / cov_iso EMA(임베딩 공간 통계) /
    augment는 어느 쪽에서도 건드리지 않는다 - 리셋 대상은 head 계열 상태뿐이다.

    scope="sync" (기본): student head와 teacher head를 같은 난수로 동기 리셋하고 center도
      0으로 되돌린다. step 0(teacher=deepcopy(student), center=0) 상태의 완전한 복원이다.
      단, 리셋 직후 teacher와 student의 logit이 거의 같아져 KL(p_t||p_s)가 공짜로 만족되므로
      "다시 배우라"는 압력이 loss에 거의 실리지 않는다(실측: KL 0.0039 -> 0.0075, step 0의
      0.127에 한참 못 미침).

    scope="student_only": student head만 난수로 되돌리고 teacher head는 학습된 상태로 둔다.
      학습된 teacher가 만들어 놓은 prototype 배치를 난수 student가 처음부터 다시 맞춰야 하므로
      리셋 직후 KL이 크게 튀고, 그 재학습 gradient가 backbone으로 흘러든다.
      center는 teacher logit의 EMA라 teacher head와 한 몸이다 - teacher를 남기면서 center만
      0으로 만들면 살아 있는 teacher의 centering(붕괴 방지 기제)을 망가뜨리므로 함께 보존한다.

    optimizer state(head 파라미터의 Adam 1/2차 모멘트와 step 카운터)는 두 scope 모두에서
    제거한다: 사라진 prototype에 대한 모멘텀이 남으면 난수 head를 첫 스텝부터 밀어버린다.
    """
    assert scope in ("sync", "student_only"), f"unknown head_reinit_scope: {scope}"
    student.head.reset_parameters()
    for p in student.head.parameters():
        optimizer.state.pop(p, None)
    if scope == "sync":
        teacher.model.head.load_state_dict(student.head.state_dict())
        dino_loss.center.zero_()


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
    embed_push_lr = cfg["loss"].get("embed_push_lr", 0.0)
    embed_uniform_push = EmbedUniformPush(student.backbone.config.hidden_size, embed_push_lr).to(device)

    cov_iso_lambda = cfg["loss"].get("cov_iso_lambda", 0.0)
    cov_ema_momentum = cfg["loss"].get("cov_ema_momentum", 0.99)
    cov_iso_start_step = cfg["loss"].get("cov_iso_start_step", 50)
    cov_iso_penalty = None
    if cov_iso_lambda > 0:
        cov_iso_penalty = CovIsoPenalty(student.backbone.config.hidden_size, cov_ema_momentum).to(device)

    velocity_head = None
    if cfg["loss"]["velocity_head"]:
        hidden = student.backbone.config.hidden_size
        velocity_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        ).to(device)

    head_lr = cfg["train"].get("head_lr", cfg["train"]["lr"])
    exclude_ln_bias_wd = cfg["train"].get("exclude_ln_bias_wd", False)
    grad_clip = cfg["train"].get("grad_clip")
    param_groups = build_param_groups(
        student, velocity_head, cfg["train"]["lr"], head_lr, cfg["train"]["weight_decay"], exclude_ln_bias_wd
    )
    optimizer = torch.optim.AdamW(param_groups)
    all_trainable_params = [p for g in param_groups for p in g["params"]]
    warmup_steps = max(1, int(cfg["train"]["warmup_frac"] * max_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    teacher_temp_static = cfg["loss"]["teacher_temp"]
    warmup_teacher_temp = cfg["loss"].get("warmup_teacher_temp")
    teacher_temp_warmup_frac = cfg["loss"].get("teacher_temp_warmup_frac")
    teacher_temp_warmup_steps = (
        max(1, int(teacher_temp_warmup_frac * max_steps)) if warmup_teacher_temp is not None else None
    )
    teacher_temp_shape = cfg["loss"].get("teacher_temp_shape", "linear")
    momentum_start = cfg["train"].get("momentum_start")
    momentum_end = cfg["train"].get("momentum_end")
    momentum_shape = cfg["train"].get("momentum_shape", "cosine")
    # ramp_frac 미지정이면 1.0 = 전 구간 ramp(기존 동작). 더 작으면 그 시점에 ramp가 끝나고
    # 이후로는 momentum_end가 유지된다.
    momentum_ramp_frac = cfg["train"].get("momentum_ramp_frac", 1.0)
    momentum_ramp_steps = max(1, int(momentum_ramp_frac * max_steps))

    koleo_lambda_max = cfg["loss"].get("koleo_lambda", 0.0)
    koleo_hold_frac = cfg["loss"].get("koleo_hold_frac")
    koleo_decay_frac = cfg["loss"].get("koleo_decay_frac")
    koleo_min_ratio = cfg["loss"].get("koleo_min_ratio")
    koleo_hold_steps = int(koleo_hold_frac * max_steps) if koleo_hold_frac is not None else None
    koleo_decay_steps = int(koleo_decay_frac * max_steps) if koleo_decay_frac is not None else None

    student_temp = cfg["loss"]["student_temp"]
    max_tokens = cfg["data"]["max_tokens"]
    batch_size = cfg["train"]["batch_size"]
    log_every = cfg["eval"]["log_every_steps"]
    # R9: head_reinit_every_steps step마다 DINO head를 난수 재초기화(0/미지정이면 off =
    # 기존 config와 완전히 동일). scope는 reinit_dino_head docstring 참고.
    head_reinit_every = cfg["train"].get("head_reinit_every_steps", 0)
    head_reinit_scope = cfg["train"].get("head_reinit_scope", "sync")
    eval_every = cfg["eval"]["every_steps"]
    # 평가할 표현 공간(evaluate.EVAL_SPACES). Final Embedding 외에 DINO head를 통과한 벡터도
    # 함께 재는 것이 기본값이다 - backbone forward를 공유하므로 평가 시간은 거의 그대로다.
    # "embedding"은 주 지표(무접미사 로그 키)라 빠져 있어도 항상 맨 앞에 넣는다.
    eval_spaces = tuple(cfg["eval"].get("spaces", EVAL_SPACES))
    if "embedding" not in eval_spaces:
        eval_spaces = ("embedding",) + eval_spaces
    n_pairs = cfg["eval"]["batch_kl_pairs"]

    # --- Part B 진단 플래그 (전부 기본 off) ---
    diag_grad_norms = cfg["train"].get("diag_grad_norms", False)
    diag_tbin_kl = cfg["train"].get("diag_tbin_kl", False)
    diag_teacher_eval = cfg["train"].get("diag_teacher_eval", False)
    diag_confidence = cfg["train"].get("diag_confidence", False)
    diag_drift = cfg["train"].get("diag_drift", False)
    dense_early_eval = cfg["train"].get("dense_early_eval", False)
    # R7 캘리브레이션 전용: cov_iso_lambda*L_iso 단독 backward로 backbone grad norm을 측정해
    # diag_grad_norm_backbone(결합 backward, DINO CE+koleo+cov_iso 전부 포함)과 비교하면
    # cov_iso가 실제로 얼마만큼의 grad를 기여하는지 알 수 있다. 계산량이 늘어나므로(추가
    # backward 1회) 캘리브레이션 dry-run에서만 켠다.
    diag_coviso_grad_split = cfg["train"].get("diag_coviso_grad_split", False)

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
            teacher_temp = teacher_temp_schedule(step, warmup_teacher_temp, teacher_temp_static,
                                                 teacher_temp_warmup_steps, teacher_temp_shape)
        else:
            teacher_temp = teacher_temp_static

        koleo_lambda = resolve_koleo_lambda(step, koleo_lambda_max, koleo_hold_steps, koleo_decay_steps, koleo_min_ratio)

        token_embeds = student.get_input_embeddings()(input_ids)
        # step 0은 이미 난수 head라 건너뛴다 - 첫 리셋은 step==head_reinit_every.
        if head_reinit_every > 0 and step > 0 and step % head_reinit_every == 0:
            reinit_dino_head(student, teacher, optimizer, dino_loss, head_reinit_scope)
            logger.info(f"[step {step}] HEAD REINIT scope={head_reinit_scope}")
            wandb.log({"head_reinit": 1.0}, step=step)
            tb_writer.add_scalar("train/head_reinit", 1.0, step)
        views = aug(token_embeds, special_mask, step)

        with torch.no_grad():
            t_embedding, t_logits, *_ = teacher(
                inputs_embeds=views.teacher_embeds, attention_mask=attention_mask,
                embed_push=embed_uniform_push.push,
            )
        embed_push_grad_norm = embed_uniform_push.step(t_embedding)

        student_logits = []
        vel_losses = []
        koleo_losses = []
        cov_iso_losses = []
        cov_iso_mean_terms = []
        cov_iso_cov_terms = []
        for k, s_embeds in enumerate(views.student_embeds):
            s_embedding, s_logits, s_hidden, *_ = student(inputs_embeds=s_embeds, attention_mask=attention_mask)
            student_logits.append(s_logits)
            if velocity_head is not None:
                v_pred = velocity_head(s_hidden)
                vel_losses.append(velocity_loss(v_pred, views.eps[k], views.x0_std, special_mask))
            if koleo_lambda > 0:
                koleo_losses.append(koleo_loss(s_embedding))
            if cov_iso_penalty is not None:
                # EMA 통계는 step 0부터 계속 갱신(자리 잡을 시간을 준다) - cov_iso_start_step
                # 이전에는 아래에서 total_loss에 더하지 않을 뿐. K개 뷰 전부에 대해 호출하므로
                # 한 스텝 안에서 버퍼가 K번 순차 갱신된다(뷰마다 다른 노이즈 레벨의 실제 샘플).
                l_iso, iso_aux = cov_iso_penalty(s_embedding)
                cov_iso_losses.append(l_iso)
                cov_iso_mean_terms.append(iso_aux["L_iso_mean"])
                cov_iso_cov_terms.append(iso_aux["L_iso_cov"])

        loss, aux = dino_loss(t_logits, student_logits, teacher_temp, student_temp)
        total_loss = loss
        if velocity_head is not None:
            l_vel = torch.stack(vel_losses).mean()
            total_loss = total_loss + cfg["loss"]["velocity_lambda"] * l_vel
            aux["L_vel"] = l_vel.detach()
        if koleo_lambda > 0:
            l_koleo = torch.stack(koleo_losses).mean()
            total_loss = total_loss + koleo_lambda * l_koleo
            aux["L_koleo"] = l_koleo.detach()
        if cov_iso_losses and step >= cov_iso_start_step:
            l_iso_total = torch.stack(cov_iso_losses).mean()
            total_loss = total_loss + cov_iso_lambda * l_iso_total
            aux["L_iso"] = l_iso_total.detach()
            aux["L_iso_mean"] = torch.stack(cov_iso_mean_terms).mean()
            aux["L_iso_cov"] = torch.stack(cov_iso_cov_terms).mean()

        if diag_tbin_kl:
            for k, t_k in enumerate(views.t_students):
                kl_k = aux["kl_per_view"][k].item()
                for t_val in t_k.tolist():
                    b = tbin_index(t_val)
                    tbin_sums[b] += kl_k
                    tbin_counts[b] += 1

        grad_norm_log = {}
        if diag_coviso_grad_split and cov_iso_losses and step >= cov_iso_start_step:
            optimizer.zero_grad()
            (cov_iso_lambda * l_iso_total).backward(retain_graph=True)
            grad_norm_log["diag_grad_norm_coviso_only_backbone"] = torch.nn.utils.clip_grad_norm_(
                list(student.backbone.parameters()), max_norm=1e10
            ).item()

        optimizer.zero_grad()
        total_loss.backward()

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

        if grad_clip is not None:
            grad_norm_log["grad_norm_total"] = torch.nn.utils.clip_grad_norm_(
                all_trainable_params, max_norm=grad_clip
            ).item()

        optimizer.step()
        scheduler.step()
        if momentum_start is not None:
            teacher.momentum = teacher_momentum_schedule(step, momentum_start, momentum_end,
                                                         momentum_ramp_steps, momentum_shape)
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
            if "L_koleo" in aux:
                log["L_koleo"] = aux["L_koleo"].item()
            if koleo_lambda_max > 0:
                log["koleo_lambda"] = koleo_lambda
            if "L_iso" in aux:
                log["L_iso"] = aux["L_iso"].item()
                log["L_iso_mean"] = aux["L_iso_mean"].item()
                log["L_iso_cov"] = aux["L_iso_cov"].item()
            if "push_grad_norm" in aux:
                log["push_grad_norm"] = aux["push_grad_norm"]
            if embed_push_lr > 0:
                log["embed_push_grad_norm"] = embed_push_grad_norm
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
            space_metrics = sts_b_dev_metrics_by_space(student, tokenizer, device, eval_spaces)
            sts, alignment, uniformity = space_metrics["embedding"]
            eff_rank, max_sv = effective_rank_metrics(student, tokenizer, rank_eval_sentences, device)
            eval_log = {
                "sts_b_dev_spearman": sts, "effective_rank": eff_rank, "max_sv_ratio": max_sv,
                "alignment": alignment, "uniformity": uniformity,
            }
            eval_msg = (
                f"[step {step}] EVAL sts_b_dev={sts:.4f} eff_rank={eff_rank:.2f} max_sv_ratio={max_sv:.4f} "
                f"alignment={alignment:.4f} uniformity={uniformity:.4f}"
            )
            # head 통과 후 공간들은 접미사를 붙여 따로 기록한다(무접미사 키는 Final Embedding 고정).
            for space in eval_spaces:
                if space == "embedding":
                    continue
                sp_sts, sp_align, sp_unif = space_metrics[space]
                eval_log[f"sts_b_dev_spearman_{space}"] = sp_sts
                eval_log[f"alignment_{space}"] = sp_align
                eval_log[f"uniformity_{space}"] = sp_unif
                eval_msg += f" | {space}: sts={sp_sts:.4f} align={sp_align:.4f} unif={sp_unif:.4f}"

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

    if cfg["train"].get("save_checkpoint", True):
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
