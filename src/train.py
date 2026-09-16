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
from src.loss import (CovIsoPenalty, DINOLoss, EmbedUniformPush, EntropyCtrl, TokenLatentPredictor,
                      batch_kl_diagnostic, koleo_loss, token_latent_loss, token_latent_targets, velocity_loss)
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


class BatchIndexSampler:
    """배치 문장 인덱스 공급기.

    order="sample"(기본): 매 step random.sample - step 간 복원 추출이다. 기존 run과 bit-identical
      하다: random.sample은 모집단 원소가 아니라 길이만 보고 위치를 고르므로, 문장 리스트 대신
      인덱스로 뽑아도 같은 위치가 나온다(스크립트로 확인).
    order="epoch": 셔플한 순열을 앞에서부터 잘라 쓰는 비복원 순회. 한 바퀴를 다 쓰면 재셔플한다.
      복원 추출이면 wiki1m(985,723문장)에서 7700x32 step은 뽑기의 약 11%, 15600x32 step은 약 22%가
      중복이라 긴 예산 비교에 교란이 된다. epoch 순회면 둘 다 한 바퀴 안이라 중복이 0이다.

    문장 문자열이 아니라 인덱스를 돌려주는 이유: 역번역(paraphrase) 같은 문장 단위 부가 데이터를
    같은 인덱스로 찾아야 하기 때문이다.
    """

    def __init__(self, n: int, batch_size: int, order: str = "sample", rng=random):
        if order not in ("sample", "epoch"):
            raise ValueError(f"unknown data_order: {order!r} (sample|epoch)")
        if batch_size > n:
            raise ValueError(f"batch_size({batch_size})가 문장 수({n})보다 크다")
        self.n, self.batch_size, self.order, self.rng = n, batch_size, order, rng
        self.epoch = 0
        self._perm: list[int] = []
        self._cursor = 0
        if order == "epoch":
            self._reshuffle()

    def _reshuffle(self) -> None:
        self._perm = list(range(self.n))
        self.rng.shuffle(self._perm)
        self._cursor = 0

    def next(self) -> list[int]:
        if self.order == "sample":
            return self.rng.sample(range(self.n), self.batch_size)
        if self._cursor + self.batch_size > self.n:   # 남은 조각은 버리고 다음 epoch로 - 배치 크기 고정
            self.epoch += 1
            self._reshuffle()
        idx = self._perm[self._cursor:self._cursor + self.batch_size]
        self._cursor += self.batch_size
        return idx


def _is_no_decay_param(name: str) -> bool:
    """bias 또는 정규화 레이어(LayerNorm/norm) 파라미터인지, 이름 기반으로 판정.
    BERT는 "LayerNorm.weight"/"...bias", ModernBERT는 bias 없이 "norm.weight"만 씀
    (직접 확인, 두 backbone 모두 소문자화 후 "norm"/"bias" 부분 문자열 검사로 커버)."""
    lname = name.lower()
    return "bias" in lname or "norm" in lname


def build_param_groups(
    student, velocity_head, lr: float, head_lr: float, weight_decay: float, exclude_ln_bias_wd: bool,
    extra_heads: list | None = None,
) -> list[dict]:
    """backbone/head(+velocity_head) x decay/no-decay 4-way(또는 그 이하) param group 구성.

    head_lr==lr 이고 exclude_ln_bias_wd=False 이면(둘 다 신규 키 미지정 시 기본값) 단일
    그룹으로 접혀 기존 `AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)`와
    파라미터 집합·순서·하이퍼파라미터가 완전히 동일 - 기존 config 재현성 보존."""
    backbone_named = list(student.backbone.named_parameters())
    head_named = list(student.head.named_parameters())
    if velocity_head is not None:
        head_named += list(velocity_head.named_parameters())
    for i, mod in enumerate(extra_heads or []):          # R15 토큰 예측기 등 - 없으면 기존과 동일
        head_named += [(f"extra{i}.{n}", q) for n, q in mod.named_parameters()]

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


def build_augment(cfg: dict, mask_embed: torch.Tensor | None = None) -> FlowNoiseAug:
    # map_location: 통계 파일이 CUDA 텐서로 저장돼 있어 CPU 실행에서 그대로 로드하면 실패한다.
    # FlowNoiseAug가 매 호출마다 입력 device로 옮기므로 GPU 실행 수치에는 영향이 없다.
    stats = torch.load(ROOT / cfg["data"]["embed_stats_path"], map_location="cpu", weights_only=True)
    a = cfg["augment"]
    warmup_steps = max(1, int(a["warmup_frac"] * cfg["train"]["max_steps"]))
    return FlowNoiseAug(
        stats["mean"], stats["std"],
        mode=a["mode"], num_student_views=a["num_student_views"],
        t_lo=a["t_lo"], t_start=a["t_start"], t_max=a["t_max"],
        warmup_steps=warmup_steps, delta_t=a["delta_t"],
        # R12-B: 둘 다 기본값이 off이며 그때 기존 config와 bit-identical이다.
        noise_corr_rho=a.get("noise_corr_rho", 0.0),
        cutoff_span_frac=a.get("cutoff_span_frac", 0.0),
        cutoff_prob=a.get("cutoff_prob", 1.0),
        cutoff_mode=a.get("cutoff_mode", "drop"),
        mask_embed=mask_embed,
        # R15: 마스킹은 token_latent_lambda > 0일 때만 켠다. token_latent_views의 기본값이 1이라
        # 람다와 무관하게 켜면 lambda=0에서도 뷰가 바뀌어 기존 run과 달라진다.
        token_latent_views=token_latent_views(cfg),
        mask_ratio=a.get("mask_ratio", 0.15),
        mask_then_noise=a.get("mask_then_noise", True),
    )


def token_latent_views(cfg: dict) -> int:
    """R15 마스킹 뷰 수. token_latent_lambda가 0이면 항상 0 (기존과 bit-identical)."""
    if cfg["loss"].get("token_latent_lambda", 0.0) <= 0:
        return 0
    return cfg["augment"].get("token_latent_views", 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="config의 train.max_steps override (스모크/3000-step 연장판용)")
    parser.add_argument("--seed", type=int, default=None, help="config의 seed override (매트릭스 다중 시드용)")
    parser.add_argument("--run-name-suffix", default="", help="wandb run_name에 덧붙일 접미사 (예: _seed43)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 가속 스위치. config로도 켤 수 있고(train.tf32 / train.batch_views / model.attn_implementation)
    # 여기서 실행 시 덮어쓸 수도 있다. 기본값 None = config를 따른다(그리고 config 기본은 off).
    parser.add_argument("--tf32", dest="tf32", action="store_true", default=None,
                        help="Ampere 이상에서 TF32 matmul 허용 (torch 기본값은 꺼짐)")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false",
                        help="config가 켜 놓았어도 TF32를 끈다")
    parser.add_argument("--batch-views", dest="batch_views", action="store_true", default=None,
                        help="K개 student 뷰를 forward 1회로 묶는다(수학적 동치)")
    parser.add_argument("--no-batch-views", dest="batch_views", action="store_false")
    parser.add_argument("--attn-impl", default=None,
                        help='backbone attention 구현 (예: "sdpa", "eager"). 미지정이면 transformers 기본')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.tf32 is not None:
        cfg["train"]["tf32"] = args.tf32
    if args.batch_views is not None:
        cfg["train"]["batch_views"] = args.batch_views
    if args.attn_impl is not None:
        cfg["model"]["attn_implementation"] = args.attn_impl
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
        cfg["model"]["backbone"], cfg["model"]["head"]["bottleneck_dim"], cfg["model"]["head"]["logit_dim"],
        attn_implementation=cfg["model"].get("attn_implementation"),
    ).to(device)
    teacher = EMATeacher(student, momentum=cfg["train"]["teacher_momentum"])

    # cutoff_mode="mask"일 때만 필요한 [MASK] 임베딩. 그 외에는 None이라 비용이 없다.
    mask_embed = None
    if (cfg["augment"].get("cutoff_span_frac", 0.0) > 0 and cfg["augment"].get("cutoff_mode") == "mask") \
            or token_latent_views(cfg) > 0:
        mask_embed = student.get_input_embeddings()(
            torch.tensor([tokenizer.mask_token_id], device=device)
        )[0].detach().clone()
    aug = build_augment(cfg, mask_embed)
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

    # R15 토큰 latent 예측. lambda=0(기본)이면 예측기도 없고 teacher도 hidden을 반환하지 않는다.
    tok_lambda = cfg["loss"].get("token_latent_lambda", 0.0)
    tok_top_k = cfg["loss"].get("token_latent_top_k", 6)
    token_predictor = None
    if tok_lambda > 0:
        token_predictor = TokenLatentPredictor(student.backbone.config.hidden_size).to(device)
        if velocity_head is not None:
            raise ValueError("token_latent_lambda와 velocity_head를 함께 쓰는 경우는 검증되지 않았다")

    head_lr = cfg["train"].get("head_lr", cfg["train"]["lr"])
    exclude_ln_bias_wd = cfg["train"].get("exclude_ln_bias_wd", False)
    grad_clip = cfg["train"].get("grad_clip")
    param_groups = build_param_groups(
        student, velocity_head, cfg["train"]["lr"], head_lr, cfg["train"]["weight_decay"], exclude_ln_bias_wd,
        extra_heads=[token_predictor] if token_predictor is not None else None,
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
    # R11-A1 후반 샤프닝 하강 국면. teacher_temp_final이 없으면(기본) r8과 bit-identical.
    tt_final = cfg["loss"].get("teacher_temp_final")
    tt_decay_start_frac = cfg["loss"].get("teacher_temp_decay_start_frac")
    tt_decay_end_frac = cfg["loss"].get("teacher_temp_decay_end_frac", 1.0)
    tt_decay_shape = cfg["loss"].get("teacher_temp_decay_shape", "cosine")
    if tt_final is not None and tt_decay_start_frac is None:
        raise ValueError("loss.teacher_temp_final을 주면 loss.teacher_temp_decay_start_frac도 필요하다")
    tt_decay_start = int(tt_decay_start_frac * max_steps) if tt_final is not None else None
    tt_decay_end = int(tt_decay_end_frac * max_steps) if tt_final is not None else None

    # R11-A2 엔트로피 목표 제어기. 켜면 A1 하강 국면은 비활성(plateau까지 개루프, 이후 인계).
    ectrl_cfg = cfg["loss"]
    entropy_ctrl = None
    if ectrl_cfg.get("entropy_ctrl", False):
        tt_final = None              # 인계 - 개루프 하강과 폐루프가 동시에 걸리지 않게
        entropy_ctrl = EntropyCtrl(
            start_step=int(ectrl_cfg.get("entropy_ctrl_start_frac", 0.45) * max_steps),
            end_step=int(ectrl_cfg.get("entropy_ctrl_end_frac", 0.85) * max_steps),
            delta=ectrl_cfg["entropy_ctrl_delta"],
            gain=ectrl_cfg.get("entropy_ctrl_gain", 0.02),
            tau_min=ectrl_cfg.get("entropy_ctrl_tau_min", 0.05),
            tau_max=ectrl_cfg.get("entropy_ctrl_tau_max", teacher_temp_static),
            tau_init=teacher_temp_static,
            max_step=ectrl_cfg.get("entropy_ctrl_max_step", 0.01),
        )

    # 붕괴 감시(전 run 공통). 기준값은 r8 plateau(step 675~1499, 2시드) 실측:
    #   H(p_bar_t) 9.0097 / eff_rank 305.16 / batch_KL 0.0620
    # cov_iso는 응축(rank 붕괴)은 막지만 p_bar_t가 한 점으로 쏠리는 경로는 막지 못한다.
    guard = cfg.get("collapse_guard", {})
    guard_on = guard.get("enabled", False)
    g_hbar = guard.get("h_p_bar_t_ref", 9.0097)
    g_rank = guard.get("eff_rank_ref", 305.16)
    g_bkl = guard.get("batch_kl_ref", 0.0620)
    # 기준값은 plateau(step 675~1499) 평균이라 초기 구간에 적용하면 오발동한다 - 실측으로
    # step 0의 batch_KL은 0.0222로 임계 0.0310 아래이고, step 120~150도 0.036으로 여유가
    # 17%뿐이다. 그래서 (1) start_frac 이전에는 감시하지 않고, (2) 연속 위반을 요구한다.
    g_start = int(guard.get("start_frac", 0.2) * max_steps)
    g_patience = guard.get("patience", 3)
    guard_warned = False
    guard_strikes = 0
    last_ok_eval_step = -1      # 중단 보고용 - 감시를 통과한 마지막 평가 step
    last_ok_eval_step = -1      # 중단 보고용 - 감시를 통과한 마지막 평가 step
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
    # 가속 스위치. 셋 다 기본 off이며 그때 기존 run과 동일한 경로다.
    #   batch_views: K개 student 뷰를 forward 1회로 묶는다(수학적 동치, 부동소수점 누적만 다름).
    #   tf32: A6000(Ampere)의 TF32 텐서코어를 matmul에 허용한다. torch 2.x 기본값이 False라
    #         지금까지 순수 FP32로 돌았다. mantissa 10비트로 bf16(7비트)보다 보수적이지만
    #         기존 측정치와 bit-identical은 아니다.
    batch_views = cfg["train"].get("batch_views", False)
    if cfg["train"].get("tf32", False):
        # TF32는 Ampere(SM 8.0) 이상에서만 의미가 있다. 그 아래에서는 조용히 무시되므로
        # 켰다고 믿고 넘어가지 않도록 여기서 확인하고 로그에 남긴다.
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        if cap >= (8, 0):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"[train] TF32 matmul 활성화 (SM {cap[0]}.{cap[1]})")
        else:
            logger.info(f"[train] TF32 요청됐으나 이 GPU(SM {cap[0]}.{cap[1]})는 Ampere 미만 - 무시")

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
    data_order = cfg["train"].get("data_order", "sample")
    batch_sampler = BatchIndexSampler(len(sentences), batch_size, data_order)
    if data_order == "epoch":
        n_needed = max_steps * batch_size
        logger.info(f"[train] data_order=epoch: {n_needed:,}문장 필요 / {len(sentences):,}문장 보유 "
                    f"({n_needed / len(sentences):.2f} epoch)")

    for step in range(max_steps):
        seen_epoch = batch_sampler.epoch
        batch_idx = batch_sampler.next()
        if batch_sampler.epoch != seen_epoch:
            logger.info(f"[step {step}] DATA EPOCH {batch_sampler.epoch} 시작 (재셔플)")
        batch_sentences = [sentences[i] for i in batch_idx]
        enc = tokenizer(
            batch_sentences, truncation=True, max_length=max_tokens, padding=True,
            return_special_tokens_mask=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        special_mask = enc["special_tokens_mask"].bool().to(device) | (~attention_mask.bool())

        if warmup_teacher_temp is not None:
            teacher_temp = teacher_temp_schedule(step, warmup_teacher_temp, teacher_temp_static,
                                                 teacher_temp_warmup_steps, teacher_temp_shape,
                                                 tt_decay_start, tt_decay_end, tt_final, tt_decay_shape)
            if entropy_ctrl is not None and step >= entropy_ctrl.start_step:
                teacher_temp = entropy_ctrl.tau      # 인계 후에는 제어기가 온도를 쥔다
        else:
            teacher_temp = teacher_temp_static

        koleo_lambda = resolve_koleo_lambda(step, koleo_lambda_max, koleo_hold_steps, koleo_decay_steps, koleo_min_ratio)

        token_embeds = student.get_input_embeddings()(input_ids)
        views = aug(token_embeds, special_mask, step, attention_mask)

        tok_target = None
        with torch.no_grad():
            t_out = teacher(
                inputs_embeds=views.teacher_embeds, attention_mask=attention_mask,
                embed_push=embed_uniform_push.push, return_all_hidden=token_predictor is not None,
            )
            t_embedding, t_logits = t_out[0], t_out[1]
            if token_predictor is not None:
                # teacher는 깨끗한 원문(t=0)을 보므로 마스크 위치의 teacher latent가 공짜 목표다.
                tok_target = token_latent_targets(t_out[4], tok_top_k)
        embed_push_grad_norm = embed_uniform_push.step(t_embedding)

        student_logits = []
        tok_losses, tok_coss = [], []
        student_embeds_norm = []   # pos_cos_raw 진단용(정규화된 pooled). r10에서 이 로깅만 이식.
        vel_losses = []
        koleo_losses = []
        cov_iso_losses = []
        cov_iso_mean_terms = []
        cov_iso_cov_terms = []
        # B2 drop 모드는 뷰마다 attention_mask가 다르다(잘린 span 제외). off면 원본 그대로.
        view_masks = (views.student_masks if views.student_masks is not None
                      else [attention_mask] * len(views.student_embeds))
        if batch_views and len(views.student_embeds) > 1:
            # K개 뷰를 배치 축으로 이어 붙여 forward 1회로 처리한다. 토큰 길이 중앙값이 27이라
            # 배치 32는 GPU에 너무 작아 커널 런치와 낮은 점유율로 시간을 버린다.
            # 수학적으로는 뷰별 forward와 동일하다(배치 간 결합 연산이 없음) - 부동소수점
            # 누적 순서만 달라져 bit-identical은 아니다. 그래서 기본값은 off다.
            cat_out = student(inputs_embeds=torch.cat(views.student_embeds, dim=0),
                              attention_mask=torch.cat(view_masks, dim=0))
            fwd = list(zip(*(x.chunk(len(views.student_embeds), dim=0) for x in cat_out[:3])))
        else:
            fwd = None
        for k, s_embeds in enumerate(views.student_embeds):
            s_mask = view_masks[k]
            if fwd is not None:
                s_embedding, s_logits, s_hidden = fwd[k]
            else:
                s_embedding, s_logits, s_hidden, *_ = student(inputs_embeds=s_embeds, attention_mask=s_mask)
            student_logits.append(s_logits)
            student_embeds_norm.append(s_embedding)
            if tok_target is not None and views.student_token_masks is not None \
                    and views.student_token_masks[k] is not None:
                l_tok, c_tok = token_latent_loss(token_predictor(s_hidden), tok_target,
                                                 views.student_token_masks[k])
                tok_losses.append(l_tok)
                tok_coss.append(c_tok)
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
        with torch.no_grad():
            # 노이즈 student 뷰 vs 깨끗한 teacher의 pooled 코사인. 뷰가 pooled 수준에서
            # 실제로 어려운지를 보는 지표다(R12의 핵심 진단). 1에 가까울수록 공짜 positive.
            aux["pos_cos_raw"] = torch.stack(
                [F.cosine_similarity(z_s, t_embedding, dim=-1) for z_s in student_embeds_norm]
            ).mean()

        if entropy_ctrl is not None:
            # 이번 스텝의 온도는 이미 위에서 확정됐다 - 여기서 만드는 tau는 다음 스텝용이다.
            entropy_ctrl.observe(aux["H_pt"].item())
            entropy_ctrl.update(step)
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
        if tok_losses:
            l_tok = torch.stack(tok_losses).mean()
            total_loss = total_loss + tok_lambda * l_tok
            aux["L_tok"] = l_tok.detach()
            aux["tok_cos"] = torch.stack(tok_coss).mean()

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
                "pos_cos_raw": aux["pos_cos_raw"].item(),
            }
            if entropy_ctrl is not None:
                log["tau_t"] = entropy_ctrl.tau
                log["H_ema"] = entropy_ctrl.h_ema
                tgt = entropy_ctrl.target(step)
                if tgt is not None:
                    log["H_target"] = tgt
            if "L_vel" in aux:
                log["L_vel"] = aux["L_vel"].item()
            if "L_tok" in aux:
                log["L_tok"] = aux["L_tok"].item()
                log["tok_cos"] = aux["tok_cos"].item()     # 마스크 위치 예측-목표 코사인 (정확도 대리)
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

            if guard_on and step >= g_start:
                # cov_iso는 응축(rank 붕괴)은 막지만 p_bar_t가 한 점으로 쏠리는 경로는 못 막는다.
                hbar = log["H_p_bar_t"]
                bad_h = hbar < g_hbar - 0.30
                bad_kl = log["batch_KL"] < g_bkl * 0.5
                if bad_h or bad_kl:
                    guard_strikes += 1
                    why = (f"H_p_bar_t={hbar:.4f} < {g_hbar - 0.30:.4f}" if bad_h
                           else f"batch_KL={log['batch_KL']:.4f} < {g_bkl * 0.5:.4f}")
                    logger.info(f"[step {step}] COLLAPSE STRIKE {guard_strikes}/{g_patience} {why}")
                    if guard_strikes >= g_patience:
                        logger.info(f"[step {step}] COLLAPSE ABORT {why} "
                                    f"(마지막 정상 평가 step={last_ok_eval_step})")
                        break
                else:
                    guard_strikes = 0        # 연속이 끊기면 초기화 - 단발 노이즈로 중단하지 않는다
                if not guard_warned and hbar < g_hbar - 0.15:
                    guard_warned = True
                    logger.info(f"[step {step}] COLLAPSE WARN H_p_bar_t={hbar:.4f} < {g_hbar - 0.15:.4f}")

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
            if guard_on and step >= g_start:
                if eff_rank < g_rank * 0.7:
                    logger.info(f"[step {step}] COLLAPSE ABORT eff_rank={eff_rank:.2f} < {g_rank * 0.7:.2f} "
                                f"(마지막 정상 평가 step={last_ok_eval_step})")
                    break
                last_ok_eval_step = step

    if cfg["train"].get("save_checkpoint", True):
        ckpt_path = ROOT / "checkpoints" / cfg["run_name"] / "last.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": student.state_dict(),
            "model_cfg": cfg["model"],
            "teacher_state_dict": teacher.model.state_dict(),
        }
        if token_predictor is not None:
            ckpt["token_predictor_state_dict"] = token_predictor.state_dict()
        torch.save(ckpt, ckpt_path)
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
