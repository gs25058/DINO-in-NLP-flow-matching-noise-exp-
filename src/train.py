"""Current experiment: wiki1m → flow-noise views → DINO+mean/cov → dev selection.

Run through scripts/gpu_guard.py; see configs/coviso_relax/base.yaml.
"""

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.augment import FlowNoiseAug
from src.evaluate import EVAL_SPACES, effective_rank_metrics, sts_b_dev_metrics_by_space
from src.loss import DINOObjective, LossTerms, backbone_grad_norm, batch_kl_diagnostic
from src.model import DinoTextModel, EMATeacher
from src.provenance import append_record, record_run

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        for key, value in cfg.items():
            base[key] = (
                {**base[key], **value}
                if isinstance(base.get(key), dict) and isinstance(value, dict)
                else value
            )
        return base
    return cfg


def load_sentences(path):
    with Path(path).open() as stream:
        return [json.loads(line)["text"] for line in stream]


class BatchIndexSampler:
    """Shuffled epochs, dropping incomplete tails; uses the seeded Python RNG."""

    def __init__(self, n, batch_size, rng=random):
        if not 0 < batch_size <= n:
            raise ValueError("batch_size must fit the dataset")
        self.n, self.batch_size, self.rng = n, batch_size, rng
        self.epoch = 0
        self._reshuffle()

    def _reshuffle(self):
        self.indices = list(range(self.n))
        self.rng.shuffle(self.indices)
        self.cursor = 0

    def next(self):
        if self.cursor + self.batch_size > self.n:
            self.epoch += 1
            self._reshuffle()
        indices = self.indices[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return indices


def build_param_groups(student, lr, head_lr, weight_decay):
    """Backbone/head learning rates; bias and normalization parameters have no decay."""
    groups = []
    for module, rate in ((student.backbone, lr), (student.head, head_lr)):
        named = list(module.named_parameters())
        for decay in (True, False):
            parameters = [
                p
                for name, p in named
                if ("bias" not in name.lower() and "norm" not in name.lower()) == decay
            ]
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": rate,
                        "weight_decay": weight_decay if decay else 0.0,
                    }
                )
    return groups


def cosine_ramp(step, start, end, duration):
    progress = min(max(step / duration, 0.0), 1.0) if duration > 0 else 1.0
    return start + (end - start) * (1 - math.cos(math.pi * progress)) / 2


def setup_logger(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("flowdino.train")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    for handler in (
        logging.FileHandler(run_dir / "train.log"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


@dataclass
class TrainingStep:
    losses: LossTerms
    student_logits: list[torch.Tensor]
    teacher_temp: float


class Trainer:
    """Training computation, optimization, and evaluation have separate methods."""

    def __init__(self, cfg, device, logger, writer, root=ROOT):
        self.cfg, self.device, self.logger, self.writer, self.root = (
            cfg,
            device,
            logger,
            writer,
            root,
        )
        train, data, model, loss = (cfg[k] for k in ("train", "data", "model", "loss"))
        self.run_dir = root / "results/logs" / cfg["run_name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if (
            train["tf32"]
            and str(device).startswith("cuda")
            and torch.cuda.get_device_capability() >= (8, 0)
        ):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled; FP32 parameters, no AMP")
        self.tokenizer = AutoTokenizer.from_pretrained(model["backbone"])
        self.sentences = load_sentences(root / data["sentences_path"])
        self.rank_sentences = load_sentences(root / data["rank_eval_path"])
        self.student = DinoTextModel(
            model["backbone"],
            model["head"]["bottleneck_dim"],
            model["head"]["logit_dim"],
        ).to(device)
        self.teacher = EMATeacher(self.student, train["momentum_start"])
        record_run(root, cfg, self.student, device, sys.argv)
        stats = torch.load(root / data["embed_stats_path"], map_location="cpu", weights_only=True)
        self.augment = FlowNoiseAug(
            stats["mean"],
            stats["std"],
            **{k: cfg["augment"][k] for k in ("num_student_views", "t_lo", "t_start", "t_max")},
            warmup_steps=max(1, int(cfg["augment"]["warmup_frac"] * train["max_steps"])),
        )
        self.objective = DINOObjective(
            loss, self.student.backbone.config.hidden_size, model["head"]["logit_dim"]
        ).to(device)
        groups = build_param_groups(
            self.student, train["lr"], train["head_lr"], train["weight_decay"]
        )
        self.optimizer = torch.optim.AdamW(groups)
        self.parameters = [p for g in groups for p in g["params"]]
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=max(1, int(train["warmup_frac"] * train["max_steps"])),
            num_training_steps=train["max_steps"],
        )
        self.temp_steps = max(1, int(loss["teacher_temp_warmup_frac"] * train["max_steps"]))
        self.momentum_steps = max(1, int(train["momentum_ramp_frac"] * train["max_steps"]))
        self.gradient_steps = set(train["component_grad_steps"])
        self.best_sts, self.best_step = -float("inf"), -1

    def training_step(self, sentences, step) -> TrainingStep:
        enc = self.tokenizer(
            sentences,
            truncation=True,
            max_length=self.cfg["data"]["max_tokens"],
            padding=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        ids, mask = (
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
        )
        special = enc["special_tokens_mask"].bool().to(self.device) | ~mask.bool()
        token_embeds = self.student.get_input_embeddings()(ids)
        views = self.augment(token_embeds, special, step)
        teacher = self.teacher(inputs_embeds=views.teacher_embeds, attention_mask=mask)
        count = len(views.student_embeds)
        student = self.student(
            inputs_embeds=torch.cat(views.student_embeds, dim=0),
            attention_mask=torch.cat([mask] * count, dim=0),
        )
        embeddings, logits = (
            student.embedding.chunk(count, dim=0),
            list(student.logits.chunk(count, dim=0)),
        )
        loss_cfg = self.cfg["loss"]
        temperature = cosine_ramp(
            step,
            loss_cfg["warmup_teacher_temp"],
            loss_cfg["teacher_temp"],
            self.temp_steps,
        )
        losses = self.objective(teacher.logits, embeddings, logits, step, temperature)
        return TrainingStep(losses, logits, temperature)

    def optimize(self, result, step):
        losses, diagnostics = result.losses, {}
        if step in self.gradient_steps:
            parameters = [p for p in self.student.backbone.parameters() if p.requires_grad]
            diagnostics["grad_dino_backbone"] = backbone_grad_norm(losses.dino, parameters)
            for name in ("mean", "cov"):
                diagnostics[f"grad_{name}_backbone"] = (
                    backbone_grad_norm(
                        getattr(losses, f"lambda_{name}") * getattr(losses, name),
                        parameters,
                    )
                    if losses.active
                    else 0.0
                )
        self.optimizer.zero_grad()
        losses.total.backward()
        diagnostics["grad_norm_total"] = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.cfg["train"]["grad_clip"], error_if_nonfinite=True
        ).item()
        if not torch.isfinite(losses.total):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        self.optimizer.step()
        self.scheduler.step()
        train = self.cfg["train"]
        self.teacher.momentum = cosine_ramp(
            step, train["momentum_start"], train["momentum_end"], self.momentum_steps
        )
        self.teacher.update(self.student)
        return diagnostics

    def log_step(self, result, gradients, step):
        with torch.no_grad():
            batch_kl = batch_kl_diagnostic(
                result.student_logits,
                self.cfg["loss"]["student_temp"],
                self.cfg["eval"]["batch_kl_pairs"],
            )
        log = {
            **result.losses.scalars(),
            **gradients,
            "batch_KL": batch_kl.item(),
            "teacher_temp": result.teacher_temp,
            "teacher_momentum": self.teacher.momentum,
        }
        append_record(
            self.run_dir / "components.jsonl",
            {"loop_step": step, "optimizer_steps": step + 1, **log},
        )
        for key, value in log.items():
            self.writer.add_scalar(f"train/{key}", value, step)
        self.logger.info(f"[step {step}] " + " ".join(f"{k}={v:.6g}" for k, v in log.items()))

    def evaluate(self, step):
        metrics = sts_b_dev_metrics_by_space(self.student, self.tokenizer, self.device, EVAL_SPACES)
        final = metrics["embedding"]
        rank, max_sv = effective_rank_metrics(
            self.student, self.tokenizer, self.rank_sentences, self.device
        )
        log = {
            "sts_b_dev_spearman": final.spearman,
            "alignment": final.alignment,
            "uniformity": final.uniformity,
            "effective_rank": rank,
            "max_sv_ratio": max_sv,
        }
        for space in ("bottleneck", "head"):
            row = metrics[space]
            log.update(
                {
                    f"sts_b_dev_spearman_{space}": row.spearman,
                    f"alignment_{space}": row.alignment,
                    f"uniformity_{space}": row.uniformity,
                }
            )
        for key, value in log.items():
            self.writer.add_scalar(f"eval/{key}", value, step)
        self.logger.info(f"[step {step}] EVAL " + " ".join(f"{k}={v:.6g}" for k, v in log.items()))
        append_record(
            self.run_dir / "evaluations.jsonl",
            {
                "loop_step": step,
                "optimizer_steps": step + 1,
                "sts_b_dev": final.spearman,
                "alignment": final.alignment,
                "uniformity": final.uniformity,
                "eff_rank": rank,
                "max_sv_ratio": max_sv,
            },
        )
        if final.spearman > self.best_sts:
            self.best_sts, self.best_step = final.spearman, step
            self.save_checkpoint("best", step, final.spearman)
        return final.spearman

    def save_checkpoint(self, which, step, sts):
        path = self.root / "checkpoints" / self.cfg["run_name"] / f"{which}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.student.state_dict(),
                "teacher_state_dict": self.teacher.model.state_dict(),
                "model_cfg": self.cfg["model"],
                "train_cfg": self.cfg["train"],
                "loss_cfg": self.cfg["loss"],
                "cov_iso_state_dict": self.objective.covariance.state_dict(),
                "loop_step": step,
                "sts_b_dev": sts,
                "best_step": self.best_step,
                "best_sts": self.best_sts,
            },
            path,
        )

    def run(self):
        train, evaluation = self.cfg["train"], self.cfg["eval"]
        steps = min(train["max_steps"], train.get("stop_after_steps", train["max_steps"]))
        sampler = BatchIndexSampler(len(self.sentences), train["batch_size"])
        self.student.train()
        for step in range(steps):
            result = self.training_step([self.sentences[i] for i in sampler.next()], step)
            gradients = self.optimize(result, step)
            logging_step = step % evaluation["log_every_steps"] == 0 or step == steps - 1
            if logging_step:
                self.log_step(result, gradients, step)
            if step % evaluation["every_steps"] == 0 or step == steps - 1:
                sts = self.evaluate(step)
            if step in self.gradient_steps and not logging_step:
                append_record(
                    self.run_dir / "gradient_diagnostics.jsonl",
                    {"loop_step": step, **gradients},
                )
        self.save_checkpoint("last", step, sts)
        (self.run_dir / "completion.json").write_text(
            json.dumps(
                {
                    "status": "smoke_complete" if steps < train["max_steps"] else "complete",
                    "completed_steps": steps,
                    "best_sts": self.best_sts,
                    "best_loop_step": self.best_step,
                    "last_sts": sts,
                    "last_loop_step": step,
                },
                indent=2,
            )
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        help="stop a smoke run without compressing its schedules",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.stop_after_steps is not None:
        if args.stop_after_steps < 1:
            parser.error("stop-after-steps must be positive")
        cfg["train"]["stop_after_steps"] = args.stop_after_steps
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    run_dir = ROOT / "results/logs" / cfg["run_name"]
    logger = setup_logger(run_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with SummaryWriter(str(ROOT / "results/tensorboard" / cfg["run_name"] / timestamp)) as writer:
        try:
            Trainer(cfg, args.device, logger, writer).run()
        except Exception:
            logger.exception("training crashed")
            raise


if __name__ == "__main__":
    main()
