"""Word-prototype DINO heads: search space, k-means cache, and RNG-free diagnostics.

Nothing here draws random numbers during training. Diagnostics read tensors the
training step already produced, or run a no_grad clean teacher pass at evaluation.
"""

import copy
import json
import math

import torch
from torch.nn import functional as F

from src import joint_search
from src.provenance import append_record

# Five fixed probe sentences; the first two are named in the request
# (the second is the known association error, "carrying a boy" → pregnancy words).
PROBE_SENTENCES = (
    "A woman is riding on a horse.",
    "A woman is carrying a boy.",
    "A man with a hard hat is dancing.",
    "The stock market fell sharply after the announcement.",
    "A cat is playing with a ball of yarn.",
)
# Student copy rate is binned by that view's interpolation time t.
T_BINS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0001))
TOP_K = 10

CONDITIONS = {
    "C1": {"head_type": "vocab", "head_freeze": False, "logit_dim": 30522},
    "C2": {"head_type": "vocab", "head_freeze": True, "logit_dim": 30522},
    "C3": {"head_type": "vocab_kmeans", "head_freeze": False, "logit_dim": 8192},
}

# Common axes: the coviso joint space that produced C0 (t13), minus its three
# absolute temperatures, which word logits make meaningless (scale ≈ 5–10x).
COMMON = {k: v for k, v in joint_search.SPACE.items()
          if k not in ("loss.teacher_temp", "loss.warmup_teacher_temp", "loss.student_temp", "train.head_lr")}
TEMPERATURE = {
    # teacher = ratio × student, so the teacher is never softer than the student.
    "_teacher_ratio": (0.1, 1.0, True),
    # initial (warmup) teacher temperature = ratio × final teacher temperature.
    "_warmup_ratio": (0.2, 1.0, False),
}
STUDENT_TEMP = {"C1": (0.3, 2.0, True), "C2": (0.3, 2.0, True), "C3": (0.05, 2.0, True)}
HEAD = {
    # Pretrained head stays in the backbone learning-rate family.
    "C1": {"_head_lr_mult": (0.25, 4.0, True)},
    "C2": {},
    # Random transform + k-means decoder: the random-head learning-rate range of C0's search.
    "C3": {"train.head_lr": joint_search.SPACE["train.head_lr"]},
}


def space(condition):
    return {**COMMON, "loss.student_temp": STUDENT_TEMP[condition], **TEMPERATURE, **HEAD[condition]}


def configure(base, condition, params, kmeans_path=None):
    """Bounds-checked config for one condition; head keys come only from CONDITIONS."""
    declared = space(condition)
    if set(params) != set(declared):
        raise ValueError("parameters must match the declared space")
    cfg = copy.deepcopy(base)
    for key, value in params.items():
        lo, hi, _ = declared[key]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"parameter outside declared search space: {key}={value}")
        if not key.startswith("_"):
            section, name = key.split(".")
            cfg[section][name] = float(value)
    aug = cfg["augment"]
    aug["t_max"] = min(0.99, aug["t_lo"] + params["_t_span"])
    aug["t_start"] = aug["t_lo"] + (aug["t_max"] - aug["t_lo"]) * params["_t_start_frac"]
    loss = cfg["loss"]
    loss["teacher_temp"] = params["_teacher_ratio"] * loss["student_temp"]
    loss["warmup_teacher_temp"] = params["_warmup_ratio"] * loss["teacher_temp"]
    train = cfg["train"]
    if condition == "C1":
        train["head_lr"] = train["lr"] * params["_head_lr_mult"]
    elif condition == "C2":
        train["head_lr"] = train["lr"]  # no head parameters are optimized
    spec = CONDITIONS[condition]
    cfg["model"]["head"] = {"bottleneck_dim": 768, "logit_dim": spec["logit_dim"],
                            "head_type": spec["head_type"], "head_freeze": spec["head_freeze"]}
    if spec["head_type"] == "vocab_kmeans":
        if kmeans_path is None:
            raise ValueError("C3 needs the k-means cache path")
        cfg["model"]["head"]["kmeans_path"] = str(kmeans_path)
    return cfg


def anchors(base, condition):
    """Two enqueued points: t13 (C0) values for every common axis, with word-scale temperatures.

    The second also uses grad_clip 3.0 (the value named in the request) and a sharper student.
    """
    t13 = joint_search.baseline_parameters(base)
    common = {k: t13[k] for k in COMMON}
    warmup = base["loss"]["warmup_teacher_temp"] / base["loss"]["teacher_temp"]
    head = {"C1": {"_head_lr_mult": 1.0}, "C2": {},
            "C3": {"train.head_lr": float(base["train"]["head_lr"])}}[condition]
    if condition == "C3":
        first = {"loss.student_temp": float(base["loss"]["student_temp"]),
                 "_teacher_ratio": base["loss"]["teacher_temp"] / base["loss"]["student_temp"]}
        second = {"loss.student_temp": 1.0, "_teacher_ratio": 0.5}
    else:
        first = {"loss.student_temp": 1.0, "_teacher_ratio": 0.5}
        second = {"loss.student_temp": 0.5, "_teacher_ratio": 0.5}
    points = [{**common, **first, "_warmup_ratio": warmup, **head},
              {**common, **second, "_warmup_ratio": warmup, **head, "train.grad_clip": 3.0}]
    for point in points:
        configure(base, condition, point, kmeans_path="/unused" if condition == "C3" else None)
    return points


def _sentence_token_sets(ids, special, labels):
    """Per sentence: the set of prototype indices its content tokens map to."""
    mapped = ids if labels is None else labels[ids]
    return [set(row[~mask].tolist()) for row, mask in zip(mapped.cpu(), special.cpu())]


def copy_rate(scores, token_sets, k=TOP_K):
    top = scores.topk(k, dim=-1).indices.cpu().tolist()
    return [sum(i in tokens for i in row) / k for row, tokens in zip(top, token_sets)]


def pair_kl(logits, temperature):
    """Symmetric KL over every ordered pair of different sentences; deterministic."""
    log_p = F.log_softmax(logits / temperature, dim=-1)
    p = log_p.exp()
    cross = p @ log_p.t()  # cross[i, j] = Σ p_i log p_j
    self_term = cross.diagonal()
    kl = self_term[:, None] - cross  # KL(p_i || p_j)
    n = len(p)
    off = ~torch.eye(n, dtype=torch.bool, device=p.device)
    return ((kl + kl.t()) / 2)[off].mean()


class VocabDiagnostics:
    def __init__(self, trainer, root):
        self.trainer = trainer
        head_cfg = trainer.cfg["model"]["head"]
        self.k = head_cfg["logit_dim"]
        self.labels, self.representatives = None, None
        if head_cfg["head_type"] == "vocab_kmeans":
            cache = torch.load(root / head_cfg["kmeans_path"], map_location="cpu", weights_only=True)
            self.labels = cache["labels"].to(trainer.device)
            self.representatives = cache["representatives"]
        student = trainer.student
        self.initial_head = {n: p.detach().clone() for n, p in student.head.named_parameters()}
        self.initial_embeddings = student.get_input_embeddings().weight.detach().clone()
        self.probe = trainer.tokenizer(list(PROBE_SENTENCES), padding=True, return_tensors="pt",
                                       return_special_tokens_mask=True)
        self.batch = None

    @torch.no_grad()
    def step_scalars(self, result):
        v = result.vocab
        ids, special, times = v["batch"]
        tokens = _sentence_token_sets(ids, special, self.labels)
        loss_cfg = self.trainer.cfg["loss"]
        centered = v["teacher_logits"] - v["center"]
        out = {
            "H_pt_norm": result.losses.diagnostics["H_pt"].item() / math.log(self.k),
            "teacher_pair_KL": pair_kl(centered, result.teacher_temp).item(),
            "copy_rate_teacher": sum(copy_rate(centered, tokens)) / len(tokens),
        }
        rates = {b: [] for b in T_BINS}
        for logits, t in zip(result.student_logits, times):
            for rate, value in zip(copy_rate(logits.detach(), tokens), t.tolist()):
                for lo, hi in T_BINS:
                    if lo <= value < hi:
                        rates[(lo, hi)].append(rate)
        for (lo, hi), values in rates.items():
            if values:
                out[f"copy_rate_student_t{lo:.2f}_{min(hi, 1.0):.2f}"] = sum(values) / len(values)
                out[f"copy_count_student_t{lo:.2f}_{min(hi, 1.0):.2f}"] = len(values)
        out["teacher_student_temp_ratio"] = result.teacher_temp / loss_cfg["student_temp"]
        return out

    @torch.no_grad()
    def eval_scalars(self, step):
        trainer = self.trainer
        student = trainer.student
        out = {}
        for name, parameter in student.head.named_parameters():
            initial = self.initial_head[name]
            out[f"head_drift_{name}"] = ((parameter - initial).norm() / initial.norm().clamp_min(1e-12)).item()
        embeddings = student.get_input_embeddings().weight
        out["input_embedding_drift"] = (
            (embeddings - self.initial_embeddings).norm() / self.initial_embeddings.norm()).item()
        out["decoder_shares_input_storage"] = float(
            student.head.decoder.weight.data_ptr() == embeddings.data_ptr())
        self._top_words(step)
        return out

    def _top_words(self, step):
        trainer, enc = self.trainer, self.probe
        device = trainer.device
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        teacher = trainer.teacher.model
        output = teacher(inputs_embeds=teacher.get_input_embeddings()(ids), attention_mask=mask)
        centered = output.logits - trainer.objective.dino.center
        tokens = _sentence_token_sets(ids, enc["special_tokens_mask"].bool().to(device) | ~mask.bool(), self.labels)
        rows = []
        for sentence, scores, rate in zip(PROBE_SENTENCES, centered, copy_rate(centered, tokens)):
            top = scores.topk(TOP_K).indices.tolist()
            if self.labels is None:
                words = trainer.tokenizer.convert_ids_to_tokens(top)
            else:
                words = ["/".join(trainer.tokenizer.convert_ids_to_tokens(
                    [j for j in self.representatives[i].tolist() if j >= 0]) or ["<empty>"]) for i in top]
            rows.append({"sentence": sentence, "top": words, "copy_rate": rate})
        append_record(trainer.run_dir / "vocab_topwords.jsonl", {"loop_step": step, "rows": rows})


def kmeans_word_embeddings(weight, k=8192, iterations=100, seed=0, device="cpu"):
    """Euclidean k-means (k-means++ init) on the word-embedding rows; deterministic for a seed."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = weight.double().to(device)
    n = len(x)
    squared = (x * x).sum(1)
    first = torch.randint(n, (1,), generator=generator).item()
    centers = [first]
    closest = (squared - 2 * x @ x[first] + squared[first]).clamp_min(0)
    for _ in range(1, k):
        probabilities = (closest / closest.sum()).cpu()
        index = torch.multinomial(probabilities, 1, generator=generator).item()
        centers.append(index)
        closest = torch.minimum(closest, (squared - 2 * x @ x[index] + squared[index]).clamp_min(0))
    c = x[centers].clone()
    history = []
    for iteration in range(iterations):
        distances = squared[:, None] - 2 * x @ c.t() + (c * c).sum(1)[None]
        labels = distances.argmin(1)
        inertia = distances.gather(1, labels[:, None]).clamp_min(0).sum().item()
        counts = torch.bincount(labels, minlength=k)
        sums = torch.zeros_like(c).index_add_(0, labels, x)
        empty = counts == 0
        updated = torch.where(empty[:, None], c, sums / counts.clamp_min(1)[:, None].double())
        moved = (updated - c).norm(dim=1).max().item()
        c = updated
        history.append({"iteration": iteration, "inertia": inertia, "empty": int(empty.sum()), "max_shift": moved})
        if moved < 1e-7:
            break
    distances = squared[:, None] - 2 * x @ c.t() + (c * c).sum(1)[None]
    labels = distances.argmin(1)
    representatives = torch.full((k, 3), -1, dtype=torch.long)
    for cluster in range(k):
        members = (labels == cluster).nonzero().flatten()
        if len(members):
            order = distances[members, cluster].argsort()[:3]
            chosen = members[order]
            representatives[cluster, : len(chosen)] = chosen.cpu()
            if len(chosen) < 3:
                representatives[cluster, len(chosen):] = chosen[0].cpu()
    return {"centers": c.float().cpu(), "labels": labels.cpu(), "representatives": representatives,
            "history": json.dumps(history), "seed": seed, "k": k}
