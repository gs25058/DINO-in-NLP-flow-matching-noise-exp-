"""Single-factor interventions from a fixed step: causal diagnosis of the late dev drop.

Config ``intervention: {kind, from_step[, t_lo, t_max]}``. Absent key → nothing runs. Before
``from_step`` training is bit-identical to the run without the key (same RNG draws, same values).
Each kind changes exactly one factor and consumes no random numbers:

- ``sharpen_off``: final teacher temperature = student temperature (no sharpening).
- ``center_freeze``: DINO center momentum 1 (center stops moving; targets still use it).
- ``iso_off``: mean and covariance weights 0.
- ``teacher_freeze``: teacher EMA momentum 1 (teacher stops tracking the student).
- ``noise``: noise range set to [t_lo, t_max] with the curriculum finished.
- ``lr_zero``: every learning rate 0 (the student stops; early-stopping reference).
"""

KINDS = ("sharpen_off", "center_freeze", "iso_off", "teacher_freeze", "noise", "lr_zero")


class Intervention:
    def __init__(self, cfg):
        self.kind = cfg["kind"]
        self.start = int(cfg["from_step"])
        if self.kind not in KINDS:
            raise ValueError(f"intervention kind must be one of {KINDS}")
        if self.start < 1:
            raise ValueError("intervention from_step must be positive")
        if self.kind == "noise":
            self.t_lo, self.t_max = float(cfg["t_lo"]), float(cfg["t_max"])
            if not 0 <= self.t_lo <= self.t_max <= 1:
                raise ValueError("noise intervention needs 0 <= t_lo <= t_max <= 1")
        elif set(cfg) - {"kind", "from_step"}:
            raise ValueError("only the noise intervention takes extra keys")
        self.applied = False

    def active(self, step):
        return step >= self.start

    def begin(self, trainer, step):
        """One-shot state changes at the first intervened step, before its forward pass."""
        if self.applied or not self.active(step):
            return
        self.applied = True
        if self.kind == "center_freeze":
            trainer.objective.dino.center_momentum = 1.0
        elif self.kind == "iso_off":
            trainer.objective.mean_weight = 0.0
            trainer.objective.cov_weight = 0.0
        elif self.kind == "noise":
            augment = trainer.augment
            augment.t_lo, augment.t_start, augment.t_max = self.t_lo, self.t_max, self.t_max
        elif self.kind == "lr_zero":
            trainer.scheduler.base_lrs = [0.0 for _ in trainer.scheduler.base_lrs]
            for group in trainer.optimizer.param_groups:
                group["lr"] = 0.0

    def teacher_temp(self, step, temperature, loss_cfg):
        if self.kind == "sharpen_off" and self.active(step):
            return float(loss_cfg["student_temp"])
        return temperature

    def teacher_momentum(self, step, momentum):
        if self.kind == "teacher_freeze" and self.active(step):
            return 1.0
        return momentum
