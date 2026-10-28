"""Opt-in collapse detection calibrated to the current base, never a global default."""
import math


class CollapseGuard:
    def __init__(self, cfg, max_steps):
        self.enabled = cfg.get("enabled", False)
        self.start = int(cfg.get("start_frac", 0.2) * max_steps)
        self.patience = cfg.get("patience", 3)
        self.strikes = 0
        if not self.enabled:
            return
        keys = ("h_p_bar_t_ref", "eff_rank_ref", "batch_kl_ref")
        if any(k not in cfg or not math.isfinite(cfg[k]) or cfg[k] < 0 for k in keys):
            raise ValueError("collapse_guard needs explicit finite reference values from its own base plateau")
        if self.patience < 1 or not 0 <= cfg.get("start_frac", 0.2) <= 1:
            raise ValueError("invalid collapse_guard patience/start_frac")
        self.hbar, self.rank, self.bkl = (cfg[k] for k in keys)

    def observe_train(self, step, hbar, bkl):
        if not self.enabled or step < self.start:
            return None
        reason = None
        if not math.isfinite(hbar) or hbar < self.hbar - 0.30:
            reason = f"H_p_bar_t={hbar:.4f} threshold={self.hbar - 0.30:.4f}"
        elif not math.isfinite(bkl) or bkl < self.bkl * 0.5:
            reason = f"batch_KL={bkl:.4f} threshold={self.bkl * 0.5:.4f}"
        self.strikes = self.strikes + 1 if reason else 0
        return reason if self.strikes >= self.patience else None

    def observe_rank(self, step, rank):
        if self.enabled and step >= self.start and (not math.isfinite(rank) or rank < self.rank * 0.7):
            return f"eff_rank={rank:.2f} threshold={self.rank * 0.7:.2f}"
        return None
