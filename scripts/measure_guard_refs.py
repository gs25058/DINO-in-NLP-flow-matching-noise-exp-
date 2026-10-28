"""Measure collapse references from completed base logs; performs no training.

Backported from origin/exp/r19-simcse-gram, with per-seed completeness checks
and source hashes so a different base/hardware cannot silently reuse references.
"""
import argparse
import hashlib
import re
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
NUMBER = r"[-+\d.eE]+"


def measure_refs(paths, max_steps, plateau_frac=0.45, start_frac=0.2):
    if max_steps < 1 or not 0 <= start_frac <= plateau_frac < 1:
        raise ValueError("require max_steps>0 and 0<=start_frac<=plateau_frac<1")
    hbar, bkl, guard_bkl, rank, sources = [], [], [], [], []
    for path in paths:
        contents = path.read_text()
        records = []
        for line in contents.splitlines():
            match = re.match(r"\[step (\d+)\] (loss|EVAL)", line)
            if match:
                records.append((int(match[1]), match[2],
                                {k: float(v) for k, v in re.findall(rf"(\w+)=({NUMBER})", line)}))
        if "COLLAPSE ABORT" in contents or not any(s == max_steps - 1 and k == "EVAL" for s, k, _ in records):
            raise ValueError(f"incomplete base run: {path}")
        own_h, own_b, own_g, own_rank = [], [], [], []
        for step, kind, values in records:
            if step >= max_steps:
                continue
            if kind == "loss":
                if step >= int(plateau_frac * max_steps):
                    own_h.append(values["H_p_bar_t"])
                    own_b.append(values["batch_KL"])
                if step >= int(start_frac * max_steps):
                    own_g.append(values["batch_KL"])
            elif step >= int(plateau_frac * max_steps):
                own_rank.append(values["eff_rank"])
        if not all((own_h, own_b, own_g, own_rank)):
            raise ValueError(f"insufficient plateau records: {path}")
        hbar.extend(own_h)
        bkl.extend(own_b)
        guard_bkl.extend(own_g)
        rank.extend(own_rank)
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not paths:
        raise ValueError("at least one base log is required")
    return {"enabled": True, "h_p_bar_t_ref": statistics.mean(hbar),
            "eff_rank_ref": statistics.mean(rank),
            "batch_kl_ref": min(statistics.mean(bkl), 2 * min(guard_bkl) / 1.47),
            "start_frac": start_frac, "patience": 3,
            "reference_max_steps": max_steps, "reference_logs": sources}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="base run without _s<seed>")
    ap.add_argument("--max-steps", required=True, type=int)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--plateau-frac", type=float, default=0.45)
    ap.add_argument("--start-frac", type=float, default=0.2)
    args = ap.parse_args()
    paths = [ROOT / "results" / "logs" / f"{args.run}_s{s}" / "train.log" for s in args.seeds]
    refs = measure_refs(paths, args.max_steps, args.plateau_frac, args.start_frac)
    print(yaml.safe_dump({"collapse_guard": refs}, sort_keys=False))


if __name__ == "__main__":
    main()
