"""Phase B schedules: neutral is legacy, u=1 equals sharpen_off exactly, freeze is exact; campaign logic."""

import json
import threading

import optuna
import pytest

from scripts import run_late_schedule as runner
from src import late_schedule as ls
from src import vocab_proto as vp
from src.provenance import write_json
from src.train import late_teacher_temp, load_config

BASE = load_config("configs/vocab_proto/base.yaml")


def params(condition, **schedule):
    return {**vp.anchors(BASE, condition)[0], **ls.NEUTRAL, **schedule}


def test_neutral_schedule_is_the_vocab_configuration_exactly():
    for c in ls.CONDITIONS:
        p = params(c)
        assert ls.configure(BASE, c, p) == vp.configure(BASE, c, {k: p[k] for k in vp.space(c)})


def test_full_relaxation_switch_equals_the_sharpen_off_intervention():
    cfg = ls.configure(BASE, "C2", params("C2", _late_relax=1.0, _late_start_frac=0.25, _late_len_frac=0.0))
    loss = cfg["loss"]
    assert loss["teacher_temp_late_ratio"] == 1.0
    assert late_teacher_temp(999, 0.37, loss, 4000) == 0.37
    assert late_teacher_temp(1000, 0.37, loss, 4000) == float(loss["student_temp"])


def test_cosine_relaxation_is_monotone_between_scheduled_and_target():
    cfg = ls.configure(BASE, "C1", params("C1", _late_relax=0.5, _late_start_frac=0.25, _late_len_frac=0.25))
    loss = cfg["loss"]
    tau_t = loss["teacher_temp"]
    target = loss["teacher_temp_late_ratio"] * loss["student_temp"]
    assert tau_t < target < loss["student_temp"]
    values = [late_teacher_temp(s, tau_t, loss, 4000) for s in range(900, 2200, 50)]
    assert values[0] == tau_t and values[-1] == pytest.approx(target)
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))
    assert late_teacher_temp(1500, tau_t, loss, 4000) == pytest.approx((tau_t + target) / 2)


def test_freeze_key_only_when_scheduled_and_space_bounds():
    assert "teacher_freeze_frac" not in ls.configure(BASE, "C1", params("C1"))["train"]
    cfg = ls.configure(BASE, "C1", params("C1", **{"train.teacher_freeze_frac": 0.25}))
    assert cfg["train"]["teacher_freeze_frac"] == 0.25 and "teacher_temp_late_ratio" not in cfg["loss"]
    with pytest.raises(ValueError):
        ls.configure(BASE, "C1", params("C1", _late_relax=1.5))


class FakeChildren:
    def __init__(self):
        self.stopping = threading.Event()

    def stop(self):
        self.stopping.set()


def test_campaign_search_confirm_ablate_report(tmp_path, monkeypatch):
    campaign = tmp_path / "phase_b"
    for name in ("results/logs", "results/analysis"):
        (campaign / name).mkdir(parents=True)
    write_json(campaign / "source_manifest.json", {"report": str(tmp_path / "r.md"), "data_root": str(tmp_path)})
    write_json(campaign / "state.json", {"stage": "search"})
    monkeypatch.setattr(runner, "TRIALS", 6)
    previous = {c: {n: {"params": vp.anchors(BASE, c)[0]} for n in ("9", "3", "28", "16", "29", "14")} for c in ls.CONDITIONS}
    calls, lock = [], threading.Lock()

    def fake_train(campaign_, manifest, children, gpu, cfg, label, seed, smoke=False, trial=None, report="best"):
        cond = label.split("_")[1]
        with lock:
            calls.append((label, seed, report))
        if label.startswith("search") and label.endswith("_t5"):
            raise optuna.TrialPruned()
        relax = cfg["loss"].get("teacher_temp_late_ratio", 0.0)
        score = {"C0": 0.73, "C1": 0.76, "C2": 0.77}[cond] + 0.01 * relax + 0.0001 * (seed % 7)
        if label.startswith("off"):
            score -= 0.02
        name = f"{label}_s{seed}"
        log = campaign_ / "results/logs" / name
        log.mkdir(parents=True, exist_ok=True)
        (log / "evaluations.jsonl").write_text("".join(
            json.dumps({"loop_step": s, "sts_b_dev": score}) + "\n" for s in (0, 3500, 3750, 3999)))
        return {"run_name": name, "label": label, "seed": seed, "condition": cond, "metrics": {"best_sts": score}}

    monkeypatch.setattr(runner.vproto, "train", fake_train)
    monkeypatch.setattr(runner, "check_reproduction", lambda *a: None)
    monkeypatch.setattr(runner, "audit", lambda *a: None)
    # Four slots: two concurrent trials per condition (constant-liar TPE).
    manifest = {"gpus": ["a", "b", "c", "d"], "kmeans_path": "/k", "data_root": str(tmp_path)}
    records = []
    found = runner.search(campaign, manifest, FakeChildren(), BASE, previous, records)
    assert {c: len(v) for c, v in found.items()} == {"C1": 6, "C2": 6}
    assert all(r == "current" for label, _, r in calls if label.startswith("search"))
    assert found["C2"]["0"]["params"]["_late_relax"] == 1.0 and found["C2"]["2"]["params"]["_late_relax"] == 0.0
    groups, selected = runner.confirm(campaign, manifest, FakeChildren(), BASE, records, found)
    assert set(groups) == {"C0", "C1", "C2"} and set(groups["C0"]) == {"101", "102", "103"}
    assert json.loads((campaign / "selection.json").read_text())["test_scores_seen"] is False
    runner.ablate(campaign, manifest, FakeChildren(), BASE, records, found, selected, groups)
    assert set(groups) == {"C0", "C1", "C2", "C1_off", "C2_off"}
    for g, seeds in groups.items():
        for s, r in seeds.items():
            for w, delta in (("best", 0.002), ("last", 0.0)):
                value = {"C0": 0.670, "C1": 0.690, "C2": 0.692, "C1_off": 0.675, "C2_off": 0.676}[g] + delta
                write_json(campaign / "results/analysis" / f"{g}_s{s}_{w}.json", {
                    "runs": {r["run_name"]: {"avg_7task": value}},
                    "experiment": {"group": g, "seed": int(s), "which": w}})
    write_json(campaign / "state.json", {"stage": "complete"})
    runner.render(campaign)
    text = (campaign / "report.md").read_text()
    assert "**R1 C1 vs C0 (STS7 raw last):** **성립**" in text
    assert "**R2 C2 후반 하락 해소" in text and "**성립**" in text.split("**R2 C2")[1].split("\n")[0]
    assert "**R3 C1 스케줄 자체 효과" in text and "**성립**" in text.split("**R3 C1")[1].split("\n")[0]
