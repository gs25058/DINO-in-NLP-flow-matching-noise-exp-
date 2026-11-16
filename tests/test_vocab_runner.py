"""Campaign logic with fake training: budgets, pruning/failure, selection frozen before test, report."""

import json
import threading

import optuna
import pytest

from scripts import report_vocab_proto as report
from scripts import run_vocab_proto as runner
from src.provenance import write_json
from src.train import load_config


class FakeChildren:
    def __init__(self):
        self.stopping = threading.Event()

    def stop(self):
        self.stopping.set()


def _write_logs(path, quality):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "components.jsonl").open("w") as stream:
        for step in range(0, 4000, 10):
            row = {"loop_step": step, "H_pt": 3.0, "H_pt_norm": 0.3, "batch_KL": 2.0, "teacher_pair_KL": 5.0,
                   "copy_rate_teacher": 0.5, "copy_rate_student_t0.50_0.70": 0.6, "copy_count_student_t0.50_0.70": 50,
                   "copy_rate_student_t0.85_1.00": 0.2, "copy_count_student_t0.85_1.00": 50}
            stream.write(json.dumps(row) + "\n")
        stream.write(json.dumps({**row, "loop_step": 3999}) + "\n")
    with (path / "evaluations.jsonl").open("w") as stream:
        for step in (0, 2000, 3999):
            stream.write(json.dumps({"loop_step": step, "sts_b_dev": quality, "eff_rank": 300.0}) + "\n")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    c = tmp_path / "vocab_campaign"
    for name in ("configs", "commands", "status", "results/analysis", "results/logs"):
        (c / name).mkdir(parents=True)
    report_path = tmp_path / "report.md"
    write_json(c / "source_manifest.json", {"report": str(report_path), "data_root": str(tmp_path)})
    write_json(c / "state.json", {"stage": "prepared"})
    c0 = tmp_path / runner.C0_CAMPAIGN / "results/logs"
    for seed in runner.FINAL_SEEDS:
        _write_logs(c0 / runner.C0_RUN.format(seed=seed), 0.74)
    return c


def test_search_confirmation_selection_and_report(campaign, monkeypatch):
    monkeypatch.setattr(runner, "TRIALS", 5)
    base = load_config("configs/vocab_proto/base.yaml")
    manifest = {"gpus": ["a", "b"], "kmeans_path": "/k.pt", "data_root": str(campaign.parent)}
    calls, lock = [], threading.Lock()

    def fake_train(campaign_, manifest_, children, gpu, cfg, label, seed, smoke=False, trial=None):
        condition = label.split("_")[1]
        n = int(label.split("_t")[-1]) if "_t" in label else 0
        with lock:
            calls.append((label, seed, cfg["loss"]["teacher_temp"] <= cfg["loss"]["student_temp"]))
        if label.startswith("search") and condition == "C2" and n == 3:
            raise optuna.TrialPruned()
        if label.startswith("search") and condition == "C3" and n == 4:
            raise RuntimeError("nonfinite loss")
        quality = 0.70 + 0.01 * n + {"C1": 0.02, "C2": 0.0, "C3": 0.01}[condition] + 0.001 * (seed - 42)
        name = runner.run_name(campaign_, label, seed)
        _write_logs(campaign_ / "results/logs" / name, quality)
        return {"run_name": name, "label": label, "seed": seed, "condition": condition,
                "metrics": {"best_sts": quality, "last_sts": quality}, "initial_backbone_sha256": f"b{seed}",
                "initial_student_sha256": f"{condition}{seed}", "first_views": {"v": seed},
                "augment_sha256": str(cfg["augment"]), "batch_order": f"o{seed}", "data": {"d": "x"}}

    monkeypatch.setattr(runner, "train", fake_train)
    records = []
    found = runner.search(campaign, manifest, FakeChildren(), base, records)
    assert {c: len(v) for c, v in found.items()} == {"C1": 5, "C2": 5, "C3": 5}
    assert found["C2"]["3"]["state"] == "PRUNED" and found["C3"]["4"]["state"] == "FAIL"
    assert all(ok for _, _, ok in calls)
    # Anchors are trial 0 and 1 of every condition.
    anchor = runner.vp.anchors(base, "C1")[1]
    assert found["C1"]["1"]["params"] == pytest.approx(anchor)
    groups = runner.confirm(campaign, manifest, FakeChildren(), base, records, found)
    shortlist = json.loads((campaign / "shortlist.json").read_text())["trials"]
    assert shortlist["C1"] == ["4", "3", "2"] and "3" not in shortlist["C2"] and "4" not in shortlist["C3"]
    selection = json.loads((campaign / "selection.json").read_text())
    assert selection["test_scores_seen"] is False and selection["selected_trials"]["C1"] == "4"
    assert set(groups["C1"]) == {"42", "43", "44"}
    assert groups["C1"]["42"]["run_name"].endswith("search_C1_t4_s42")
    confirm_seeds = sorted({s for label, s, _ in calls if label.startswith("confirm")})
    assert confirm_seeds == [43, 44]
    # Fake test scores, then the report computes the pre-registered verdicts.
    for c in ("C0", "C1", "C2", "C3"):
        for s in runner.FINAL_SEEDS:
            for w in ("best", "last"):
                value = {"C0": 0.670, "C1": 0.680, "C2": 0.679, "C3": 0.6795}[c] + 0.0001 * s
                name = runner.C0_RUN.format(seed=s) if c == "C0" else groups[c][str(s)]["run_name"]
                write_json(campaign / "results/analysis" / f"{c}_s{s}_{w}.json", {
                    "runs": {name: {"avg_7task": value}},
                    "postprocessed": {name: {"center": {"avg_7task": value + .01},
                                             "center_pc2": {"avg_7task": value + .02}}},
                    "experiment": {"condition": c, "seed": s, "which": w}})
    write_json(campaign / "state.json", {"stage": "complete"})
    report.render(campaign)
    text = (campaign / "report.md").read_text()
    assert "**P1** C1 > C0 (STS7 raw, dev-best ckpt): **성립**" in text
    assert "**P2**" in text and "**성립**" in text.split("**P2**")[1].split("\n")[0]
    assert "prototype 방향의 의미가 핵심" in text
    assert "**P4** C1 − C2 (적응 효과): **판정불가**" in text
    assert "**P5**" in text and "**기각**" in text.split("**P5**")[1].split("\n")[0]
