import json

import pytest

from scripts.plot_dense_trajectory import aggregate, analyze, load_runs


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(dict(model="test", sts_b_dev=0.7, alignment=0.2,
                                            uniformity=-2, **row)) for row in rows))


def test_aggregation_uses_common_measured_steps_and_actual_seed_count(tmp_path):
    path = tmp_path / "measurements.jsonl"
    write_rows(path, [{"seed": seed, "optimizer_steps": step} for seed, steps in
                     [(7, [0, 2, 4]), (9, [0, 3, 4]), (11, [0, 4])] for step in steps])
    means = aggregate(load_runs(path), "test")
    assert [row["optimizer_steps"] for row in means] == [0, 4]
    assert all(row["n_seeds"] == 3 for row in means)


def test_appended_duplicate_run_is_rejected(tmp_path):
    path = tmp_path / "measurements.jsonl"
    write_rows(path, [{"seed": 7, "optimizer_steps": step} for step in [0, 2, 0]])
    with pytest.raises(ValueError, match="duplicate/non-increasing"):
        load_runs(path)


def test_analysis_preserves_input_and_refuses_existing_output(tmp_path):
    path = tmp_path / "measurements.jsonl"
    write_rows(path, [{"seed": 7, "optimizer_steps": step} for step in [0, 2]])
    original = path.read_bytes()
    out = tmp_path / "analysis"
    result = analyze(path, out)
    assert path.read_bytes() == original
    assert result["series"]["test"]["n_seeds"] == 1
    assert (out / "alignment_uniformity.svg").exists()
    with pytest.raises(FileExistsError):
        analyze(path, out)
