"""Admission decisions and cross-launcher pending reservations; no GPU calls."""

import os

import pytest

from scripts.gpu_guard import admission_reason, registry, select_gpu


@pytest.fixture
def gpus():
    return {str(i): {"uuid": f"GPU-{i}", "util": 0, "free": 48000} for i in range(4)}


def test_idle_gpu_admits_but_foreign_or_blocked_owner_does_not(gpus):
    assert admission_reason("0", gpus, [], [], "me") is None
    for owner in ("gs25009", "gs25049", "someone_else"):
        processes = [{"uuid": "GPU-0", "pid": 4, "owner": owner}]
        assert admission_reason("0", gpus, processes, [], "me") is not None


def test_account_limit_counts_evaluation_and_jobs_on_other_gpus(gpus):
    processes = [{"uuid": f"GPU-{i}", "pid": i + 10, "owner": "me"} for i in (1, 2)]
    assert "limit" in admission_reason("0", gpus, processes, [], "me")


def test_pending_starts_count_without_double_counting_running_jobs(gpus):
    reservations = [{"uuid": "GPU-1"}, {"uuid": "GPU-2"}]
    assert "limit" in admission_reason("0", gpus, [], reservations, "me")
    processes = [{"uuid": "GPU-1", "pid": 10, "owner": "me"}]
    assert admission_reason("0", gpus, processes, reservations[:1], "me") is None
    assert admission_reason("1", gpus, [], reservations[:1], "me") is not None


def test_memory_and_utilization_are_both_required(gpus):
    gpus["0"]["free"] = 34000
    assert admission_reason("0", gpus, [], [], "me") is not None
    gpus["0"].update(free=48000, util=98)
    assert admission_reason("0", gpus, [], [], "me") is not None


def test_auto_selection_excludes_blocked_occupied_and_reserved_gpus(gpus):
    processes = [
        {"uuid": "GPU-0", "pid": 10, "owner": "gs25049"},
        {"uuid": "GPU-1", "pid": 11, "owner": "other"},
    ]
    assert select_gpu("auto", gpus, processes, [{"uuid": "GPU-2"}], "me") == ("3", None)
    selected, reason = select_gpu(
        "auto", gpus, processes, [{"uuid": "GPU-2"}, {"uuid": "GPU-3"}], "me"
    )
    assert selected is None
    assert "account GPU process limit" in reason
    assert select_gpu("0", gpus, processes, [], "me")[0] is None


def test_registry_persists_pending_jobs_across_invocations(tmp_path):
    reservation = {"guard_pid": os.getpid(), "child_pid": os.getpid(), "uuid": "GPU-1"}
    with registry(tmp_path) as rows:
        rows.append(reservation)
    with registry(tmp_path) as rows:
        assert rows == [reservation]
        rows.clear()
    with registry(tmp_path) as rows:
        assert rows == []


def test_timeout_stops_only_the_launched_child_and_releases_reservation(
    tmp_path, monkeypatch, gpus
):
    import sys

    from scripts import gpu_guard

    monkeypatch.setattr(gpu_guard, "snapshot", lambda: (gpus, []))
    monkeypatch.setattr(gpu_guard, "registry", lambda _: registry(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu_guard",
            "--gpu",
            "0",
            "--timeout",
            "0.05",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(5)",
        ],
    )
    real_popen = gpu_guard.subprocess.Popen
    children = []

    def launch(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(gpu_guard.subprocess, "Popen", launch)
    term_handler, int_handler = (
        gpu_guard.signal.getsignal(gpu_guard.signal.SIGTERM),
        gpu_guard.signal.getsignal(gpu_guard.signal.SIGINT),
    )
    try:
        assert gpu_guard.main() == 124
    finally:
        gpu_guard.signal.signal(gpu_guard.signal.SIGTERM, term_handler)
        gpu_guard.signal.signal(gpu_guard.signal.SIGINT, int_handler)
    assert len(children) == 1 and children[0].poll() is not None
    with registry(tmp_path) as rows:
        assert rows == []
