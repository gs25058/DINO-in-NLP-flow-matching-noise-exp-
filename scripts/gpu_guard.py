"""Shared-server admission for a single GPU process; see GPU_POLICY.md.

Uses only stdlib. Admission and pending starts are serialized across launchers
of this account. Unmanaged jobs are included through nvidia-smi snapshots.
"""
import argparse
import csv
import fcntl
import json
import os
import pwd
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

BLOCKED_OWNERS = {"gs25009", "gs25049"}
MAX_PROCESSES = 2


def snapshot():
    def query(kind, fields):
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"], text=True)
        return list(csv.reader(output.splitlines(), skipinitialspace=True))

    gpus = {}
    for index, uuid, util, used, total in query("gpu", "index,uuid,utilization.gpu,memory.used,memory.total"):
        gpus[index] = {"uuid": uuid, "util": int(util), "free": int(total) - int(used)}
    processes = []
    for uuid, pid, _name, _memory in query("compute-apps", "gpu_uuid,pid,process_name,used_memory"):
        result = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True, check=False)
        owner = result.stdout.strip()
        if not owner:
            raise RuntimeError(f"cannot establish GPU process owner for PID {pid}; retry after refreshing")
        processes.append({"uuid": uuid, "pid": int(pid), "owner": owner})
    return gpus, processes


def admission_reason(gpu, gpus, processes, reservations, owner, min_free_mib=35000, max_util=30):
    """None admits the run; otherwise return a reason to wait/fail closed."""
    if gpu not in gpus:
        return f"GPU {gpu} not found"
    selected = gpus[gpu]
    own = {p["pid"] for p in processes if p["owner"] == owner}
    own_uuids = {p["uuid"] for p in processes if p["owner"] == owner}
    pending = sum(r["uuid"] not in own_uuids for r in reservations)
    if len(own) + pending >= MAX_PROCESSES:
        return "account GPU process limit (2), including pending starts"
    occupants = [p for p in processes if p["uuid"] == selected["uuid"]]
    if any(p["owner"] in BLOCKED_OWNERS for p in occupants):
        return "GPU is used by a blocked owner (gs25009/gs25049)"
    if occupants or any(r["uuid"] == selected["uuid"] for r in reservations):
        return "GPU already has a process or a pending reservation"
    if selected["free"] < min_free_mib or selected["util"] >= max_util:
        return f"GPU requires free memory >= {min_free_mib} MiB and utilization < {max_util}%"
    return None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def registry(directory):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or directory.stat().st_uid != os.getuid():
        raise RuntimeError(f"unsafe GPU reservation directory: {directory}")
    with (directory / "admission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "reservations.json"
        rows = json.loads(path.read_text()) if path.exists() else []
        rows = [r for r in rows if alive(r["guard_pid"]) and alive(r["child_pid"])]
        try:
            yield rows
        finally:
            path.write_text(json.dumps(rows))


def stop_child(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            child.wait()
            return
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", required=True, help="physical nvidia-smi GPU index")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=30)
    ap.add_argument("--min-free-mib", type=int, default=35000)
    ap.add_argument("--max-util", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=None, help="child runtime limit, seconds")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.poll_seconds <= 0 or (args.timeout is not None and args.timeout <= 0):
        ap.error("command and positive timeout/poll-seconds required")
    directory = Path(f"/tmp/flowdino-gpu-{os.getuid()}")
    child = None
    starting = False
    pending_signal = None

    def interrupted(signum, _frame):
        nonlocal pending_signal
        if starting:
            pending_signal = signum
            return
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while child is None:
            with registry(directory) as rows:
                gpus, processes = snapshot()
                reason = admission_reason(args.gpu, gpus, processes, rows, pwd.getpwuid(os.getuid()).pw_name,
                                          args.min_free_mib, args.max_util)
                if reason is None:
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[args.gpu]["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
                    # Defer interruption until cleanup has the child's PID.
                    starting = True
                    try:
                        child = subprocess.Popen(command, env=env, start_new_session=True)
                        rows.append({"uuid": gpus[args.gpu]["uuid"], "guard_pid": os.getpid(),
                                     "child_pid": child.pid})
                    finally:
                        starting = False
                    if pending_signal is not None:
                        raise SystemExit(128 + pending_signal)
            if child is None:
                print(f"[gpu_guard] GPU {args.gpu}: {reason}", file=sys.stderr, flush=True)
                if not args.wait:
                    return 2
                time.sleep(args.poll_seconds)
        try:
            return child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print("[gpu_guard] runtime timeout; stopping this job's process group", file=sys.stderr)
            return 124
    finally:
        if child is not None:
            stop_child(child)
            with registry(directory) as rows:
                rows[:] = [r for r in rows if r["guard_pid"] != os.getpid()]


if __name__ == "__main__":
    sys.exit(main())
