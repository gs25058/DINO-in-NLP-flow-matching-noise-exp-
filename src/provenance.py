"""Frozen source snapshots and per-run data, initialization, and execution evidence."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import torch
import yaml


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def git_info(root):
    def run(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": run("status", "--short"),
        "diff": run("diff", "HEAD", "--binary"),
    }


def record_run(root, cfg, student, device, argv):
    out = root / "results" / "logs" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    source_manifest = cfg.get("experiment", {}).get("source_manifest")
    source = json.loads(Path(source_manifest).read_text()) if source_manifest else git_info(root)
    data = {}
    for key in ("sentences_path", "rank_eval_path", "embed_stats_path"):
        path = root / cfg["data"][key]
        row = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".jsonl":
            with path.open("rb") as stream:
                row["records"] = sum(1 for _ in stream)
        data[key] = row
    info = {
        "source": source,
        "seed": cfg["seed"],
        "command": argv,
        "device": str(device),
        "initial_student_sha256": model_fingerprint(student),
        "data": data,
        "python": platform.python_version(),
        "libraries": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "datasets", "numpy", "scipy")
        },
        "precision": {
            "amp": False,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
        "fresh_pretrained_start": True,
    }
    if str(device).startswith("cuda"):
        info["gpu"] = {
            "name": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
        }
    (out / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def append_record(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def prepare(campaign, workspace, gpus, report):
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / "source"
    source.mkdir()
    for directory in ("src", "scripts", "configs", "tests"):
        shutil.copytree(
            workspace / directory,
            source / directory,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
    for name in (
        "pyproject.toml",
        "uv.lock",
        "METHOD.md",
        "CLAUDE.md",
        "GPU_POLICY.md",
    ):
        shutil.copy2(workspace / name, source / name)
    for name in ("OPTUNA_SPEC.md", "baseline_provenance.json", "validation.json"):
        shutil.copy2(workspace / "docs/experiments/coviso_relax" / name, campaign / name)
    files = {
        str(path.relative_to(source)): sha256_file(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    manifest = {
        "git": git_info(workspace),
        "workspace": str(workspace),
        "campaign": str(campaign),
        "gpus": gpus,
        "report": str(report),
        "created_unix": time.time(),
        "source_files": files,
        "spec_sha256": sha256_file(campaign / "OPTUNA_SPEC.md"),
        "baseline": read_json(campaign / "baseline_provenance.json"),
    }
    write_json(campaign / "source_manifest.json", manifest)
    (campaign / "source_diff.patch").write_text(manifest["git"]["diff"] + "\n")
    for name in ("results", "checkpoints"):
        target = campaign / name
        target.mkdir()
        (source / name).symlink_to(target, target_is_directory=True)
    for relative in files:
        (source / relative).chmod(0o444)
    (campaign / "configs").mkdir()
    (campaign / "commands").mkdir()
    (campaign / "status").mkdir()
    write_json(campaign / "state.json", {"stage": "prepared", "metrics": {}, "commands": []})
    return manifest


def verify_source(campaign, manifest):
    for relative, expected in manifest["source_files"].items():
        if sha256_file(campaign / "source" / relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")


def read_records(path):
    """Ignore a final line that another process is still writing."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines(keepends=True)
        if line.endswith("\n")
    ]
