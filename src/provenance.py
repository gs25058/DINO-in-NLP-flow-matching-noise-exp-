"""Opt-in run provenance for controlled experiments."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

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
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "status": run("status", "--short"), "diff": run("diff", "HEAD", "--binary")}


def record_run(root, cfg, student, device, argv):
    out = root / "results" / "logs" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    source_manifest = cfg.get("experiment", {}).get("source_manifest")
    source = json.loads(Path(source_manifest).read_text()) if source_manifest else git_info(root)
    data = {}
    for key in ("sentences_path", "rank_eval_path", "embed_stats_path"):
        path = root / cfg["data"][key]
        row = {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        if path.suffix == ".jsonl":
            with path.open("rb") as stream:
                row["records"] = sum(1 for _ in stream)
        data[key] = row
    info = {"source": source, "seed": cfg["seed"], "command": argv, "device": str(device),
            "initial_student_sha256": model_fingerprint(student), "data": data,
            "python": platform.python_version(), "libraries": {p: importlib.metadata.version(p) for p in
            ("torch", "transformers", "datasets", "numpy", "scipy")},
            "precision": {"amp": False, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "cudnn_tf32": torch.backends.cudnn.allow_tf32},
            "fresh_pretrained_start": True}
    if str(device).startswith("cuda"):
        info["gpu"] = {"name": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()}
    (out / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def append_record(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
