"""Optuna 하이퍼파라미터 탐색.

각 trial은 base config를 복사해 탐색 공간의 값을 덮어쓴 임시 yaml을 만들고,
`uv run python -m src.train --config <임시 yaml> --max-steps <budget>`을 서브프로세스로
실행한 뒤, 그 run의 results/logs/<run_name>/train.log에서 마지막 EVAL 줄의
sts_b_dev를 목적함수 값으로 파싱해 Optuna에 보고한다(maximize).

CLAUDE.md 원칙("기존 Table 1 하이퍼파라미터는 근거 없이 변경 금지 - 노이즈/신규
하이퍼파라미터만 탐색 대상")에 따라 SEARCH_SPACES는 이 프로젝트에서 새로 도입한
파라미터로만 구성한다.

주의: --max-steps는 base config의 실제 max_steps보다 짧게(기본 750) 잡아 trial당
비용을 낮췄다 - 이 프로젝트의 핵심 실패 모드(후반 붕괴)는 보통 step 500~1000 사이에
이미 드러나므로 750은 실용적 타협점이지만, 완전히 충실한 탐색을 원하면
--max-steps를 base config의 실제 길이(예: 1500)로 올릴 것 (trial당 비용 ~2배).

실행:
    uv run python scripts/tune.py --base-config configs/r2_uniform_push.yaml \
        --space uniform_push --n-trials 20 --max-steps 750 --n-jobs 4 --gpus 0,1,2,3
"""
import argparse
import copy
import os
import queue
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train import load_config  # noqa: E402  (extends 체인 해석 재사용)

ROOT = Path(__file__).resolve().parent.parent

# 탐색 공간: {space_name: {"cfg.dotted.path": ("float"|"int"|"categorical", suggest_* kwargs)}}
SEARCH_SPACES: dict[str, dict[str, tuple[str, dict]]] = {
    "uniform_push": {
        "loss.uniform_push_lr": ("float", {"low": 0.05, "high": 5.0, "log": True}),
    },
    "uniform_push_wide": {
        "loss.uniform_push_lr": ("float", {"low": 0.05, "high": 5.0, "log": True}),
        "loss.center_momentum": ("float", {"low": 0.80, "high": 0.99}),
        "loss.teacher_temp": ("float", {"low": 0.06, "high": 0.15}),
    },
    "r5_schedules": {
        "loss.teacher_temp": ("float", {"low": 0.09, "high": 0.20}),
        "loss.teacher_temp_warmup_frac": ("float", {"low": 0.1, "high": 0.5}),
        "train.momentum_end": ("float", {"low": 0.998, "high": 0.9999}),
    },
}


def set_by_path(cfg: dict, dotted: str, value) -> None:
    """"loss.uniform_push_lr" 같은 점(dot) 경로로 중첩 dict 값을 설정한다."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def parse_final_sts(log_path: Path) -> float | None:
    """train.log에서 마지막 'EVAL sts_b_dev=...' 값을 읽는다. 없으면 None(실패/미완주)."""
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"EVAL sts_b_dev=([-\d.eE+]+)", text)
    return float(matches[-1]) if matches else None


class GpuPool:
    """스레드 안전한 GPU id 풀. n_jobs>1일 때 trial마다 GPU를 하나씩 빌렸다 반납한다."""

    def __init__(self, gpu_ids: list[int]):
        self.q: queue.Queue[int] = queue.Queue()
        for g in gpu_ids:
            self.q.put(g)

    def acquire(self) -> int:
        return self.q.get()

    def release(self, gpu_id: int) -> None:
        self.q.put(gpu_id)


def run_trial(trial: optuna.Trial, base_cfg: dict, space_name: str, max_steps: int,
              seed: int, gpu_pool: GpuPool, timeout: int) -> float:
    cfg = copy.deepcopy(base_cfg)
    for dotted, (kind, kwargs) in SEARCH_SPACES[space_name].items():
        if kind == "float":
            value = trial.suggest_float(dotted, **kwargs)
        elif kind == "int":
            value = trial.suggest_int(dotted, **kwargs)
        elif kind == "categorical":
            value = trial.suggest_categorical(dotted, **kwargs)
        else:
            raise ValueError(f"unknown suggest kind: {kind}")
        set_by_path(cfg, dotted, value)

    cfg["run_name"] = f"{base_cfg['run_name']}_optuna_t{trial.number}"
    cfg["train"]["max_steps"] = max_steps
    cfg["seed"] = seed

    tmp_path = None
    gpu_id = gpu_pool.acquire()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(ROOT / "configs")
        ) as f:
            yaml.safe_dump(cfg, f)
            tmp_path = f.name

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.train", "--config", tmp_path],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout,
        )
    finally:
        gpu_pool.release(gpu_id)
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)

    log_path = ROOT / "results" / "logs" / cfg["run_name"] / "train.log"
    sts = parse_final_sts(log_path)
    if sts is None:
        print(f"[tune] trial {trial.number} FAILED params={trial.params}\n"
              f"--- stderr tail ---\n{result.stderr[-2000:]}")
        raise optuna.TrialPruned()
    print(f"[tune] trial {trial.number}: sts_b_dev={sts:.4f} params={trial.params}")
    return sts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--space", default="uniform_push", choices=list(SEARCH_SPACES))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=750,
                         help="trial당 학습 step 예산 (base config의 실제 max_steps보다 짧게 권장)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1, help="동시 실행 trial 수 (GPU 개수 이하로)")
    parser.add_argument("--gpus", default="0", help="쉼표로 구분된 GPU id 목록, 앞에서 n_jobs개 사용")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--timeout", type=int, default=3600, help="trial 하나당 최대 대기(초)")
    args = parser.parse_args()

    base_cfg = load_config(Path(args.base_config))

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    gpu_pool = GpuPool(gpu_ids[: max(args.n_jobs, 1)])

    study_name = args.study_name or f"tune_{Path(args.base_config).stem}_{args.space}"
    (ROOT / "results" / "analysis").mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{ROOT}/results/analysis/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize", load_if_exists=True
    )

    study.optimize(
        lambda trial: run_trial(trial, base_cfg, args.space, args.max_steps, args.seed, gpu_pool, args.timeout),
        n_trials=args.n_trials, n_jobs=args.n_jobs,
    )

    print(f"[tune] best value: {study.best_value:.4f}")
    print(f"[tune] best params: {study.best_params}")

    out_csv = ROOT / "results" / "analysis" / f"{study_name}_trials.csv"
    study.trials_dataframe().to_csv(out_csv, index=False)
    print(f"[tune] wrote {out_csv}")


if __name__ == "__main__":
    main()
