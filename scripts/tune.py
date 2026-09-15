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
    # 스케줄 자체(값 범위/방식/기간)를 전면 탐색. 이 프로젝트에서 스케줄 축은 사실상 한 번도
    # 튜닝된 적이 없다 - R5에서 "warmup을 넣으면 좋아진다"까지만 확인하고 값은 DINO 원본
    # 기본값(0.082->0.11, 0.997->0.9995, frac 0.3)을 그대로 써왔다.
    #
    # 핵심 동기 두 가지:
    # (1) 교락 해제 - train.warmup_frac(LR)과 loss.teacher_temp_warmup_frac이 둘 다 0.3으로
    #     묶여 있어 warmup 종료(step 450)에 생기는 전환이 LR 때문인지 온도 때문인지 구분이
    #     불가능했다. 두 축을 독립적으로 탐색해 분리한다.
    # (2) 타이밍 불일치 - 학습의 실질적 이득은 step ~250에서 대부분 끝나는데(모든 1500-step
    #     run에서 관찰) warmup은 450까지 이어진다. 기간이 짧은 쪽이 유리한지 확인한다.
    #
    # 주의: 이 공간은 max_steps=1500(배포 길이)에서 돌려야 한다. 모든 파라미터가
    # frac*max_steps라서 750-step 예산으로 튜닝하면 절대 step이 절반이 되어 결과가 전이되지
    # 않는다(cov_iso t18에서 750-step 우위가 1500-step에서 사라진 전례도 있다).
    #     uv run python scripts/tune.py --base-config <챔피언 config> \
    #         --space r5_schedules_full --max-steps 1500 --n-trials 40 --n-jobs 2 --gpus <idle>
    "r5_schedules_full": {
        # 값 범위
        "loss.warmup_teacher_temp": ("float", {"low": 0.04, "high": 0.10}),
        "loss.teacher_temp": ("float", {"low": 0.09, "high": 0.20}),
        "train.momentum_end": ("float", {"low": 0.998, "high": 0.9999}),
        # 기간
        "loss.teacher_temp_warmup_frac": ("float", {"low": 0.05, "high": 0.6}),
        "train.warmup_frac": ("float", {"low": 0.05, "high": 0.5}),
        "train.momentum_ramp_frac": ("float", {"low": 0.2, "high": 1.0}),
        # 방식
        "loss.teacher_temp_shape": ("categorical", {"choices": ["linear", "cosine"]}),
        "train.momentum_shape": ("categorical", {"choices": ["linear", "cosine"]}),
    },
    # KoLeo(loss.koleo_lambda)와 lrsplit 안정화 축(train.lr/head_lr/grad_clip)이 각각 단독
    # 효과가 거의 그대로 누적되는 것을 수동 grid(r6_bert_koleo_lrsplit_a/b/c, lr=2e-4 고정)
    # 에서 확인했다 - 이 4개를 joint로 탐색해 개별 최적점의 조합이 grid의 최고값(raw
    # STS-B dev 0.666, koleo_lambda=0.2 + lr=2e-4)을 넘는지 확인한다.
    # base config는 configs/r5d_bert_lrsplit_c.yaml 권장 (exclude_ln_bias_wd=true 등
    # lrsplit 안정화 구조가 이미 갖춰져 있고, train.lr/head_lr/grad_clip은 아래 탐색값으로
    # 덮어써진다). train.lr 상한(5.4e-4)은 Table 1 원값 - BERT에서 이 값은 lrsplit
    # 안정화 스택 없이는 붕괴했었다(r5d_bert_lrsplit_c까지의 안정화 축 참고).
    "koleo_lrsplit": {
        "loss.koleo_lambda": ("float", {"low": 0.01, "high": 0.5, "log": True}),
        "train.lr": ("float", {"low": 3.0e-5, "high": 5.4e-4, "log": True}),
        "train.head_lr": ("float", {"low": 1.0e-4, "high": 1.0e-3, "log": True}),
        "train.grad_clip": ("float", {"low": 1.0, "high": 6.0}),
    },
    # R7 cov_iso의 "제대로 된" 검증용 공간. 수동 grid(r7_coviso_bert_lam{0.5,1,2}, EMA
    # momentum/start_step 고정)에서는 λ를 4배 늘려도 STS가 0.607~0.612에 갇혀 게이트가
    # 걸렸는데, 그게 방법 자체의 한계인지 파라미터 선택 문제인지 구분하려면 cov_iso 자체
    # 파라미터(λ/EMA momentum/시작 시점)와 안정화 축(lr/head_lr/grad_clip)을 joint로 넓게
    # 봐야 한다.
    # base config는 koleo_lrsplit 탐색과 동일하게 configs/r5d_bert_lrsplit_c.yaml을 쓴다 -
    # KoLeo가 최고값(trial 27, sts_b_dev=0.6822@750)을 낸 것과 정확히 같은 조건(같은 base,
    # 같은 lr/head_lr/grad_clip 축)에서 정규화 항만 바꿔 비교하기 위함. 이 base는
    # koleo_lambda를 설정하지 않아 기본값 0.0 - KoLeo 오염 없음.
    # λ 상한(8.0)은 수동 grid 최대값(2.0)의 4배 - "더 세게 걸면 되는가"를 확인하는 범위.
    "cov_iso_full": {
        "loss.cov_iso_lambda": ("float", {"low": 0.1, "high": 8.0, "log": True}),
        "loss.cov_ema_momentum": ("float", {"low": 0.80, "high": 0.999}),
        "loss.cov_iso_start_step": ("int", {"low": 0, "high": 200}),
        "train.lr": ("float", {"low": 3.0e-5, "high": 5.4e-4, "log": True}),
        "train.head_lr": ("float", {"low": 1.0e-4, "high": 1.0e-3, "log": True}),
        "train.grad_clip": ("float", {"low": 1.0, "high": 6.0}),
    },
    # 노이즈 분포 자체. 이 프로젝트에서 augment.* 는 단 한 번도 탐색된 적이 없다 - 지금까지의
    # study 4개(146 trial)는 전부 정규화 항(koleo/cov_iso/uniform_push)과 최적화·스케줄
    # (lr/head_lr/grad_clip/teacher_temp/momentum)만 건드렸다. 정작 CLAUDE.md가 탐색을
    # 허용한 축("노이즈 관련 신규 하이퍼파라미터만 탐색 대상")이 비어 있었다.
    #
    # 챔피언 config의 노이즈 설정은 프로젝트 시작부터 t ~ U(0.35, 0.5) 고정이고
    # warmup_frac=0.0이라 curriculum(t_start->t_max)은 아예 꺼져 있다 - t_start/t_max가
    # 죽은 값이다. 이 범위의 근거는 어디에도 없다.
    #
    # r10b의 t 제어기가 "양성 쌍 난이도가 일정하게 유지되는 t"를 스스로 찾아 0.63에
    # 평형한 것도 현재 중앙값 0.425가 낮게 잡혔을 가능성을 시사한다(align 항과 교락돼
    # 있어 강한 증거는 아니다).
    #
    # 파라미터화 주의: t ~ U(t_lo, t_hi(step))라 t_lo < t_hi가 항상 성립해야 한다.
    # t_max/t_start를 독립 샘플링하면 무효 조합이 생기므로, 폭(_t_span)과 비율
    # (_t_start_frac)로 받아 DERIVED에서 유효한 값으로 환산한다(아래 _derive_flow_noise_t).
    # 밑줄로 시작하는 키는 config 경로가 아니라 파생용 의사 파라미터다.
    #
    # max_steps는 반드시 1500(배포 길이)로 돌릴 것 - warmup_frac이 frac*max_steps라서
    # 750-step 예산에서는 절대 step이 절반이 되어 결과가 전이되지 않는다(r5_schedules_full
    # 주석의 전례와 동일한 함정).
    # R13 가설별 튜닝. 손으로 고른 값 때문에 가설이 기각된 것이 아님을 확인하려는 것이므로,
    # base는 r8이 아니라 현 챔피언(r12_bert_noise_optuna_t35, 2시드 0.7333)이다 - 각 기제를
    # "지금 실제로 도달한 상태" 위에서 평가한다.
    #
    # A: 엔트로피 제어기. R13 매트릭스에서 delta 0.066/0.152를 손으로 골랐고 둘 다 무반응이었다.
    # 도달 가능 범위가 실측상 -0.66 nats까지이므로 상한을 0.8로 열고, 개입 시점·기간·이득도 함께 본다.
    "r13_sharpen_ctrl": {
        "loss.entropy_ctrl_delta": ("float", {"low": 0.02, "high": 0.8, "log": True}),
        "loss.entropy_ctrl_gain": ("float", {"low": 0.05, "high": 1.0, "log": True}),
        "loss.entropy_ctrl_start_frac": ("float", {"low": 0.15, "high": 0.70}),
        "_ctrl_span_frac": ("float", {"low": 0.10, "high": 0.80}),   # end = min(1.0, start + span)
    },
    # B-cutoff: R13에서 유일하게 alignment/uniformity/rank를 동시에 개선한 기제.
    # span 비율만 손으로 골랐고(0.1) prob/mode는 고정이었다 - 셋 다 연다.
    "r13_view_cutoff": {
        "augment.cutoff_span_frac": ("float", {"low": 0.02, "high": 0.35}),
        "augment.cutoff_prob": ("float", {"low": 0.3, "high": 1.0}),
        "augment.cutoff_mode": ("categorical", {"choices": ["drop", "mask"]}),
    },
    # B-rho: 상관 노이즈. rho를 넣으면 유효 난이도가 바뀌는데 챔피언의 t는 rho=0에서 튜닝된
    # 값이라, t를 통째로 재탐색하지 않고 배율 하나만 열어 난이도만 재조정하게 한다
    # (t 모양까지 다시 찾으면 노이즈 study를 중복하게 되고 10 trial로는 과소표본이다).
    "r13_view_corr": {
        "augment.noise_corr_rho": ("float", {"low": 0.05, "high": 1.0}),
        "_t_scale": ("float", {"low": 0.6, "high": 1.3}),
    },
    "flow_noise_t": {
        "augment.t_lo": ("float", {"low": 0.0, "high": 0.6}),
        "_t_span": ("float", {"low": 0.05, "high": 0.6}),      # t_max = t_lo + span (1.0 상한 clip)
        "_t_start_frac": ("float", {"low": 0.1, "high": 1.0}),  # 1.0이면 램프 없음(t_start == t_max)
        "augment.warmup_frac": ("float", {"low": 0.0, "high": 0.6}),
        "augment.num_student_views": ("int", {"low": 2, "high": 4}),
    },
}


def _derive_flow_noise_t(cfg: dict, pseudo: dict) -> None:
    """폭/비율로 받은 의사 파라미터를 t_max/t_start로 환산한다.

    t_lo < t_start <= t_max <= 1.0 이 구성상 보장되므로 무효 trial이 생기지 않는다.
    _t_start_frac=1.0이면 t_start == t_max라 curriculum이 사실상 꺼진 상태(현 챔피언과 동일)."""
    t_lo = cfg["augment"]["t_lo"]
    t_max = min(1.0, t_lo + pseudo["_t_span"])
    cfg["augment"]["t_max"] = t_max
    cfg["augment"]["t_start"] = t_lo + (t_max - t_lo) * pseudo["_t_start_frac"]


def _derive_r13_sharpen_ctrl(cfg: dict, pseudo: dict) -> None:
    """end_frac을 start_frac + span으로 만들어 end < start 무효 조합을 없앤다."""
    start = cfg["loss"]["entropy_ctrl_start_frac"]
    cfg["loss"]["entropy_ctrl_end_frac"] = min(1.0, start + pseudo["_ctrl_span_frac"])


def _derive_r13_view_corr(cfg: dict, pseudo: dict) -> None:
    """base(챔피언)의 t 범위에 배율 하나를 곱한다. 모양은 유지하고 난이도만 조정."""
    a = cfg["augment"]
    scale = pseudo["_t_scale"]
    a["t_lo"] = min(0.95, a["t_lo"] * scale)
    a["t_max"] = min(1.0, a["t_max"] * scale)
    a["t_start"] = min(a["t_start"] * scale, a["t_max"])
    if a["t_lo"] >= a["t_max"]:                 # 배율로 뒤집히는 일은 없지만 방어적으로
        a["t_lo"] = a["t_max"] * 0.5


#: space_name -> 파생 파라미터 환산 함수. 없는 space는 의사 파라미터도 없다.
DERIVED = {
    "flow_noise_t": _derive_flow_noise_t,
    "r13_sharpen_ctrl": _derive_r13_sharpen_ctrl,
    "r13_view_corr": _derive_r13_view_corr,
}


def set_by_path(cfg: dict, dotted: str, value) -> None:
    """"loss.uniform_push_lr" 같은 점(dot) 경로로 중첩 dict 값을 설정한다."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def parse_collapsed_sts(log_path: Path) -> float | None:
    """붕괴 감시로 중단된 run의 마지막 유효 sts_b_dev를 돌려준다(중단이 아니면 None).

    중단된 trial을 pruned로 버리면 TPE가 그 영역을 "값 없음"으로 보고 계속 재탐색한다.
    낮은 점수로 기록해야 탐색이 붕괴 영역을 피해 간다. 평가가 한 번도 없었으면 None이고,
    그때는 호출부가 아주 낮은 sentinel을 준다.
    """
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    if "COLLAPSE ABORT" not in text:
        return None
    hits = re.findall(r"EVAL sts_b_dev=([-\d.eE+]+)", text)
    return float(hits[-1]) if hits else None


def parse_final_sts(log_path: Path, min_step: int = 0) -> float | None:
    """train.log에서 마지막 'EVAL sts_b_dev=...' 값을 읽는다. 없으면 None(실패/미완주).

    min_step: 마지막 EVAL이 이 step 미만이면 None을 반환한다(중도 사망 처리). 이게 없으면
    step 0 EVAL만 남기고 죽은 run이 "사전학습 초기값(BERT 기준 0.5931)"을 정상 점수로
    보고해버려서, Optuna가 그 영역을 '탐색했는데 나쁨'으로 학습한다 - 실제로
    tune_r5d_bert_lrsplit_c_cov_iso_full study의 40 trial 중 11개가 이렇게 기록됐다.
    """
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"\[step (\d+)\] EVAL sts_b_dev=([-\d.eE+]+)", text)
    if not matches:
        return None
    last_step, last_sts = matches[-1]
    if int(last_step) < min_step:
        return None
    return float(last_sts)


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
              seed: int, gpu_pool: GpuPool, timeout: int, study_name: str) -> float:
    cfg = copy.deepcopy(base_cfg)
    pseudo: dict = {}
    for dotted, (kind, kwargs) in SEARCH_SPACES[space_name].items():
        if kind == "float":
            value = trial.suggest_float(dotted, **kwargs)
        elif kind == "int":
            value = trial.suggest_int(dotted, **kwargs)
        elif kind == "categorical":
            value = trial.suggest_categorical(dotted, **kwargs)
        else:
            raise ValueError(f"unknown suggest kind: {kind}")
        if dotted.startswith("_"):
            pseudo[dotted] = value      # config 경로가 아닌 파생용 값
        else:
            set_by_path(cfg, dotted, value)
    if space_name in DERIVED:
        DERIVED[space_name](cfg, pseudo)

    # study_name(base config stem + search space)을 그대로 재사용한다 - base_cfg['run_name']만
    # 쓰면 어떤 search space를 탐색 중인지(예: koleo_lambda)가 trial 이름에서 사라진다.
    cfg["run_name"] = f"{study_name}_t{trial.number}"
    cfg["train"]["max_steps"] = max_steps
    cfg["train"]["save_checkpoint"] = False  # trial 결과는 train.log의 sts_b_dev만으로 충분 - .pt는 아무도 안 읽음
    cfg["seed"] = seed
    # train.py가 학습 종료 시 TensorBoard HPARAMS 탭에 이 trial을 기록하도록 전달한다.
    cfg["_optuna"] = {"study_name": study_name, "trial_number": trial.number, "params": dict(trial.params)}

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
    # train.py는 step == max_steps-1에서 반드시 EVAL을 찍으므로, 마지막 EVAL이 거기 못 미치면
    # 완주하지 못한 run이다(중도 사망) - 점수 대신 pruned로 처리한다.
    sts = parse_final_sts(log_path, min_step=max_steps - 1)
    if sts is None:
        # 붕괴 감시로 중단된 run은 "실패"가 아니라 "나쁜 설정"이다. 마지막 유효 평가값으로
        # 채점해야 TPE가 그 영역을 피한다 - pruned로 버리면 값이 없어 계속 재탐색한다.
        collapsed = parse_collapsed_sts(log_path)
        if collapsed is not None:
            print(f"[tune] trial {trial.number}: COLLAPSED, 마지막 유효 sts_b_dev={collapsed:.4f} "
                  f"params={trial.params}")
            return collapsed
        if "COLLAPSE ABORT" in (log_path.read_text(errors="ignore") if log_path.exists() else ""):
            print(f"[tune] trial {trial.number}: COLLAPSED (평가 이력 없음) params={trial.params}")
            return 0.0      # 평가 한 번도 못 한 붕괴 - 최저점
        print(f"[tune] trial {trial.number} FAILED/INCOMPLETE params={trial.params}\n"
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
        lambda trial: run_trial(
            trial, base_cfg, args.space, args.max_steps, args.seed, gpu_pool, args.timeout, study_name
        ),
        n_trials=args.n_trials, n_jobs=args.n_jobs,
    )

    print(f"[tune] best value: {study.best_value:.4f}")
    print(f"[tune] best params: {study.best_params}")

    out_csv = ROOT / "results" / "analysis" / f"{study_name}_trials.csv"
    study.trials_dataframe().to_csv(out_csv, index=False)
    print(f"[tune] wrote {out_csv}")


if __name__ == "__main__":
    main()
