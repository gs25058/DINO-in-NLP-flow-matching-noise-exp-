#!/usr/bin/env bash
# Optuna study 무인 감독 실행기 (space 무관 범용).
#
# "아무도 안 봐도 끝까지 완주"가 목표다:
#   1. 우리 계정의 다른 학습이 끝날 때까지 대기 (동시 GPU 2개 제한 준수)
#   2. base config: BASE_CONFIG로 지정하거나, 미지정 시 1500-step 완주 run 중 시드 평균 최고를 자동 선택
#   3. gs25009/gs25049가 쓰는 GPU를 제외하고, 진짜 유휴인 GPU만 선택
#      (util < 30% 그리고 여유 메모리 > 25GB - 여유 메모리만 보면 98% 점유 GPU를 고르게 된다)
#      2개를 못 찾으면 1개로라도 진행하고, 하나도 없으면 5분 후 재시도
#   4. tune.py 실행 - 중간에 죽으면 Optuna SQLite에서 이어받아 자동 재시작
#      (남은 trial 수만 다시 요청하므로 총 목표치를 넘기지 않는다)
#   5. 목표 trial 수를 채우거나 재시도 한도에 도달하면 종료
#
# 사용:
#   SPACE=flow_noise_t BASE_CONFIG=r8_bert_coviso_sched_optuna_t50 TARGET_TRIALS=40 \
#     setsid nohup bash scripts/run_study.sh > results/logs/noise_study_supervisor.log 2>&1 &
#
# (구 이름 run_schedule_study.sh - r5_schedules_full study 전용이었던 것을 일반화했다.)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET_TRIALS="${TARGET_TRIALS:-60}"
MAX_STEPS="${MAX_STEPS:-1500}"
SPACE="${SPACE:-r5_schedules_full}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
N_JOBS=2
# 절대 쓰지 않을 GPU 소유자 (사용자 지시: 팀원/타 연구실 GPU 회피)
BLOCKED_OWNERS="gs25009|gs25049"

log() { echo "[supervisor $(date '+%m-%d %H:%M:%S')] $*"; }

# ---------- 1. 우리 계정의 기존 학습이 끝날 때까지 대기 ----------
log "기존 학습 종료 대기 중..."
while pgrep -u "$(id -u)" -f "venv/bin/python3 -m src.train" >/dev/null 2>&1; do
  sleep 30
done
log "기존 학습 없음 - 진행"

source "$ROOT/scripts/env.sh" 2>/dev/null || true
export HF_HUB_OFFLINE=1

# ---------- 2. base config 자동 선택 (1500-step 완주한 run 중 시드 평균 최고) ----------
if [[ -n "${BASE_CONFIG:-}" ]]; then
  log "base config 지정됨: $BASE_CONFIG"
else
BASE_CONFIG="$(uv run python - <<'PY'
from pathlib import Path
import re, collections
LOG = Path("results/logs")
CANDS = ["r8_bert_coviso_sched_optuna_t50", "r6_bert_koleo_lrsplit_optuna_best_t27"] + [f"r7_bert_coviso_optuna_t{t}" for t in (18, 35, 34, 30, 39)]
EVAL = re.compile(r"\[step (\d+)\] EVAL sts_b_dev=([-\d.eE+]+)")
scores = collections.defaultdict(list)
for c in CANDS:
    if not Path(f"configs/{c}.yaml").exists():
        continue
    for s in (42, 43):
        f = LOG / f"{c}_s{s}" / "train.log"
        if not f.exists():
            continue
        m = EVAL.findall(f.read_text(errors="ignore"))
        if m and int(m[-1][0]) >= 1499:      # 완주한 run만 인정
            scores[c].append(float(m[-1][1]))
best = max(scores, key=lambda c: sum(scores[c]) / len(scores[c])) if scores else "r6_bert_koleo_lrsplit_optuna_best_t27"
for c, v in sorted(scores.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
    print(f"#   {c}: mean={sum(v)/len(v):.4f} seeds={v}", flush=True)
print(best)
PY
)"
# 마지막 줄이 선택된 config, 그 앞은 후보 점수표(로그용)
echo "$BASE_CONFIG" | grep '^#' || true
BASE_CONFIG="$(echo "$BASE_CONFIG" | grep -v '^#' | tail -1)"
fi
STUDY_NAME="tune_${BASE_CONFIG}_${SPACE}"
log "base config 선택: $BASE_CONFIG"
log "study 이름: $STUDY_NAME (SQLite 재개 가능)"

# ---------- 3. 안전한 유휴 GPU 2개 선택 ----------
pick_gpus() {
  uv run python - <<'PY'
import subprocess, re
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
# GPU별 (uuid -> index, 여유메모리)
MIN_FREE_MIB = 25000   # 우리 run 1개가 ~8GB - 여유를 두고 남의 작업과 충돌하지 않을 선
MAX_UTIL = 30          # 여유 메모리만 보면 98% 점유 중인 GPU를 "여유 있다"고 고르게 된다
gpus = {}
for line in sh("nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits").strip().splitlines():
    idx, uuid, used, total, util = [x.strip() for x in line.split(",")]
    gpus[uuid] = {"index": int(idx), "free": int(total) - int(used), "util": int(util), "blocked": False}
# 차단 소유자가 쓰는 GPU 표시
for line in sh("nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader").strip().splitlines():
    if not line.strip():
        continue
    uuid, pid = [x.strip() for x in line.split(",")]
    owner = sh(f"ps -o user= -p {pid}").strip()
    if re.search(r"gs25009|gs25049", owner) and uuid in gpus:
        gpus[uuid]["blocked"] = True
ok = [g for g in gpus.values()
      if not g["blocked"] and g["util"] < MAX_UTIL and g["free"] > MIN_FREE_MIB]
ok.sort(key=lambda g: (g["util"], -g["free"]))   # 가장 한가한 것 우선
print(",".join(str(g["index"]) for g in ok[:2]))
PY
}

# ---------- 4. 완료 trial 수 조회 ----------
completed_trials() {
  uv run python - "$STUDY_NAME" <<'PY' 2>/dev/null || echo 0
import sys, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
name = sys.argv[1]
try:
    st = optuna.load_study(study_name=name, storage=f"sqlite:///results/analysis/{name}.db")
    print(sum(1 for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE))
except Exception:
    print(0)
PY
}

# ---------- 5. 재시작 루프 ----------
for attempt in $(seq 1 "$MAX_RESTARTS"); do
  done_n="$(completed_trials | tail -1)"
  remaining=$(( TARGET_TRIALS - done_n ))
  if [[ "$remaining" -le 0 ]]; then
    log "목표 달성: ${done_n}/${TARGET_TRIALS} trial 완료. 종료."
    break
  fi

  GPUS="$(pick_gpus | tail -1)"
  if [[ -z "$GPUS" ]]; then
    log "진짜 유휴인 GPU가 없음 (util<30% & 여유>25GB 조건) - 5분 후 재시도"
    sleep 300
    continue
  fi
  # 1개만 잡혀도 그걸로 진행한다 - 2개를 기다리다 밤새 아무것도 못 도는 것보다 낫다.
  JOBS=$N_JOBS
  if [[ "$GPUS" != *,* ]]; then
    JOBS=1
    log "유휴 GPU가 1개뿐 (GPU=${GPUS}) - n_jobs=1로 진행"
  fi

  log "시도 ${attempt}/${MAX_RESTARTS}: 완료 ${done_n}/${TARGET_TRIALS}, 남은 ${remaining} trial, GPU=${GPUS}"
  uv run python scripts/tune.py \
    --base-config "configs/${BASE_CONFIG}.yaml" \
    --space "$SPACE" \
    --n-trials "$remaining" \
    --max-steps "$MAX_STEPS" \
    --n-jobs "$JOBS" \
    --gpus "$GPUS" \
    --study-name "$STUDY_NAME" \
    --seed 42
  rc=$?
  log "tune.py 종료 (exit=$rc)"

  after="$(completed_trials | tail -1)"
  if [[ "$after" -ge "$TARGET_TRIALS" ]]; then
    log "목표 달성: ${after}/${TARGET_TRIALS} trial 완료."
    break
  fi
  if [[ "$after" -le "$done_n" ]]; then
    # 한 번도 진전이 없었으면 (즉시 실패) 백오프를 길게 잡아 크래시 루프 방지
    log "진전 없음(${done_n} -> ${after}) - 3분 대기 후 재시도"
    sleep 180
  else
    log "진전 있음(${done_n} -> ${after}) - 30초 후 재개"
    sleep 30
  fi
done

final="$(completed_trials | tail -1)"
log "===== 감독 종료: ${final}/${TARGET_TRIALS} trial 완료 ====="
if [[ "$final" -gt 0 ]]; then
  uv run python - "$STUDY_NAME" <<'PY'
import sys, optuna, json
optuna.logging.set_verbosity(optuna.logging.WARNING)
name = sys.argv[1]
st = optuna.load_study(study_name=name, storage=f"sqlite:///results/analysis/{name}.db")
print("[supervisor] best value:", st.best_value)
print("[supervisor] best params:", json.dumps(st.best_params, indent=2, ensure_ascii=False))
st.trials_dataframe().to_csv(f"results/analysis/{name}_trials.csv", index=False)
print(f"[supervisor] wrote results/analysis/{name}_trials.csv")
PY
fi
