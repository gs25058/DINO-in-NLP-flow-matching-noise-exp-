#!/usr/bin/env bash
# 스케줄 튜닝(r5_schedules_full) 무인 감독 실행기.
#
# 이 스크립트는 "사용자가 자는 동안 아무도 안 봐도 끝까지 완주"를 목표로 한다:
#   1. 우리 계정의 다른 학습이 끝날 때까지 대기 (동시 GPU 2개 제한 준수)
#   2. 1500-step 검증이 끝난 run들 중 최고 성능 config를 base로 자동 선택
#   3. gs25009/gs25049가 쓰는 GPU를 제외하고 가장 여유 있는 GPU 2개 자동 선택
#   4. tune.py 실행 - 중간에 죽으면 Optuna SQLite에서 이어받아 자동 재시작
#      (남은 trial 수만 다시 요청하므로 총 목표치를 넘기지 않는다)
#   5. 목표 trial 수를 채우거나 재시도 한도에 도달하면 종료
#
# 사용: setsid nohup bash scripts/run_schedule_study.sh > results/logs/schedule_study_supervisor.log 2>&1 &
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
BASE_CONFIG="$(uv run python - <<'PY'
from pathlib import Path
import re, collections
LOG = Path("results/logs")
CANDS = ["r6_bert_koleo_lrsplit_optuna_best_t27"] + [f"r7_bert_coviso_optuna_t{t}" for t in (18, 35, 34, 30, 39)]
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
STUDY_NAME="tune_${BASE_CONFIG}_${SPACE}"
log "base config 선택: $BASE_CONFIG"
log "study 이름: $STUDY_NAME (SQLite 재개 가능)"

# ---------- 3. 안전한 유휴 GPU 2개 선택 ----------
pick_gpus() {
  uv run python - <<'PY'
import subprocess, re
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
# GPU별 (uuid -> index, 여유메모리)
gpus = {}
for line in sh("nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits").strip().splitlines():
    idx, uuid, used, total = [x.strip() for x in line.split(",")]
    gpus[uuid] = {"index": int(idx), "free": int(total) - int(used), "blocked": False}
# 차단 소유자가 쓰는 GPU 표시
for line in sh("nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader").strip().splitlines():
    if not line.strip():
        continue
    uuid, pid = [x.strip() for x in line.split(",")]
    owner = sh(f"ps -o user= -p {pid}").strip()
    if re.search(r"gs25009|gs25049", owner) and uuid in gpus:
        gpus[uuid]["blocked"] = True
ok = [g for g in gpus.values() if not g["blocked"]]
ok.sort(key=lambda g: -g["free"])
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
  if [[ -z "$GPUS" || "$GPUS" != *,* ]]; then
    log "사용 가능한 GPU 2개를 못 찾음 (got='${GPUS}') - 5분 후 재시도"
    sleep 300
    continue
  fi

  log "시도 ${attempt}/${MAX_RESTARTS}: 완료 ${done_n}/${TARGET_TRIALS}, 남은 ${remaining} trial, GPU=${GPUS}"
  uv run python scripts/tune.py \
    --base-config "configs/${BASE_CONFIG}.yaml" \
    --space "$SPACE" \
    --n-trials "$remaining" \
    --max-steps "$MAX_STEPS" \
    --n-jobs "$N_JOBS" \
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
