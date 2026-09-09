#!/usr/bin/env bash
# R9: 주기적 DINO head 리셋(1000 step마다) vs 대조군, 3000 step x 2시드.
#
#   lane A (GPU 1개): 대조군    s42 -> s43
#   lane B (GPU 1개): head 리셋 s42 -> s43
# 동시 GPU 프로세스는 항상 2개 (사용자 지시: 공용 서버 2개 제한).
# lane당 순차 실행이라 ~20분 뒤에는 두 조건의 s42가 모두 끝나 조기 비교가 가능하다.
#
# 사용: setsid nohup bash scripts/run_r9_headreinit.sh > results/logs/r9_headreinit_supervisor.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CTRL=r9_bert_t50_steps3000
TREAT=r9_bert_t50_steps3000_headreinit1000
SEEDS=(42 43)
BLOCKED_OWNERS="gs25009|gs25049"   # 사용자 지시: 이 계정들이 쓰는 GPU는 절대 쓰지 않는다

log() { echo "[r9 $(date '+%m-%d %H:%M:%S')] $*"; }

# ---------- 우리 계정의 기존 학습이 끝날 때까지 대기 ----------
while pgrep -u "$(id -u)" -f "python3 -m src.train" >/dev/null 2>&1; do
  log "기존 학습 진행 중 - 30초 후 재확인"
  sleep 30
done

source "$ROOT/scripts/env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------- 안전한 유휴 GPU 2개 선택 (run_schedule_study.sh와 동일 로직) ----------
pick_gpus() {
  uv run python - <<'PY'
import subprocess, re
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
gpus = {}
for line in sh("nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits").strip().splitlines():
    idx, uuid, used, total = [x.strip() for x in line.split(",")]
    gpus[uuid] = {"index": int(idx), "free": int(total) - int(used), "blocked": False}
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

GPUS="$(pick_gpus | tail -1)"
if [[ -z "$GPUS" || "$GPUS" != *,* ]]; then
  log "GPU 2개를 찾지 못함 (got='${GPUS}') - 중단"
  exit 1
fi
GPU_A="${GPUS%%,*}"
GPU_B="${GPUS##*,}"
log "GPU 선택: lane A=${GPU_A} (대조군), lane B=${GPU_B} (head 리셋)"

run_lane() {
  local gpu="$1" cfg="$2"
  for seed in "${SEEDS[@]}"; do
    local run="${cfg}_s${seed}"
    mkdir -p "results/logs/${run}"
    log "시작: ${run} (GPU ${gpu})"
    CUDA_VISIBLE_DEVICES="$gpu" uv run python -m src.train \
      --config "configs/${cfg}.yaml" \
      --seed "$seed" \
      --run-name-suffix "_s${seed}" \
      --device cuda > "results/logs/${run}/launch.log" 2>&1
    log "종료: ${run} (exit=$?)"
  done
}

run_lane "$GPU_A" "$CTRL"  &
pid_a=$!
run_lane "$GPU_B" "$TREAT" &
pid_b=$!
wait $pid_a; wait $pid_b

log "===== 4개 run 완료 ====="
for cfg in "$CTRL" "$TREAT"; do
  for seed in "${SEEDS[@]}"; do
    f="results/logs/${cfg}_s${seed}/train.log"
    [[ -f "$f" ]] && echo "--- ${cfg}_s${seed} ---" && grep EVAL "$f" | tail -3
  done
done
