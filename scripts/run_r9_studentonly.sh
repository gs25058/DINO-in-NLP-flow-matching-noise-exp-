#!/usr/bin/env bash
# R9 처치군 B: student head만 리셋(teacher head 유지) 1000 step 주기, 3000 step x 2시드.
#
# 대조군(r9_bert_t50_steps3000)은 이미 별도 lane에서 돌고 있고 리셋 조건과 무관한 공통
# 기준선이므로 여기서 다시 돌리지 않는다. 이 스크립트는 lane 1개(순차 s42 -> s43)만 띄우고,
# 대조군 lane과 합쳐 동시 GPU 프로세스는 항상 2개를 넘지 않는다(사용자 지시).
#
# 사용: setsid nohup bash scripts/run_r9_studentonly.sh > results/logs/r9_studentonly_supervisor.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG=r9_bert_t50_steps3000_headreinit1000_studentonly
SEEDS=(42 43)

log() { echo "[r9-so $(date '+%m-%d %H:%M:%S')] $*"; }

# ---------- 동시 2개 제한: 우리 학습이 2개 이상이면 자리가 날 때까지 대기 ----------
own_trains() { pgrep -u "$(id -u)" -f "python3 -m src.train --config" | wc -l; }
while [[ "$(own_trains)" -ge 2 ]]; do
  log "우리 학습 $(own_trains)개 실행 중 - 자리 대기"
  sleep 30
done

source "$ROOT/scripts/env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------- GPU 선택: 차단 소유자 + 우리가 이미 쓰는 GPU 제외, 여유 메모리 최대 ----------
pick_gpu() {
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
    # gs25009/gs25049: 사용자 지시로 회피. gs25058: 우리 대조군 lane과 같은 GPU에 겹치지 않게.
    if re.search(r"gs25009|gs25049|gs25058", owner) and uuid in gpus:
        gpus[uuid]["blocked"] = True
ok = [g for g in gpus.values() if not g["blocked"]]
ok.sort(key=lambda g: -g["free"])
print(ok[0]["index"] if ok else "")
PY
}

for seed in "${SEEDS[@]}"; do
  GPU="$(pick_gpu | tail -1)"
  if [[ -z "$GPU" ]]; then
    log "쓸 수 있는 GPU가 없음 - 5분 후 재시도"
    sleep 300
    GPU="$(pick_gpu | tail -1)"
    [[ -z "$GPU" ]] && { log "여전히 없음 - 중단"; exit 1; }
  fi
  run="${CFG}_s${seed}"
  mkdir -p "results/logs/${run}"
  log "시작: ${run} (GPU ${GPU})"
  CUDA_VISIBLE_DEVICES="$GPU" uv run python -m src.train \
    --config "configs/${CFG}.yaml" \
    --seed "$seed" \
    --run-name-suffix "_s${seed}" \
    --device cuda > "results/logs/${run}/launch.log" 2>&1
  log "종료: ${run} (exit=$?)"
done

log "===== student_only 2시드 완료 ====="
for seed in "${SEEDS[@]}"; do
  f="results/logs/${CFG}_s${seed}/train.log"
  [[ -f "$f" ]] && echo "--- ${CFG}_s${seed} ---" && grep EVAL "$f" | tail -3
done
