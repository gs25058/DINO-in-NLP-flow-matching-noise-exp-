#!/usr/bin/env bash
# R14 예산 스케일링 매트릭스: 3 config x 시드 {42, 43}, 각 run 완주 직후 7-task 평가.
#
# 레인 2개(GPU 1장씩), 레인당 순차 - 동시 GPU 프로세스는 항상 2개(공용 서버 하드룰).
# 순서는 7700 -> 7700_mom999 -> 15600: 게이트(P-14a/b)와 P-14c 판정에 필요한 7700 계열을 먼저 끝낸다.
# 완주한 run(최종 EVAL + 7-task json)은 건너뛰므로 끊겨도 같은 명령으로 이어진다.
#
# 사용: GPUS=a,b ACCEL=1 setsid nohup bash scripts/run_r14.sh > results/logs/r14.log 2>&1 &
#   ACCEL=1 이면 --batch-views --tf32 (가속 재현 확인을 통과했을 때만)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export UV_CACHE_DIR=/src/gs25058/scratch/.uv_cache HF_HOME=/src/gs25058/scratch/.hf_home
export HF_HUB_OFFLINE=1 PYTHONPATH="$ROOT"
PY=/src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python
FLAGS=""; [[ "${ACCEL:-0}" == "1" ]] && FLAGS="--batch-views --tf32"
IFS=',' read -r GPU_A GPU_B <<< "${GPUS:?GPUS=a,b 필요}"
CONFIGS=(r14_bert_champ_7700 r14_bert_champ_7700_mom999 r14_bert_champ_15600)
OUT=results/analysis/r14_r17; mkdir -p "$OUT" results/logs
log() { echo "[r14 $(date '+%m-%d %H:%M:%S')] $*"; }

run_lane() {
  local gpu="$1" seed="$2"
  for cfg in "${CONFIGS[@]}"; do
    local run="${cfg}_s${seed}"
    local max; max=$(grep -E "^  max_steps:" "configs/${cfg}.yaml" | awk '{print $2}')
    if ! grep -q "\[step $((max - 1))\] EVAL" "results/logs/${run}/train.log" 2>/dev/null; then
      log "학습 시작: $run (GPU $gpu, flags='${FLAGS}')"
      CUDA_VISIBLE_DEVICES="$gpu" $PY -m src.train --config "configs/${cfg}.yaml" --seed "$seed" \
        --run-name-suffix "_s${seed}" $FLAGS > "results/logs/${run}.out" 2>&1
      log "학습 종료: $run (exit=$?)"
    else
      log "학습 건너뜀(완주): $run"
    fi
    if [[ ! -f "$OUT/sts7_${run}.json" ]] && [[ -f "checkpoints/${run}/last.pt" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=0 $PY scripts/eval_sts7.py --config "configs/${cfg}.yaml" \
        --runs "$run" --out "$OUT/sts7_${run}.json" > "results/logs/${run}.sts7.out" 2>&1
      log "7-task 완료: $run (exit=$?) $(grep -oE 'avg_7task.*' "$OUT/sts7_${run}.json" 2>/dev/null | head -c 40)"
    fi
  done
}

log "레인 A: GPU $GPU_A 시드 42 / 레인 B: GPU $GPU_B 시드 43"
run_lane "$GPU_A" 42 & PID_A=$!
run_lane "$GPU_B" 43 & PID_B=$!
wait "$PID_A" "$PID_B"      # disown 금지 - 명시적 PID 대기
log "전체 완료"
