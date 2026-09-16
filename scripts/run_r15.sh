#!/usr/bin/env bash
# R15 토큰 latent 예측 매트릭스: 4 config x 시드 {42, 43}, 각 run 뒤 7-task 평가.
# configs/r15_base.yaml에 "GATE: PENDING"이 남아 있으면 시작하지 않는다(R14 게이트 판정 전 실행 방지).
#
# 사용: GPUS=a,b setsid nohup bash scripts/run_r15.sh > results/logs/r15.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
if grep -q "GATE: PENDING" configs/r15_base.yaml; then
  echo "[r15] configs/r15_base.yaml의 R14 게이트가 확정되지 않았다 - 중단"; exit 2
fi
export UV_CACHE_DIR=/src/gs25058/scratch/.uv_cache HF_HOME=/src/gs25058/scratch/.hf_home
export HF_HUB_OFFLINE=1 PYTHONPATH="$ROOT"
PY=/src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python
FLAGS=""; [[ "${ACCEL:-0}" == "1" ]] && FLAGS="--batch-views --tf32"
IFS=',' read -r GPU_A GPU_B <<< "${GPUS:?GPUS=a,b 필요}"
CONFIGS=(r15_bert_tok_lam1.0 r15_bert_tok_lam0.5 r15_bert_tok_lam2.0 r15_bert_tok_lam1.0_mask0.30)
OUT=results/analysis/r14_r17; mkdir -p "$OUT" results/logs
log() { echo "[r15 $(date '+%m-%d %H:%M:%S')] $*"; }

BASE_RUN=r14_bert_champ_7700   # r15_base.yaml이 extends하는 run - P-15b(7-task +1.5) 판정의 기준

run_lane() {
  local gpu="$1" seed="$2"
  # base의 7-task가 없으면 먼저 잰다(R14 실행기를 R15 우선으로 멈추면서 빠진 평가)
  if [[ ! -f "$OUT/sts7_${BASE_RUN}_s${seed}.json" ]] && [[ -f "checkpoints/${BASE_RUN}_s${seed}/last.pt" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=0 $PY scripts/eval_sts7.py --config "configs/${BASE_RUN}.yaml" \
      --runs "${BASE_RUN}_s${seed}" --out "$OUT/sts7_${BASE_RUN}_s${seed}.json" > "results/logs/${BASE_RUN}_s${seed}.sts7.out" 2>&1
    log "base 7-task 완료: ${BASE_RUN}_s${seed} (exit=$?)"
  fi
  for cfg in "${CONFIGS[@]}"; do
    local run="${cfg}_s${seed}"
    local max; max=$($PY -c "from src.train import load_config; print(load_config('configs/${cfg}.yaml')['train']['max_steps'])")
    if grep -q "COLLAPSE ABORT" "results/logs/${run}/train.log" 2>/dev/null; then
      log "건너뜀(붕괴 중단 기록됨): $run"; continue
    fi
    if ! grep -q "\[step $((max - 1))\] EVAL" "results/logs/${run}/train.log" 2>/dev/null; then
      log "학습 시작: $run (GPU $gpu, flags='${FLAGS}')"
      CUDA_VISIBLE_DEVICES="$gpu" $PY -m src.train --config "configs/${cfg}.yaml" --seed "$seed" \
        --run-name-suffix "_s${seed}" $FLAGS > "results/logs/${run}.out" 2>&1
      log "학습 종료: $run (exit=$?)$(grep -q 'COLLAPSE ABORT' results/logs/${run}/train.log 2>/dev/null && echo ' [붕괴 중단]')"
    fi
    if [[ ! -f "$OUT/sts7_${run}.json" ]] && [[ -f "checkpoints/${run}/last.pt" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=0 $PY scripts/eval_sts7.py --config "configs/${cfg}.yaml" \
        --runs "$run" --out "$OUT/sts7_${run}.json" > "results/logs/${run}.sts7.out" 2>&1
      log "7-task 완료: $run (exit=$?)"
    fi
  done
}

log "레인 A: GPU $GPU_A 시드 42 / 레인 B: GPU $GPU_B 시드 43"
run_lane "$GPU_A" 42 & PID_A=$!
run_lane "$GPU_B" 43 & PID_B=$!
wait "$PID_A" "$PID_B"
log "전체 완료"
