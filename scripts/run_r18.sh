#!/usr/bin/env bash
# R18 iBOT 토큰 CE 매트릭스. GPU 1장에서 한 번에 한 run씩 순차로 돈다.
# 이 서버(RTX 6000 Ada) 벤치마크에서 같은 GPU에 run을 겹치면 총 처리량이 떨어졌다
# (단독 4.84 -> 2개 3.61 -> 3개 2.85 steps/s, 4개는 OOM). 수치는 겹쳐도 bit-identical이었다.
#
# 단계 (STAGE):
#   gate     r18a_lam0.3 s42를 돌리며 step 1000 EVAL의 eff_rank를 본다. 240 미만이면 run을 죽이고 exit 3
#            (R15식 앵커 - 매트릭스 진행 전 원인 확인). 통과하면 끝까지 돌리고 대조군 r18_bert_ctrl s42/s43.
#   matrix   나머지 arm x 시드 {42, 43}. configs/r18_base.yaml에 "GUARD: PENDING"이 남아 있으면 거부.
#   base1500 챔피언 1500-step 새 서버 기준선 s42/s43: 가속 없이(_ada) + 가속으로(_ada_accel).
#   sts7     RUNS="run_s42 run_s43 ..." 의 7-task 평가 -> $OUT/sts7_<첫 run>.json
# 완주한 run(마지막 step EVAL 또는 COLLAPSE ABORT)은 건너뛴다. 미완주 로그는 옆으로 치우고 다시 돈다
# (train.log는 append라 그대로 두면 두 실행이 섞인다).
#
# 사용: STAGE=gate setsid nohup bash scripts/run_r18.sh > results/logs/r18_gate.out 2>&1 &
#   OUT: 7-task json 위치 (기본 <repo>/results/analysis/r18). GPU: CUDA 장치 번호 (기본 0).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
source "$ROOT/scripts/env.sh"      # UV_CACHE_DIR, HF_HOME, PY
export PYTHONPATH="$ROOT" WANDB_MODE="${WANDB_MODE:-disabled}"   # TensorBoard + train.log에 전부 남는다
STAGE="${STAGE:?STAGE=gate|matrix|base1500|sts7 필요}"
GPU="${GPU:-0}"
OUT="${OUT:-$ROOT/results/analysis/r18}"
ACCEL="--batch-views --tf32"
GATE_STEP=1000; GATE_MIN_RANK=240
mkdir -p "$OUT" results/logs
log() { echo "[r18 $(date '+%m-%d %H:%M:%S')] $*"; }

max_steps() { $PY -c "from src.train import load_config; print(load_config('configs/$1.yaml')['train']['max_steps'])"; }

done_run() {  # run max
  local f="results/logs/$1/train.log"
  [[ -f "$f" ]] && { grep -q "\[step $(( $2 - 1 ))\] EVAL" "$f" || grep -q "COLLAPSE ABORT" "$f"; }
}

# train <cfg> <seed> <suffix> <flags> [gate]
train() {
  local cfg="$1" seed="$2" suffix="$3" flags="$4" gate="${5:-}"
  local run="${cfg}${suffix}_s${seed}" max; max=$(max_steps "$cfg")
  if done_run "$run" "$max"; then log "건너뜀(완주): $run"; return 0; fi
  if [[ -d "results/logs/$run" ]]; then
    mv "results/logs/$run" "results/logs/${run}.incomplete.$(date +%s)"; log "미완주 로그 치움: $run"
  fi
  log "시작: $run (max_steps=$max, flags='$flags')"
  CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 $PY -m src.train --config "configs/${cfg}.yaml" --seed "$seed" \
    --run-name-suffix "${suffix}_s${seed}" $flags > "results/logs/${run}.out" 2>&1 &
  local pid=$!
  if [[ -n "$gate" ]]; then
    local f="results/logs/$run/train.log" line=""
    while kill -0 "$pid" 2>/dev/null; do
      line=$(grep -m1 "\[step $GATE_STEP\] EVAL" "$f" 2>/dev/null) && break
      sleep 20
    done
    if [[ -z "$line" ]]; then
      wait "$pid"; log "게이트 전에 종료: $run (exit=$?) - results/logs/${run}.out 확인"; return 2
    fi
    local rank; rank=$(sed -E 's/.*eff_rank=([0-9.]+).*/\1/' <<< "$line")
    if awk -v r="$rank" -v m="$GATE_MIN_RANK" 'BEGIN{exit !(r < m)}'; then
      kill "$pid"; wait "$pid" 2>/dev/null
      log "GATE FAIL: $run step $GATE_STEP eff_rank=$rank < $GATE_MIN_RANK - 중단"; return 3
    fi
    log "GATE PASS: $run step $GATE_STEP eff_rank=$rank >= $GATE_MIN_RANK"
  fi
  wait "$pid"; local rc=$?
  log "종료: $run (exit=$rc) $(grep "\[step $(( max - 1 ))\] EVAL" "results/logs/$run/train.log" 2>/dev/null | grep -oE 'sts_b_dev=[0-9.]+ eff_rank=[0-9.]+ max_sv_ratio=[0-9.]+ alignment=[0-9.]+ uniformity=-?[0-9.]+')"
  return "$rc"
}

case "$STAGE" in
  gate)
    train r18a_lam0.3 42 "" "$ACCEL" gate || exit $?
    for s in 42 43; do train r18_bert_ctrl "$s" "" "$ACCEL"; done
    ;;
  matrix)
    if grep -q "GUARD: PENDING" configs/r18_base.yaml; then
      log "configs/r18_base.yaml의 붕괴 감시 기준값이 확정되지 않았다 - 중단"; exit 2
    fi
    for cfg in r18a_lam0.3 r18a_lam1.0 r18a_ratio0.5 r18b_sep_ratio0.5 r18a_ratio0.5_mask0.3; do
      for s in 42 43; do train "$cfg" "$s" "" "$ACCEL"; done
    done
    ;;
  base1500)
    for s in 42 43; do
      train r12_bert_noise_optuna_t35 "$s" "_ada" ""
      train r12_bert_noise_optuna_t35 "$s" "_ada_accel" "$ACCEL"
    done
    ;;
  sts7)
    read -r -a runs <<< "${RUNS:?RUNS=\"run_s42 run_s43\" 필요}"
    CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/eval_sts7.py --config configs/r18_base.yaml --runs "${runs[@]}" \
      --out "$OUT/sts7_${runs[0]}.json"
    ;;
  *) log "알 수 없는 STAGE: $STAGE"; exit 2 ;;
esac
log "단계 완료: $STAGE"
