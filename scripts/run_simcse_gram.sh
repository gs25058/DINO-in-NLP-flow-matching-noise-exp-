#!/usr/bin/env bash
# R19: SimCSE(비지도) InfoNCE의 cos 유사도 행렬을 gram 변환(exp -> Sinkhorn-Knopp -> SVD top-k 제거)
# 으로 바꿨을 때의 효과. 대조군은 원본 SimCSE 그대로(같은 데이터·예산·평가).
# GPU 1장에서 한 번에 한 run씩 순차로 돈다(C-4: 겹치면 총 처리량만 떨어진다).
#
# STAGE:
#   ctrl   원본 SimCSE 1 epoch, 시드 42/43
#   gram   gram 변환 arm, 시드 42/43 (TOPK/SCALE/ITERS 환경변수로 조정)
#   matrix ctrl + gram5(scale 1, 8) x 시드 42/43 = 6 run 순차
#   sts7   RUNS="a b ..."의 7-task STS (cls pooling, SimCSE 공식 평가 방식)
#
# 사용: STAGE=ctrl setsid nohup bash scripts/run_simcse_gram.sh > results/logs/r19_ctrl.out 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/venvs/flowdino}"
source "$ROOT/scripts/env.sh" >/dev/null
export PYTHONPATH="$ROOT" WANDB_MODE="${WANDB_MODE:-disabled}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
STAGE="${STAGE:?STAGE=ctrl|gram|matrix|short|sts7 필요}"
GPU="${GPU:-0}"
TOPK="${TOPK:-5}"; SCALE="${SCALE:-0}"; ITERS="${ITERS:-3}"
EVAL_STEPS="${EVAL_STEPS:-125}"
STEPS="${STEPS:-0}"          # >0이면 1 epoch 대신 이 step 수로 자른다(빠른 비교용)
OUT="${OUT:-$ROOT/results/analysis/r19}"
mkdir -p "$OUT" results/logs
log() { echo "[r19 $(date '+%m-%d %H:%M:%S')] $*"; }

train() {  # run-name  extra-args...
  local run="$1"; shift
  if grep -q "best STS-B dev" "results/logs/$run/train.log" 2>/dev/null; then
    log "건너뜀(완주): $run"; return 0
  fi
  [[ -d "results/logs/$run" ]] && mv "results/logs/$run" "results/logs/${run}.incomplete.$(date +%s)"
  log "시작: $run ($*)"
  local budget=(--epochs 1)
  [[ "$STEPS" -gt 0 ]] && budget=(--max-steps "$STEPS")
  CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/train_simcse_baseline.py --run-name "$run" \
    "${budget[@]}" --eval-steps "$EVAL_STEPS" "$@" > "results/logs/${run}.out" 2>&1
  local rc=$?
  log "종료: $run (exit=$rc) $(grep 'best STS-B dev' "results/logs/$run/train.log" 2>/dev/null)"
  return "$rc"
}

case "$STAGE" in
  ctrl)
    for s in 42 43; do train "simcse_repro_1epoch_s${s}" --seed "$s"; done
    ;;
  gram)
    for s in 42 43; do
      train "r19_simcse_gram${TOPK}_scale${SCALE}_s${s}" --seed "$s" \
        --gram-topk "$TOPK" --gram-scale "$SCALE" --sinkhorn-iters "$ITERS"
    done
    ;;
  short)
    # 빠른 비교용: STEPS step x (대조군, gram5 scale=1) x 시드 42/43. 시드 42가 먼저 다 끝난다.
    for s in 42 43; do
      train "simcse_repro_${STEPS}_s${s}" --seed "$s"
      train "r19_simcse_gram5_scale1_${STEPS}_s${s}" --seed "$s" --gram-topk 5 --gram-scale 1
    done
    ;;
  matrix)
    # 시드별로 대조군 -> arm 순서. 중간에 멈춰도 s42 전체 비교가 먼저 완성된다.
    for s in 42 43; do
      train "simcse_repro_1epoch_s${s}" --seed "$s"
      train "r19_simcse_gram5_scale1_s${s}" --seed "$s" --gram-topk 5 --gram-scale 1
      train "r19_simcse_gram5_scale8_s${s}" --seed "$s" --gram-topk 5 --gram-scale 8
    done
    ;;
  sts7)
    read -r -a runs <<< "${RUNS:?RUNS=\"run_s42 run_s43\" 필요}"
    CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/eval_sts7.py --config configs/base_bert.yaml \
      --runs "${runs[@]}" --which "${WHICH:-best}" --pooling cls --out "$OUT/sts7_${runs[0]}.json"
    ;;
  *) log "알 수 없는 STAGE: $STAGE"; exit 2 ;;
esac
log "단계 완료: $STAGE"
