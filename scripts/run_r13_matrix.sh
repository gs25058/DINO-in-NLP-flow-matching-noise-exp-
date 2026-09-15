#!/usr/bin/env bash
# R13 매트릭스: A(후반 teacher 샤프닝) 4 arm + B(pooling을 견디는 뷰) 5 arm, 각 2시드.
#
# 레인 2개(GPU 1장씩), 레인당 순차 - 동시 GPU 프로세스는 항상 2개(공용 서버 하드룰).
# A를 먼저 돌려 붕괴 게이트가 걸리는지 빨리 드러나게 한다. 이미 완주한 run은 건너뛰므로
# 중간에 끊겨도 이어서 돌릴 수 있다.
#
# 사용: GPUS=9,5 setsid nohup bash scripts/run_r13_matrix.sh > results/logs/r13_matrix.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export UV_CACHE_DIR=/src/gs25058/scratch/.uv_cache
export HF_HOME=/src/gs25058/scratch/.hf_home
export HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT"
PY=/src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python

CONFIGS=(
  r13_bert_ttdecay0.07 r13_bert_ttdecay0.05          # A 개루프
  r13_bert_hctrl0.066  r13_bert_hctrl0.152           # A 폐루프 (같은 H 도달점)
  r13_bert_rho0.5      r13_bert_rho1.0               # B 상관 노이즈
  r13_bert_cutoff0.1                                 # B span cutoff
  r13_bert_iid_t0.45-0.55 r13_bert_iid_t0.50-0.60    # B iid 대조군 (pooled 난이도 짝)
)
IFS=',' read -r GPU_A GPU_B <<< "${GPUS:-9,5}"
log() { echo "[r13 $(date '+%m-%d %H:%M:%S')] $*"; }

run_lane() {
  local gpu="$1" seed="$2"
  for cfg in "${CONFIGS[@]}"; do
    local run="${cfg}_s${seed}"
    if grep -q "\[step 1499\] EVAL" "results/logs/${run}/train.log" 2>/dev/null; then
      log "건너뜀(완주): $run"; continue
    fi
    if grep -q "COLLAPSE ABORT" "results/logs/${run}/train.log" 2>/dev/null; then
      log "건너뜀(붕괴 중단 기록됨): $run"; continue
    fi
    log "시작: $run (GPU $gpu)"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m src.train \
      --config "configs/${cfg}.yaml" --seed "$seed" --run-name-suffix "_s${seed}" \
      > "results/logs/${run}.out" 2>&1
    local rc=$?
    local note=""
    grep -q "COLLAPSE ABORT" "results/logs/${run}/train.log" 2>/dev/null && note=" [붕괴 중단]"
    log "완료: $run (exit=$rc)$note"
  done
}

log "레인 A: GPU $GPU_A 시드 42 / 레인 B: GPU $GPU_B 시드 43 - config ${#CONFIGS[@]}개씩"
run_lane "$GPU_A" 42 &  PID_A=$!
run_lane "$GPU_B" 43 &  PID_B=$!
wait "$PID_A" "$PID_B"      # disown 금지 - 명시적 PID 대기
log "전체 완료"

log "=== 2시드 요약 (기준선 r8 = 0.7120) ==="
$PY - <<'PY'
import re, statistics
from pathlib import Path
E=re.compile(r"\[step (\d+)\] EVAL sts_b_dev=([-\d.]+).*?eff_rank=([-\d.]+).*?alignment=([-\d.]+) uniformity=([-\d.]+)")
CFG=["r13_bert_ttdecay0.07","r13_bert_ttdecay0.05","r13_bert_hctrl0.066","r13_bert_hctrl0.152",
     "r13_bert_rho0.5","r13_bert_rho1.0","r13_bert_cutoff0.1",
     "r13_bert_iid_t0.45-0.55","r13_bert_iid_t0.50-0.60"]
print(f"{'config':<30} {'STS':>8} {'시드별':>18} {'align':>7} {'rank':>7} {'상태':>10}")
for c in CFG:
    vals=[]; state=""
    for s in (42,43):
        f=Path(f"results/logs/{c}_s{s}/train.log")
        if not f.exists(): continue
        txt=f.read_text(errors="ignore")
        if "COLLAPSE ABORT" in txt:
            state="중단"
            m=re.search(r"COLLAPSE ABORT.*?마지막 정상 평가 step=(-?\d+)", txt)
            if m: state=f"중단@{m.group(1)}"
        m=E.findall(txt)
        if m and int(m[-1][0])>=1499: vals.append(tuple(float(x) for x in m[-1][1:]))
    if not vals:
        print(f"{c:<30} {'-':>8} {'-':>18} {'-':>7} {'-':>7} {state or '미완주':>10}")
        continue
    print(f"{c:<30} {statistics.mean(v[0] for v in vals):>8.4f} "
          f"{'/'.join(f'{v[0]:.4f}' for v in vals):>18} "
          f"{statistics.mean(v[2] for v in vals):>7.4f} {statistics.mean(v[1] for v in vals):>7.1f} {state:>10}")
PY
