#!/usr/bin/env bash
# 노이즈 축 튜닝 상위 5 config를 2시드로 검증한다.
#
# trial 값은 전부 단일 시드(42)라 상위권 내부 순위는 시드 노이즈(챔피언 기준 0.011) 안에 있다.
# 2시드 평균이 나와야 "챔피언 0.7120을 실제로 넘었는가"를 말할 수 있다.
#
# 레인 2개(GPU 1장씩), 레인당 순차 실행 - 동시 GPU 프로세스는 항상 2개(공용 서버 제한).
# 사용: GPUS=0,9 setsid nohup bash scripts/run_r12_validation.sh > results/logs/r12_validation.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
source "$ROOT/scripts/env.sh" 2>/dev/null || true
export HF_HUB_OFFLINE=1

CONFIGS=(r12_bert_noise_optuna_t12 r12_bert_noise_optuna_t16 r12_bert_noise_optuna_t33
         r12_bert_noise_optuna_t35 r12_bert_noise_optuna_t38)
IFS=',' read -r GPU_A GPU_B <<< "${GPUS:-0,9}"
log() { echo "[r12 $(date '+%m-%d %H:%M:%S')] $*"; }

# 레인 A = 시드 42, 레인 B = 시드 43. 각 레인은 config 5개를 순차로 돈다.
run_lane() {
  local gpu="$1" seed="$2"
  for cfg in "${CONFIGS[@]}"; do
    local run="${cfg}_s${seed}"
    if grep -q "\[step 1499\] EVAL" "results/logs/${run}/train.log" 2>/dev/null; then
      log "건너뜀(이미 완주): $run"; continue
    fi
    log "시작: $run (GPU $gpu)"
    CUDA_VISIBLE_DEVICES="$gpu" uv run python -m src.train \
      --config "configs/${cfg}.yaml" --seed "$seed" --run-name-suffix "_s${seed}" \
      > "results/logs/${run}.out" 2>&1
    log "완료: $run (exit=$?)"
  done
}

log "레인 A: GPU $GPU_A, 시드 42 / 레인 B: GPU $GPU_B, 시드 43 - config ${#CONFIGS[@]}개씩"
run_lane "$GPU_A" 42 &  PID_A=$!
run_lane "$GPU_B" 43 &  PID_B=$!
wait "$PID_A" "$PID_B"      # disown 금지 - 명시적 PID로 대기(과거 즉시 종료 사고)
log "전체 완료"

log "=== 2시드 요약 ==="
uv run python - <<'PY'
import re, statistics
from pathlib import Path
E=re.compile(r"\[step (\d+)\] EVAL sts_b_dev=([-\d.]+).*?eff_rank=([-\d.]+).*?alignment=([-\d.]+) uniformity=([-\d.]+)")
rows=[]
for c in ["r12_bert_noise_optuna_t12","r12_bert_noise_optuna_t16","r12_bert_noise_optuna_t33",
          "r12_bert_noise_optuna_t35","r12_bert_noise_optuna_t38"]:
    v=[]
    for s in (42,43):
        f=Path(f"results/logs/{c}_s{s}/train.log")
        if not f.exists(): continue
        m=E.findall(f.read_text(errors="ignore"))
        if m and int(m[-1][0])>=1499: v.append(tuple(float(x) for x in m[-1][1:]))
    if v: rows.append((statistics.mean(x[0] for x in v), c, v))
print(f"{'config':<32} {'STS 평균':>9} {'시드별':>18} {'align':>7} {'rank':>7}")
for mean,c,v in sorted(rows, reverse=True):
    seeds="/".join(f"{x[0]:.4f}" for x in v)
    print(f"{c:<32} {mean:>9.4f} {seeds:>18} {statistics.mean(x[2] for x in v):>7.4f} {statistics.mean(x[1] for x in v):>7.1f}")
print("\n기준선 r8_bert_coviso_sched_optuna_t50: 0.7120 (0.7173/0.7066)")
PY
