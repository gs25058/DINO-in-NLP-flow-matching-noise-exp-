#!/usr/bin/env bash
# R13 가설별 Optuna 튜닝 - 손으로 고른 값 때문에 기각된 것이 아님을 확인한다.
#
# base는 전부 현 챔피언(r12_bert_noise_optuna_t35, 2시드 0.7333)이다. R13 매트릭스는 r8
# (t=U(0.35,0.50)) 위에서 돌았는데 그 t 범위는 이후 열등한 것으로 판명됐다.
#
# 세 study를 순차 실행한다(동시 GPU 프로세스 2개 제한). 각각 SQLite로 재개 가능.
# 사용: TRIALS=10 setsid nohup bash scripts/run_r13_tuning.sh > results/logs/r13_tuning.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export UV_CACHE_DIR=/src/gs25058/scratch/.uv_cache
export HF_HOME=/src/gs25058/scratch/.hf_home
export HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT"
PY=/src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python

TRIALS="${TRIALS:-10}"
MAX_STEPS="${MAX_STEPS:-1500}"
MAX_UTIL="${MAX_UTIL:-30}"
MIN_FREE="${MIN_FREE:-25000}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
log() { echo "[tune $(date '+%m-%d %H:%M:%S')] $*"; }

# space:base config
STUDIES=(
  "r13_sharpen_ctrl:r13_bert_hctrl_tunebase"
  "r13_view_cutoff:r13_bert_cutoff_tunebase"
  "r13_view_corr:r13_bert_rho_tunebase"
)

pick_gpus() {
  MAX_UTIL="$MAX_UTIL" MIN_FREE="$MIN_FREE" $PY - <<'PY'
import subprocess, re, os
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
MAXU, MINF = int(os.environ["MAX_UTIL"]), int(os.environ["MIN_FREE"])
gpus = {}
for line in sh("nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits").strip().splitlines():
    i,u,used,tot,ut = [x.strip() for x in line.split(",")]
    gpus[u] = {"index": int(i), "free": int(tot)-int(used), "util": int(ut), "blocked": False}
for line in sh("nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader").strip().splitlines():
    if not line.strip(): continue
    u, p = [x.strip() for x in line.split(",")]
    if re.search(r"gs25009|gs25049", sh(f"ps -o user= -p {p}").strip()) and u in gpus:
        gpus[u]["blocked"] = True
ok = [g for g in gpus.values() if not g["blocked"] and g["util"] < MAXU and g["free"] > MINF]
ok.sort(key=lambda g: (g["util"], -g["free"]))
print(",".join(str(g["index"]) for g in ok[:2]))
PY
}

completed() {
  $PY - "$1" <<'PY' 2>/dev/null || echo 0
import sys, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
n = sys.argv[1]
try:
    st = optuna.load_study(study_name=n, storage=f"sqlite:///results/analysis/{n}.db")
    print(sum(1 for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE))
except Exception:
    print(0)
PY
}

for entry in "${STUDIES[@]}"; do
  SPACE="${entry%%:*}"; BASE="${entry##*:}"
  STUDY="tune_${BASE}_${SPACE}"
  for attempt in $(seq 1 "$MAX_RESTARTS"); do
    done_n="$(completed "$STUDY" | tail -1)"
    remaining=$(( TRIALS - done_n ))
    if [[ "$remaining" -le 0 ]]; then
      log "$SPACE 완료: ${done_n}/${TRIALS}"; break
    fi
    GPUS="$(pick_gpus | tail -1)"
    if [[ -z "$GPUS" ]]; then
      log "$SPACE - 조건 만족 GPU 없음(util<${MAX_UTIL}% & 여유>${MIN_FREE}MiB) - 5분 후 재시도"
      sleep 300; continue
    fi
    JOBS=2; [[ "$GPUS" != *,* ]] && JOBS=1
    log "$SPACE 시도 ${attempt}: 완료 ${done_n}/${TRIALS}, 남은 ${remaining}, GPU=${GPUS}, jobs=${JOBS}"
    $PY scripts/tune.py --base-config "configs/${BASE}.yaml" --space "$SPACE" \
      --n-trials "$remaining" --max-steps "$MAX_STEPS" --n-jobs "$JOBS" --gpus "$GPUS" \
      --study-name "$STUDY" --seed 42
    log "$SPACE tune.py 종료 (exit=$?)"
  done
done

log "=== 세 study 요약 (챔피언 단일시드(42) 기준 0.7309) ==="
$PY - <<'PY'
import optuna
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.WARNING)
for study, label in [("tune_r13_bert_hctrl_tunebase_r13_sharpen_ctrl","A 엔트로피 제어기"),
                     ("tune_r13_bert_cutoff_tunebase_r13_view_cutoff","B cutoff"),
                     ("tune_r13_bert_rho_tunebase_r13_view_corr","B 상관 노이즈")]:
    try:
        s = optuna.load_study(study_name=study, storage=f"sqlite:///results/analysis/{study}.db")
        d = [t for t in s.trials if t.state == TrialState.COMPLETE]
        if not d:
            print(f"\n  {label}: 완료 trial 없음"); continue
        print(f"\n  {label}: {len(d)} trial, best={s.best_value:.4f} (trial {s.best_trial.number})")
        for t in sorted(d, key=lambda t: -t.value)[:5]:
            ps = " ".join(f"{k.split('.')[-1]}={v:.4g}" if isinstance(v, float) else f"{k.split('.')[-1]}={v}"
                          for k, v in t.params.items())
            print(f"    {t.value:.4f}  {ps}")
    except Exception as e:
        print(f"\n  {label}: {e}")
PY
