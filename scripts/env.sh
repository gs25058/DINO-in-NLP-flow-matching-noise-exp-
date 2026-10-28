#!/usr/bin/env bash
# 공용 서버용: 홈 쿼터 보호를 위해 캐시를 스크래치로.
# 이 서버엔 /scratch가 없음 (확인됨) — $HOME이 이미 대용량 디스크(/src)에 있으므로 그 아래를 스크래치로 사용.
_FLOWDINO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_FLOWDINO_GPU_GUARD="$_FLOWDINO_ROOT/scripts/gpu_guard.py"
SCRATCH="${SCRATCH:-$HOME/scratch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.uv_cache}"
export HF_HOME="${HF_HOME:-$SCRATCH/.hf_home}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$_FLOWDINO_ROOT/.venv}"
if [[ -z "${PY:-}" ]]; then
  if [[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
    PY="$UV_PROJECT_ENVIRONMENT/bin/python"
  else
    PY="uv run python"
  fi
fi
export PY
unset _FLOWDINO_ROOT

# Keep interpreter arguments as an array; callers can use flowdino_python safely.
# PY is an executable path or a simple command (e.g. "uv run python"), never shell code.
flowdino_python() {
  if [[ -x "$PY" ]]; then
    "$PY" "$@"
  else
    local -a python_command
    read -r -a python_command <<< "$PY"
    "${python_command[@]}" "$@"
  fi
}

flowdino_gpu() {
  local gpu="$1"; shift
  flowdino_python "$_FLOWDINO_GPU_GUARD" \
    --gpu "$gpu" --wait -- "$@"
}

flowdino_gpu_python() {
  local gpu="$1"; shift
  if [[ -x "$PY" ]]; then
    flowdino_gpu "$gpu" "$PY" "$@"
  else
    local -a python_command
    read -r -a python_command <<< "$PY"
    flowdino_gpu "$gpu" "${python_command[@]}" "$@"
  fi
}

# 계산 노드가 오프라인이면 주석 해제 (로그인 노드에서 prepare_data.py 선행 필수)
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

echo "[env] UV_CACHE_DIR=$UV_CACHE_DIR"
echo "[env] HF_HOME=$HF_HOME"
echo "[env] PY=$PY"
