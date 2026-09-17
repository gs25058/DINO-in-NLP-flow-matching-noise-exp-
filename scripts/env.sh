#!/usr/bin/env bash
# 실행 환경 변수. 셸에서 `source scripts/env.sh`로 쓰고, 실행기(run_*.sh)도 이 파일을 source한다.
# 이미 설정된 값은 덮어쓰지 않는다 - 서버마다 다른 값은 셸 프로필이나 실행 시 환경변수로 준다.
#
#   SCRATCH                 캐시 기본 위치 (기본 $HOME/scratch)
#   UV_CACHE_DIR            uv 캐시 (기본 $SCRATCH/.uv_cache)
#   HF_HOME                 HF 모델/데이터셋 캐시 (기본 $SCRATCH/.hf_home)
#   UV_PROJECT_ENVIRONMENT  venv 위치 (기본 <repo>/.venv). 저장소가 네트워크 FS에 있으면 로컬 디스크로
#                           빼는 것이 빠르다. 설정해 두면 worktree에서 `uv run`을 해도 venv를 새로 만들지 않는다.
#   PY                      실행기가 쓸 파이썬 (기본: venv의 python, 없으면 "uv run python")
_FLOWDINO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH:-$HOME/scratch}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.uv_cache}"
export HF_HOME="${HF_HOME:-$SCRATCH/.hf_home}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

if [[ -z "${PY:-}" ]]; then
  _venv="${UV_PROJECT_ENVIRONMENT:-$_FLOWDINO_ROOT/.venv}"
  if [[ -x "$_venv/bin/python" ]]; then PY="$_venv/bin/python"; else PY="uv run python"; fi
  unset _venv
fi
export PY
unset _FLOWDINO_ROOT

# 계산 노드가 오프라인이면 주석 해제 (로그인 노드에서 prepare_data.py 선행 필수)
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

echo "[env] UV_CACHE_DIR=$UV_CACHE_DIR"
echo "[env] HF_HOME=$HF_HOME"
echo "[env] PY=$PY"
