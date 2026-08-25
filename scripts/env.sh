#!/usr/bin/env bash
# 공용 서버용: 홈 쿼터 보호를 위해 캐시를 스크래치로.
# 이 서버엔 /scratch가 없음 (확인됨) — $HOME이 이미 대용량 디스크(/src)에 있으므로 그 아래를 스크래치로 사용.
SCRATCH="${SCRATCH:-$HOME/scratch}"
mkdir -p "$SCRATCH/.uv_cache" "$SCRATCH/.hf_home"

export UV_CACHE_DIR="$SCRATCH/.uv_cache"
export HF_HOME="$SCRATCH/.hf_home"

# 계산 노드가 오프라인이면 주석 해제 (로그인 노드에서 prepare_data.py 선행 필수)
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

echo "[env] UV_CACHE_DIR=$UV_CACHE_DIR"
echo "[env] HF_HOME=$HF_HOME"
