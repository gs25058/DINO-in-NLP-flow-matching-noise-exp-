#!/usr/bin/env bash
# Thin wrapper around TensorBoard: sources env.sh and points --logdir at
# results/tensorboard/ directly. Stale runs are no longer hidden via a
# symlink farm — run scripts/archive_old_runs.sh periodically to move
# finished runs out to results/tensorboard_archive/, and use TensorBoard's
# own run-name regex filter (top-left search box in the UI) if you need to
# narrow the view further.
#
# Usage: scripts/tb_recent.sh [port]
#   port  TensorBoard port (default: 6006)
set -euo pipefail

PORT="${1:-6006}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$ROOT/results/tensorboard"

if [[ ! -d "$LOGDIR" ]]; then
  echo "[tb_recent] no such dir: $LOGDIR" >&2
  exit 1
fi

source "$ROOT/scripts/env.sh" 2>/dev/null || true
# env.sh가 정한 PY와 같은 venv의 tensorboard를 쓴다. `uv run`은 cwd 프로젝트 환경을 찾아 동기화하므로
# venv가 저장소 밖(UV_PROJECT_ENVIRONMENT)에 있으면 여기에 5.6GB짜리 .venv를 새로 만든다.
TB="$(dirname "${PY:-}")/tensorboard"
if [[ -x "$TB" ]]; then
  exec "$TB" --logdir "$LOGDIR" --port "$PORT" --host 0.0.0.0
fi
exec uv run tensorboard --logdir "$LOGDIR" --port "$PORT" --host 0.0.0.0
