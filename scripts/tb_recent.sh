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
exec uv run tensorboard --logdir "$LOGDIR" --port "$PORT" --host 0.0.0.0
