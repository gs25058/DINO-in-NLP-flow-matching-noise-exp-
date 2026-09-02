#!/usr/bin/env bash
# TensorBoard has no built-in "sort by recency" or "hide old runs" option — it just
# lists every subdirectory under --logdir. This wrapper builds a symlink farm of only
# the runs whose most recent event file was touched within the last N hours, and
# points TensorBoard at that instead, so the initial screen isn't cluttered with
# dozens of stale runs.
#
# Usage: scripts/tb_recent.sh [hours] [port]
#   hours  runs with no event-file update in the last HOURS are hidden (default: 48)
#   port   TensorBoard port (default: 6006)
#
# Original data under results/tensorboard/ is untouched (symlinks only).
set -euo pipefail

HOURS="${1:-48}"
PORT="${2:-6006}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/results/tensorboard"
DST="$ROOT/results/tensorboard_recent"

if [[ ! -d "$SRC" ]]; then
  echo "[tb_recent] no such dir: $SRC" >&2
  exit 1
fi

rm -rf "$DST"
mkdir -p "$DST"

# this server's `find` is bfs, which (unlike GNU findutils) rejects relative
# "-N hours" timestamps for -newermt and wants an absolute ISO 8601 one instead.
CUTOFF="$(date -d "-${HOURS} hours" +%Y-%m-%dT%H:%M:%S)"

total=0
shown=0
for run_dir in "$SRC"/*/; do
  [[ -d "$run_dir" ]] || continue
  run_name="$(basename "$run_dir")"
  total=$((total + 1))
  if find "$run_dir" -name "events.out.tfevents*" -newermt "$CUTOFF" -print -quit | grep -q .; then
    ln -s "$run_dir" "$DST/$run_name"
    shown=$((shown + 1))
  fi
done

echo "[tb_recent] showing ${shown}/${total} runs updated in the last ${HOURS}h"
if [[ "$shown" -eq 0 ]]; then
  echo "[tb_recent] nothing matched — widen the window, e.g. scripts/tb_recent.sh 168" >&2
fi

source "$ROOT/scripts/env.sh" 2>/dev/null || true
exec uv run tensorboard --logdir "$DST" --port "$PORT" --host 0.0.0.0
