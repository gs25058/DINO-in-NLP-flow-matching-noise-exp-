#!/usr/bin/env bash
# Discover recent campaign logs while TensorBoard is running.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$ROOT/results/tensorboard"
mkdir -p "$LOGDIR"
link_recent() {
  for events in "$ROOT"/results/experiments/*/results/tensorboard; do
    [[ -d "$events" ]] || continue
    if [[ -n "$(find "$events" -type f -name 'events.out.tfevents*' -mtime -3 -print -quit)" ]]; then
      experiment="$(basename "$(dirname "$(dirname "$events")")")"
      target="$LOGDIR/$experiment"
      if [[ ! -e "$target" && ! -L "$target" ]]; then
        ln -s "../experiments/$experiment/results/tensorboard" "$target" || true
      fi
    fi
  done
}
if [[ "${1:-}" == --watch ]]; then
  while :; do link_recent; sleep 10; done
fi
link_recent
bash "$0" --watch &
watcher=$!
"$ROOT/.venv/bin/python" "$ROOT/scripts/tensorboard_filtered.py" &
filter=$!
trap 'kill "$watcher" "$filter" 2>/dev/null || true' EXIT
"$ROOT/.venv/bin/python" -m tensorboard.main \
  --logdir "$ROOT/results/tensorboard_filtered" --host 127.0.0.1 \
  --port "${PORT:-6011}" --reload_interval 10 --load_fast=false "$@"
