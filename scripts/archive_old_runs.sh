#!/usr/bin/env bash
# Moves stale run folders out of results/tensorboard/ so TensorBoard's initial
# run list doesn't accumulate dozens of finished experiments. A run is "stale"
# when none of its events.out.tfevents* files have been modified within the
# last DAYS days. Stale runs are `mv`d (never deleted) into
# results/tensorboard_archive/, preserving their contents; point TensorBoard
# there too (or symlink it back under tensorboard/) if you ever need to look
# at an archived run again.
#
# Usage: scripts/archive_old_runs.sh [--dry-run] [threshold]
#   --dry-run   print what would be archived without moving anything
#   threshold   staleness threshold, anything `date -d "-<threshold>"` accepts
#               (default: "3 days"; e.g. "12 hours", "7 days")
set -euo pipefail

DRY_RUN=0
THRESHOLD="3 days"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) THRESHOLD="$arg" ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/results/tensorboard"
DST="$ROOT/results/tensorboard_archive"

if [[ ! -d "$SRC" ]]; then
  echo "[archive_old_runs] no such dir: $SRC" >&2
  exit 1
fi

# this server's `find` is bfs, which (unlike GNU findutils) rejects relative
# "-N days" timestamps for -newermt and wants an absolute ISO 8601 one instead
# (see scripts/tb_recent.sh for the same workaround).
cutoff="$(date -d "-${THRESHOLD}" +%Y-%m-%dT%H:%M:%S)"

archived=0
kept=0
skipped=0

for run_dir in "$SRC"/*/; do
  [[ -d "$run_dir" ]] || continue
  run_name="$(basename "$run_dir")"

  if ! find "$run_dir" -name "events.out.tfevents*" -print -quit | grep -q .; then
    echo "[skip] $run_name (no event file)"
    skipped=$((skipped + 1))
    continue
  fi

  if find "$run_dir" -name "events.out.tfevents*" -newermt "$cutoff" -print -quit | grep -q .; then
    kept=$((kept + 1))
  else
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[would archive] $run_name"
    else
      mkdir -p "$DST"
      mv "$run_dir" "$DST/$run_name"
      echo "[archived] $run_name"
    fi
    archived=$((archived + 1))
  fi
done

verb="archived"
[[ "$DRY_RUN" -eq 1 ]] && verb="would archive"
echo "[archive_old_runs] ${verb} ${archived}, kept ${kept}, skipped ${skipped} (no event file) — cutoff: ${THRESHOLD} (${cutoff})"
