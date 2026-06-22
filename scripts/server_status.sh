#!/usr/bin/env bash
# Check status of a running sweep on a server (run on the server itself).
#
# Usage:  bash scripts/server_status.sh [GROUP]
#   GROUP defaults to scanning all group_* dirs.
set -euo pipefail
REPO_DIR="${REPO_DIR:-/root/cracks-yolo}"
cd "$REPO_DIR/output" 2>/dev/null || { echo "no output/ dir yet"; exit 0; }

for gdir in group_*; do
  [ -d "$gdir" ] || continue
  G="${gdir#group_}"
  echo "=== group $G ==="
  SCHED="$gdir/scheduler"
  if [ -f "$SCHED/results.jsonl" ]; then
    OK=$(grep -c '"status": "ok"' "$SCHED/results.jsonl" 2>/dev/null || echo 0)
    echo "  completed (ok): $OK"
  fi
  if [ -f "$SCHED/errors.jsonl" ]; then
    FAIL=$(wc -l < "$SCHED/errors.jsonl" 2>/dev/null || echo 0)
    [ "$FAIL" -gt 0 ] && echo "  FAILED: $FAIL  (cat $SCHED/errors.jsonl)"
  fi
  # latest experiment log tail
  latest_log=$(ls -t "$SCHED"/*.log 2>/dev/null | head -1)
  if [ -n "$latest_log" ]; then
    echo "  latest: $(basename "$latest_log")"
    tail -2 "$latest_log" 2>/dev/null | sed 's/^/    /'
  fi
  # scheduler process alive?
  if [ -f "$gdir/scheduler.pid" ]; then
    PID=$(cat "$gdir/scheduler.pid")
    kill -0 "$PID" 2>/dev/null && echo "  scheduler: RUNNING (pid $PID)" || echo "  scheduler: EXITED"
  fi
done

echo ""
echo "=== zip finished groups for download ==="
for gdir in group_*; do
  [ -d "$gdir" ] || continue
  G="${gdir#group_}"
  if [ -f "$gdir/scheduler.pid" ] && ! kill -0 "$(cat "$gdir/scheduler.pid")" 2>/dev/null; then
    ZIP="/root/autodl-fs/group_${G}.zip"
    [ -f "$ZIP" ] || { echo "  zipping group_$G -> $ZIP"; cd "$gdir" && zip -qr "$ZIP" . && cd -; }
  fi
done
