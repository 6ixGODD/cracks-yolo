#!/usr/bin/env bash
# Quick-start: clone + install + data + predownload weights + launch training.
# No set -e (fragile pipe exits killed the script). Each step checks its own errors.
#
# Usage (on the server, as root):
#   GROUP=1 bash scripts/quickstart_server.sh
#   GROUP=2 bash scripts/quickstart_server.sh
#   GROUP=3 bash scripts/quickstart_server.sh
#   GROUP=4 bash scripts/quickstart_server.sh
GROUP="${GROUP:-}"
[ -z "$GROUP" ] && { echo "ERROR: set GROUP=1|2|3|4"; exit 1; }

REPO_URL="${REPO_URL:-https://ghfast.top/https://github.com/6ixGODD/cracks-yolo.git}"
BRANCH="${BRANCH:-master}"
DATA_SRC="${DATA_SRC:-/root/autodl-fs/CrackDetection_Augmentation.v1.yolov5pytorch}"
DATA_DST="${DATA_DST:-/root/autodl-tmp/CrackDetection_Augmentation.v1.yolov5pytorch}"
REPO_DIR="${REPO_DIR:-/root/cracks-yolo}"
PY="$(command -v python || command -v python3)"

echo "=== [1/7] clone repo ==="
if [ -d "$REPO_DIR/.git" ]; then
  cd "$REPO_DIR" && git fetch --all && git reset --hard "origin/$BRANCH"
else
  git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi

echo "=== [2/7] copy dataset to fast local disk ==="
if [ ! -d "$DATA_DST" ]; then
  cp -r "$DATA_SRC" "$DATA_DST"
else
  echo "  dataset already at $DATA_DST"
fi
mkdir -p data
ln -sfn "$DATA_DST" "data/CrackDetection_Augmentation.v1.yolov5pytorch"

echo "=== [3/7] install deps ==="
$PY -m pip install -q ultralytics thop torchsummary pycocotools opencv-python \
  pyyaml loguru pydantic scipy scikit-learn statsmodels seaborn pandas matplotlib \
  tabulate tqdm requests 2>&1 | tail -3
$PY -c "import torch; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== [4/7] predownload ALL YOLO weights via ghfast.top mirror ==="
# ultralytics downloads from github at runtime (slow/corrupt from China).
# Pre-fetch every asset the sweep needs via the ghfast.top mirror, placing them
# where ultralytics' YOLO() looks first (cwd). This avoids runtime download failures.
# Also purge any corrupt partial .pt files (ultralytics detects + retries endlessly).
MIRROR="https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0"
ASSETS="yolov5nu yolov5su yolov5mu yolov5lu yolov5xu \
        yolov8n yolov8s yolov8m yolov8l yolov8x \
        yolov9t yolov9s yolov9m yolov9c yolov9e \
        yolov10n yolov10s yolov10m yolov10b yolov10l yolov10x \
        yolov3u yolov3-tinyu yolov3-sppu"
# purge corrupt/empty .pt files in cwd
for f in *.pt; do
  [ -f "$f" ] || continue
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$sz" -lt 1000000 ]; then
    echo "  purging corrupt $f (${sz} bytes)"
    rm -f "$f"
  fi
done
for asset in $ASSETS; do
  f="${asset}.pt"
  if [ -f "$f" ] && [ $(stat -c%s "$f" 2>/dev/null || echo 0) -gt 1000000 ]; then
    echo "  $f already present"
  else
    echo -n "  downloading $f ... "
    # retry up to 5 times
    for attempt in 1 2 3 4 5; do
      if curl -sL --connect-timeout 30 --max-time 600 -o "$f" "${MIRROR}/${f}"; then
        sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if [ "$sz" -gt 1000000 ]; then
          echo "OK (${sz} bytes)"
          break
        fi
      fi
      echo "retry $attempt..."
      rm -f "$f"
    done
    if [ ! -f "$f" ] || [ $(stat -c%s "$f" 2>/dev/null || echo 0) -lt 1000000 ]; then
      echo "FAILED — try: curl -L -o $f ${MIRROR}/${f}"
    fi
  fi
done

echo "=== [5/7] free disk ==="
rm -rf "$REPO_DIR/output" 2>/dev/null
mkdir -p "$REPO_DIR/output/group_$GROUP"

echo "=== [6/7] launch group_$GROUP ==="
cd "$REPO_DIR"
OUT="output/group_$GROUP"
LOG="$OUT/scheduler_main.log"
nohup $PY -m scripts.schedule_experiments \
  --config "experiments/compose_4/group_$GROUP.yaml" \
  --output-dir "$OUT" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/scheduler.pid"
echo "  scheduler pid=$PID"
echo "  log: $LOG"
echo ""
echo "  scheduler is running in background (nohup). You can disconnect SSH safely."

echo "=== [7/7] quick health check (10s) ==="
sleep 10
if kill -0 "$PID" 2>/dev/null; then
  echo "  scheduler ALIVE after 10s — looks good."
  echo "  latest log:"
  tail -5 "$LOG" 2>/dev/null | sed 's/^/    /'
else
  echo "  scheduler EXITED within 10s — check log:"
  tail -20 "$LOG" 2>/dev/null | sed 's/^/    /'
fi

echo ""
echo "=========================================="
echo "MONITOR:"
echo "  tail -f $LOG"
echo "  ls $OUT/scheduler/"
echo "  cat $OUT/scheduler/results.jsonl   # done"
echo "  cat $OUT/scheduler/errors.jsonl    # failed"
echo ""
echo "AFTER SWEEP:"
echo "  cd $OUT && zip -r /root/autodl-fs/group_${GROUP}.zip ."
echo "=========================================="
