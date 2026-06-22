#!/usr/bin/env bash
# Quick-start: clone + install + data + launch training on a fresh autodl server.
#
# Usage (run on the server, as root):
#   GROUP=1 bash <(curl -sL https://raw.githubusercontent.com/6ixGODD/cracks-yolo/master/scripts/quickstart_server.sh)
#   # or, after cloning manually:
#   GROUP=1 bash scripts/quickstart_server.sh
#
# Env vars:
#   GROUP     = 1|2|3|4  (which compose_4/group_N.yaml to run)  [required]
#   REPO_URL  = clone URL (default ghfast.top mirror)
#   DATA_SRC  = dataset source path on autodl  (default /root/autodl-fs/...)
#   BRANCH    = master
#
# After launch: tail -f /root/cracks-yolo/output/group_${GROUP}/scheduler/*.log
set -euo pipefail

GROUP="${GROUP:-}"
[ -z "$GROUP" ] && { echo "ERROR: set GROUP=1|2|3|4"; exit 1; }

REPO_URL="${REPO_URL:-https://ghfast.top/https://github.com/6ixGODD/cracks-yolo.git}"
BRANCH="${BRANCH:-master}"
DATA_SRC="${DATA_SRC:-/root/autodl-fs/CrackDetection_Augmentation.v1.yolov5pytorch}"
DATA_DST="${DATA_DST:-/root/autodl-tmp/CrackDetection_Augmentation.v1.yolov5pytorch}"
REPO_DIR="${REPO_DIR:-/root/cracks-yolo}"

echo "=== [1/6] clone repo ==="
if [ -d "$REPO_DIR/.git" ]; then
  cd "$REPO_DIR" && git fetch --all && git reset --hard "origin/$BRANCH"
else
  git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi
git submodule update --init --recursive || true  # deps may be large; ok to skip

echo "=== [2/6] copy dataset to fast local disk ==="
if [ ! -d "$DATA_DST" ]; then
  cp -r "$DATA_SRC" "$DATA_DST"
else
  echo "  dataset already at $DATA_DST, skipping copy"
fi
# symlink data/ so the YAML paths resolve
mkdir -p data
ln -sfn "$DATA_DST" "data/CrackDetection_Augmentation.v1.yolov5pytorch"

echo "=== [3/6] install deps ==="
# Prefer conda python if present (autodl default), else system python3.
PY="$(command -v python || command -v python3)"
$PY -m pip install -q --upgrade pip
$PY -m pip install -q ultralytics thop torchsummary pycocotools opencv-python \
  pyyaml loguru pydantic scipy scikit-learn statsmodels seaborn pandas matplotlib \
  tabulate tqdm requests 2>&1 | tail -2 || true
echo "  torch check:"
$PY -c "import torch; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== [4/6] free disk (clear any stale output) ==="
rm -rf "$REPO_DIR/output" 2>/dev/null || true
mkdir -p "$REPO_DIR/output/group_$GROUP"

echo "=== [5/6] launch group_$GROUP (nohup, background) ==="
cd "$REPO_DIR"
OUT="output/group_$GROUP"
LOG="$OUT/scheduler_main.log"
nohup $PY -m scripts.schedule_experiments \
  --config "experiments/compose_4/group_$GROUP.yaml" \
  --output-dir "$OUT" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "  started scheduler pid=$PID"
echo "  log: $LOG"
echo "  PID written to: $OUT/scheduler.pid"
echo "$PID" > "$OUT/scheduler.pid"

echo "=== [6/6] monitor (first 60s) ==="
sleep 5
for i in $(seq 1 12); do
  sleep 5
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "  scheduler EXITED early! tail of log:"
    tail -20 "$LOG"
    exit 1
  fi
  echo "  --- t=${i}*5s ---"
  tail -3 "$LOG" 2>/dev/null | sed 's/^/    /'
done

echo ""
echo "=== LAUNCHED. To monitor: ==="
echo "  tail -f $LOG"
echo "  ls $OUT/scheduler/         # per-experiment logs"
echo "  cat $OUT/scheduler/results.jsonl   # completed"
echo "  cat $OUT/scheduler/errors.jsonl    # failed"
echo ""
echo "=== After sweep finishes, zip + move to autodl-fs: ==="
echo "  cd $OUT && zip -r /root/autodl-fs/group_${GROUP}.zip . && cd -"
