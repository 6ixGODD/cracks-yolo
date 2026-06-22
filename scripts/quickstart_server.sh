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

echo "=== [4/7] predownload ALL pretrained weights to shared autodl-fs ==="
# Shared cache on autodl-fs (persists across servers). github assets via
# ghfast.top mirror (China-friendly); pytorch.org + fbaipublicfiles direct.
# Existing files are skipped (size-checked) so re-runs are fast.
WEIGHTS_DIR="${WEIGHTS_DIR:-/root/autodl-fs/weights}"
mkdir -p "$WEIGHTS_DIR"
GH="https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0"
GH_V7="https://ghfast.top/https://github.com/WongKinYiu/yolov7/releases/download/v0.1"

download() {  # download <url> <filename> [mirror_url]
  local url="$1" f="$WEIGHTS_DIR/$2" mirror="${3:-}"
  if [ -f "$f" ] && [ "$(stat -c%s "$f" 2>/dev/null || echo 0)" -gt 1000000 ]; then
    echo "  $2 already present ($(stat -c%s "$f") bytes)"
    return 0
  fi
  echo -n "  $2 ... "
  for attempt in 1 2 3 4 5; do
    local u="$url"
    [ "$attempt" -le 2 ] && [ -n "$mirror" ] && u="$mirror"
    if curl -sL --connect-timeout 60 --max-time 1800 --retry 3 --retry-delay 10 -o "$f" "$u"; then
      local sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
      if [ "$sz" -gt 1000000 ]; then echo "OK (${sz}B)"; return 0; fi
    fi
    echo -n "retry$attempt "
    rm -f "$f"; sleep 5
  done
  echo "FAILED"
  return 1
}

echo "  -- YOLO (ultralytics, github mirror) --"
for a in yolov5nu yolov5su yolov5mu yolov5lu yolov5xu \
         yolov8n yolov8s yolov8m yolov8l yolov8x \
         yolov9t yolov9s yolov9m yolov9c yolov9e \
         yolov10n yolov10s yolov10m yolov10b yolov10l yolov10x \
         yolov3u yolov3-tinyu yolov3-sppu; do
  download "${GH}/${a}.pt" "${a}.pt" || true
done
echo "  -- YOLOv7 (WongKinYiu, github mirror) --"
download "${GH_V7}/yolov7.pt" "yolov7.pt" || true
echo "  -- torchvision (pytorch.org -> mirror first, direct fallback) --"
PT_MIRROR="https://ghfast.top/https://download.pytorch.org/models"  # fast mirror, fallback direct
download "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth" \
  "retinanet_resnet50_fpn_coco.pth" "${PT_MIRROR}/retinanet_resnet50_fpn_coco-eeacb38b.pth" || true
download "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" \
  "fasterrcnn_resnet50_fpn_coco.pth" "${PT_MIRROR}/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" || true
download "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth" \
  "maskrcnn_resnet50_fpn_coco.pth" "${PT_MIRROR}/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth" || true
download "https://download.pytorch.org/models/fcos_resnet50_fpn_coco-99b0c9b7.pth" \
  "fcos_resnet50_fpn_coco.pth" "${PT_MIRROR}/fcos_resnet50_fpn_coco-99b0c9b7.pth" || true
download "https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth" \
  "ssd300_vgg16_coco.pth" "${PT_MIRROR}/ssd300_vgg16_coco-b556d3b4.pth" || true
download "https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth" \
  "ssdlite320_mobilenet_v3_large_coco.pth" "${PT_MIRROR}/ssdlite320_mobilenet_v3_large_coco-a79551df.pth" || true
echo "  -- DETR (fbaipublicfiles, direct) --"
download "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth" "detr-r50-e632da11.pth" || true

echo "=== [4b/7] symlink weights to where each loader looks ==="
cd "$REPO_DIR"
# (a) ultralytics: looks for <asset>.pt in cwd.
for f in "$WEIGHTS_DIR"/*.pt; do
  [ -f "$f" ] || continue
  ln -sfn "$f" "$(basename "$f")"
done
# (b) torchvision: looks in ~/.cache/torch/hub/checkpoints/<official-name>.pth
TV_CACHE="$HOME/.cache/torch/hub/checkpoints"
mkdir -p "$TV_CACHE"
declare -A TV_MAP=(
  ["retinanet_resnet50_fpn_coco.pth"]="retinanet_resnet50_fpn_coco-eeacb38b.pth"
  ["fasterrcnn_resnet50_fpn_coco.pth"]="fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
  ["maskrcnn_resnet50_fpn_coco.pth"]="maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
  ["fcos_resnet50_fpn_coco.pth"]="fcos_resnet50_fpn_coco-99b0c9b7.pth"
  ["ssd300_vgg16_coco.pth"]="ssd300_vgg16_coco-b556d3b4.pth"
  ["ssdlite320_mobilenet_v3_large_coco.pth"]="ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
)
for friendly in "${!TV_MAP[@]}"; do
  official="${TV_MAP[$friendly]}"
  if [ -f "$WEIGHTS_DIR/$friendly" ]; then
    ln -sfn "$WEIGHTS_DIR/$friendly" "$TV_CACHE/$official"
  fi
done
# (c) cracks_yolo weights/loader.py: caches to weights/{key}.pt (key = spec.key).
#     v7 key=yolov7w -> weights/yolov7w.pt; detr key=detr_r50 -> weights/detr_r50.pt
mkdir -p "$REPO_DIR/weights"
[ -f "$WEIGHTS_DIR/yolov7.pt" ] && ln -sfn "$WEIGHTS_DIR/yolov7.pt" "$REPO_DIR/weights/yolov7w.pt"
[ -f "$WEIGHTS_DIR/detr-r50-e632da11.pth" ] && ln -sfn "$WEIGHTS_DIR/detr-r50-e632da11.pth" "$REPO_DIR/weights/detr_r50.pt"
echo "  symlinks created (cwd for ultralytics, ~/.cache for torchvision, weights/ for v7+detr)"

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
