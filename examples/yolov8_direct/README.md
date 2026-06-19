# Example — YOLOv8 direct train + test

[English](README.md) | [中文](README.zh-CN.md)

Smoke test: train the YOLOv8 baseline (no SAC, no cross-validation) on the original train/valid split, then evaluate `best.pt` on the held-out test split.

## Commands

```bash
# 1. Train (validation on the valid split during training).
uv run python -m scripts.train \
    --model yolov8s \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --output-dir examples/yolov8_direct/output/train \
    --epochs 10 \
    --batch-size 8 \
    --lr 1e-3 \
    --num-workers 0 \
    --device cuda

# 2. Test (evaluate the resulting best.pt on the test split).
uv run python -m scripts.test \
    --model yolov8s \
    --weights examples/yolov8_direct/output/train/best.pt \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --split test \
    --output-dir examples/yolov8_direct/output/test \
    --batch-size 8 \
    --num-workers 0 \
    --device cuda
```

## Configuration

- **Model**: `yolov8s` (YOLOv8s baseline, anchor-free, no SAC).
- **Dataset**: `CrackDetection_Augmentation.v1.yolov5pytorch` — original splits respected (train 770 / valid 220 / test 110), 1 class (`cracks`).
- **Training**: 10 epochs, batch 8, lr 1e-3, AMP on, seed 42. Validation runs on the `valid` split each epoch.
- `--num-workers 0` is required on Windows (multi-worker DataLoaders hit shared-memory limits).

## Artifacts

All artifacts are gitignored under `examples/yolov8_direct/output/`:

**Train** (`output/train/`):
- `metrics.csv` — per-epoch losses + validation metrics.
- `loss_curve.png`, `metric_curve.png`, `config.yaml`, `best.pt`.
- `run.log.jsonl`.

**Test** (`output/test/`):
- `metrics.csv` — accuracy + efficiency (FPS, latency, params, GFLOPs, peak VRAM).
- `model_analysis.json` — full efficiency report.
- `per_image/<id>.json`, `predictions/<id>.jpg`, `curves/{pr,roc,confusion}.png`.
- `run.log.jsonl`.
