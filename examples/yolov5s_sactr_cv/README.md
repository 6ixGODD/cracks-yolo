# Example — YOLOv5s-SACTR 5-fold cross-validation

[English](README.md) | [中文](README.zh-CN.md)

Smoke test: run YOLOv5s-SACTR (SAC + TR) through 5-fold cross-validation on the tongue surface crack dataset and verify the full CV artifact set.

## Command

```bash
uv run python -m scripts.train \
    --model yolov5s_sactr \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --output-dir examples/yolov5s_sactr_cv/output/cv \
    --epochs 10 \
    --batch-size 8 \
    --lr 1e-3 \
    --num-workers 0 \
    --cross-val --n-folds 5 --val-fraction 0.1 \
    --device cuda
```

## Configuration

- **Model**: `yolov5s_sactr` (YOLOv5s backbone with Switchable Atrous Convolution + Transformer block).
- **Dataset**: `CrackDetection_Augmentation.v1.yolov5pytorch` — 770 / 220 / 110 (train / valid / test), 1 class (`cracks`). CV mode merges all three splits into one pool (1100 records).
- **Splits per fold**: held-out fold = **test** (220 records); remaining 880 split into train (90% = 792) + val (10% = 88) via `train_test_split(random_state=seed+fold, stratify=...)`.
- **Training**: 10 epochs, batch 8, lr 1e-3, AMP on, seed 42.
- `--num-workers 0` is required on Windows (multi-worker DataLoaders hit shared-memory limits).

## Artifacts

All artifacts are written to `examples/yolov5s_sactr_cv/output/cv/` (gitignored):

- `cv_summary.csv` — per-fold test metrics (accuracy + efficiency).
- `cv_report.json` — per-fold train/test summaries + aggregated mean ± std.
- `fold_0` … `fold_4/` — each fold's full train + test run (see `docs/pipeline.md`).
- `fold_<i>/test/metrics.csv`, `model_analysis.json`, `per_image/`, `predictions/`, `curves/`.
- `run.log.jsonl` — structured log records.
