# Experiment configurations

[English](README.md) | [中文](README.zh-CN.md)

Two YAML configs drive the 26-model sweep for tongue surface crack detection. Both are read by `scripts/schedule_experiments.py` and run via `python -m scripts.train` / `python -m scripts.test` in subprocesses (see `docs/scheduler.md`).

## Files

- **`all_models_direct.yaml`** — 26 models × 2 experiments (train + test) = 52 experiments. Each model is trained on the `train` split (validation on the `valid` split during training), then `best.pt` is evaluated on the held-out `test` split. Produces one direct train→test metric per model.
- **`all_models_cv5.yaml`** — 26 models × 1 cross-validation experiment = 26 experiments. Each experiment merges the original train+valid+test splits into one pool, runs `StratifiedKFold(n_splits=5)`. Per fold: held-out fold = **TEST**, remaining records split into train (90%) + val (10%, `val_fraction=0.1`) for backprop validation. Produces per-fold metrics + aggregated mean ± std per model.

## Usage

```bash
# Direct sweep (52 experiments).
python -m scripts.schedule_experiments \
    --config experiments/all_models_direct.yaml \
    --output-dir output/all_models_direct

# 5-fold CV sweep (26 experiments).
python -m scripts.schedule_experiments \
    --config experiments/all_models_cv5.yaml \
    --output-dir output/all_models_cv5

# Retry any failed experiments.
python -m scripts.schedule_experiments \
    --retry-failed output/all_models_direct/scheduler/errors.jsonl \
    --output-dir output/all_models_direct_retry
```

## Multi-GPU

Set `scheduler.max_parallel` to the GPU count and add `env: {CUDA_VISIBLE_DEVICES: "N"}` to each experiment (N = 0/1/2/...). Batch sizes can then be reduced to ~1/N. See the YAML header comments for details.

## Batch sizes (single high-VRAM GPU)

| Family | Batch size |
| --- | --- |
| YOLOv8n | 128 |
| YOLOv5s / YOLOv8s / YOLOv10s | 64 |
| YOLOv7w / YOLOv8m / YOLOv9c | 32 |
| YOLOv8l | 16 |
| YOLOv8x / Faster-RCNN / Mask-RCNN | 8 |
| RetinaNet / FCOS | 16 |
| SSD300 | 32 |
| SSDlite320 | 64 |

If a run OOMs, halve the batch size and retry. YOLO uses `lr=0.01`; torchvision detectors use `lr=0.0001` (lower — wrappers default to SGD with `lr=1e-4`).
