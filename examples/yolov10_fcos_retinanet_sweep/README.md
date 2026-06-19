# Example — Scheduler sweep: YOLOv10 / FCOS / RetinaNet

[English](README.md) | [中文](README.zh-CN.md)

Smoke test: drive three detector paradigms through the unattended experiment scheduler in one sweep — train + test each, 5 epochs.

## Command

```bash
uv run python -m scripts.schedule_experiments \
    --config examples/yolov10_fcos_retinanet_sweep/sweep.yaml \
    --output-dir examples/yolov10_fcos_retinanet_sweep/output
```

## Configuration

Defined in [`sweep.yaml`](sweep.yaml). One scheduler run, 6 experiments (3 models × train + test), sequential (`max_parallel=1`):

| Model | Paradigm | Batch | LR |
| --- | --- | --- | --- |
| `yolov10s` | anchor-free, NMS-free | 8 | 1e-3 |
| `fcos_r50` | anchor-free, center-ness | 4 | 1e-4 |
| `retinanet_r50` | anchor-based, focal loss | 4 | 1e-4 |

- **Dataset**: `CrackDetection_Augmentation.v1.yolov5pytorch` — original splits (train 770 / valid 220 / test 110), 1 class (`cracks`).
- **Training**: 5 epochs each, AMP on, seed 42. The two R50 detectors use batch 4 + lr 1e-4 to fit a 4 GB GPU.
- `num_workers: 0` everywhere (Windows shared-memory constraint).

## Artifacts

All artifacts are gitignored under `examples/yolov10_fcos_retinanet_sweep/output/`:

- `scheduler/results.jsonl` — one record per successful experiment (elapsed, output_dir).
- `scheduler/errors.jsonl` — failures (exit code + log path), created only on error.
- `scheduler/<exp_name>.log` — full stdout/stderr per experiment.
- `<model>/` — each model's train run (`metrics.csv`, `loss_curve.png`, `best.pt`, …) and `test/` subdir (`metrics.csv` with efficiency, `model_analysis.json`, `per_image/`, `predictions/`, `curves/`).
