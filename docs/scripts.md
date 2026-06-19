# Scripts

`scripts/` contains all CLI entry points. Each script is a thin wrapper
around `cracks_yolo.*` modules. All accept `--config <yaml>` and individual
`--flags`, and write to `--output-dir`.

## train.py

```bash
python -m scripts.train \
    --model yolov5s_sactr \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --epochs 100 --batch-size 32 --input-size 640 \
    --output-dir output/yolov5s_sactr \
    --seed 42
```

Flags:
- `--model` — ZOO key (see `cracks_yolo.zoo.ZOO`).
- `--dataset` — YOLOv5-format dataset root (contains `data.yaml` + `train/`, `valid/`, `test/`).
- `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--input-size` — training hyperparameters.
- `--amp` / `--no-amp` — toggle AMP (default off).
- `--num-workers` — DataLoader workers (default 4).
- `--device` — `cuda` or `cpu` (default `cuda`).
- `--seed` — reproducibility seed (default 42).
- `--val-interval` — validate every N epochs (default 1).
- `--log-every-n-steps` — train-step log frequency (default 10).
- `--cross-val` — switch to 5-fold CV mode (uses `--n-folds`, `--val-split`, `--train-split`).

Emits to `output-dir`: `run.log.jsonl`, `metrics.csv`, `loss_curve.png`,
`metric_curve.png`, `config.yaml`, `best.pt`.

## test.py

```bash
python -m scripts.test \
    --model yolov5s_sactr \
    --weights output/yolov5s_sactr/best.pt \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --input-size 640 \
    --output-dir output/yolov5s_sactr_test
```

Emits: `metrics.csv`, `per_image/<id>.json`, `predictions/<id>.jpg`,
`curves/{pr,roc,confusion}.png`, `TestLog` in `run.log.jsonl`.

## convert_dataset.py

```bash
python -m scripts.convert_dataset \
    --input data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --from yolo --to coco \
    --output data/Crack_coco
```

## heatmap.py

```bash
python -m scripts.heatmap \
    --model yolov5s_sactr \
    --weights output/yolov5s_sactr/best.pt \
    --input data/CrackDetection_Augmentation.v1.yolov5pytorch/test \
    --layers backbone.8,backbone.9 \
    --output-dir output/heatmaps
```

Generates Grad-CAM heatmaps for the specified backbone layers. Per image per
layer: `heatmaps/<image_id>/<layer>.png` + `feature_maps/<image_id>/<layer>.npy`.

**Layer naming**: use dot-notation relative to the model's top-level
attributes. YOLOv5s backbone has 10 children (indices 0-9), so valid layers
are `backbone.0` through `backbone.9`. Invalid indices raise
`IndexError: index N is out of range` — check `len(model.backbone)` first.

See `docs/heatmap.md` for methodology.

## analyze_dataset.py

```bash
python -m scripts.analyze_dataset \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --output-dir output/dataset_analysis
```

Emits: `class_distribution.png`, `bbox_size_distribution.png`,
`bbox_position_heatmap.png`, `image_size_distribution.png`,
`diversity_metrics.json` (Shannon entropy, unique bbox aspect-ratio buckets,
spatial coverage).

## analyze_model.py

```bash
python -m scripts.analyze_model \
    --model yolov5s_sactr \
    --input-size 640 \
    --output-dir output/model_analysis
```

Emits: `params.csv`, `macs.csv` (via `fvcore.nn.FlopCountAnalysis`),
`latency.csv` (p50/p95 over 100 runs, CPU + CUDA), `vram.csv` (peak
`torch.cuda.max_memory_allocated`), `comparison_plot.png`.

Use `--all` to run on every ZOO entry:

```bash
python -m scripts.analyze_model --all --output-dir output/model_analysis_all
```

## schedule_experiments.py

YAML-driven batch scheduler. See `docs/scheduler.md`.

```bash
python -m scripts.schedule_experiments --config experiments.yaml --output-dir output/scheduler
python -m scripts.schedule_experiments --retry-failed output/scheduler/errors.jsonl --output-dir output/scheduler_retry
```

## compare_models.py

```bash
python -m scripts.compare_models \
    --models yolov5s,yolov5s_sactr,yolov8s,yolov10s,yolov9c,retinanet_r50,faster_rcnn_r50 \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --n-folds 5 --epochs 100 \
    --metric map50 \
    --output-dir output/comparison
```

Runs 5-fold CV for each model, then per-fold paired t-test on the chosen
metric. Emits `comparison.csv`, `paired_t_test.csv`, `comparison_plot.png`.
