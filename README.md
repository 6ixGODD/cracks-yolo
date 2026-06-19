# cracks-yolo

[English](README.md) | [中文](README.zh-CN.md)

Self-contained PyTorch model zoo for **tongue surface crack detection**,
with SAC (Switchable Atrous Convolution) and TR (Transformer) enhancements
across YOLOv5 / v7 / v8 / v9 / v10, plus torchvision RetinaNet, Faster R-CNN,
Mask R-CNN, FCOS, SSD300, and SSDlite320 baselines for cross-paradigm
comparison. Every model is a single `nn.Module` class that owns its layers,
loss, optimizer-builder, and pretrained-weight loader. No runtime YAML
parsing, no external framework coupling.

## Models (26 ZOO entries)

### YOLO family (anchor-based)

| Key | Class | Notes |
| --- | --- | --- |
| `yolov5s` | `YOLOv5s` | baseline |
| `yolov5s_sac` | `YOLOv5sSAC` | + SAC in backbone |
| `yolov5s_tr` | `YOLOv5sTR` | + TR (Transformer) |
| `yolov5s_sactr` | `YOLOv5sSACTR` | + SAC + TR |
| `yolov7w` | `YOLOv7w` | OTA loss, RepConv |
| `yolov7w_sac` | `YOLOv7wSAC` | + SAC |

### YOLO family (anchor-free)

| Key | Class | Notes |
| --- | --- | --- |
| `yolov8n` / `yolov8s` / `yolov8m` / `yolov8l` / `yolov8x` | `YOLOv8{n,s,m,l,x}` | n/s/m/l/x sizes, DFL, CIoU |
| `yolov8n_sac` ... `yolov8x_sac` | `YOLOv8{n,s,m,l,x}SAC` | + SAC in C2f |
| `yolov9c` | `YOLOv9c` | GELAN backbone, SPPELAN neck (no PGI) |
| `yolov9c_sac` | `YOLOv9cSAC` | + SAC (C2fSAC fallback) |
| `yolov10s` | `YOLOv10s` | NMS-free, dual one2many/one2one head |
| `yolov10s_sac` | `YOLOv10sSAC` | + SAC |

### Cross-paradigm baselines (torchvision)

| Key | Class | Notes |
| --- | --- | --- |
| `retinanet_r50` | `RetinaNetR50` | single-stage anchor-based, FocalLoss |
| `faster_rcnn_r50` | `FasterRCNNR50` | two-stage, RPN + ROI |
| `mask_rcnn_r50` | `MaskRCNNR50` | two-stage + mask head (bbox-fill masks) |
| `fcos_r50` | `FCOSR50` | anchor-free, centerness branch |
| `ssd300_vgg16` | `SSD300VGG16` | single-stage classic, 300x300 input |
| `ssdlite320_mobilenetv3` | `SSDlite320MobileNetV3` | lightweight, 320x320 input |

## Quickstart

```bash
uv sync          # or: pip install -e .

# For CUDA 11.8 support:
uv pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

```python
import torch
from cracks_yolo.zoo import ZOO

model = ZOO["yolov5s_sactr"](num_classes=1)
model.train()

x = torch.randn(2, 3, 640, 640)
preds = model(x)

targets = torch.tensor(
    [[0, 0, 0.50, 0.50, 0.20, 0.20],
     [1, 0, 0.40, 0.40, 0.15, 0.25]],
    dtype=torch.float32,
)
loss, parts = model.compute_loss(preds, targets, imgs=x)
loss.backward()
```

Load COCO pretrained weights (baseline variants only -- SAC/TR layers
have no COCO weights, so partial-load with `strict=False`):

```python
from cracks_yolo.zoo import YOLOv5s
model = YOLOv5s.from_pretrained(num_classes=1)  # downloads + strict=False load
```

### Train / test / cross-val / compare

```bash
# Single train with COCO pretrained init.
python -m scripts.train --model yolov5s_sactr --pretrained \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --epochs 100 --batch-size 32 --output-dir output/yolov5s_sactr

# 5-fold cross-validation. Merges train+valid+test into one pool,
# held-out fold = test, remaining 90/10 split for train/val.
python -m scripts.train --model yolov5s_sactr --cross-val --n-folds 5 \
    --val-fraction 0.1 --pretrained \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --epochs 100 --batch-size 32 --output-dir output/yolov5s_sactr_cv

# Multi-model comparison with paired t-test.
python -m scripts.compare_models \
    --models yolov5s,yolov5s_sactr,yolov8s,yolov9c,retinanet_r50,faster_rcnn_r50 \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --n-folds 5 --epochs 100 --output-dir output/comparison

# Batch scheduling with subprocess isolation + error capture + retry.
python -m scripts.schedule_experiments \
    --config experiments/all_models_direct.yaml \
    --output-dir output/all_models_direct
```

### Full 26-model sweeps

Two ready-to-run configs in [`experiments/`](experiments/):

- **`experiments/all_models_direct.yaml`** -- 26 models x 2 (train + test) =
  52 experiments. Direct train to test on original splits.
- **`experiments/all_models_cv5.yaml`** -- 26 models x 1 CV experiment = 26
  experiments. 5-fold CV on the merged pool (held-out fold = test).

After `git clone` + `uv sync` + cu118 torch install:

```bash
# Direct sweep (52 experiments).
python -m scripts.schedule_experiments \
    --config experiments/all_models_direct.yaml \
    --output-dir output/all_models_direct

# 5-fold CV sweep (26 experiments).
python -m scripts.schedule_experiments \
    --config experiments/all_models_cv5.yaml \
    --output-dir output/all_models_cv5
```

For multi-GPU servers, bump `scheduler.max_parallel` and add
`env: {CUDA_VISIBLE_DEVICES: "N"}` per experiment (see
[`experiments/README.md`](experiments/README.md) for batch-size tuning).

## Layout

```
cracks_yolo/
  ops/         # Conv, CSP, transformer, detect heads, SAC/TR, YOLOv9 ops.
  losses/      # ComputeLoss (v5), ComputeLossOTA (v7), v8DetectionLoss, E2ELoss (v10).
  zoo/         # 26 model classes. base.py = DetectorModel Protocol + PretrainedSpec.
  weights/     # load_pretrained: download, key-remap, strict=False + LoadReport.
  logging/     # loguru JSONL sink + TypedDict log record schemas.
  metrics/     # COCOMetricsCalculator + PR/ROC/confusion + paired t-test/Wilcoxon/bootstrap CI.
  pipeline/    # TrainPipelineImpl / TestPipelineImpl / crossval / compare.
  dataset/     # YOLOSource, COCOSource, DetectionDataset, transforms, yolo<->coco convert.
  viz/         # loss/metric/PR/ROC curves, confusion matrix, Grad-CAM, dataset plots.
  analysis/    # DatasetAnalysisReport, ModelAnalysisReport.
scripts/       # train, test, convert_dataset, heatmap, analyze_dataset, analyze_model,
               # schedule_experiments, compare_models.
```

## Documentation

**English**:

- [`docs/architecture.md`](docs/architecture.md) -- design philosophy, package layout, the Protocol-based contract.
- [`docs/ops.md`](docs/ops.md) -- every operator (math, constructor, when to use, SAC/TR write-ups).
- [`docs/models.md`](docs/models.md) -- per-model architecture, loss formula, SAC/TR insertion points.
- [`docs/metrics.md`](docs/metrics.md) -- every metric (mAP, AR, precision/recall, statistical tests).
- [`docs/pretrained.md`](docs/pretrained.md) -- `from_pretrained` semantics, key remapping, SAC/TR partial-load.
- [`docs/logging.md`](docs/logging.md) -- log record schemas, JSONL format, post-hoc queries.
- [`docs/usage.md`](docs/usage.md) -- end-to-end tutorial.
- [`docs/development.md`](docs/development.md) -- how to add a new model variant.
- [`docs/dataset.md`](docs/dataset.md) -- formats, conversion, transforms, target tensor conventions.
- [`docs/pipeline.md`](docs/pipeline.md) -- TrainPipeline/TestPipeline usage, 5-fold CV, multi-model comparison.
- [`docs/scheduler.md`](docs/scheduler.md) -- YAML format, retry workflow, parallel execution.
- [`docs/scripts.md`](docs/scripts.md) -- every script's purpose, args, input/output.
- [`docs/heatmap.md`](docs/heatmap.md) -- Grad-CAM methodology, layer selection, output structure.
- [`docs/cross_validation.md`](docs/cross_validation.md) -- 5-fold mechanics, paired t-test, interpretation.
- [`docs/cuda_setup.md`](docs/cuda_setup.md) -- cu118 install, VRAM scaling, AMP, multi-GPU.
- [`experiments/README.md`](experiments/README.md) -- ready-to-run sweep configs.

**中文** ([`README.zh-CN.md`](README.zh-CN.md)):

- [`docs/architecture.zh-CN.md`](docs/architecture.zh-CN.md)
- [`docs/ops.zh-CN.md`](docs/ops.zh-CN.md)
- [`docs/models.zh-CN.md`](docs/models.zh-CN.md)
- [`docs/metrics.zh-CN.md`](docs/metrics.zh-CN.md)
- [`docs/pretrained.zh-CN.md`](docs/pretrained.zh-CN.md)
- [`docs/logging.zh-CN.md`](docs/logging.zh-CN.md)
- [`docs/usage.zh-CN.md`](docs/usage.zh-CN.md)
- [`docs/development.zh-CN.md`](docs/development.zh-CN.md)
- [`docs/dataset.zh-CN.md`](docs/dataset.zh-CN.md)
- [`docs/pipeline.zh-CN.md`](docs/pipeline.zh-CN.md)
- [`docs/scheduler.zh-CN.md`](docs/scheduler.zh-CN.md)
- [`docs/scripts.zh-CN.md`](docs/scripts.zh-CN.md)
- [`docs/heatmap.zh-CN.md`](docs/heatmap.zh-CN.md)
- [`docs/cross_validation.zh-CN.md`](docs/cross_validation.zh-CN.md)
- [`docs/cuda_setup.zh-CN.md`](docs/cuda_setup.zh-CN.md)

## Verification

```bash
uv run ruff check cracks_yolo tests scripts
uv run mypy --strict cracks_yolo tests scripts
uv run pytest -q
```

All three are required green before merge.

## License

See `LICENSE` (or your project's license file).
