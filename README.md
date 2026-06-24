# cracks-yolo

Self-contained PyTorch detection model zoo for tongue surface crack detection.
45 models across YOLOv3/v5/v6/v8/v9/v10/v11/v12/v26, RT-DETR, and six torchvision
baselines (RetinaNet, Faster R-CNN, Mask R-CNN, FCOS, SSD300, SSDlite320). Every
model is an explicit `nn.Module` subclass that owns its layers, loss, optimizer
builder, and pretrained-weight loader. No runtime YAML parsing, no ultralytics
monkey-patching, no abstract-base-class hook system.

## Quickstart

```bash
pip install -e .
```

```bash
# Single experiment: train then auto-test on best checkpoint.
cy run -c experiments/models/yolov5s_sactr.yaml

# Or with full CLI flags:
cy train -m yolov8s -d data/dataset -o output/run1 -e 300 -b 64 --pretrained
cy test -m yolov8s --weights output/run1/weights/best.pt -d data/dataset -o output/run1/test

# Batch scheduling with subprocess isolation.
cy compose -c experiments/compose_all.yaml -o output/all_models -p 2
```

## CLI

| Command | Purpose |
| --- | --- |
| `cy train` | Train a single model with full hyperparameter control. |
| `cy test` | Evaluate a trained checkpoint on test and validation splits. |
| `cy run` | Run one experiment from a YAML config (train, then auto-test). |
| `cy compose` | Batch-schedule experiments from a compose YAML with `$include`. |

Key flags for `cy train`:

| Flag | Default | Description |
| --- | --- | --- |
| `-m, --model` | (required) | ZOO key, e.g. `yolov8s_sac` |
| `-d, --dataset` | (required) | Path to dataset root |
| `-o, --output-dir` | (required) | Output directory |
| `-e, --epochs` | 300 | Number of epochs |
| `-b, --batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--pretrained / --no-pretrained` | `--pretrained` | Load COCO weights |
| `--optimizer` | adamw | `adamw` or `sgd` |
| `--cosine-lr / --no-cosine-lr` | `--cosine-lr` | Cosine LR schedule |
| `--ema / --no-ema` | `--ema` | Exponential moving average |
| `--patience` | 100 | Early stopping patience (epochs) |
| `--device` | cuda | `cuda` or `cpu` |
| `--seed` | 42 | Random seed |

## Model zoo

45 explicit classes in `cracks_yolo.zoo.ZOO`, each hardcoding its architecture
cfg, pretrained asset, SAC/TR injection indices, and decode format.

### YOLO families (39 models)

| Family | Keys | Sizes | SAC variants |
| --- | --- | --- | --- |
| YOLOv3 | `yolov3` | 1 | -- |
| YOLOv5 | `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x` | 5 | `yolov5s_sac`, `yolov5s_tr`, `yolov5s_sactr` |
| YOLOv6 | `yolov6n` | 1 | `yolov6n_sac` |
| YOLOv8 | `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x` | 5 | `yolov8n_sac`, `yolov8s_sac` |
| YOLOv9 | `yolov9t`, `yolov9s`, `yolov9m`, `yolov9c`, `yolov9e` | 5 | `yolov9c_sac` |
| YOLOv10 | `yolov10n`, `yolov10s`, `yolov10m`, `yolov10b`, `yolov10l`, `yolov10x` | 6 | `yolov10s_sac` |
| RT-DETR | `rtdetr_r50` | 1 | `rtdetr_r50_sac` |
| YOLO11 | `yolo11n`, `yolo11s` | 2 | -- |
| YOLO12 | `yolo12n`, `yolo12s` | 2 | -- |
| YOLO26 | `yolo26n`, `yolo26s` | 2 | -- |

### Cross-paradigm baselines (6 models)

| Key | Architecture | Paradigm |
| --- | --- | --- |
| `retinanet_r50` | RetinaNet, ResNet-50 FPN | Single-stage, anchor-based, Focal Loss |
| `faster_rcnn_r50` | Faster R-CNN, ResNet-50 FPN | Two-stage, RPN + RoI |
| `mask_rcnn_r50` | Mask R-CNN, ResNet-50 FPN | Two-stage + mask head |
| `fcos_r50` | FCOS, ResNet-50 FPN | Anchor-free, centerness |
| `ssd300_vgg16` | SSD300, VGG-16 | Single-stage, 300x300 |
| `ssdlite320_mobilenetv3` | SSDlite320, MobileNetV3-Large | Lightweight, 320x320 |

## Key features

**SAC and TR injection.** Switchable Atrous Convolution (SAC) replaces selected
C3/C2f blocks in the backbone with atrous variants; C3TR substitutes
transformer blocks. Injection points are per-class constants -- no config files,
no runtime dispatch. Supported on YOLOv5s, YOLOv6n, YOLOv8n/s, YOLOv9c,
YOLOv10s, and RT-DETR-R50.

**Explicit model classes.** Each ZOO entry is a concrete class (e.g.
`YOLOv8sSAC`) that hardcodes its YAML cfg, pretrained asset, SAC/TR indices,
and decode format. No abstract factory, no registry indirection, no
`isinstance` branching in pipelines.

**Pretrained weight loading.** `from_pretrained()` downloads COCO weights via
ultralytics, intersects the state dict by key and shape, and loads with
`strict=False`. SAC/TR layers receive random init; matched backbone layers
get COCO transfer.

**Pipeline contract via base class.** `BaseModel` defines `train_model`,
`inference`, `save`, `load`, `from_pretrained`, and `analyze`. A 3-state
machine (`UNINITIALIZED -> PRETRAINED -> TRAINED`) enforces lifecycle
correctness at runtime. Pipelines depend only on this interface.

**Batch scheduling with `cy compose`.** YAML-driven experiment scheduler with
`$include` composition, per-experiment env overrides (`CUDA_VISIBLE_DEVICES`),
subprocess isolation, and `errors.jsonl` for retry workflows.

**Model analysis.** `model.analyze()` returns `ModelAnalysisReport` with
parameter counts, MACs/GFLOPs (via thop), FPS/latency percentiles, peak VRAM,
and a 3-level structure tree. Available as `cy analyze` or programmatically.

## Output structure

After `cy train` or `cy run`:

```
output_dir/
  weights/
    best.pt             # Best-validation checkpoint
    last.pt             # Final-epoch checkpoint
  results.csv           # Per-epoch metrics (loss, mAP50, mAP50-95)
  metrics.csv           # Alias / copy of results.csv
  args.yaml             # Effective training arguments
  train_logs/           # Ultralytics training logs
  test/                 # Auto-test artifacts (from cy run)
    per_image/          # Per-image COCO-format prediction JSONs
    predictions/        # Annotated prediction images
    curves/
      pr.png            # Precision-recall curve
      roc.png           # ROC curve
      confusion.png     # Confusion matrix
    metrics_summary.json
```

## Package layout

```
cracks_yolo/
  ops/                  # SAC, C3TR, and shared operator modules
  losses/               # Loss functions (v5, v7 OTA, v8 DFL, v10 E2E)
  zoo/                  # Model classes and ZOO registry
    ultralytics/        # UltralyticsAdapter + 39 explicit YOLO/RT-DETR classes
    torchvision/        # 6 torchvision wrapper classes
  weights/              # Pretrained download, key remapping, partial load
  logging/              # loguru JSONL sink, typed log record schemas
  metrics/              # COCO mAP, PR/ROC/confusion, statistical tests
  pipeline/             # train, test, compose (batch scheduler)
  dataset/              # YOLO/COCO loaders, transforms, augmentations
  viz/                  # Curves, confusion matrix, Grad-CAM, dataset plots
  analysis/             # DatasetAnalysisReport, ModelAnalysisReport
cli.py                  # Typer CLI (train, test, run, compose)
```

## Verification

```bash
ruff check cracks_yolo tests
mypy --strict cracks_yolo tests
pytest -q
```

All three must pass with zero errors before merge.

## Documentation

| Document | Content |
| --- | --- |
| `docs/models.md` | Per-model architecture, loss formulas, SAC/TR insertion points |
| `docs/ops.md` | Operator reference (SAC, C3TR, Conv, CSP, detect heads) |
| `docs/pipeline.md` | TrainPipeline, TestPipeline, compose scheduler |
| `docs/dataset.md` | Data formats, conversion, transforms, target conventions |
| `docs/metrics.md` | COCO mAP, PR/ROC, statistical tests (t-test, Wilcoxon, bootstrap) |
| `docs/logging.md` | JSONL log schema, loguru configuration |
| `docs/usage.md` | End-to-end tutorial |
| `docs/heatmap.md` | Grad-CAM methodology and output structure |
| `docs/scripts.md` | CLI reference (all commands, all flags) |
| `docs/scheduler.md` | Compose YAML format, `$include`, retry workflow |
| `docs/models.md` | How to add a new model variant |

Chinese translations: `docs/*.zh-CN.md`.

## License

See `LICENSE`.
