# Models

## Architecture

Every model in `cracks-yolo` is a single, self-contained subclass of `cracks_yolo.zoo.base.BaseModel` -- a concrete `nn.Module` that owns its layers, its loss, its optimizer-building logic, and its pretrained-weight loader. There are no abstract hooks, no plugin registries, and no runtime YAML parsing for model structure. The class itself is the sole source of truth for its architecture.

`BaseModel` enforces a three-state lifecycle via `ModelState`: `UNINITIALIZED` (freshly constructed), `PRETRAINED` (COCO weights loaded), and `TRAINED` (fine-tuned on the target dataset). It declares five abstract methods that every subclass must implement:

| Method | Purpose |
|---|---|
| `train_model(config: TrainConfig) -> TrainReport` | Run a full training loop |
| `inference(images: Tensor) -> list[InferenceResult]` | Decode raw outputs into xyxy boxes, scores, and labels |
| `save(path, torchscript, onnx)` | Persist weights to disk |
| `load(path)` | Restore weights from a checkpoint |
| `from_pretrained(cls, num_classes, **kwargs) -> BaseModel` | Build from COCO pretrained weights |

Two concrete subclasses cover all model families:

- **`UltralyticsAdapter`** -- wraps ultralytics `DetectionModel` instances for YOLO families (v3 through v26, RT-DETR). Injects SAC/TR operators at designated backbone indices during construction, and delegates training to ultralytics' native `DetectionTrainer` (or `RTDETRTrainer`) while preserving the injected architecture.
- **`TorchvisionBase`** -- wraps `torchvision.models.detection` detectors. Rewires the standard torchvision training loop (model-level `.forward(loss_dict)`) into a per-image step, and decodes their heterogeneous output formats into the unified `InferenceResult`.

The pipeline contract is structural: pipelines access `model.decode_format` and
`model._inner` directly, never branching on `isinstance(model, ...)`.

## ZOO Registry

`cracks_yolo.zoo.ZOO` maps short names to 45 concrete classes (39 ultralytics + 6
torchvision). Assembled from `cracks_yolo.zoo.ultralytics.ZOO` and
`cracks_yolo.zoo.torchvision.ZOO`.

```python
from cracks_yolo.zoo import ZOO
model = ZOO["yolov5s_sactr"](num_classes=1).to("cuda")
```

## SAC and TR Injection

SAC (Switchable Atrous Convolution) and TR (Transformer) are the two architectural interventions proposed in this work for tongue surface crack detection.

**SAC** replaces the standard 3x3 convolution inside CSP bottleneck blocks with `SAConv2d` (defined in `cracks_yolo.ops.sac`). Each instance learns a per-pixel soft switch that fuses two parallel atrous convolutions -- one at the base dilation rate, one at 3x that rate -- enabling the network to select its receptive field dynamically. The operator also applies pre-context aggregation (global average pooling into a 1x1 conv then broadcast addition) and a symmetric post-context residual before batch normalization and activation.

**TR** replaces a whole C3 block with `C3TR`, which routes one CSP branch through a `TransformerBlock` (1x1 projection + learned positional embedding + stacked `TransformerLayer` modules, each applying QKV multi-head self-attention followed by a two-layer MLP, both with residual connections).

Injection is performed by `apply_sac_tr(model, sac_indices, tr_indices)` from `cracks_yolo.zoo.ultralytics.sac_injection`. The function iterates over the model's `model.model` `Sequential`, replacing blocks at the given indices with their SAC/TR counterparts (`C3` becomes `C3SAC` or `C3TR`; `C2f` becomes `C2fSAC`). Shared convolution weights are copied from the original block into the replacement; only SAC-specific tensors (switch conv, weight-difference parameter, pre/post context convs) remain randomly initialized. Routing metadata (`f`, `i`, `type`, `np`) is preserved so the replaced layer integrates transparently into the forward graph. SAC insertion points per family:

- **YOLOv5s**: C3 blocks at backbone indices (2, 4, 6); C3TR at (8,).
- **YOLOv6n/YOLOv8n/YOLOv8s/YOLOv9c/YOLOv10s/RT-DETR-R50**: at indices (2, 4, 6, 8).

When loading COCO pretrained weights with `strict=False`, SAC/TR layers appear in the missing-keys report; they are randomly initialized.

## Adding a New Model Variant

1. Add a class in `cracks_yolo/zoo/ultralytics/__init__.py` (YOLO family) or a new file under `cracks_yolo/zoo/torchvision/` (torchvision family). The class must subclass `UltralyticsAdapter` or `TorchvisionBase`. Its `__init__` calls `_build_detection_model` with the appropriate YAML config name, optionally injects SAC/TR via `apply_sac_tr`, and passes all metadata to `super().__init__()`.
2. Register the class in the appropriate `ZOO` dict with a short, descriptive key (e.g., `"yolov8n_sac"`).
3. Add tests in `tests/zoo/test_<arch>.py`: forward shape on `(2,3,640,640)`, `compute_loss` finite and produces non-zero gradients, `build_optimizer` returns a `torch.optim.Optimizer`, and `from_pretrained` partial-loads without error.
4. Update this document.

The naming convention is `{baseline}_{improvements}` with `_sac` and `_tr` suffixes. No loss or optimizer abbreviations appear in class names -- those are owned by `TrainConfig`.

## Available Model Families

### YOLO families (via `UltralyticsAdapter`)

| Family | Scales | Variants |
|---|---|---|
| YOLOv3 | -- | baseline |
| YOLOv5 | n, s, m, l, x | baseline; s only: `_sac`, `_tr`, `_sactr` |
| YOLOv6 | n | baseline, `_sac` |
| YOLOv8 | n, s, m, l, x | baseline; n, s: `_sac` |
| YOLOv9 | t, s, m, c, e | baseline; c only: `_sac` |
| YOLOv10 | n, s, m, b, l, x | baseline; s only: `_sac` |
| RT-DETR | r50 | baseline, `_sac` |
| YOLO11 | n, s | baseline |
| YOLO12 | n, s | baseline |
| YOLO26 | n, s | baseline |

### Torchvision detectors (via `TorchvisionBase`)

| Detector | Backbone | Paradigm |
|---|---|---|
| RetinaNet | ResNet-50 | Single-stage, anchor-based, Focal Loss |
| Faster R-CNN | ResNet-50 | Two-stage, RPN + RoI heads |
| Mask R-CNN | ResNet-50 | Two-stage + instance segmentation |
| FCOS | ResNet-50 | Anchor-free, center-ness |
| SSD300 | VGG-16 | Single-stage, multi-scale anchors |
| SSDlite320 | MobileNetV3-Large | Lightweight single-stage |

All models accept `num_classes: int = 1` and `input_size: int = 640`. For torchvision detectors, pretrained backbones (COCO, ImageNet) are available via `from_pretrained` using torchvision's weight API. For YOLO families, pretrained COCO weights are fetched through ultralytics' model hub.

## Loss Device-Sync Convention

Loss modules hold internal tensors (anchors, BCE pos-weights, stride) that are not `nn.Parameter`s and therefore do not move with `.to()`. Each loss's `__call__` must sync `self.device`, `self.anchors`, `self.stride`, and BCE submodules to `preds[0].device` at entry. See `cracks_yolo/losses/yolov5.py` for the canonical pattern.
