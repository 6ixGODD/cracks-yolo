# Models

[English](models.md) | [中文](models.zh-CN.md)

Every model in `cracks_yolo.zoo` is a self-contained `nn.Module` that owns its layers, loss modules, optimizer-builder, and `from_pretrained` classmethod. Long class names encode every choice — `YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU` — while short aliases live in `cracks_yolo.zoo.__init__` for ergonomics. There are 26 entries in the `ZOO` registry, covering 7 model families with baseline and SAC/TR variants.

Each zoo class declares two class attributes the pipeline reads instead of branching on class name:

- `loss_parts_schema: tuple[str, ...]` — names of each entry in the `parts` tensor returned by `compute_loss`. v5/v7 = `("box","cls","obj")`; v8/v9/v10 = `("box","cls","dfl")`; torchvision wrappers = `("total","cls","box_reg","rpn_box_reg")`.
- `decode_format: str` — `"anchor_free"` (v8/v9/v10: `(B, 4+nc, N)`) or `"anchor_based"` (v5/v7 + torchvision wrappers: `(B, N, nc+5)` or `(B, N_max, 6)`).

All `compute_loss` signatures accept `imgs: torch.Tensor | None = None` (v7 uses it for OTA assignment; others ignore). The pipeline always passes `imgs=images`.

## Registry

`cracks_yolo.zoo.ZOO: dict[str, type[nn.Module]]` maps short names to classes:

| Key | Class | Family |
| --- | --- | --- |
| `yolov5s` | `YOLOv5s` | YOLOv5 |
| `yolov5s_sac` | `YOLOv5sSAC` | YOLOv5 + SAC |
| `yolov5s_tr` | `YOLOv5sTR` | YOLOv5 + TR |
| `yolov5s_sactr` | `YOLOv5sSACTR` | YOLOv5 + SAC + TR |
| `yolov7w` | `YOLOv7w` | YOLOv7 |
| `yolov7w_sac` | `YOLOv7wSAC` | YOLOv7 + SAC |
| `yolov8n` | `YOLOv8n` | YOLOv8 nano |
| `yolov8n_sac` | `YOLOv8nSAC` | YOLOv8 nano + SAC |
| `yolov8s` | `YOLOv8s` | YOLOv8 small |
| `yolov8s_sac` | `YOLOv8sSAC` | YOLOv8 small + SAC |
| `yolov8m` | `YOLOv8m` | YOLOv8 medium |
| `yolov8m_sac` | `YOLOv8mSAC` | YOLOv8 medium + SAC |
| `yolov8l` | `YOLOv8l` | YOLOv8 large |
| `yolov8l_sac` | `YOLOv8lSAC` | YOLOv8 large + SAC |
| `yolov8x` | `YOLOv8x` | YOLOv8 xlarge |
| `yolov8x_sac` | `YOLOv8xSAC` | YOLOv8 xlarge + SAC |
| `yolov9c` | `YOLOv9c` | YOLOv9 compact |
| `yolov9c_sac` | `YOLOv9cSAC` | YOLOv9 compact + SAC |
| `yolov10s` | `YOLOv10s` | YOLOv10 small |
| `yolov10s_sac` | `YOLOv10sSAC` | YOLOv10 small + SAC |
| `retinanet_r50` | `RetinaNetR50` | RetinaNet R50 |
| `faster_rcnn_r50` | `FasterRCNNR50` | Faster R-CNN R50 |
| `mask_rcnn_r50` | `MaskRCNNR50` | Mask R-CNN R50 |
| `fcos_r50` | `FCOSR50` | FCOS R50 |
| `ssd300_vgg16` | `SSD300VGG16` | SSD300 VGG16 |
| `ssdlite320_mobilenetv3` | `SSDlite320MobileNetV3` | SSDlite320 MobileNetV3 |

---

## YOLOv5s (`cracks_yolo/zoo/yolov5.py`)

### Architecture

```
Input (B, 3, 640, 640)
  |
  +- Backbone (stride 8/16/32)
  |   Conv-P1/2 -> Conv-P2/4 -> C3 -> Conv-P3/8 -> C3 -> Conv-P4/16 -> C3
  |   -> Conv-P5/32 -> C3 -> SPPF
  |
  +- Neck (FPN + PAN)
  |   Conv -> Upsample -> Concat(P4) -> C3 -> Conv -> Upsample
  |   -> Concat(P3) -> C3 (P3/8-small)
  |   -> Conv -> Concat(neck-P4) -> C3 (P4/16-medium)
  |   -> Conv -> Concat(neck-P5) -> C3 (P5/32-large)
  |
  +- Head: DetectAnchorBased (3 scales x 3 anchors, hardcoded COCO anchors)
       -> (B, 25200, nc+5)  [25200 = (80^2 + 40^2 + 20^2) x 3]
```

### Variants

- **SAC:** swap backbone C3 -> `C3SAC` at P2/P3/P4/P5 stages. SAC layers are randomly initialized when loading COCO weights (no COCO SAC weights exist).
- **TR:** swap one backbone C3 -> `C3TR` (TransformerBlock). The TransformerBlock is randomly initialized on COCO load.
- **SACTR:** both SAC and TR applied simultaneously.

### Loss (`ComputeLoss`)

CIoU + BCEobj + BCEcls with IoU-aware obj targets and anchor_t=4.0 grid-offset matching. Ported from upstream YOLOv5 loss.

$$L = \lambda_\text{box} \cdot L_\text{CIoU} + \lambda_\text{obj} \cdot L_\text{BCEobj} + \lambda_\text{cls} \cdot L_\text{BCEcls}$$

Gains: $\lambda_\text{box}=0.05$, $\lambda_\text{obj}=0.7$, $\lambda_\text{cls}=0.3$. The obj target is scaled by the IoU between the prediction and the matched ground-truth box, so the obj branch learns "how well this anchor matches a GT" rather than a binary 0/1.

### Optimizer

`build_optimizer()` returns `torch.optim.AdamW(model.parameters(), lr=1e-3)`.

### Anchors (COCO, hardcoded)

```
P3/8:  [10,13], [16,30],   [33,23]
P4/16: [30,61], [62,45],   [59,119]
P5/32: [116,90], [156,198], [373,326]
```

Divided by stride on init: `head.anchors /= head.stride.view(-1, 1, 1)`.

### Stride init

A dummy training forward at `s=256` produces feature maps; stride is computed as `[s / f.shape[-2] for f in feats]` and stored on the head.

### Paper

- YOLOv5: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- SAC: [Switchable Atrous Convolution (ICCV 2021)](https://arxiv.org/abs/1908.07698)
- TR: Transformer block from [Attention Is All You Need (NeurIPS 2017)](https://arxiv.org/abs/1706.03762)

---

## YOLOv7w (`cracks_yolo/zoo/yolov7.py`)

### Architecture

YOLOv7-w (wide) variant. Uses `RepConv` (train-time multi-branch, eval-time fused), `SPPCSPC`, and `IDetect` (with `ImplicitA`/`ImplicitM`). First conv is `Conv(3, 32, 3, 2)` to set stride=2 for P1/2.

```
Input (B, 3, 640, 640)
  |
  +- Backbone
  |   Conv-P1/2 -> Conv-P2/4 -> C3 -> Conv-P3/8 -> C3
  |   -> Conv-P4/16 -> C3 -> Conv-P5/32 -> C3 -> SPPCSPC
  |
  +- Neck (FPN + PAN with RepConv)
  |
  +- Head: IDetect (ImplicitA + ImplicitM, COCO anchors)
       -> (B, 3, H, W, nc+5) per scale -> decode to (B, N, nc+5)
```

The SAC variant replaces backbone C3 stages with `C3SAC`.

### Loss (`ComputeLossOTA`)

Optimal Transport Assignment (SimOTA) -- dynamic-k matching. Cost matrix = `cls_loss + 3 * iou_loss`; for each GT, the top-k anchors (k computed from IoU sum) are selected as positives. **Requires `imgs` argument** to `compute_loss` because OTA uses image dimensions for the dynamic-k calculation.

### Deployment

Call `model.fuse()` before eval to fuse `RepConv` multi-branches into single convs and `ImplicitA`/`ImplicitM` into preceding convs.

### Paper

- YOLOv7: [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art (CVPR 2023)](https://arxiv.org/abs/2207.02696)
- SimOTA: from [YOLOX (2021)](https://arxiv.org/abs/2107.08430)

---

## YOLOv8 {n, s, m, l, x} (`cracks_yolo/zoo/yolov8.py`)

### Architecture

Anchor-free. C2f stages (C3 + split + re-use) in the backbone, SPPF at P5, FPN+PAN neck, `DetectAnchorFree` head with separate `cv2` (box, reg_max=16) and `cv3` (cls) branches. Five size variants via `width_mult` and `depth_mult` scaling.

```
Input (B, 3, 640, 640)
  |
  +- Backbone (stride 8/16/32)
  |   Conv-P1/2 -> Conv-P2/4 -> C2f -> Conv-P3/8 -> C2f
  |   -> Conv-P4/16 -> C2f -> Conv-P5/32 -> C2f -> SPPF
  |
  +- Neck (FPN + PAN)
  |   Upsample -> Concat(P4) -> C2f -> Upsample
  |   -> Concat(P3) -> C2f (P3/8-small)
  |   -> Conv -> Concat(neck-P4) -> C2f (P4/16-medium)
  |   -> Conv -> Concat(neck-P5) -> C2f (P5/32-large)
  |
  +- Head: DetectAnchorFree (DFL box + cls branches)
       Training: {"boxes": (B, 4*reg_max, N), "scores": (B, nc, N), "feats": [...]}
       Eval:     (B, 4+nc, N)  where N = 8400 at 640x640
```

### Size variants

| Variant | width_mult | depth_mult | P3 ch | P4 ch | P5 ch |
| ------- | ---------- | ---------- | ----- | ----- | ----- |
| nano    | 0.25       | 0.33       | 64    | 128   | 256   |
| small   | 0.50       | 0.33       | 128   | 256   | 512   |
| medium  | 0.75       | 0.67       | 192   | 384   | 768   |
| large   | 1.00       | 1.00       | 256   | 512   | 1024  |
| xlarge  | 1.25       | 1.00       | 320   | 640   | 1280  |

### Loss (`v8DetectionLoss`)

- `TaskAlignedAssigner` (topk=10, alpha=0.5, beta=6.0) for positive assignment.
- `BboxLoss` with CIoU + DFL (Distribution Focal Loss).
- **No obj loss** (v8 dropped the objectness branch).
- Returns a `(3,)` tensor of `[box, cls, dfl]`. `compute_loss` sums to a scalar before backward.

### SAC variant

`C2f` -> `C2fSAC` in the four backbone C2f stages (P2/P3/P4/P5). Available for all five sizes.

### Paper

- YOLOv8: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- DFL: from [Generalized Focal Loss (NeurIPS 2020)](https://arxiv.org/abs/2006.04388)

---

## YOLOv9c (`cracks_yolo/zoo/yolov9.py`)

### Architecture

YOLOv9-c (compact), simplified -- no PGI auxiliary branch. Backbone uses GELAN-style `RepNCSPELAN4` stages with `ADown` downsampling. Neck: `SPPELAN` bottom + FPN/PAN with `RepNCSPELAN4` stages. Head: v8 `DetectAnchorFree` (separate cv2/cv3 branches + DFL). Loss: `v8DetectionLoss`.

The upstream PGI (Programmable Gradient Information) auxiliary supervision branch (`DualDDetect` head + `CBFuse` fusion) is omitted. This makes the YOLOv9 entry a fair same-Protocol comparison baseline rather than a perfect reproduction.

```
Input (B, 3, 640, 640)
  |
  +- Backbone
  |   Conv-P1/2 -> Conv-P2/4 -> RepNCSPELAN4 -> ADown
  |   -> RepNCSPELAN4 (P3) -> ADown -> RepNCSPELAN4 (P4)
  |   -> ADown -> RepNCSPELAN4 (P5)
  |
  +- Neck (SPPELAN + FPN/PAN with RepNCSPELAN4)
  |
  +- Head: DetectAnchorFree (v8-style, DFL)
```

The SAC variant replaces `RepNCSPELAN4` with `C2fSAC` at the same backbone positions (P2/P3/P4/P5). Since `RepConvN` is structurally incompatible with SAC (re-parameterization), the SAC variant falls back to `C2fSAC` stages. COCO weights load with `strict=False`; SAC layers are randomly initialized.

### Paper

- YOLOv9: [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information (2024)](https://arxiv.org/abs/2402.13616)

---

## YOLOv10s (`cracks_yolo/zoo/yolov10.py`)

### Architecture

YOLOv10 backbone: C2f -> C2fCIB + SCDown at P3/P4/P5, `PSA` (Partial Self-Attention) at P5. Neck: same FPN+PAN as v8 but with `SCDown` for downsampling. Head: `v10Detect` (dual head).

```
Input (B, 3, 640, 640)
  |
  +- Backbone
  |   Conv-P1/2 -> Conv-P2/4 -> C2f -> Conv-P3/8 -> C2fCIB
  |   -> SCDown -> C2fCIB (P4) -> SCDown -> C2fCIB (P5) -> PSA
  |
  +- Neck (FPN+PAN with SCDown)
  |
  +- Head: v10Detect (dual head)
       Training: {"one2many": {...}, "one2one": {...}}
       Eval:     (B, 4+nc, N)  [one2one head only -- NMS-free]
```

### Loss (`E2ELoss`)

Two `v8DetectionLoss` instances:

- **one2many** (topk=10): standard training, one GT matches many predictions.
- **one2one** (topk=7, decay schedule): one GT matches one prediction -- this enables NMS-free inference.

The one2one loss weight follows a decay schedule so it ramps up over training. Returns `(3,)` tensor; `compute_loss` sums to a scalar.

### NMS-free inference

Eval-mode forward runs only the one2one head, producing one detection per GT -- no NMS needed. Top-K filter is applied in the decode step.

### Paper

- YOLOv10: [YOLOv10: Real-Time End-to-End Object Detection (NeurIPS 2024)](https://arxiv.org/abs/2405.14458)

---

## Torchvision detectors (`cracks_yolo/zoo/torchvision_detectors.py`)

### Architecture

Six torchvision detection models are wrapped to satisfy the `DetectorModel` Protocol, providing cross-paradigm comparison baselines.

| Key | Inner model | Paradigm | Backbone |
| --- | ----------- | -------- | -------- |
| `retinanet_r50` | `retinanet_resnet50_fpn` | Single-stage, anchor-based | ResNet-50 FPN |
| `faster_rcnn_r50` | `fasterrcnn_resnet50_fpn` | Two-stage, anchor-based | ResNet-50 FPN |
| `mask_rcnn_r50` | `maskrcnn_resnet50_fpn` | Two-stage, anchor-based + mask | ResNet-50 FPN |
| `fcos_r50` | `fcos_resnet50_fpn` | Single-stage, anchor-free | ResNet-50 FPN |
| `ssd300_vgg16` | `ssd300_vgg16` | Single-stage, anchor-based | VGG-16 |
| `ssdlite320_mobilenetv3` | `ssdlite320_mobilenet_v3_large` | Single-stage, anchor-based | MobileNetV3-Large |

### Adaptation strategy

- `forward(x)` in train mode stashes images and returns `{"_tv_images": x}`. In eval mode it calls the inner model and returns the `list[dict]`.
- `compute_loss(preds, targets, imgs=None)` converts `(N, 6)` YOLO targets to torchvision's `list[dict]` format with `boxes`/`labels`, calls the inner model, sums the loss dict, returns `(total_loss, parts_tensor)` where `parts_tensor` is `(total, cls, box_reg, rpn_box_reg)`.
- `decode(preds)` converts eval-mode `list[dict]` to `(B, N_max, 6)` of `(x1, y1, x2, y2, score, class_id)` -- the anchor-based format. Shorter detections are zero-padded; the pipeline's NMS step ignores zero-score rows.

### Loss

Torchvision models use their own built-in loss heads: Focal Loss (RetinaNet, FCOS, SSD), Cross-Entropy + Smooth L1 (Faster/Mask R-CNN). COCO pretrained weights load via torchvision's own weight machinery (`weights="DEFAULT"`), not through the `cracks_yolo` weights registry.

### Paper

- RetinaNet: [Focal Loss for Dense Object Detection (ICCV 2017)](https://arxiv.org/abs/1708.02002)
- Faster R-CNN: [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (NeurIPS 2015)](https://arxiv.org/abs/1506.01497)
- Mask R-CNN: [Mask R-CNN (ICCV 2017)](https://arxiv.org/abs/1703.06870)
- FCOS: [FCOS: Fully Convolutional One-Stage Object Detection (ICCV 2019)](https://arxiv.org/abs/1904.01355)
- SSD: [SSD: Single Shot MultiBox Detector (ECCV 2016)](https://arxiv.org/abs/1512.02325)

---

## SAC insertion points (summary)

- v5: replace backbone `C3` with `C3SAC` at P2/P3/P4/P5 (indices 2, 4, 6, 8 in the standard v5s backbone).
- v8/v10: replace backbone `C2f` with `C2fSAC` at the same positions.
- v7: replace `RepConv`/Bottleneck sequences with `BottleneckSAC`-based equivalents.
- v9: replace `RepNCSPELAN4` with `C2fSAC` at P2/P3/P4/P5 (structural compromise -- `RepConvN` cannot host SAC).

When loading COCO weights, SAC layers appear in `LoadReport.missing` -- they are randomly initialized.

## TR insertion points

- v5: replace one backbone `C3` (typically the P5/32 stage) with `C3TR`. The `TransformerBlock` is randomly initialized on COCO load.

## Pretrained weight semantics

- Baseline variants declare a `PretrainedSpec` class attribute pointing to official COCO release URLs.
- SAC/TR variants declare `pretrained_spec = None` (no COCO weights exist for the enhanced layers).
- `from_pretrained` uses `strict=False` and returns a `LoadReport` with `matched`, `missing`, and `unexpected` key lists.
- Torchvision wrappers use torchvision's built-in weight API (`weights="DEFAULT"`), not the `cracks_yolo` weights registry.

## Adding a new variant

See `docs/development.md`.

## Current state

- 26 ZOO entries across 7 model families.
- 318 tests green; `ruff` and `mypy --strict` clean on 97 source files.
- Loss device-sync convention: loss modules sync internal tensors (anchors, stride, BCE pos-weights) to `preds[0].device` at entry. See `cracks_yolo/losses/yolov5.py` for the canonical pattern.
