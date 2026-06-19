# Models (`cracks_yolo.zoo`)

Each model class is a self-contained `nn.Module` owning its layers, loss
modules, optimizer-builder, and `from_pretrained` classmethod. Long class
names encode every choice — `YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU`.
Short aliases live in `cracks_yolo.zoo.__init__` for ergonomics.

## Registry

`cracks_yolo.zoo.ZOO: dict[str, type[nn.Module]]` maps short names →
classes:

| Key | Class | Variants |
| --- | --- | --- |
| `yolov5s` | `YOLOv5s` | baseline |
| `yolov5s_sac` | `YOLOv5sSAC` | SAC in backbone C3 stages |
| `yolov5s_tr` | `YOLOv5sTR` | TR (Transformer) in one C3 stage |
| `yolov5s_sactr` | `YOLOv5sSACTR` | both SAC and TR |
| `yolov7w` | `YOLOv7w` | baseline |
| `yolov7w_sac` | `YOLOv7wSAC` | SAC |
| `yolov8s` | `YOLOv8s` | baseline (anchor-free) |
| `yolov8s_sac` | `YOLOv8sSAC` | SAC in C2f stages |
| `yolov10s` | `YOLOv10s` | baseline (NMS-free) |
| `yolov10s_sac` | `YOLOv10sSAC` | SAC |

---

## YOLOv5s (`cracks_yolo/zoo/yolov5.py`)

### Architecture

```
Input (B, 3, 640, 640)
  │
  ├─ Backbone (stride 8/16/32)
  │   Conv-P1/2 → Conv-P2/4 → C3 → Conv-P3/8 → C3 → Conv-P4/16 → C3
  │   → Conv-P5/32 → C3 → SPPF
  │
  ├─ Neck (FPN + PAN)
  │   Conv → Upsample → Concat(P4) → C3 → Conv → Upsample
  │   → Concat(P3) → C3 (P3/8-small)
  │   → Conv → Concat(neck-P4) → C3 (P4/16-medium)
  │   → Conv → Concat(neck-P5) → C3 (P5/32-large)
  │
  └─ Head: DetectAnchorBased (3 scales × 3 anchors, hardcoded COCO anchors)
       → (B, 25200, nc+5)  [25200 = (80² + 40² + 20²) × 3]
```

### Variants

- **SAC:** swap backbone C3 → `C3SAC` at P2/P3/P4/P5 stages. SAC layers
  are randomly initialized when loading COCO weights (no COCO SAC weights
  exist).
- **TR:** swap one backbone C3 → `C3TR` (TransformerBlock). The
  TransformerBlock is randomly initialized on COCO load.

### Loss (`ComputeLoss`)

CIoU + BCEobj + BCEcls with IoU-aware obj targets and anchor_t=4.0
grid-offset matching. Ported verbatim from `deps/yolov5/utils/loss.py`.

$$L = \lambda_\text{box} \cdot L_\text{CIoU} + \lambda_\text{obj} \cdot L_\text{BCEobj} + \lambda_\text{cls} \cdot L_\text{BCEcls}$$

with gains $\lambda_\text{box}=0.05$, $\lambda_\text{obj}=0.7$,
$\lambda_\text{cls}=0.3$ by default. The obj target is scaled by the IoU
between the prediction and the matched ground-truth box, so the obj
branch learns "how well this anchor matches a GT" rather than a binary
0/1.

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

A dummy training forward at `s=256` produces feature maps; the stride
is computed as `[s / f.shape[-2] for f in feats]` and stored on the head.

---

## YOLOv7w (`cracks_yolo/zoo/yolov7.py`)

### Architecture

YOLOv7-w (wide) variant. Uses `RepConv` (train-time multi-branch,
eval-time fused), `SPPCSPC`, and `IDetect` (with `ImplicitA`/`ImplicitM`).
First conv is `Conv(3, 32, 3, 2)` to set stride=2 for P1/2.

### Loss (`ComputeLossOTA`)

Optimal Transport Assignment (SimOTA) — dynamic-k matching. Cost matrix
= `cls_loss + 3 * iou_loss`; for each GT, the top-k anchors (k computed
from IoU sum) are selected as positives. **Requires `imgs` argument** to
`compute_loss` because OTA uses image dimensions for the dynamic-k
calculation.

### Deployment

Call `model.fuse()` before eval to fuse `RepConv` multi-branches into
single convs and `ImplicitA`/`ImplicitM` into preceding convs.

---

## YOLOv8s (`cracks_yolo/zoo/yolov8.py`)

### Architecture

Anchor-free. C2f stages (C3 + split + re-use) in the backbone, SPPF at
P5, FPN+PAN neck, `DetectAnchorFree` head with separate `cv2` (box,
reg_max=16) and `cv3` (cls) branches.

```
Output (training): {"boxes": (B, 4*reg_max, N), "scores": (B, nc, N), "feats": [...]}
Output (eval):     (B, 4+nc, N)  where N = 8400 at 640×640
                   = (80² + 40² + 20²) anchor-free grid points
```

### Loss (`v8DetectionLoss`)

- `TaskAlignedAssigner` (topk=10, alpha=0.5, beta=6.0) for positive
  assignment.
- `BboxLoss` with CIoU + DFL (Distribution Focal Loss).
- **No obj loss** (v8 dropped the objectness branch).
- Returns a `(3,)` tensor of `[box, cls, dfl]`. `compute_loss` sums to
  a scalar before backward.

### SAC variant

`C2f` → `C2fSAC` in the four backbone C2f stages (P2/P3/P4/P5).

---

## YOLOv10s (`cracks_yolo/zoo/yolov10.py`)

### Architecture

YOLOv10 backbone: C2f → C2fCIB + SCDown at P3/P4/P5, `PSA` (Partial
Self-Attention) at P5. Neck: same FPN+PAN as v8 but with `SCDown` for
downsampling. Head: `v10Detect` (dual head).

```
Training: {"one2many": {...}, "one2one": {...}}  — dual head
Eval:     (B, 4+nc, N)  [one2one head only — NMS-free]
```

### Loss (`E2ELoss`)

Two `v8DetectionLoss` instances:
- **one2many** (topk=10): standard training, one GT matches many
  predictions.
- **one2one** (topk=7, decay schedule): one GT matches one prediction —
  this is what enables NMS-free inference.

The one2one loss weight follows a decay schedule so it ramps up over
training. Returns `(3,)` tensor; `compute_loss` sums to a scalar.

### NMS-free inference

Eval-mode forward runs only the one2one head, producing one detection
per GT — no NMS needed. Top-K filter is applied in the decode step.

---

## SAC insertion points (summary)

For v5: replace backbone `C3` with `C3SAC` at P2/P3/P4/P5 (indices 2, 4,
6, 8 in the standard v5s backbone).
For v8/v10: replace backbone `C2f` with `C2fSAC` at the same positions.
For v7: replace `RepConv`/Bottleneck sequences with `BottleneckSAC`-based
equivalents.

When loading COCO weights, SAC layers appear in `LoadReport.missing` —
they're randomly initialized.

## TR insertion points

For v5: replace one backbone `C3` (typically the P5/32 stage) with
`C3TR`. The TransformerBlock is randomly initialized on COCO load.

## Adding a new variant

See `docs/development.md`.
