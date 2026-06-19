# Usage

End-to-end tutorial. The dataset adapter and pipeline **implementations**
land in a later pass; for now this shows model instantiation, pretrained
loading, forward, loss, and decode.

## Install (development)

```bash
git clone <repo>
cd cracks-yolo
uv sync          # or: pip install -e .
```

Requires Python 3.11 or 3.12, PyTorch ≥ 2.2.

## Quickstart: forward + loss + backward

```python
import torch
from cracks_yolo.zoo import ZOO

# Instantiate any registered model.
model = ZOO["yolov5s_sactr"](num_classes=1)
model.train()

# Forward (training mode returns raw head outputs).
x = torch.randn(2, 3, 640, 640)
preds = model(x)

# Build YOLO-format targets: (N, 6) = (img_idx, cls, x, y, w, h) normalized.
targets = torch.tensor(
    [
        [0, 0, 0.50, 0.50, 0.20, 0.20],
        [0, 0, 0.30, 0.70, 0.10, 0.10],
        [1, 0, 0.40, 0.40, 0.15, 0.25],
    ],
    dtype=torch.float32,
)

# Compute loss.
# v7 needs the image batch (OTA assignment uses image dimensions).
if model.__class__.__name__.startswith("YOLOv7"):
    loss, parts = model.compute_loss(preds, targets, imgs=x)
else:
    loss, parts = model.compute_loss(preds, targets)

loss.backward()
assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
```

## Eval-mode forward + decode

```python
model.eval()
with torch.no_grad():
    out = model(x)               # eval forward decodes internally
    decoded = model.decode(out)  # returns (B, N, nc+5) or (B, 4+nc, N)

print(decoded.shape)
# v5/v7: torch.Size([2, 25200, 6])  — (B, anchors, nc+5)
# v8/v10: torch.Size([2, 5, 8400])  — (B, 4+nc, grid_cells)
```

## Load COCO pretrained weights

```python
from cracks_yolo.zoo import YOLOv5s

# Baseline (has pretrained_spec) — downloads + loads with strict=False.
model = YOLOv5s.from_pretrained(num_classes=1)

# SAC/TR variants return random init (pretrained_spec is None).
from cracks_yolo.zoo import YOLOv5sSACTR
model = YOLOv5sSACTR(num_classes=1)
```

To inspect the load report:

```python
from cracks_yolo.weights.loader import load_pretrained
from cracks_yolo.zoo import YOLOv5s

model = YOLOv5s(num_classes=1)
report = load_pretrained(
    model=model,
    spec=YOLOv5s.pretrained_spec,
    weights_dir=None,  # defaults to ./weights
    strict=False,
)
print(f"matched: {len(report.matched)}")
print(f"missing: {report.missing[:5]} ... ({len(report.missing)} total)")
print(f"unexpected: {len(report.unexpected)}")
```

## Build the optimizer

```python
model = ZOO["yolov8s_sac"](num_classes=1)
optimizer = model.build_optimizer()
# torch.optim.AdamW(model.parameters(), lr=1e-3)
```

## List all available models

```python
from cracks_yolo.zoo import ZOO

for key, cls in ZOO.items():
    print(f"{key:18s} -> {cls.__name__}")
```

Output:

```
yolov5s            -> YOLOv5s_CIoU_BCEObj_BCECls_AdamW_SILU
yolov5s_sac        -> YOLOv5sSAC_CIoU_BCEObj_BCECls_AdamW_SILU
yolov5s_tr         -> YOLOv5sTR_CIoU_BCEObj_BCECls_AdamW_SILU
yolov5s_sactr      -> YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU
yolov7w            -> YOLOv7w_CIoU_BCEObj_BCECls_AdamW_SILU
yolov7w_sac        -> YOLOv7wSAC_CIoU_BCEObj_BCECls_AdamW_SILU
yolov8s            -> YOLOv8s_CIoU_DFL_AdamW_SILU
yolov8s_sac        -> YOLOv8sSAC_CIoU_DFL_AdamW_SILU
yolov10s           -> YOLOv10s_CIoU_DFL_AdamW_SILU
yolov10s_sac       -> YOLOv10sSAC_CIoU_DFL_AdamW_SILU
```

## Structured logging (for future pipelines)

```python
from pathlib import Path
from loguru import logger
from cracks_yolo.logging.configure import configure_logger
from cracks_yolo.logging.schema import TrainStepLog

configure_logger(output_dir=Path("output/run1"))

record: TrainStepLog = {
    "record_type": "train_step",
    "step": 0, "epoch": 0,
    "total_loss": 1.23, "box_loss": 0.4, "cls_loss": 0.5,
    "obj_loss": 0.33, "dfl_loss": None,
    "lr": 1e-3, "timestamp": "2026-06-18T00:00:00",
}
logger.bind(**record).info("step done")
# Writes one JSON line to output/run1/run.log.jsonl
```

## What's next

- Train/test pipeline **implementations** land in a later pass.
- Dataset adapter (`cracks_yolo.dataset`) lands in a later pass pending
  the COCO-format cracks dataset.
- Metrics **implementation** (pycocotools + torchvision) lands in a later
  pass; the `MetricsCalculator` Protocol and `MetricReport` schema are
  fixed today.
