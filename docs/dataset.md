# Dataset

[English](dataset.md) | [中文](dataset.zh-CN.md)

`cracks_yolo.dataset` provides format-agnostic dataset loading, conversion between YOLOv5-PyTorch and COCO formats, a torchvision-compatible `DetectionDataset`, and a conservative transform pipeline. The default dataset (`data/CrackDetection_Augmentation.v1.yolov5pytorch`) is a tongue surface crack image collection with a single class (`crack`) in YOLOv5 PyTorch format.

## Supported formats

| Format | Layout | Loader |
| --- | --- | --- |
| YOLOv5 PyTorch | `data.yaml` + `train/`, `valid/`, `test/` with `images/` + `labels/*.txt` | `cracks_yolo.dataset.yolo.YOLOSource` |
| COCO | `instances_{train,val}.json` + `train2017/`, `val2017/` (image dir) | `cracks_yolo.dataset.coco.COCOSource` |

Both sources expose a uniform `load_split(split: str) -> list[RawDetection]` API. `RawDetection` is the format-agnostic record type:

```python
@dataclass
class RawDetection:
    image_id: int
    image_path: Path
    boxes_norm: np.ndarray   # (N, 4) xyxy normalized to [0, 1]
    labels: np.ndarray       # (N,) int64
```

## DetectionDataset (torch.utils.data.Dataset)

`DetectionDataset` wraps a list of `RawDetection` + a `DetectionTransform` and returns `DetectionSample(image, boxes, labels, image_id)` per item. Construct via the classmethods:

```python
from cracks_yolo.dataset import DetectionDataset

# From a YOLOv5-format root.
ds = DetectionDataset.from_yolo(
    root="data/CrackDetection_Augmentation.v1.yolov5pytorch",
    split="train",
    input_size=640,
    train=True,
)

# From a COCO-format root.
ds = DetectionDataset.from_coco(
    root="data/coco",
    split="train",
    input_size=640,
    train=True,
)

# From pre-built records (for 5-fold CV splits).
ds = DetectionDataset.from_records(records=subset, input_size=640, train=True)
```

## DataLoader

`build_dataloader` returns a `torch.utils.data.DataLoader` with a custom `detection_collate` that handles variable-N boxes per image:

```python
from cracks_yolo.dataset import build_dataloader

loader = build_dataloader(ds, batch_size=32, num_workers=4, shuffle=True, pin_memory=True)
for images, targets in loader:
    # images: (B, 3, H, W) float in [0, 1]
    # targets: list[dict] of {"boxes": (N,4) xyxy abs, "labels": (N,), "image_id": Tensor}
    ...
```

## Transforms

`DetectionTransform` is a small callable class (not torchvision v2). The default train pipeline is intentionally conservative — only horizontal flip — so baseline comparisons are not confounded by heavy augmentation.

- **Resize**: bilinear to `input_size x input_size`.
- **Normalize**: pixels scaled to `[0, 1]` (no mean/std subtraction — YOLO convention).
- **Train augmentation**: random horizontal flip (p=0.5). Boxes are flipped accordingly.
- **Eval**: resize + normalize only.

`build_transforms(input_size, train, augment)` is the factory used by `DetectionDataset.from_*`.

## Format conversion

`cracks_yolo.dataset.convert` provides:

- `yolo_to_coco(yolo_root, out_json)` — emits a single COCO `instances.json` per split.
- `coco_to_yolo(coco_json, image_dir, out_labels_dir)` — emits one `.txt` per image.

Use `scripts/convert_dataset.py` for the CLI:

```bash
python -m scripts.convert_dataset \
    --input data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --from yolo --to coco \
    --output data/Crack_coco
```

## Target tensor conventions

The pipeline's `targets_to_yolo` helper converts the dataloader's `list[dict]` format to the `(N, 6)` YOLO target tensor consumed by all YOLO losses:

| Column | Meaning |
| --- | --- |
| 0 | image index (0-based, within the batch) |
| 1 | class id (0-based) |
| 2 | x_center (normalized) |
| 3 | y_center (normalized) |
| 4 | width (normalized) |
| 5 | height (normalized) |

The torchvision detector wrappers (`cracks_yolo/zoo/torchvision_detectors.py`) convert this `(N, 6)` format back to torchvision's `list[dict]` with xyxy absolute-pixel boxes and 1-indexed labels (background = 0).
