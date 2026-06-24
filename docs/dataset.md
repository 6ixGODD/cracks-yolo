# Dataset

The `cracks_yolo.dataset` module provides format-agnostic ingestion of YOLOv5-PyTorch and
COCO detection datasets. All sources emit `RawDetection` records --- a shared intermediate
representation consumed by a single `DetectionDataset` wrapper that yields
`(image_tensor, targets)` batches for training and inference.

## YOLO format on disk

The module expects the Roboflow-export layout:

```
<root>/
  data.yaml            # nc, names, train/val/test path hints
  train/{images,labels}/
  valid/{images,labels}/
  test/{images,labels}/
```

`data.yaml` declares `nc` (integer), `names` (class label strings), and `train`/`val`/`test`
path hints. The path hints are **not** consulted; the module resolves each split by convention
(`<root>/<split>/images/`), making it robust to absolute-path pollution in Roboflow exports.

Each label file (`<split>/labels/<stem>.txt`) encodes one annotation per line:

```
<cls> <xc> <yc> <w> <h>
```

All fields are floating-point. `cls` is zero-indexed; `xc`, `yc`, `w`, `h` are
centre-normalised to `[0, 1]`. The reader converts to `xyxy` normalised boxes internally.
Accepted image formats: `.jpg`, `.jpeg`, `.png`.

## YOLOSource

`YOLOSource(root)` parses `data.yaml`. Two methods:

- `list_splits()` returns the subset of `("train", "valid", "test")` with an existing
  `<split>/images/` directory. `"val"` is normalised to `"valid"`.
- `load_split(split)` reads the labels directory and JPEG headers (PIL decode only; pixels
  are loaded later by `DetectionDataset`). Returns `list[RawDetection]` with `boxes_norm` as
  `xyxy` in `[0, 1]` and `image_id` set to the file enumeration index.

`YOLODataset` is a backwards-compatible alias.

## COCOSource

`COCOSource(root, image_dir=None)` reads standard COCO `instances_*.json` layouts. The
`image_dir` parameter overrides default heuristics (`<root>/<split>/` or `<root>/images/`),
covering Roboflow COCO exports with non-standard image nesting. `list_splits` scans for
`instances_{split}.json`, `{split}.json`, or `instances_{split}2017.json`. `load_split`
returns `RawDetection` records with COCO `xywh` absolute-pixel boxes converted to normalised
`xyxy`. `COCODataset` is the alias.

## RawDetection

Frozen dataclass (`cracks_yolo.dataset.types`):

| Field        | Type                                      | Description                                     |
|------------- |-------------------------------------------|-------------------------------------------------|
| `image_path` | `Path`                                    | Path to the image file                          |
| `image_id`   | `int`                                     | Unique identifier within the split              |
| `width`      | `int`                                     | Original pixel width (from JPEG header)         |
| `height`     | `int`                                     | Original pixel height                           |
| `boxes_norm` | `list[tuple[float, float, float, float]]` | `xyxy` boxes, normalised to `[0, 1]`            |
| `labels`     | `list[int]`                               | Class indices, 0-indexed                        |

This decouples annotation parsing from pixel loading: sources only read image headers,
deferring full decode to `DetectionDataset.__getitem__`.

## DetectionDataset

`DetectionDataset(records, transform)` is a `torch.utils.data.Dataset[DetectionSample]`
subclass. `__getitem__` opens the JPEG, converts to RGB, applies the transform, and returns
a `DetectionSample(image: Tensor (3,H,W), boxes: Tensor (N,4) xyxy abs, labels: Tensor (N,),
image_id: int)`.

Three classmethods:

| Classmethod    | Signature                                                  | Use case                         |
|--------------- |------------------------------------------------------------|----------------------------------|
| `from_yolo`    | `(root, split, input_size, train, augment)`                | Single YOLO split                |
| `from_coco`    | `(root, split, input_size, train, augment, image_dir)`     | Single COCO split                |
| `from_records` | `(records, input_size, train, augment)`                    | Arbitrary records (e.g. CV fold) |

All three build a `DetectionTransform` via `build_transforms` internally.

## build_transforms

`build_transforms(input_size, train=False, augment=True) -> DetectionTransform` returns a
callable that (a) bilinear-resizes to `(input_size, input_size)`; (b) converts to `Tensor`
normalised to `[0, 1]`; (c) applies random horizontal flip (p = 0.5) when `train` and
`augment`. No mean/std subtraction --- YOLO convention.

Signature of `DetectionTransform.__call__`:
`(image: PIL.Image, boxes_norm, labels) -> (tensor (3,H,W), boxes_xyxy_abs (N,4), labels (N,))`.

## build_dataloader

`build_dataloader(dataset, batch_size, num_workers=0, shuffle=False, pin_memory=False) ->
DataLoader` wraps the dataset in a `torch.utils.data.DataLoader` with `detection_collate` as
`collate_fn`. The collate function stacks images into `(B, 3, H, W)` and returns targets as a
`list[dict]` of `{"boxes": (N,4), "labels": (N,), "image_id": int}`, avoiding lossy padding
across variable-`N` boxes.

## COCO conversion utilities

Two standalone functions in `cracks_yolo.dataset.convert`:

| Function       | Signature                                         | Description                                    |
|--------------- |---------------------------------------------------|------------------------------------------------|
| `yolo_to_coco` | `(yolo_root, out_dir) -> dict[str, Path]`         | Writes `instances_<split>.json` per YOLO split |
| `coco_to_yolo` | `(coco_json, image_dir, out_labels_dir) -> int`   | Writes YOLO `.txt` per image                  |

Images are **not** copied by either direction; only annotations are regenerated.
`coco_to_yolo` writes empty `.txt` files for negative samples (YOLO convention).
