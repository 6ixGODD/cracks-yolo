"""Dataset adapters: YOLOv5 PyTorch format + COCO format.

Both adapters return the same contract — ``(img_tensor, targets_dict)``
where ``targets_dict = {"boxes": (N,4) xyxy absolute-pixel, "labels": (N,)}``
— so downstream code (DetectionDataset, transforms, dataloader) is
format-agnostic.

The YOLO format on disk (Roboflow export style) is::

    <root>/
      data.yaml            # nc, names, train/val/test paths
      train/
        images/*.jpg
        labels/*.txt       # one row per box: "cls xc yc w h" normalized
      valid/...
      test/...

The COCO format on disk is::

    <root>/
      instances_train.json
      instances_val.json
      train/<image_id>.jpg
      ...
"""

from __future__ import annotations

from cracks_yolo.dataset.coco import COCODataset
from cracks_yolo.dataset.coco import COCOSource
from cracks_yolo.dataset.convert import coco_to_yolo
from cracks_yolo.dataset.convert import yolo_to_coco
from cracks_yolo.dataset.torchadapter import DetectionDataset
from cracks_yolo.dataset.torchadapter import build_dataloader
from cracks_yolo.dataset.transforms import build_transforms
from cracks_yolo.dataset.yolo import YOLODataset
from cracks_yolo.dataset.yolo import YOLOSource

__all__ = [
    "COCODataset",
    "COCOSource",
    "DetectionDataset",
    "YOLODataset",
    "YOLOSource",
    "build_dataloader",
    "build_transforms",
    "coco_to_yolo",
    "yolo_to_coco",
]
