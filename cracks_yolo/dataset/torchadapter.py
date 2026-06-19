"""torch.utils.data.Dataset wrapper + dataloader builder.

``DetectionDataset`` is format-agnostic — it consumes a list of
:class:`RawDetection` (from YOLOSource or COCOSource) and applies a
:class:`DetectionTransform`. Returns ``(image_tensor, target_dict)`` where
``target_dict = {"boxes": (N,4) xyxy abs, "labels": (N,), "image_id": int}``.

``build_dataloader`` returns a ``torch.utils.data.DataLoader`` with a
custom collate_fn that handles variable-N boxes per image (returns a list
of target dicts, not a padded tensor).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from cracks_yolo.dataset.transforms import DetectionTransform
from cracks_yolo.dataset.transforms import build_transforms
from cracks_yolo.dataset.types import RawDetection


@dataclass
class DetectionSample:
    """One item returned by DetectionDataset.__getitem__."""

    image: torch.Tensor  # (3, H, W) float in [0, 1]
    boxes: torch.Tensor  # (N, 4) xyxy absolute pixels
    labels: torch.Tensor  # (N,) long
    image_id: int


class DetectionDataset(Dataset[DetectionSample]):
    """Wraps a list of RawDetection records + a transform.

    Use the ``from_yolo`` / ``from_coco`` classmethods for the common case
    (load a split from disk). For 5-fold CV, build the list of
    RawDetection externally and pass it via ``from_records``.
    """

    def __init__(
        self,
        records: list[RawDetection],
        transform: DetectionTransform,
    ) -> None:
        self.records = records
        self.transform = transform

    @classmethod
    def from_yolo(
        cls,
        root: str | Path,
        split: str,
        input_size: int,
        train: bool = False,
        augment: bool = True,
    ) -> DetectionDataset:
        from cracks_yolo.dataset.yolo import YOLOSource

        src = YOLOSource(root)
        records = src.load_split(split)  # type: ignore[arg-type]
        transform = build_transforms(input_size, train=train, augment=augment)
        return cls(records=records, transform=transform)

    @classmethod
    def from_coco(
        cls,
        root: str | Path,
        split: str,
        input_size: int,
        train: bool = False,
        augment: bool = True,
        image_dir: str | Path | None = None,
    ) -> DetectionDataset:
        from cracks_yolo.dataset.coco import COCOSource

        src = COCOSource(root, image_dir=image_dir)
        records = src.load_split(split)  # type: ignore[arg-type]
        transform = build_transforms(input_size, train=train, augment=augment)
        return cls(records=records, transform=transform)

    @classmethod
    def from_records(
        cls,
        records: list[RawDetection],
        input_size: int,
        train: bool = False,
        augment: bool = True,
    ) -> DetectionDataset:
        transform = build_transforms(input_size, train=train, augment=augment)
        return cls(records=records, transform=transform)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> DetectionSample:
        rec = self.records[idx]
        with Image.open(rec.image_path) as im:
            rgb = im.convert("RGB")
            image_tensor, boxes, labels = self.transform(rgb, rec.boxes_norm, rec.labels)
        return DetectionSample(
            image=image_tensor,
            boxes=boxes,
            labels=labels,
            image_id=rec.image_id,
        )


def detection_collate(
    batch: list[DetectionSample],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Collate function for variable-N boxes per image.

    Returns:
        (images (B, 3, H, W), targets) where ``targets`` is a list of
        dicts ``{"boxes": (N,4), "labels": (N,), "image_id": Tensor(int)}``.
    """
    images = torch.stack([s.image for s in batch], dim=0)
    targets = [
        {
            "boxes": s.boxes,
            "labels": s.labels,
            "image_id": torch.tensor(s.image_id, dtype=torch.long),
        }
        for s in batch
    ]
    return images, targets


def build_dataloader(
    dataset: DetectionDataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = False,
) -> DataLoader[tuple[torch.Tensor, list[dict[str, torch.Tensor]]]]:
    """Build a DataLoader with the detection-aware collate_fn."""
    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate,
    )
