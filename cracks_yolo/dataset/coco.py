"""COCO format reader.

Layout::

    <root>/
      instances_train.json
      instances_val.json
      <image_dir>/<file_name>

The JSON follows the standard COCO ``instances_*.json`` schema:
``images``, ``annotations``, ``categories``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cracks_yolo.dataset.types import RawDetection
from cracks_yolo.dataset.types import SplitName


def _normalize_split(name: SplitName) -> str:
    return "valid" if name == "val" else name


def _find_annotations_json(root: Path, split: str) -> Path:
    """Locate the COCO instances JSON for ``split``.

    Accepts any of: ``instances_<split>.json``, ``<split>.json``,
    ``instances_<split>2017.json``.
    """
    candidates = [
        root / f"instances_{split}.json",
        root / f"{split}.json",
        root / f"instances_{split}2017.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"no COCO instances JSON for split {split!r} under {root}; "
        f"tried: {[str(c) for c in candidates]}"
    )


class COCOSource:
    """Reads a COCO-format dataset directory.

    The image dir defaults to ``<root>/<split>`` but can be overridden
    (Roboflow COCO exports nest images under ``<root>/<split>/``).
    """

    def __init__(self, root: str | Path, image_dir: str | Path | None = None) -> None:
        self.root = Path(root)
        self._image_dir_override = Path(image_dir) if image_dir is not None else None

    def _image_dir(self, split: str) -> Path:
        if self._image_dir_override is not None:
            return self._image_dir_override
        candidates = [self.root / split, self.root / "images"]
        for c in candidates:
            if c.is_dir():
                return c
        return self.root / split

    def list_splits(self) -> list[str]:
        """Return splits with a discoverable instances JSON."""
        out: list[str] = []
        for s in ("train", "valid", "val", "test"):
            try:
                _find_annotations_json(self.root, _normalize_split(s))
            except FileNotFoundError:
                continue
            out.append(s)
        return list(dict.fromkeys(out))

    def load_split(self, split: SplitName) -> list[RawDetection]:
        s = _normalize_split(split)
        json_path = _find_annotations_json(self.root, s)
        with json_path.open(encoding="utf-8") as f:
            coco: dict[str, Any] = json.load(f)

        images: list[dict[str, Any]] = list(coco.get("images", []))
        annotations: list[dict[str, Any]] = list(coco.get("annotations", []))
        categories: list[dict[str, Any]] = list(coco.get("categories", []))
        self.class_names: list[str] = [str(c["name"]) for c in categories]
        self.num_classes: int = len(categories)

        by_image: dict[int, list[tuple[float, float, float, float, int]]] = {}
        for ann in annotations:
            img_id = int(ann["image_id"])
            x, y, w, h = (float(v) for v in ann["bbox"])  # COCO = xywh absolute pixels
            cat = int(ann["category_id"])
            by_image.setdefault(img_id, []).append((x, y, w, h, cat))

        image_dir = self._image_dir(s)
        out: list[RawDetection] = []
        for img in images:
            img_id = int(img["id"])
            width = int(img.get("width", 0))
            height = int(img.get("height", 0))
            file_name = str(img["file_name"])
            img_path = image_dir / file_name
            raw_boxes = by_image.get(img_id, [])
            boxes_norm: list[tuple[float, float, float, float]] = []
            labels: list[int] = []
            for x, y, w, h, cat in raw_boxes:
                if width > 0 and height > 0:
                    boxes_norm.append((x / width, y / height, (x + w) / width, (y + h) / height))
                else:
                    boxes_norm.append((0.0, 0.0, 0.0, 0.0))
                labels.append(cat)
            out.append(
                RawDetection(
                    image_path=img_path,
                    image_id=img_id,
                    width=width,
                    height=height,
                    boxes_norm=boxes_norm,
                    labels=labels,
                )
            )
        return out


# Alias for symmetry with YOLODataset.
COCODataset = COCOSource
"""Alias — ``COCODataset`` reads the JSON; ``DetectionDataset`` wraps it."""
