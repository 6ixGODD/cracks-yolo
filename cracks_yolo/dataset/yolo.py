"""YOLOv5 PyTorch format reader.

Roboflow-export layout::

    <root>/
      data.yaml            # nc, names, train/val/test path hints
      train/{images,labels}/
      valid/{images,labels}/
      test/{images,labels}/

Label files: one row per box, ``cls xc yc w h`` all normalized to [0, 1].
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import yaml

from cracks_yolo.dataset.types import RawDetection
from cracks_yolo.dataset.types import SplitName


def _normalize_split(name: SplitName) -> str:
    """``val`` -> ``valid``; everything else passes through."""
    return "valid" if name == "val" else name


class YOLOSource:
    """Reads a YOLOv5-format dataset directory.

    The ``data.yaml`` ``train``/``val``/``test`` fields are *hints* — we
    ignore their values and look for ``<root>/<split>/images/`` directly.
    This makes the source robust to the absolute-path pollution that
    Roboflow exports ship.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        yaml_path = self.root / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
        with yaml_path.open(encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
        self.class_names: list[str] = list(cfg.get("names", []))
        self.num_classes: int = int(cfg.get("nc", len(self.class_names)))
        if not self.class_names:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]

    def list_splits(self) -> list[str]:
        """Return the splits that actually exist on disk."""
        return [s for s in ("train", "valid", "test") if (self.root / s / "images").is_dir()]

    def load_split(self, split: SplitName) -> list[RawDetection]:
        """Load all detections for one split.

        Image dimensions are read from the JPEG header via PIL — we don't
        decode the pixels here, that happens in ``DetectionDataset``.
        """
        s = _normalize_split(split)
        images_dir = self.root / s / "images"
        labels_dir = self.root / s / "labels"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"images dir not found: {images_dir}")

        out: list[RawDetection] = []
        image_files = sorted(
            p for p in images_dir.iterdir() if p.suffix in {".jpg", ".jpeg", ".png"}
        )
        for idx, img_path in enumerate(image_files):
            label_path = labels_dir / (img_path.stem + ".txt")
            boxes_norm: list[tuple[float, float, float, float]] = []
            labels: list[int] = []
            if label_path.exists():
                with label_path.open(encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls = int(float(parts[0]))
                        xc, yc, w, h = (float(x) for x in parts[1:5])
                        x1 = max(0.0, xc - w / 2)
                        y1 = max(0.0, yc - h / 2)
                        x2 = min(1.0, xc + w / 2)
                        y2 = min(1.0, yc + h / 2)
                        boxes_norm.append((x1, y1, x2, y2))
                        labels.append(cls)
            with Image.open(img_path) as im:
                width, height = im.size
            out.append(
                RawDetection(
                    image_path=img_path,
                    image_id=idx,
                    width=width,
                    height=height,
                    boxes_norm=boxes_norm,
                    labels=labels,
                )
            )
        return out


# Backwards-friendly alias. Tests and docs may use either name.
YOLODataset = YOLOSource
"""Alias — ``YOLODataset`` reads the directory; ``DetectionDataset`` wraps it."""
