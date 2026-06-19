"""YOLO <-> COCO format converters.

Used by ``scripts/convert_dataset.py``. Both directions preserve the
image files (no copying — the new layout references the same files);
only the annotation files are regenerated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from cracks_yolo.dataset.yolo import YOLOSource


def yolo_to_coco(yolo_root: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Convert a YOLOv5-format dataset to COCO format.

    Writes one ``instances_<split>.json`` per split that exists in the
    YOLO root. Images are *not* copied — the COCO JSON's ``file_name``
    field points to the original YOLO image paths (relative to
    ``out_dir/<split>/``).

    Args:
        yolo_root: Source directory with ``data.yaml`` + ``train/valid/test``.
        out_dir: Where to write the COCO JSONs.

    Returns:
        Dict mapping split name -> output JSON path.
    """
    yolo_root = Path(yolo_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = YOLOSource(yolo_root)
    out: dict[str, Path] = {}
    for split in src.list_splits():
        records = src.load_split(split)  # type: ignore[arg-type]
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        categories = [
            {"id": i + 1, "name": name, "supercategory": "cracks"}
            for i, name in enumerate(src.class_names)
        ]
        for rec in records:
            images.append({
                "id": rec.image_id,
                "file_name": str(rec.image_path.name),
                "width": rec.width,
                "height": rec.height,
            })
            for j, (x1, y1, x2, y2) in enumerate(rec.boxes_norm):
                w = x2 - x1
                h = y2 - y1
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": rec.image_id,
                    "category_id": rec.labels[j] + 1,
                    "bbox": [x1 * rec.width, y1 * rec.height, w * rec.width, h * rec.height],
                    "area": w * rec.width * h * rec.height,
                    "iscrowd": 0,
                })
        coco = {"images": images, "annotations": annotations, "categories": categories}
        out_path = out_dir / f"instances_{split}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        out[split] = out_path
    return out


def coco_to_yolo(
    coco_json: str | Path,
    image_dir: str | Path,
    out_labels_dir: str | Path,
) -> int:
    """Convert a single COCO instances JSON to YOLOv5 label files.

    Writes one ``.txt`` per image at ``out_labels_dir/<file_stem>.txt``.
    Skips images that have no annotations (writes an empty file — YOLO
    convention for negative samples).

    Returns:
        Number of label files written.
    """
    coco_json = Path(coco_json)
    image_dir = Path(image_dir)
    out_labels_dir = Path(out_labels_dir)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    with coco_json.open(encoding="utf-8") as f:
        coco: dict[str, Any] = json.load(f)

    images_by_id: dict[int, dict[str, Any]] = {int(im["id"]): im for im in coco.get("images", [])}
    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        by_image.setdefault(int(ann["image_id"]), []).append(ann)

    written = 0
    for img_id, img in images_by_id.items():
        file_name = str(img["file_name"])
        stem = Path(file_name).stem
        w = int(img.get("width", 0))
        h = int(img.get("height", 0))
        if w <= 0 or h <= 0:
            # Fall back to reading from the JPEG header.
            try:
                with Image.open(image_dir / file_name) as im:
                    w, h = im.size
            except FileNotFoundError:
                continue
        lines: list[str] = []
        for ann in by_image.get(img_id, []):
            x, y, bw, bh = (float(v) for v in ann["bbox"])
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            cls = int(ann["category_id"]) - 1  # COCO ids are 1-indexed
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        (out_labels_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        written += 1
    return written


__all__ = ["coco_to_yolo", "yolo_to_coco"]
