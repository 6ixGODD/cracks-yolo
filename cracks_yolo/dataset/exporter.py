from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import typing as t
import uuid

from cracks_yolo.dataset import Dataset
from cracks_yolo.dataset import SplitRatio

logger = logging.getLogger(__name__)


class NamingStrategy(t.Protocol):
    """Protocol for file naming strategies."""

    def gen_name(self, origin: str, source: str | None, image_id: str, /) -> str:
        """Generate a new file name based on the original name, source, and
        image ID.

        Args:
            origin: Original file name
            source: Source name (if any)
            image_id: Image ID

        Returns:
            New file name
        """


class OriginalNaming:
    def gen_name(self, origin: str, _source: str | None, _image_id: str, /) -> str:
        return origin


class PrefixNaming:
    def gen_name(self, origin: str, source: str | None, _image_id: str, /) -> str:
        if source:
            stem = pathlib.Path(origin).stem
            suffix = pathlib.Path(origin).suffix
            return f"{source}_{stem}{suffix}"
        return origin


class UUIDNaming:
    def __init__(self, with_source_prefix: bool = True):
        self.with_source_prefix = with_source_prefix

    def gen_name(self, origin: str, source_name: str | None, _image_id: str, /) -> str:
        suffix = pathlib.Path(origin).suffix
        unique_id = str(uuid.uuid4())[:8]

        if self.with_source_prefix and source_name:
            return f"{source_name}_{unique_id}{suffix}"
        return f"{unique_id}{suffix}"


class SequentialNaming:
    def __init__(self, with_source_prefix: bool = True):
        self.with_source_prefix = with_source_prefix
        self.counter = 0

    def gen_name(self, original_name: str, source_name: str | None, _image_id: str) -> str:
        suffix = pathlib.Path(original_name).suffix
        self.counter += 1

        if self.with_source_prefix and source_name:
            return f"{source_name}_{self.counter:06d}{suffix}"
        return f"{self.counter:06d}{suffix}"


def export_coco(
    dataset: Dataset,
    output_dir: str | os.PathLike[str],
    split_ratio: SplitRatio | None = None,
    seed: int | None = None,
    naming_strategy: NamingStrategy | None = None,
    copy_images: bool = True,
) -> None:
    """Export dataset in COCO format.

    Args:
        dataset: Dataset to export
        output_dir: Output directory path
        split_ratio: Optional train/val/test split ratios
        seed: Random seed for reproducibility
        naming_strategy: File naming strategy (default: OriginalNaming)
        copy_images: Whether to copy image files
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    naming_strategy = naming_strategy or OriginalNaming()

    logger.info(f"Exporting COCO dataset to {output_dir}")

    if split_ratio is None:
        all_image_ids = list(dataset.images.keys())
        _export_coco_split(
            dataset,
            output_dir,
            "train",
            all_image_ids,
            naming_strategy,
            copy_images,
        )
    else:
        splits = dataset.split(split_ratio, seed)
        for split_name, image_ids in splits.items():
            if image_ids:
                _export_coco_split(
                    dataset,
                    output_dir,
                    split_name,
                    image_ids,
                    naming_strategy,
                    copy_images,
                )

    logger.info(f"\nCOCO dataset exported to: {output_dir}")


def _export_coco_split(
    dataset: t.Any,
    output_dir: pathlib.Path,
    split_name: str,
    image_ids: list[str],
    naming_strategy: NamingStrategy,
    copy_images: bool,
) -> None:
    """Export a single split in COCO format."""
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    coco_output = {
        "info": {
            "description": dataset.name,
            "version": "1.0",
            "year": 2025,
            "contributor": "6ixGODD",
            "date_created": "2025-11-14",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    for cat_id, cat_name in dataset.categories.items():
        coco_output["categories"].append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "object",
        })

    ann_id = 1
    images_copied = 0
    images_skipped = 0

    # Track renamed files to avoid conflicts
    used_names: set[str] = set()

    for img_id in image_ids:
        img_info = dataset.get_image(img_id)
        if img_info is None:
            logger.warning(f"Image {img_id} not found, skipping")
            continue

        # Generate new filename
        source = dataset.get_image_source(img_id)
        new_filename = naming_strategy.gen_name(img_info.file_name, source, img_id)

        # Handle filename conflicts
        if new_filename in used_names:
            stem = pathlib.Path(new_filename).stem
            suffix = pathlib.Path(new_filename).suffix
            counter = 1
            while f"{stem}_{counter}{suffix}" in used_names:
                counter += 1
            new_filename = f"{stem}_{counter}{suffix}"

        used_names.add(new_filename)

        coco_output["images"].append({
            "id": int(img_id),
            "file_name": new_filename,
            "width": img_info.width,
            "height": img_info.height,
        })

        # Copy image file
        if copy_images and img_info.path and img_info.path.exists():
            dest_path = images_dir / new_filename
            if not dest_path.exists():
                try:
                    shutil.copy2(img_info.path, dest_path)
                    images_copied += 1
                except OSError as e:
                    logger.error(f"Failed to copy {img_info.path}: {e}")
                    images_skipped += 1
        elif copy_images:
            images_skipped += 1

        # Add annotations
        for ann in dataset.get_annotations(img_id):
            x, y, w, h = ann.bbox.xywh
            coco_output["annotations"].append({
                "id": int(ann.annotation_id) if ann.annotation_id else ann_id,
                "image_id": int(img_id),
                "category_id": ann.category_id,
                "bbox": [x, y, w, h],
                "area": ann.get_area(),
                "iscrowd": ann.iscrowd,
                "segmentation": [],
            })
            ann_id += 1

    # Save annotation file
    annotation_file = output_dir / f"annotations_{split_name}.json"
    with annotation_file.open(mode="w") as f:
        json.dump(coco_output, f, indent=2)

    logger.info(
        f"COCO {split_name} exported: {len(coco_output['images'])} images, "
        f"{images_copied} copied, {images_skipped} skipped"
    )


def export_yolov5(
    dataset: Dataset,
    output_dir: str | os.PathLike[str],
    split_ratio: SplitRatio | None = None,
    seed: int | None = None,
    naming_strategy: NamingStrategy | None = None,
    copy_images: bool = True,
) -> None:
    """Export dataset in YOLOv5 format."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    naming_strategy = naming_strategy or OriginalNaming()

    logger.info(f"Exporting YOLOv5 dataset to {output_dir}")

    if split_ratio is None:
        all_image_ids = list(dataset.images.keys())
        _export_yolov5_split(
            dataset,
            output_dir,
            "train",
            all_image_ids,
            naming_strategy,
            copy_images,
        )
        splits_to_write = ["train"]
    else:
        splits = dataset.split(split_ratio, seed)
        splits_to_write = []
        for split_name, image_ids in splits.items():
            if image_ids:
                _export_yolov5_split(
                    dataset,
                    output_dir,
                    split_name,
                    image_ids,
                    naming_strategy,
                    copy_images,
                )
                splits_to_write.append(split_name)

    # Create data.yaml
    yaml_path = output_dir / "data.yaml"
    with yaml_path.open(mode="w") as f:
        f.write("# YOLOv5 dataset configuration\n")
        f.write(f"# Generated from {dataset.name}\n\n")
        f.write(f"path: {output_dir.absolute()}\n")

        for split_name in ["train", "val", "test"]:
            if split_name in splits_to_write:
                f.write(f"{split_name}: images/{split_name}\n")

        f.write(f"\nnc: {dataset.num_categories()}\n\n")
        f.write("names:\n")
        for i, cat_id in enumerate(sorted(dataset.categories.keys())):
            f.write(f"  {i}: {dataset.categories[cat_id]}\n")

    logger.info(f"\nYOLOv5 dataset exported to: {output_dir}")


def _export_yolov5_split(
    dataset: t.Any,
    output_dir: pathlib.Path,
    split_name: str,
    image_ids: list[str],
    naming_strategy: NamingStrategy,
    copy_images: bool,
) -> None:
    """Export a single split in YOLOv5 format."""
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_copied = 0
    images_skipped = 0
    labels_created = 0

    used_names: set[str] = set()

    for img_id in image_ids:
        img_info = dataset.get_image(img_id)
        if img_info is None:
            continue

        # Generate new filename
        source = dataset.get_image_source(img_id)
        new_filename = naming_strategy.gen_name(img_info.file_name, source, img_id)

        # Handle conflicts
        if new_filename in used_names:
            stem = pathlib.Path(new_filename).stem
            suffix = pathlib.Path(new_filename).suffix
            counter = 1
            while f"{stem}_{counter}{suffix}" in used_names:
                counter += 1
            new_filename = f"{stem}_{counter}{suffix}"

        used_names.add(new_filename)

        # Copy image
        if copy_images and img_info.path and img_info.path.exists():
            dest_img_path = images_dir / new_filename
            if not dest_img_path.exists():
                try:
                    shutil.copy2(img_info.path, dest_img_path)
                    images_copied += 1
                except OSError as e:
                    logger.error(f"Failed to copy {img_info.path}: {e}")
                    images_skipped += 1
                    continue
        elif copy_images:
            images_skipped += 1
            continue

        # Create label file
        label_file_name = pathlib.Path(new_filename).stem + ".txt"
        label_path = labels_dir / label_file_name

        anns = dataset.get_annotations(img_id)
        with label_path.open(mode="w") as f:
            for ann in anns:
                cx, cy, w, h = ann.bbox.cxcywh

                # Normalize
                cx_norm = cx / img_info.width
                cy_norm = cy / img_info.height
                w_norm = w / img_info.width
                h_norm = h / img_info.height

                # 0-indexed class IDs
                class_idx = ann.category_id - 1 if ann.category_id > 0 else ann.category_id

                f.write(f"{class_idx} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        labels_created += 1

    logger.info(f"YOLOv5 {split_name} exported: {images_copied} images, {labels_created} labels")
