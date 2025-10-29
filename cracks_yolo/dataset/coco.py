"""COCO format dataset implementation."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import typing as t

from cracks_yolo.dataset import BaseDataset
from cracks_yolo.dataset.types import Annotation
from cracks_yolo.dataset.types import BBox
from cracks_yolo.dataset.types import ImageInfo
from cracks_yolo.dataset.types import SplitRatio

logger = logging.getLogger(__name__)


class COCODataset(BaseDataset):
    """COCO format dataset implementation. Supports loading, exporting, and
    manipulating COCO-style datasets.

    Args:
        name: Dataset name identifier
    """

    __slots__ = (*BaseDataset.__slots__, "info", "licenses")

    def __init__(self, name: str = "coco_dataset") -> None:
        super().__init__(name)
        self.info: dict[str, t.Any] = {}
        self.licenses: list[dict[str, t.Any]] = []

    @classmethod
    def from_coco_file(
        cls,
        annotation_path: str | os.PathLike[str],
        images_dir: str | os.PathLike[str] | None = None,
    ) -> COCODataset:
        """Load dataset from COCO annotation file.

        Args:
            annotation_path: Path to COCO annotation JSON file
            images_dir: Optional custom path to images directory.
                        If None, will try to find images in common locations.

        Returns:
            Loaded COCODataset instance
        """
        annotation_path = pathlib.Path(annotation_path)

        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        # Try to locate images directory
        if images_dir is not None:
            images_dir = pathlib.Path(images_dir)
            if not images_dir.exists():
                raise FileNotFoundError(f"Images directory not found: {images_dir}")
        else:
            # Try common locations
            possible_locations = [
                annotation_path.parent / "images",  # Same level as annotation
                annotation_path.parent.parent / "images",  # One level up
                annotation_path.parent,  # Annotation and images in same dir
            ]

            images_dir = None
            for loc in possible_locations:
                if loc.exists() and loc.is_dir():
                    images_dir = loc
                    logger.info(f"Found images directory: {images_dir}")
                    break

            if images_dir is None:
                logger.warning(
                    f"Could not find images directory near {annotation_path}. "
                    f"Image paths will be set to None."
                )

        with annotation_path.open(mode="w") as f:
            coco_data = json.load(f)

        dataset_name = annotation_path.stem
        dataset = cls(name=dataset_name)

        dataset.info = coco_data.get("info", {})
        dataset.licenses = coco_data.get("licenses", [])

        logger.info(f"Loading COCO dataset from {annotation_path}")

        # Load categories
        for cat in coco_data.get("categories", []):
            dataset.add_category(cat["id"], cat["name"])

        logger.info(f"Loaded {len(dataset.categories)} categories")

        # Load images
        images_found = 0
        images_missing = 0

        for img in coco_data.get("images", []):
            img_path = None

            if images_dir is not None:
                # Try to locate the actual image file
                potential_path = images_dir / img["file_name"]

                if potential_path.exists():
                    img_path = potential_path
                    images_found += 1
                else:
                    # Try without subdirectories (flatten path)
                    flat_filename = pathlib.Path(img["file_name"]).name
                    flat_path = images_dir / flat_filename

                    if flat_path.exists():
                        img_path = flat_path
                        images_found += 1
                    else:
                        images_missing += 1
                        logger.debug(f"Image not found: {img['file_name']}")

            img_info = ImageInfo(
                image_id=str(img["id"]),
                file_name=img["file_name"],
                width=img["width"],
                height=img["height"],
                path=img_path,
            )
            dataset.add_image(img_info)

        logger.info(
            f"Loaded {len(dataset.images)} images "
            f"(found: {images_found}, missing: {images_missing})"
        )

        # Load annotations
        for ann in coco_data.get("annotations", []):
            bbox_xywh = ann["bbox"]
            bbox = BBox.from_xywh(*bbox_xywh)

            cat_name = dataset.get_category_name(ann["category_id"])
            if cat_name is None:
                logger.warning(f"Unknown category ID: {ann['category_id']}, skipping annotation")
                continue

            annotation = Annotation(
                bbox=bbox,
                category_id=ann["category_id"],
                category_name=cat_name,
                image_id=str(ann["image_id"]),
                annotation_id=str(ann["id"]),
                area=ann.get("area", bbox.area),
                iscrowd=ann.get("iscrowd", 0),
            )
            dataset.add_annotation(annotation)

        logger.info(f"Loaded {dataset.num_annotations()} annotations")

        if images_missing > 0:
            logger.warning(
                f"{images_missing} images were not found. "
                f"Export operations may fail for these images."
            )

        return dataset

    @classmethod
    def from_coco_dir(
        cls, coco_dir: str | os.PathLike[str], annotation_file: str = "annotations.json"
    ) -> COCODataset:
        """Load dataset from COCO directory structure.

        Expected structure:
        coco_dir/
        ├── annotations.json
        └── images/
            ├── image1.jpg
            └── image2.jpg

        Args:
            coco_dir: Path to COCO directory
            annotation_file: Name of annotation JSON file

        Returns:
            Loaded COCODataset instance
        """
        coco_dir = pathlib.Path(coco_dir)
        annotation_path = coco_dir / annotation_file
        images_dir = coco_dir / "images"

        # Check if images dir exists, if not, try parent/images
        if not images_dir.exists():
            images_dir = coco_dir.parent / "images"

        return cls.from_coco_file(
            annotation_path,
            images_dir=images_dir if images_dir.exists() else None,
        )

    def _export_coco_split(
        self,
        output_dir: pathlib.Path,
        split_name: str,
        image_ids: list[str],
    ) -> None:
        """Export a single split in COCO format.

        Args:
            output_dir: Output directory path
            split_name: Split name (train/val/test)
            image_ids: List of image IDs for this split
        """
        logger.info(f"Exporting COCO {split_name} split to {output_dir}")

        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        coco_output = {
            "info": self.info
            if self.info
            else {
                "description": self.name,
                "version": "1.0",
                "year": 2025,
                "contributor": "6ixGODD",
                "date_created": "2025-10-28",
            },
            "licenses": self.licenses if self.licenses else [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        for cat_id, cat_name in self.categories.items():
            coco_output["categories"].append({
                "id": cat_id,
                "name": cat_name,
                "supercategory": "object",
            })

        ann_id = 1
        images_copied = 0
        images_skipped = 0

        for img_id in image_ids:
            img_info = self.get_image(img_id)
            if img_info is None:
                logger.warning(f"Image {img_id} not found in dataset, skipping")
                continue

            coco_output["images"].append({
                "id": int(img_id),
                "file_name": img_info.file_name,
                "width": img_info.width,
                "height": img_info.height,
            })

            # Copy image file if it exists
            if img_info.path and img_info.path.exists():
                # Use only the filename, not any subdirectory structure
                dest_filename = pathlib.Path(img_info.file_name).name
                dest_path = images_dir / dest_filename

                if not dest_path.exists():
                    try:
                        shutil.copy2(img_info.path, dest_path)
                        images_copied += 1
                        logger.debug(f"Copied image: {img_info.file_name}")
                    except OSError as e:
                        logger.error(f"Failed to copy {img_info.path} to {dest_path}: {e}")
                        images_skipped += 1
            else:
                logger.warning(f"Image file not found: {img_info.file_name}")
                images_skipped += 1

            # Add annotations for this image
            for ann in self.get_annotations(img_id):
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
            f"COCO {split_name} split exported: "
            f"{len(coco_output['images'])} images, "
            f"{len(coco_output['annotations'])} annotations, "
            f"{images_copied} images copied, "
            f"{images_skipped} images skipped"
        )

        print(f"COCO {split_name} split exported:")
        print(f"  Images: {len(coco_output['images'])}")
        print(f"  Annotations: {len(coco_output['annotations'])}")
        print(f"  Copied: {images_copied}, Skipped: {images_skipped}")

    def export_coco(
        self,
        output_dir: str | os.PathLike[str],
        split_ratio: SplitRatio | None = None,
        seed: int | None = None,
    ) -> None:
        """Export dataset in COCO format with optional splitting.

        Args:
            output_dir: Output directory path
            split_ratio: Optional train/val/test split ratios
            seed: Random seed for reproducibility
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting COCO dataset to {output_dir}")

        if split_ratio is None:
            all_image_ids = list(self.images.keys())
            self._export_coco_split(output_dir, "train", all_image_ids)
        else:
            splits = self._split_image_ids(split_ratio, seed)
            for split_name, image_ids in splits.items():
                if image_ids:
                    self._export_coco_split(output_dir, split_name, image_ids)

        print(f"\nCOCO dataset exported to: {output_dir}")
        print(f"Total categories: {self.num_categories()}")

    def _export_yolov5_split(
        self,
        output_dir: pathlib.Path,
        split_name: str,
        image_ids: list[str],
    ) -> None:
        """Export a single split in YOLOv5 format.

        Args:
            output_dir: Output directory path
            split_name: Split name (train/val/test)
            image_ids: List of image IDs for this split
        """
        logger.info(f"Exporting YOLOv5 {split_name} split to {output_dir}")

        images_dir = output_dir / "images" / split_name
        labels_dir = output_dir / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        images_copied = 0
        images_skipped = 0
        labels_created = 0

        for img_id in image_ids:
            img_info = self.get_image(img_id)
            if img_info is None:
                logger.warning(f"Image {img_id} not found in dataset, skipping")
                continue

            # Copy image file
            if img_info.path and img_info.path.exists():
                # Use only the filename, not any subdirectory structure
                dest_filename = pathlib.Path(img_info.file_name).name
                dest_img_path = images_dir / dest_filename

                if not dest_img_path.exists():
                    try:
                        shutil.copy2(img_info.path, dest_img_path)
                        images_copied += 1
                        logger.debug(f"Copied image: {img_info.file_name}")
                    except OSError as e:
                        logger.error(f"Failed to copy {img_info.path} to {dest_img_path}: {e}")
                        images_skipped += 1
                        continue
            else:
                logger.warning(f"Image file not found: {img_info.file_name}, skipping")
                images_skipped += 1
                continue

            # Create label file
            label_file_name = pathlib.Path(img_info.file_name).stem + ".txt"
            label_path = labels_dir / label_file_name

            anns = self.get_annotations(img_id)
            with label_path.open(mode="w") as f:
                for ann in anns:
                    cx, cy, w, h = ann.bbox.cxcywh

                    # Normalize coordinates
                    cx_norm = cx / img_info.width
                    cy_norm = cy / img_info.height
                    w_norm = w / img_info.width
                    h_norm = h / img_info.height

                    # YOLOv5 uses 0-indexed class IDs
                    class_idx = ann.category_id - 1 if ann.category_id > 0 else ann.category_id

                    f.write(f"{class_idx} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            labels_created += 1

        logger.info(
            f"YOLOv5 {split_name} split exported: "
            f"{images_copied} images copied, "
            f"{images_skipped} images skipped, "
            f"{labels_created} labels created"
        )

        print(f"YOLOv5 {split_name} split exported:")
        print(f"  Images: {images_copied}, Skipped: {images_skipped}")
        print(f"  Labels: {labels_created}")

    def export_yolov5(
        self,
        output_dir: str | os.PathLike[str],
        split_ratio: SplitRatio | None = None,
        seed: int | None = None,
    ) -> None:
        """Export dataset in YOLOv5 format with optional splitting.

        YOLOv5 format structure:
        output_dir/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── data.yaml

        Args:
            output_dir: Output directory path
            split_ratio: Optional train/val/test split ratios
            seed: Random seed for reproducibility
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting YOLOv5 dataset to {output_dir}")

        if split_ratio is None:
            all_image_ids = list(self.images.keys())
            self._export_yolov5_split(output_dir, "train", all_image_ids)
            splits_to_write = ["train"]
        else:
            splits = self._split_image_ids(split_ratio, seed)
            for split_name, image_ids in splits.items():
                if image_ids:
                    self._export_yolov5_split(output_dir, split_name, image_ids)
            splits_to_write = [name for name, ids in splits.items() if ids]

        # Create data.yaml
        yaml_path = output_dir / "data.yaml"
        with yaml_path.open(mode="w") as f:
            f.write("# YOLOv5 dataset configuration\n")
            f.write(f"# Generated from {self.name}\n\n")
            f.write(f"path: {output_dir.absolute()}\n")

            for split_name in ["train", "val", "test"]:
                if split_name in splits_to_write:
                    f.write(f"{split_name}: images/{split_name}\n")

            f.write(f"\nnc: {self.num_categories()}\n\n")
            f.write("names:\n")
            for i, cat_id in enumerate(sorted(self.categories.keys())):
                f.write(f"  {i}: {self.categories[cat_id]}\n")

        logger.info(f"Created data.yaml at {yaml_path}")

        print(f"\nYOLOv5 dataset exported to: {output_dir}")
        print(f"Total classes: {self.num_categories()}")
