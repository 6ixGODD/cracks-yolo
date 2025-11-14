from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import typing as t

import yaml

from cracks_yolo.dataset import BaseDataset
from cracks_yolo.dataset.types import Annotation
from cracks_yolo.dataset.types import BBox
from cracks_yolo.dataset.types import ImageInfo
from cracks_yolo.dataset.types import SplitRatio

logger = logging.getLogger(__name__)


class YOLOv5Dataset(BaseDataset):
    """YOLOv5 format dataset implementation. Supports loading, exporting, and
    manipulating YOLOv5-style datasets.

    Args:
        name: Dataset name identifier
    """

    __slots__ = (*BaseDataset.__slots__, "yaml_config")

    def __init__(self, name: str = "yolov5_dataset") -> None:
        super().__init__(name)
        self.yaml_config: dict[str, t.Any] = {}

    @classmethod
    def from_yolo_dir(
        cls,
        yolo_dir: str | os.PathLike[str],
        split: t.Literal["train", "val", "test"] = "train",
        yaml_file: str = "data.yaml",
    ) -> YOLOv5Dataset:
        """Load dataset from YOLOv5 directory structure.

        Expected structure:
        yolo_dir/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/

        Args:
            yolo_dir: Path to YOLOv5 root directory
            split: Which split to load (train/val/test)
            yaml_file: Name of YAML configuration file

        Returns:
            Loaded YOLOv5Dataset instance
        """
        yolo_dir = pathlib.Path(yolo_dir)
        yaml_path = yolo_dir / yaml_file

        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        # Load YAML configuration
        with yaml_path.open(mode="r") as f:
            yaml_config = yaml.safe_load(f)

        dataset = cls(name=f"yolov5_{split}")
        dataset.yaml_config = yaml_config

        logger.info(f"Loading YOLOv5 dataset from {yolo_dir}, split: {split}")

        # Load categories
        if "names" in yaml_config:
            names = yaml_config["names"]
            if isinstance(names, dict):
                # Format: {0: 'class1', 1: 'class2'}
                for cat_id, cat_name in names.items():
                    # YOLOv5 uses 0-indexed, we convert to 1-indexed for consistency
                    dataset.add_category(int(cat_id) + 1, cat_name)
            elif isinstance(names, list):
                # Format: ['class1', 'class2']
                for i, cat_name in enumerate(names):
                    dataset.add_category(i + 1, cat_name)
        else:
            raise ValueError("YAML configuration must contain 'names' field")

        logger.info(f"Loaded {len(dataset.categories)} categories")

        # Determine paths
        images_dir = yolo_dir / "images" / split
        labels_dir = yolo_dir / "labels" / split

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        if not labels_dir.exists():
            logger.warning(f"Labels directory not found: {labels_dir}")
            labels_dir = None

        # Load images and annotations
        image_files = sorted(images_dir.glob("*"))
        image_files = [
            f
            for f in image_files
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        ]

        images_loaded = 0
        annotations_loaded = 0

        for img_path in image_files:
            try:
                from PIL import Image

                with Image.open(img_path) as img:
                    width, height = img.size

                img_id = str(images_loaded + 1)
                img_info = ImageInfo(
                    image_id=img_id,
                    file_name=img_path.name,
                    width=width,
                    height=height,
                    path=img_path,
                )
                dataset.add_image(img_info)
                images_loaded += 1

                # Load corresponding label file
                if labels_dir is not None:
                    label_path = labels_dir / (img_path.stem + ".txt")
                    if label_path.exists():
                        with label_path.open(mode="r") as f:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if not line:
                                    continue

                                try:
                                    parts = line.split()
                                    if len(parts) != 5:
                                        logger.warning(
                                            f"Invalid annotation format in {label_path}:{line_num}"
                                        )
                                        continue

                                    class_idx = int(parts[0])
                                    cx_norm = float(parts[1])
                                    cy_norm = float(parts[2])
                                    w_norm = float(parts[3])
                                    h_norm = float(parts[4])

                                    # Convert normalized coordinates to absolute
                                    cx = cx_norm * width
                                    cy = cy_norm * height
                                    w = w_norm * width
                                    h = h_norm * height

                                    bbox = BBox.from_cxcywh(cx, cy, w, h)

                                    # Convert 0-indexed to 1-indexed category ID
                                    cat_id = class_idx + 1
                                    cat_name = dataset.get_category_name(cat_id)

                                    if cat_name is None:
                                        logger.warning(
                                            f"Unknown category ID: {cat_id} in {label_path}:{line_num}"
                                        )
                                        continue

                                    annotation = Annotation(
                                        bbox=bbox,
                                        category_id=cat_id,
                                        category_name=cat_name,
                                        image_id=img_id,
                                        annotation_id=str(annotations_loaded + 1),
                                        area=bbox.area,
                                        iscrowd=0,
                                    )
                                    dataset.add_annotation(annotation)
                                    annotations_loaded += 1

                                except (ValueError, IndexError) as e:
                                    logger.warning(
                                        f"Error parsing annotation in {label_path}:{line_num}: {e}"
                                    )
                                    continue

            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                continue

        logger.info(f"Loaded {images_loaded} images and {annotations_loaded} annotations")

        return dataset

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
            "info": {
                "description": self.name,
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
