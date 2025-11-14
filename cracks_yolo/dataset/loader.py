from __future__ import annotations

import json
import logging
import os
import pathlib
import typing as t

import PIL.Image as Image
import yaml

from cracks_yolo.dataset import Dataset
from cracks_yolo.dataset.types import Annotation
from cracks_yolo.dataset.types import BBox
from cracks_yolo.dataset.types import ImageInfo
from cracks_yolo.exceptions import DatasetMergeError

logger = logging.getLogger(__name__)


def load_coco(
    annotation_path: str | os.PathLike[str],
    images_dir: str | os.PathLike[str] | None = None,
    name: str | None = None,
) -> Dataset:
    """Load dataset from COCO annotation file.

    Args:
        annotation_path: Path to COCO annotation JSON file
        images_dir: Optional custom path to images directory.
                    If None, will try to find images in common locations.
        name: Optional dataset name (default: use annotation file stem)

    Returns:
        Loaded Dataset instance
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

    with annotation_path.open(mode="r") as f:
        coco_data = json.load(f)

    dataset_name = name or annotation_path.stem
    dataset = Dataset(name=dataset_name)

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
        dataset.add_image(img_info, source_name=dataset_name)

    logger.info(
        f"Loaded {len(dataset.images)} images (found: {images_found}, missing: {images_missing})"
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
            f"{images_missing} images were not found. Export operations may fail for these images."
        )

    return dataset


def load_yolo(
    yolo_dir: str | os.PathLike[str],
    splits: t.Literal["train", "val", "test"]
    | list[t.Literal["train", "val", "test"]]
    | None = None,
    yaml_file: str = "data.yaml",
    name: str | None = None,
) -> Dataset:
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
        splits: Which split(s) to load. Can be:
            - None: Load all available splits (train, val, test)
            - Single split: "train", "val", or "test"
            - List of splits: ["train", "val"]
        yaml_file: Name of YAML configuration file
        name: Optional dataset name (default: use yolo_dir name)

    Returns:
        Loaded Dataset instance
    """
    yolo_dir = pathlib.Path(yolo_dir)
    yaml_path = yolo_dir / yaml_file

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    # Load YAML configuration
    with yaml_path.open(mode="r") as f:
        yaml_config = yaml.safe_load(f)

    dataset_name = name or yolo_dir.name
    dataset = Dataset(name=dataset_name)

    logger.info(f"Loading YOLOv5 dataset from {yolo_dir}")

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

    # Determine which splits to load
    if splits is None:
        # Load all available splits
        splits_to_load = ["train", "val", "test"]
    elif isinstance(splits, str):
        splits_to_load = [splits]
    else:
        splits_to_load = splits

    # Load each split
    total_images = 0
    total_annotations = 0

    for split in splits_to_load:
        images_dir = yolo_dir / "images" / split
        labels_dir = yolo_dir / "labels" / split

        if not images_dir.exists():
            logger.warning(f"Images directory not found for {split} split: {images_dir}")
            continue

        if not labels_dir.exists():
            logger.warning(f"Labels directory not found for {split} split: {labels_dir}")
            labels_dir = None

        logger.info(f"Loading {split} split...")

        # Load images and annotations
        image_files = sorted(images_dir.glob("*"))
        image_files = [
            f
            for f in image_files
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        ]

        split_images = 0
        split_annotations = 0

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size

                img_id = str(total_images + 1)
                img_info = ImageInfo(
                    image_id=img_id,
                    file_name=img_path.name,
                    width=width,
                    height=height,
                    path=img_path,
                )
                dataset.add_image(img_info, source_name=dataset_name)
                total_images += 1
                split_images += 1

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
                                        annotation_id=str(total_annotations + 1),
                                        area=bbox.area,
                                        iscrowd=0,
                                    )
                                    dataset.add_annotation(annotation)
                                    total_annotations += 1
                                    split_annotations += 1

                                except (ValueError, IndexError) as e:
                                    logger.warning(
                                        f"Error parsing annotation in {label_path}:{line_num}: {e}"
                                    )
                                    continue

            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                continue

        logger.info(f"Loaded {split} split: {split_images} images, {split_annotations} annotations")

    logger.info(f"Total loaded: {total_images} images and {total_annotations} annotations")

    return dataset


def merge(
    *datasets: Dataset,
    name: str = "merged_dataset",
    resolve_conflicts: t.Literal["skip", "rename", "error"] = "skip",
    preserve_sources: bool = True,
    fix_duplicates: bool = True,
) -> Dataset:
    """Merge multiple datasets into one.

    Args:
        *datasets: Datasets to merge
        name: Name for the merged dataset
        resolve_conflicts: How to handle category conflicts between datasets
            - 'skip': Merge categories with the same name (recommended for same categories)
            - 'rename': Rename conflicting categories from later datasets
            - 'error': Raise error on any category name conflict
        preserve_sources: Whether to preserve source information
        fix_duplicates: Whether to fix duplicate category names within each dataset before merging

    Returns:
        Merged dataset
    """
    if not datasets:
        logger.error("No datasets provided for merging")
        raise DatasetMergeError("No datasets provided for merging")

    logger.info(
        f"Merging {len(datasets)} datasets into '{name}' "
        f"(fix_duplicates={fix_duplicates}, resolve_conflicts={resolve_conflicts})"
    )

    # Fix duplicates in each dataset first if requested
    if fix_duplicates:
        for i, dataset in enumerate(datasets):
            logger.debug(f"Fixing duplicates in dataset {i + 1}/{len(datasets)}: {dataset.name}")
            dataset.fix_duplicate_categories()

    merged = Dataset(name=name)

    for i, dataset in enumerate(datasets):
        logger.debug(f"Merging dataset {i + 1}/{len(datasets)}: {dataset.name}")
        merged = merged.merge(
            dataset,
            resolve_conflicts=resolve_conflicts,
            preserve_sources=preserve_sources,
        )

    logger.info(f"Successfully merged {len(datasets)} datasets")

    return merged
