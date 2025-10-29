from __future__ import annotations

import abc
import collections as coll
import logging
import os
import pathlib
import random
import typing as t

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cracks_yolo.dataset.types import Annotation
from cracks_yolo.dataset.types import DatasetStatistics
from cracks_yolo.dataset.types import ImageInfo
from cracks_yolo.dataset.types import SplitRatio

logger = logging.getLogger(__name__)


class BaseDataset(abc.ABC):
    """Abstract base class for dataset management. Provides common operations
    for annotations and images.

    Args:
        name: Dataset name identifier
    """

    __slots__ = (
        "annotations",
        "categories",
        "category_name_to_id",
        "images",
        "name",
    )

    def __init__(self, name: str = "dataset") -> None:
        self.name = name
        self.images: dict[str, ImageInfo] = {}
        self.annotations: dict[str, list[Annotation]] = coll.defaultdict(list)
        self.categories: dict[int, str] = {}
        self.category_name_to_id: dict[str, int] = {}
        logger.debug(f"Initialized dataset: {name}")

    def add_category(self, category_id: int, category_name: str) -> None:
        """Add a category to the dataset.

        Args:
            category_id: Unique category identifier
            category_name: Category name
        """
        self.categories[category_id] = category_name
        self.category_name_to_id[category_name] = category_id
        logger.debug(f"Added category: {category_id} -> {category_name}")

    def add_image(self, image_info: ImageInfo) -> None:
        """Add image metadata to dataset.

        Args:
            image_info: Image information object
        """
        self.images[image_info.image_id] = image_info
        logger.debug(f"Added image: {image_info.image_id} ({image_info.file_name})")

    def add_annotation(self, annotation: Annotation) -> None:
        """Add annotation to dataset.

        Args:
            annotation: Annotation object
        """
        self.annotations[annotation.image_id].append(annotation)
        logger.debug(
            f"Added annotation {annotation.annotation_id} for image {annotation.image_id}, "
            f"category: {annotation.category_name}"
        )

    def get_image(self, image_id: str) -> ImageInfo | None:
        """Get image information by ID."""
        return self.images.get(image_id)

    def get_annotations(self, image_id: str) -> list[Annotation]:
        """Get all annotations for a specific image."""
        return self.annotations.get(image_id, [])

    def get_category_name(self, category_id: int) -> str | None:
        """Get category name by ID."""
        return self.categories.get(category_id)

    def get_category_id(self, category_name: str) -> int | None:
        """Get category ID by name."""
        return self.category_name_to_id.get(category_name)

    def num_images(self) -> int:
        """Get total number of images."""
        return len(self.images)

    def num_annotations(self) -> int:
        """Get total number of annotations."""
        return sum(len(anns) for anns in self.annotations.values())

    def num_categories(self) -> int:
        """Get total number of categories."""
        return len(self.categories)

    def fix_duplicate_categories(self) -> dict[int, int]:
        """Fix duplicate category names by merging them.

        When multiple category IDs have the same name, keep the first one
        and remap all annotations to use it.

        Returns:
            Mapping from old category IDs to new category IDs
        """
        logger.info(f"Fixing duplicate categories in dataset: {self.name}")

        # Find duplicate category names
        name_to_ids: dict[str, list[int]] = coll.defaultdict(list)
        for cat_id, cat_name in self.categories.items():
            name_to_ids[cat_name].append(cat_id)

        # Build mapping from old ID to canonical ID
        category_mapping: dict[int, int] = {}
        duplicates_found = 0

        for cat_name, cat_ids in name_to_ids.items():
            if len(cat_ids) > 1:
                # Sort IDs to keep the smallest one as canonical
                cat_ids.sort()
                canonical_id = cat_ids[0]
                duplicates_found += len(cat_ids) - 1

                logger.warning(
                    f"Found duplicate category '{cat_name}' with IDs {cat_ids}, "
                    f"keeping ID {canonical_id}"
                )

                for cat_id in cat_ids:
                    category_mapping[cat_id] = canonical_id
            else:
                # No duplicates, map to itself
                category_mapping[cat_ids[0]] = cat_ids[0]

        if duplicates_found == 0:
            logger.info("No duplicate categories found")
            return category_mapping

        # Rebuild categories with only canonical IDs
        new_categories: dict[int, str] = {}
        new_category_name_to_id: dict[str, int] = {}

        for cat_name, cat_ids in name_to_ids.items():
            canonical_id = min(cat_ids)
            new_categories[canonical_id] = cat_name
            new_category_name_to_id[cat_name] = canonical_id

        self.categories = new_categories
        self.category_name_to_id = new_category_name_to_id

        # Remap all annotations to use canonical category IDs
        annotations_updated = 0
        for img_id in list(self.annotations.keys()):
            new_anns = []
            for ann in self.annotations[img_id]:
                if ann.category_id != category_mapping[ann.category_id]:
                    # Need to remap
                    new_cat_id = category_mapping[ann.category_id]
                    new_ann = Annotation(
                        bbox=ann.bbox,
                        category_id=new_cat_id,
                        category_name=ann.category_name,
                        image_id=ann.image_id,
                        annotation_id=ann.annotation_id,
                        area=ann.area,
                        iscrowd=ann.iscrowd,
                    )
                    new_anns.append(new_ann)
                    annotations_updated += 1
                else:
                    new_anns.append(ann)

            self.annotations[img_id] = new_anns

        logger.info(
            f"Fixed {duplicates_found} duplicate categories, "
            f"updated {annotations_updated} annotations"
        )

        return category_mapping

    def get_statistics(self) -> DatasetStatistics:
        """Calculate dataset statistics.

        Returns:
            DatasetStatistics containing various metrics
        """
        logger.info(f"Calculating statistics for dataset: {self.name}")

        total_annotations = 0
        category_counts: coll.Counter[str] = coll.Counter()
        annotations_per_image: list[int] = []
        bbox_areas: list[float] = []

        for _image_id, anns in self.annotations.items():
            annotations_per_image.append(len(anns))
            total_annotations += len(anns)

            for ann in anns:
                category_counts[ann.category_name] += 1
                bbox_areas.append(ann.get_area())

        stats = DatasetStatistics(
            num_images=self.num_images(),
            num_annotations=total_annotations,
            num_categories=self.num_categories(),
            category_distribution=dict(category_counts),
            avg_annotations_per_image=float(np.mean(annotations_per_image))
            if annotations_per_image
            else 0.0,
            std_annotations_per_image=float(np.std(annotations_per_image))
            if annotations_per_image
            else 0.0,
            min_annotations_per_image=min(annotations_per_image) if annotations_per_image else 0,
            max_annotations_per_image=max(annotations_per_image) if annotations_per_image else 0,
            avg_bbox_area=float(np.mean(bbox_areas)) if bbox_areas else 0.0,
            median_bbox_area=float(np.median(bbox_areas)) if bbox_areas else 0.0,
        )

        logger.info(
            f"Statistics calculated: {stats['num_images']} images, "
            f"{stats['num_annotations']} annotations, "
            f"{stats['num_categories']} categories"
        )

        return stats

    def print_statistics(self) -> None:
        """Print dataset statistics in a readable format."""
        logger.info(f"Printing statistics for dataset: {self.name}")
        stats = self.get_statistics()

        print(f"\n{'=' * 60}")
        print(f"Dataset Statistics: {self.name}")
        print(f"{'=' * 60}")
        print(f"Total Images: {stats['num_images']}")
        print(f"Total Annotations: {stats['num_annotations']}")
        print(f"Total Categories: {stats['num_categories']}")
        print("\nAnnotations per Image:")
        print(f"  Average: {stats['avg_annotations_per_image']:.2f}")
        print(f"  Std Dev: {stats['std_annotations_per_image']:.2f}")
        print(f"  Min: {stats['min_annotations_per_image']}")
        print(f"  Max: {stats['max_annotations_per_image']}")
        print("\nBounding Box Area:")
        print(f"  Average: {stats['avg_bbox_area']:.2f}")
        print(f"  Median: {stats['median_bbox_area']:.2f}")
        print("\nCategory Distribution:")
        for cat_name, count in sorted(
            stats["category_distribution"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {cat_name}: {count}")
        print(f"{'=' * 60}\n")

    def visualize_sample(
        self,
        image_id: str,
        figsize: tuple[int, int] = (12, 8),
        show_labels: bool = True,
        save_path: pathlib.Path | None = None,
    ) -> None:
        """Visualize a single image with its annotations.

        Args:
            image_id: Image identifier
            figsize: Figure size for matplotlib
            show_labels: Whether to display category labels
            save_path: Optional path to save the visualization
        """
        logger.info(f"Visualizing sample image: {image_id}")

        img_info = self.get_image(image_id)
        if img_info is None:
            logger.error(f"Image {image_id} not found in dataset")
            raise ValueError(f"Image {image_id} not found in dataset")

        if img_info.path is None or not img_info.path.exists():
            logger.error(f"Image file not found: {img_info.path}")
            raise FileNotFoundError(f"Image file not found: {img_info.path}")

        img = Image.open(img_info.path)
        anns = self.get_annotations(image_id)
        logger.debug(f"Image loaded: {img_info.file_name}, annotations: {len(anns)}")

        _fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(img)

        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_categories()))  # type: ignore
        category_colors = {cat_id: colors[i] for i, cat_id in enumerate(self.categories.keys())}

        for ann in anns:
            bbox = ann.bbox
            color = category_colors[ann.category_id]

            rect = patches.Rectangle(
                (bbox.x_min, bbox.y_min),
                bbox.x_max - bbox.x_min,
                bbox.y_max - bbox.y_min,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            if show_labels:
                ax.text(
                    bbox.x_min,
                    bbox.y_min - 5,
                    ann.category_name,
                    color="white",
                    fontsize=10,
                    bbox={"facecolor": color, "alpha": 0.7, "edgecolor": "none", "pad": 2},
                )

        ax.axis("off")
        ax.set_title(f"Image: {img_info.file_name} | Annotations: {len(anns)}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info(f"Visualization saved to: {save_path}")

        plt.show()

    def visualize_category_distribution(
        self, figsize: tuple[int, int] = (12, 6), save_path: pathlib.Path | None = None
    ) -> None:
        """Visualize category distribution as a bar chart.

        Args:
            figsize: Figure size for matplotlib
            save_path: Optional path to save the visualization
        """
        logger.info(f"Visualizing category distribution for dataset: {self.name}")

        stats = self.get_statistics()
        cat_dist = stats["category_distribution"]

        categories = list(cat_dist.keys())
        counts = list(cat_dist.values())

        _fig, ax = plt.subplots(1, figsize=figsize)
        bars = ax.bar(categories, counts, color="skyblue", edgecolor="navy", alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Category Distribution - {self.name}", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info(f"Category distribution saved to: {save_path}")

        plt.show()

    def merge(
        self,
        other: BaseDataset,
        resolve_conflicts: t.Literal["skip", "rename", "error"] = "skip",
    ) -> BaseDataset:
        """Merge another dataset into a new dataset.

        Args:
            other: Another dataset to merge
            resolve_conflicts: How to handle category conflicts ('skip', 'rename', 'error')

        Returns:
            New merged dataset
        """
        logger.info(
            f"Merging dataset '{other.name}' into '{self.name}' with conflict resolution: {resolve_conflicts}"
        )

        merged = self.__class__(name=f"{self.name}_merged")

        category_mapping: dict[int, int] = {}
        next_category_id = max(self.categories.keys()) + 1 if self.categories else 1

        # Merge categories from self
        for cat_id, cat_name in self.categories.items():
            merged.add_category(cat_id, cat_name)
            category_mapping[cat_id] = cat_id

        # Merge categories from other
        conflicts_resolved = 0
        for cat_id, cat_name in other.categories.items():
            if cat_name in merged.category_name_to_id:
                if resolve_conflicts == "skip":
                    category_mapping[cat_id] = merged.category_name_to_id[cat_name]
                    logger.debug(
                        f"Merging duplicate category '{cat_name}' (ID {cat_id} -> {merged.category_name_to_id[cat_name]})"
                    )
                    conflicts_resolved += 1
                elif resolve_conflicts == "rename":
                    new_name = f"{cat_name}_other"
                    merged.add_category(next_category_id, new_name)
                    category_mapping[cat_id] = next_category_id
                    logger.debug(f"Renamed conflicting category: {cat_name} -> {new_name}")
                    next_category_id += 1
                    conflicts_resolved += 1
                elif resolve_conflicts == "error":
                    logger.error(f"Category name conflict: {cat_name}")
                    raise ValueError(f"Category name conflict: {cat_name}")
            else:
                merged.add_category(cat_id, cat_name)
                category_mapping[cat_id] = cat_id

        if conflicts_resolved > 0:
            logger.info(f"Merged {conflicts_resolved} duplicate categories")

        # Merge images and annotations from self
        for img_id, img_info in self.images.items():
            merged.add_image(img_info)
            for ann in self.get_annotations(img_id):
                merged.add_annotation(ann)

        logger.debug(f"Merged {len(self.images)} images from '{self.name}'")

        # Merge images and annotations from other
        image_id_offset = (
            max(int(img_id) for img_id in self.images if img_id.isdigit()) + 1 if self.images else 1
        )

        image_id_conflicts = 0
        for img_id, img_info in other.images.items():
            if img_id in merged.images:
                new_img_id = str(image_id_offset)
                image_id_offset += 1
                image_id_conflicts += 1
                logger.debug(f"Image ID conflict resolved: {img_id} -> {new_img_id}")
            else:
                new_img_id = img_id

            new_img_info = ImageInfo(
                image_id=new_img_id,
                file_name=img_info.file_name,
                width=img_info.width,
                height=img_info.height,
                path=img_info.path,
            )
            merged.add_image(new_img_info)

            for ann in other.get_annotations(img_id):
                new_cat_name = merged.get_category_name(category_mapping[ann.category_id])
                if new_cat_name is None:
                    logger.error(f"Category mapping failed for category {ann.category_id}")
                    raise ValueError(f"Category mapping failed for category {ann.category_id}")

                new_ann = Annotation(
                    bbox=ann.bbox,
                    category_id=category_mapping[ann.category_id],
                    category_name=new_cat_name,
                    image_id=new_img_id,
                    annotation_id=ann.annotation_id,
                    area=ann.area,
                    iscrowd=ann.iscrowd,
                )
                merged.add_annotation(new_ann)

        if image_id_conflicts > 0:
            logger.info(f"Resolved {image_id_conflicts} image ID conflicts")

        logger.debug(f"Merged {len(other.images)} images from '{other.name}'")
        logger.info(
            f"Merge completed: {merged.num_images()} total images, "
            f"{merged.num_annotations()} annotations, "
            f"{merged.num_categories()} categories"
        )

        return merged

    def _split_image_ids(
        self, split_ratio: SplitRatio, seed: int | None = None
    ) -> dict[str, list[str]]:
        """Split image IDs into train/val/test sets.

        Args:
            split_ratio: Train/val/test split ratios
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping split names to image ID lists
        """
        logger.info(
            f"Splitting dataset with ratio: train={split_ratio.train}, "
            f"val={split_ratio.val}, test={split_ratio.test}, seed={seed}"
        )

        split_ratio.validate()

        image_ids = list(self.images.keys())
        if seed is not None:
            random.seed(seed)
        random.shuffle(image_ids)

        total = len(image_ids)
        train_end = int(total * split_ratio.train)
        val_end = train_end + int(total * split_ratio.val)

        splits = {
            "train": image_ids[:train_end],
            "val": image_ids[train_end:val_end],
            "test": image_ids[val_end:],
        }

        logger.info(
            f"Split sizes: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )

        return splits

    @abc.abstractmethod
    def export_coco(
        self,
        output_dir: str | os.PathLike[str],
        split_ratio: SplitRatio | None = None,
        seed: int | None = None,
    ) -> None:
        """Export dataset in COCO format with optional splitting."""
        pass

    @abc.abstractmethod
    def export_yolov5(
        self,
        output_dir: str | os.PathLike[str],
        split_ratio: SplitRatio | None = None,
        seed: int | None = None,
    ) -> None:
        """Export dataset in YOLOv5 format with optional splitting."""
        pass


def merge(
    *datasets: BaseDataset,
    output_name: str = "merged_dataset",
    resolve_conflicts: t.Literal["skip", "rename", "error"] = "skip",
    fix_duplicates: bool = True,
) -> BaseDataset:
    """Merge multiple datasets into one.

    Args:
        *datasets: Datasets to merge
        output_name: Name for the merged dataset
        resolve_conflicts: How to handle category conflicts between datasets
            - 'skip': Merge categories with the same name (recommended for same categories)
            - 'rename': Rename conflicting categories from later datasets
            - 'error': Raise error on any category name conflict
        fix_duplicates: Whether to fix duplicate category names within each dataset before merging

    Returns:
        Merged dataset
    """
    if not datasets:
        logger.error("No datasets provided for merging")
        raise ValueError("No datasets provided for merging")

    logger.info(
        f"Merging {len(datasets)} datasets into '{output_name}' "
        f"(fix_duplicates={fix_duplicates}, resolve_conflicts={resolve_conflicts})"
    )

    # Fix duplicates in each dataset first if requested
    if fix_duplicates:
        for i, dataset in enumerate(datasets):
            logger.debug(f"Fixing duplicates in dataset {i + 1}/{len(datasets)}: {dataset.name}")
            dataset.fix_duplicate_categories()

    merged = datasets[0].__class__(name=output_name)

    for i, dataset in enumerate(datasets):
        logger.debug(f"Merging dataset {i + 1}/{len(datasets)}: {dataset.name}")
        merged = merged.merge(dataset, resolve_conflicts=resolve_conflicts)

    logger.info(f"Successfully merged {len(datasets)} datasets")

    return merged
