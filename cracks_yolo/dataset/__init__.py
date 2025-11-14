from __future__ import annotations

import collections as coll
import logging
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


class Dataset:
    """Base class for dataset management with multi-source support.

    Features:
    - Load from COCO or YOLO format
    - Export to COCO or YOLO format
    - Multi-source dataset management
    - Flexible file naming strategies
    - Comprehensive statistics including per-source analysis

    Args:
        name: Dataset name identifier
    """

    __slots__ = (
        "annotations",
        "categories",
        "category_name_to_id",
        "images",
        "name",
        "source_info",
    )

    def __init__(self, name: str = "dataset") -> None:
        self.name = name
        self.images: dict[str, ImageInfo] = {}
        self.annotations: dict[str, list[Annotation]] = coll.defaultdict(list)
        self.categories: dict[int, str] = {}
        self.category_name_to_id: dict[str, int] = {}
        self.source_info: dict[str, str] = {}  # image_id -> source_name
        logger.debug(f"Initialized dataset: {name}")

    def add_category(self, category_id: int, category_name: str) -> None:
        """Add a category to the dataset."""
        self.categories[category_id] = category_name
        self.category_name_to_id[category_name] = category_id
        logger.debug(f"Added category: {category_id} -> {category_name}")

    def add_image(self, image_info: ImageInfo, source_name: str | None = None) -> None:
        """Add image metadata to dataset with optional source tracking."""
        self.images[image_info.image_id] = image_info
        if source_name:
            self.source_info[image_info.image_id] = source_name
        logger.debug(f"Added image: {image_info.image_id} from source: {source_name}")

    def add_annotation(self, annotation: Annotation) -> None:
        """Add annotation to dataset."""
        self.annotations[annotation.image_id].append(annotation)

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

    def get_image_source(self, image_id: str) -> str | None:
        """Get the source name for an image."""
        return self.source_info.get(image_id)

    def num_images(self) -> int:
        """Get total number of images."""
        return len(self.images)

    def num_annotations(self) -> int:
        """Get total number of annotations."""
        return sum(len(anns) for anns in self.annotations.values())

    def num_categories(self) -> int:
        """Get total number of categories."""
        return len(self.categories)

    def get_sources(self) -> set[str]:
        """Get all unique source names in the dataset."""
        return set(self.source_info.values())

    def get_statistics(
        self, by_source: bool = False
    ) -> DatasetStatistics | dict[str, DatasetStatistics]:
        """Calculate dataset statistics.

        Args:
            by_source: If True, return statistics grouped by source

        Returns:
            DatasetStatistics or dict mapping source names to statistics
        """
        if by_source:
            return self._get_statistics_by_source()

        return self._calculate_statistics(self.images.keys())

    def _calculate_statistics(self, image_ids: t.Iterable[str]) -> DatasetStatistics:
        """Calculate statistics for a specific set of images."""
        image_ids = list(image_ids)

        total_annotations = 0
        category_counts: coll.Counter[str] = coll.Counter()
        annotations_per_image: list[int] = []
        bbox_areas: list[float] = []

        for img_id in image_ids:
            anns = self.get_annotations(img_id)
            annotations_per_image.append(len(anns))
            total_annotations += len(anns)

            for ann in anns:
                category_counts[ann.category_name] += 1
                bbox_areas.append(ann.get_area())

        return DatasetStatistics(
            num_images=len(image_ids),
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

    def _get_statistics_by_source(self) -> dict[str, DatasetStatistics]:
        """Get statistics grouped by source."""
        stats_by_source = {}

        # Group images by source
        source_images: dict[str, list[str]] = coll.defaultdict(list)
        for img_id, source in self.source_info.items():
            source_images[source].append(img_id)

        # Calculate stats for each source
        for source, img_ids in source_images.items():
            stats_by_source[source] = self._calculate_statistics(img_ids)

        return stats_by_source

    def print_statistics(self, by_source: bool = False) -> None:
        """Print dataset statistics.

        Args:
            by_source: If True, print statistics for each source separately
        """
        if by_source:
            self._print_statistics_by_source()
        else:
            self._print_single_statistics(self.get_statistics(by_source=False), self.name)

    def _print_single_statistics(self, stats: DatasetStatistics, title: str) -> None:
        """Print statistics for a single dataset or source."""
        print(f"\n{'=' * 60}")
        print(f"Dataset Statistics: {title}")
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

    def _print_statistics_by_source(self) -> None:
        """Print statistics grouped by source."""
        stats_by_source = self.get_statistics(by_source=True)

        # Print overall statistics first
        print("\n" + "=" * 80)
        print(f"MULTI-SOURCE DATASET OVERVIEW: {self.name}")
        print("=" * 80)
        print(f"Total Sources: {len(stats_by_source)}")
        print(f"Sources: {', '.join(stats_by_source.keys())}")

        overall_stats = self.get_statistics(by_source=False)
        print(f"\nOverall Images: {overall_stats['num_images']}")
        print(f"Overall Annotations: {overall_stats['num_annotations']}")
        print("=" * 80)

        # Print per-source statistics
        for source, stats in stats_by_source.items():
            self._print_single_statistics(stats, f"{self.name} - Source: {source}")

    def fix_duplicate_categories(self) -> dict[int, int]:
        """Fix duplicate category names by merging them."""
        logger.info(f"Fixing duplicate categories in dataset: {self.name}")

        name_to_ids: dict[str, list[int]] = coll.defaultdict(list)
        for cat_id, cat_name in self.categories.items():
            name_to_ids[cat_name].append(cat_id)

        category_mapping: dict[int, int] = {}
        duplicates_found = 0

        for cat_name, cat_ids in name_to_ids.items():
            if len(cat_ids) > 1:
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
                category_mapping[cat_ids[0]] = cat_ids[0]

        if duplicates_found == 0:
            logger.info("No duplicate categories found")
            return category_mapping

        # Rebuild categories
        new_categories: dict[int, str] = {}
        new_category_name_to_id: dict[str, int] = {}

        for cat_name, cat_ids in name_to_ids.items():
            canonical_id = min(cat_ids)
            new_categories[canonical_id] = cat_name
            new_category_name_to_id[cat_name] = canonical_id

        self.categories = new_categories
        self.category_name_to_id = new_category_name_to_id

        # Remap annotations
        annotations_updated = 0
        for img_id in list(self.annotations.keys()):
            new_anns = []
            for ann in self.annotations[img_id]:
                new_cat_id = category_mapping[ann.category_id]
                if ann.category_id != new_cat_id:
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

    def visualize_sample(
        self,
        image_id: str,
        figsize: tuple[int, int] = (12, 8),
        show_labels: bool = True,
        save_path: pathlib.Path | None = None,
    ) -> None:
        """Visualize a single image with its annotations."""
        img_info = self.get_image(image_id)
        if img_info is None or img_info.path is None:
            raise ValueError(f"Image {image_id} not found or has no path")

        img = Image.open(img_info.path)
        anns = self.get_annotations(image_id)
        source = self.get_image_source(image_id)

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
        title = f"Image: {img_info.file_name} | Annotations: {len(anns)}"
        if source:
            title += f" | Source: {source}"
        ax.set_title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info(f"Visualization saved to: {save_path}")

        plt.show()

    def merge(
        self,
        other: Dataset,
        resolve_conflicts: t.Literal["skip", "rename", "error"] = "skip",
        preserve_sources: bool = True,
    ) -> Dataset:
        """Merge another dataset into a new dataset.

        Args:
            other: Another dataset to merge
            resolve_conflicts: How to handle category conflicts
            preserve_sources: Whether to preserve source information

        Returns:
            New merged dataset
        """
        logger.info(f"Merging '{other.name}' into '{self.name}'")

        merged = Dataset(name=f"{self.name}_merged")

        # Merge categories
        category_mapping: dict[int, int] = {}
        next_category_id = max(self.categories.keys()) + 1 if self.categories else 1

        for cat_id, cat_name in self.categories.items():
            merged.add_category(cat_id, cat_name)
            category_mapping[cat_id] = cat_id

        for cat_id, cat_name in other.categories.items():
            if cat_name in merged.category_name_to_id:
                if resolve_conflicts == "skip":
                    category_mapping[cat_id] = merged.category_name_to_id[cat_name]
                elif resolve_conflicts == "rename":
                    new_name = f"{cat_name}_other"
                    merged.add_category(next_category_id, new_name)
                    category_mapping[cat_id] = next_category_id
                    next_category_id += 1
                elif resolve_conflicts == "error":
                    raise ValueError(f"Category name conflict: {cat_name}")
            else:
                merged.add_category(cat_id, cat_name)
                category_mapping[cat_id] = cat_id

        # Merge from self
        for img_id, img_info in self.images.items():
            source = self.source_info.get(img_id, self.name) if preserve_sources else None
            merged.add_image(img_info, source_name=source)
            for ann in self.get_annotations(img_id):
                merged.add_annotation(ann)

        # Merge from other
        image_id_offset = (
            max(int(img_id) for img_id in merged.images if img_id.isdigit()) + 1
            if merged.images
            else 1
        )

        for img_id, img_info in other.images.items():
            if img_id in merged.images:
                new_img_id = str(image_id_offset)
                image_id_offset += 1
            else:
                new_img_id = img_id

            new_img_info = ImageInfo(
                image_id=new_img_id,
                file_name=img_info.file_name,
                width=img_info.width,
                height=img_info.height,
                path=img_info.path,
            )

            source = other.source_info.get(img_id, other.name) if preserve_sources else None
            merged.add_image(new_img_info, source_name=source)

            for ann in other.get_annotations(img_id):
                new_cat_name = merged.get_category_name(category_mapping[ann.category_id])
                if new_cat_name is None:
                    raise ValueError(f"Category mapping failed for {ann.category_id}")

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

        logger.info(f"Merge completed: {merged.num_images()} images")
        return merged

    def visualize_category_distribution(
        self,
        figsize: tuple[int, int] = (12, 6),
        save_path: pathlib.Path | None = None,
    ) -> None:
        """Visualize category distribution as a bar chart.

        Args:
            figsize: Figure size for matplotlib
            save_path: Optional path to save the visualization
        """
        logger.info(f"Visualizing category distribution for dataset: {self.name}")

        stats = self.get_statistics(by_source=False)
        cat_dist = stats["category_distribution"]

        if not cat_dist:
            logger.warning("No categories to visualize")
            return

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

    def split(self, split_ratio: SplitRatio, seed: int | None = None) -> dict[str, list[str]]:
        split_ratio.validate()

        image_ids = list(self.images.keys())
        if seed is not None:
            random.seed(seed)
        random.shuffle(image_ids)

        total = len(image_ids)
        train_end = int(total * split_ratio.train)
        val_end = train_end + int(total * split_ratio.val)

        return {
            "train": image_ids[:train_end],
            "val": image_ids[train_end:val_end],
            "test": image_ids[val_end:],
        }

    def __add__(self, other: object) -> Dataset:
        if not isinstance(other, Dataset):
            return NotImplemented
        return self.merge(other)

    def __len__(self) -> int:
        return self.num_images()
