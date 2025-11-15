from __future__ import annotations

import logging
import typing as t

from cracks_yolo.dataset import Dataset
from cracks_yolo.dataset.exporter import OriginalNaming
from cracks_yolo.dataset.exporter import PrefixNaming
from cracks_yolo.dataset.exporter import SequentialNaming
from cracks_yolo.dataset.exporter import UUIDNaming
from cracks_yolo.dataset.exporter import export_coco
from cracks_yolo.dataset.exporter import export_yolov5
from cracks_yolo.dataset.loader import load_coco
from cracks_yolo.dataset.loader import load_yolo

if t.TYPE_CHECKING:
    from cracks_yolo.dataset.exporter import NamingStrategy
    from cracks_yolo.dataset.types import Annotation
    from cracks_yolo.dataset.types import ImageInfo

logger = logging.getLogger(__name__)


class AnnotationController:
    def __init__(self):
        self.datasets: dict[str, Dataset] = {}
        self.current_split: str | None = None
        self.current_index: int = 0
        self.image_ids_by_split: dict[str, list[str]] = {}

        # Cache for modified annotations
        self.modified_annotations: dict[str, list[Annotation]] = {}

    def load_dataset(
        self,
        path: str,
        format_type: str,
        splits: list[t.Literal["train", "val", "test"]],
    ) -> None:
        logger.info(f"Loading {format_type} dataset from {path}, splits: {splits}")

        self.datasets.clear()
        self.image_ids_by_split.clear()
        self.modified_annotations.clear()

        import pathlib

        path_obj = pathlib.Path(path)

        if format_type == "coco":
            for split in splits:
                ann_file = path_obj / f"annotations_{split}.json"
                if ann_file.exists():
                    dataset = load_coco(str(ann_file), name=f"dataset_{split}")
                    self.datasets[split] = dataset
                    self.image_ids_by_split[split] = list(dataset.images.keys())
                else:
                    logger.warning(f"Annotation file not found: {ann_file}")

        elif format_type == "yolo":
            for split in splits:
                split_dir = path_obj / "images" / split
                if split_dir.exists():
                    dataset = load_yolo(path, splits=split, name=f"dataset_{split}")
                    self.datasets[split] = dataset
                    self.image_ids_by_split[split] = list(dataset.images.keys())
                else:
                    logger.warning(f"Split directory not found: {split_dir}")

        if splits and self.datasets:
            self.current_split = (
                splits[0] if splits[0] in self.datasets else next(iter(self.datasets.keys()))
            )
            self.current_index = 0

        logger.info(f"Loaded {len(self.datasets)} split(s)")

    def export_dataset(
        self,
        output_dir: str,
        format_type: str,
        naming_strategy: str,
    ) -> None:
        import pathlib

        from cracks_yolo.dataset.loader import merge

        logger.info(f"Exporting to {output_dir} as {format_type}")

        # Apply all modified annotations before export
        self._apply_all_modifications()

        # Merge all splits
        if len(self.datasets) > 1:
            merged = merge(*self.datasets.values(), name="exported_dataset")
        else:
            merged = next(iter(self.datasets.values()))

        strategy = self._get_naming_strategy(naming_strategy)
        output_path = pathlib.Path(output_dir)

        if format_type == "coco":
            export_coco(merged, output_path, naming_strategy=strategy, copy_images=True)
        elif format_type == "yolo":
            export_yolov5(merged, output_path, naming_strategy=strategy, copy_images=True)

        logger.info(f"Export completed to {output_dir}")

    def _apply_all_modifications(self) -> None:
        for img_id, anns in self.modified_annotations.items():
            # Find which split this image belongs to
            for _, dataset in self.datasets.items():
                if img_id in dataset.images:
                    # Update annotations in dataset
                    updated_anns = []
                    for ann in anns:
                        updated_ann = Annotation(
                            bbox=ann.bbox,
                            category_id=ann.category_id,
                            category_name=ann.category_name,
                            image_id=img_id,
                            annotation_id=ann.annotation_id,
                            area=ann.area,
                            iscrowd=ann.iscrowd,
                        )
                        updated_anns.append(updated_ann)

                    dataset.annotations[img_id] = updated_anns
                    logger.debug(f"Applied {len(updated_anns)} annotations to image {img_id}")
                    break

    def _get_naming_strategy(self, name: str) -> NamingStrategy:
        """Get naming strategy by name."""
        strategies = {
            "original": OriginalNaming(),
            "prefix": PrefixNaming(),
            "uuid": UUIDNaming(with_source_prefix=False),
            "uuid_prefix": UUIDNaming(with_source_prefix=True),
            "sequential": SequentialNaming(with_source_prefix=False),
            "sequential_prefix": SequentialNaming(with_source_prefix=True),
        }
        return strategies.get(name, OriginalNaming())

    def has_dataset(self) -> bool:
        """Check if dataset is loaded."""
        return len(self.datasets) > 0

    def get_splits(self) -> list[str]:
        return list(self.datasets.keys())

    def set_current_split(self, split: str) -> None:
        if split in self.datasets:
            self.current_split = split
            self.current_index = 0

    def get_images_in_split(self, split: str) -> list[tuple[str, str]]:
        dataset = self.datasets.get(split)
        if dataset:
            return [(img_id, img_info.file_name) for img_id, img_info in dataset.images.items()]
        return []

    def get_current_dataset(self) -> Dataset | None:
        if self.current_split:
            return self.datasets.get(self.current_split)
        return None

    def get_image_info(self, image_id: str) -> ImageInfo | None:
        dataset = self.get_current_dataset()
        if dataset:
            return dataset.get_image(image_id)
        return None

    def get_annotations(self, image_id: str) -> list[Annotation]:
        # Check if we have cached modifications
        if image_id in self.modified_annotations:
            return self.modified_annotations[image_id].copy()

        # Otherwise get from dataset
        dataset = self.get_current_dataset()
        if dataset:
            return dataset.get_annotations(image_id)
        return []

    def get_image_source(self, image_id: str) -> str | None:
        dataset = self.get_current_dataset()
        if dataset:
            return dataset.get_image_source(image_id)
        return None

    def get_categories(self) -> dict[int, str]:
        dataset = self.get_current_dataset()
        if dataset:
            return dataset.categories
        return {}

    def next_image(self) -> str | None:
        if not self.current_split:
            return None

        images = self.image_ids_by_split.get(self.current_split, [])
        if self.current_index < len(images) - 1:
            self.current_index += 1
            return images[self.current_index]

        return None

    def prev_image(self) -> str | None:
        if not self.current_split:
            return None

        images = self.image_ids_by_split.get(self.current_split, [])
        if self.current_index > 0:
            self.current_index -= 1
            return images[self.current_index]

        return None

    def get_current_image_id(self) -> str | None:
        if not self.current_split:
            return None

        images = self.image_ids_by_split.get(self.current_split, [])
        if 0 <= self.current_index < len(images):
            return images[self.current_index]

        return None

    def get_current_index(self) -> int:
        return self.current_index + 1

    def get_split_size(self) -> int:
        if self.current_split:
            return len(self.image_ids_by_split.get(self.current_split, []))
        return 0

    def total_images(self) -> int:
        return sum(len(ids) for ids in self.image_ids_by_split.values())

    def get_dataset_info(self) -> dict[str, int]:
        total_images = self.total_images()
        total_annotations = sum(ds.num_annotations() for ds in self.datasets.values())
        num_categories = len(self.get_categories())

        return {
            "images": total_images,
            "annotations": total_annotations,
            "categories": num_categories,
            "splits": len(self.datasets),
        }

    def update_annotations(self, image_id: str, annotations: list[Annotation]) -> None:
        self.modified_annotations[image_id] = annotations.copy()
        logger.debug(f"Cached {len(annotations)} annotations for image {image_id}")

    def has_unsaved_changes(self) -> bool:
        return len(self.modified_annotations) > 0
