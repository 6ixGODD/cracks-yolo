from __future__ import annotations

import os
import pathlib
import random
import typing as t

from cracks_yolo.cli.helper import display
from cracks_yolo.dataset import Dataset
from cracks_yolo.dataset.exporter import NamingStrategy
from cracks_yolo.dataset.exporter import OriginalNaming
from cracks_yolo.dataset.exporter import PrefixNaming
from cracks_yolo.dataset.exporter import SequentialNaming
from cracks_yolo.dataset.exporter import SplitRatio
from cracks_yolo.dataset.exporter import UUIDNaming
from cracks_yolo.dataset.exporter import export_coco
from cracks_yolo.dataset.exporter import export_yolov5
from cracks_yolo.dataset.loader import load_coco
from cracks_yolo.dataset.loader import load_yolo
from cracks_yolo.exceptions import DatasetExportError
from cracks_yolo.exceptions import DatasetFormatError
from cracks_yolo.exceptions import DatasetLoadError
from cracks_yolo.exceptions import DatasetNotFoundError


def load_dataset(
    input_path: str | os.PathLike[str],
    format: t.Literal["coco", "yolo"],
    yolo_splits: t.Literal["train", "val", "test"]
    | list[t.Literal["train", "val", "test"]]
    | None = None,
    name: str | None = None,
) -> Dataset:
    """Load a dataset from the specified path and format.

    Args:
        input_path: Path to dataset
        format: Dataset format ('coco' or 'yolo')
        yolo_splits: Split to load (for YOLO only)
        name: Optional name for the dataset (if None, will use default from file)

    Returns:
        Loaded Dataset instance
    """
    path = pathlib.Path(input_path)

    if not path.exists():
        raise DatasetNotFoundError(str(input_path))

    try:
        if format == "coco":
            if not path.is_file():
                raise DatasetFormatError(format, "Expected annotation JSON file")
            dataset = load_coco(input_path, name=name)

        elif format == "yolo":
            if not path.is_dir():
                raise DatasetFormatError(format, "Expected dataset directory")
            if not (path / "data.yaml").exists():
                raise DatasetFormatError(format, "data.yaml not found")

            if yolo_splits is not None:
                dataset = load_yolo(
                    input_path,
                    splits=yolo_splits,  # type: ignore
                    name=name,
                )
            else:
                # Load all splits
                dataset = load_yolo(input_path, splits=None, name=name)

        else:
            raise DatasetFormatError(format)

        return dataset

    except (DatasetNotFoundError, DatasetFormatError):
        raise
    except Exception as e:
        raise DatasetLoadError(str(e)) from e


def export_dataset(
    dataset: Dataset,
    output_dir: pathlib.Path,
    format: t.Literal["coco", "yolo"] = "coco",
    split_ratio: SplitRatio | None = None,
    naming_strategy: NamingStrategy | None = None,
    seed: int | None = None,
    copy_images: bool = True,
    unified_structure: bool = False,
) -> None:
    """Export dataset to the specified format.

    Args:
        dataset: Dataset to export
        output_dir: Output directory
        format: Output format ('coco' or 'yolo')
        split_ratio: Optional split ratios
        naming_strategy: File naming strategy
        seed: Random seed for splits
        copy_images: Whether to copy image files
        unified_structure: Whether to export as unified structure
    """
    try:
        if format == "coco":
            export_coco(
                dataset=dataset,
                output_dir=output_dir,
                split_ratio=split_ratio,
                seed=seed,
                naming_strategy=naming_strategy,
                copy_images=copy_images,
                unified_structure=unified_structure,
            )
        elif format == "yolo":
            export_yolov5(
                dataset=dataset,
                output_dir=output_dir,
                split_ratio=split_ratio,
                seed=seed,
                naming_strategy=naming_strategy,
                copy_images=copy_images,
                unified_structure=unified_structure,
            )
        else:
            raise DatasetExportError(f"Unknown format: {format}")

    except DatasetExportError:
        raise
    except Exception as e:
        raise DatasetExportError(str(e)) from e


def get_naming_strategy(strategy_name: str) -> NamingStrategy:
    """Get naming strategy instance by name."""
    strategies = {
        "original": OriginalNaming(),
        "prefix": PrefixNaming(),
        "uuid": UUIDNaming(with_source_prefix=False),
        "uuid_prefix": UUIDNaming(with_source_prefix=True),
        "sequential": SequentialNaming(with_source_prefix=False),
        "sequential_prefix": SequentialNaming(with_source_prefix=True),
    }

    strategy = strategies.get(strategy_name)
    if strategy is None:
        raise ValueError(f"Unknown naming strategy: {strategy_name}")

    return strategy


def print_dataset_info(dataset: Dataset, by_source: bool = False) -> None:
    """Print dataset information."""
    display.header(f"Dataset: {dataset.name}")

    overview = {
        "Images": len(dataset),
        "Annotations": dataset.num_annotations(),
        "Categories": dataset.num_categories(),
    }

    if dataset.get_sources():
        overview["Sources"] = len(dataset.get_sources())

    display.key_value(overview)

    if by_source:
        stats_by_source = dataset.get_statistics(by_source=True)
        for source, stats in stats_by_source.items():
            print()
            print(f"Source: {source}")
            print("-" * (len(source) + 8))
            display.key_value(
                {
                    "Images": stats["num_images"],
                    "Annotations": stats["num_annotations"],
                    "Avg per image": f"{stats['avg_annotations_per_image']:.2f}",
                },
                indent=1,
            )
    else:
        stats = dataset.get_statistics(by_source=False)
        print()
        print("Statistics")
        print("-" * 10)
        display.key_value(
            {
                "Avg annotations/image": f"{stats['avg_annotations_per_image']:.2f}",
                "Min annotations/image": stats["min_annotations_per_image"],
                "Max annotations/image": stats["max_annotations_per_image"],
            },
            indent=1,
        )

    # Categories
    stats = dataset.get_statistics(by_source=False)
    print()
    print("Categories")
    print("-" * 10)

    total = sum(stats["category_distribution"].values())
    for cat_name, count in sorted(
        stats["category_distribution"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        pct = count / total * 100
        print(f"  {cat_name}: {count} ({pct:.1f}%)")


def visualize_dataset(
    dataset: Dataset,
    output_dir: pathlib.Path,
    num_samples: int = 5,
    show_distribution: bool = True,
    seed: int | None = None,
) -> None:
    """Generate dataset visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if show_distribution:
        display.info("Generating category distribution...")
        dist_path = output_dir / "category_distribution.png"
        dataset.visualize_category_distribution(save_path=dist_path)
        display.success(f"Saved: {dist_path.name}")

    if num_samples > 0:
        display.info(f"Visualizing {num_samples} samples...")

        image_ids = list(dataset.images.keys())
        if seed is not None:
            random.seed(seed)

        num_to_sample = min(num_samples, len(image_ids))
        sampled_ids = random.sample(image_ids, num_to_sample)

        for i, img_id in enumerate(sampled_ids, 1):
            img_info = dataset.get_image(img_id)
            if img_info and img_info.path:
                sample_path = output_dir / f"sample_{i:03d}_{img_info.file_name}"
                try:
                    dataset.visualize_sample(
                        image_id=img_id,
                        save_path=sample_path,
                    )
                    display.info(f"  {i}/{num_to_sample}: {sample_path.name}")
                except Exception as e:
                    display.warning(f"  {i}/{num_to_sample}: Failed - {e}")


def load_dataset_info(
    input_path: str | os.PathLike[str],
    format: t.Literal["coco", "yolo"],
    splits: list[t.Literal["train", "val", "test"]] | None = None,
) -> dict[str, Dataset]:
    """Load dataset(s) for info display, potentially multiple splits.

    Args:
        input_path: Path to dataset
        format: Dataset format
        splits: Specific splits to load, or None for all available

    Returns:
        Dict mapping split names to Dataset objects
    """
    from cracks_yolo.dataset.loader import load_coco
    from cracks_yolo.dataset.loader import load_yolo

    path = pathlib.Path(input_path)

    if not path.exists():
        raise DatasetNotFoundError(str(input_path))

    datasets = {}

    try:
        if format == "coco":
            # For COCO, look for annotations_<split>.json files
            if not path.is_dir():
                # Single annotation file, treat as single split
                dataset = load_coco(input_path)
                split_name = path.stem.replace("annotations_", "") or "dataset"
                datasets[split_name] = dataset
            else:
                # Directory, look for split files
                available_splits = []

                # Determine which splits to load
                if splits is None:
                    # Auto-detect available splits
                    for split in ["train", "val", "test"]:
                        ann_file = path / f"annotations_{split}.json"
                        if ann_file.exists():
                            available_splits.append(split)

                    if not available_splits:
                        # No split files found, look for any annotations*.json
                        ann_files = list(path.glob("annotations*.json"))
                        if ann_files:
                            # Use the first one
                            dataset = load_coco(ann_files[0])
                            datasets["dataset"] = dataset
                        else:
                            raise DatasetNotFoundError(f"No annotation files found in {input_path}")
                else:
                    available_splits = splits

                # Load each split
                for split in available_splits:
                    ann_file = path / f"annotations_{split}.json"
                    if ann_file.exists():
                        dataset = load_coco(ann_file, name=f"{path.name}_{split}")
                        datasets[split] = dataset
                    else:
                        display.warning(f"Annotation file not found for {split} split: {ann_file}")

        elif format == "yolo":
            if not path.is_dir():
                raise DatasetFormatError(format, "Expected dataset directory")

            if not (path / "data.yaml").exists():
                raise DatasetFormatError(format, "data.yaml not found")

            # For YOLO, load specified splits or all available
            if splits is None:
                # Load all available splits
                available_splits = []
                for split in ["train", "val", "test"]:
                    if (path / "images" / split).exists():
                        available_splits.append(split)

                if not available_splits:
                    raise DatasetFormatError(
                        format, f"No split directories found in {input_path}/images"
                    )

                # Load each split separately
                for split in available_splits:
                    dataset = load_yolo(input_path, splits=split, name=f"{path.name}_{split}")
                    datasets[split] = dataset
            else:
                # Load specific splits
                for split in splits:
                    if not (path / "images" / str(split)).exists():
                        display.warning(f"Split directory not found: {split}")
                        continue

                    dataset = load_yolo(input_path, splits=split, name=f"{path.name}_{split}")
                    datasets[split] = dataset

        else:
            raise DatasetFormatError(format)

        if not datasets:
            raise DatasetLoadError("No datasets loaded")

        return datasets

    except (DatasetNotFoundError, DatasetFormatError):
        raise
    except Exception as e:
        raise DatasetLoadError(str(e)) from e
