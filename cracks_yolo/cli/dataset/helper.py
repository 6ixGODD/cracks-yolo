"""Helper functions for dataset CLI operations."""

from __future__ import annotations

import pathlib
import typing as t

from cracks_yolo.cli.helper.display import loading
from cracks_yolo.cli.helper.display import success
from cracks_yolo.dataset import Dataset
from cracks_yolo.dataset.exporter import OriginalNaming
from cracks_yolo.dataset.exporter import PrefixNaming
from cracks_yolo.dataset.exporter import SequentialNaming
from cracks_yolo.dataset.exporter import SplitRatio
from cracks_yolo.dataset.exporter import UUIDNaming
from cracks_yolo.dataset.exporter import export_coco
from cracks_yolo.dataset.exporter import export_yolov5
from cracks_yolo.dataset.loader import load_coco
from cracks_yolo.dataset.loader import load_yolo

if t.TYPE_CHECKING:
    from cracks_yolo.dataset.exporter import NamingStrategy


def load_dataset(
    input_path: str,
    format: t.Literal["coco", "yolo"],
    split: t.Literal["train", "test", "val"] | None = None,
) -> Dataset:
    """Load a dataset from the specified path and format.

    Args:
        input_path: Path to dataset
        format: Dataset format ('coco' or 'yolo')
        split: Split to load (for YOLO only)

    Returns:
        Loaded Dataset instance
    """
    if format == "coco":
        return load_coco(input_path)
    if format == "yolo":
        return load_yolo(input_path, split=split if split is not None else "train")
    raise ValueError(f"Unknown format: {format}")


def export_dataset(
    dataset: Dataset,
    output_dir: pathlib.Path,
    format: t.Literal["coco", "yolo"] = "coco",
    split_ratio: SplitRatio | None = None,
    naming_strategy: NamingStrategy | None = None,
    seed: int | None = None,
    copy_images: bool = True,
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
    """
    with loading(f"Exporting to {format.upper()} format"):
        if format == "coco":
            export_coco(
                dataset=dataset,
                output_dir=output_dir,
                split_ratio=split_ratio,
                seed=seed,
                naming_strategy=naming_strategy,
                copy_images=copy_images,
            )
        elif format == "yolo":
            export_yolov5(
                dataset=dataset,
                output_dir=output_dir,
                split_ratio=split_ratio,
                seed=seed,
                naming_strategy=naming_strategy,
                copy_images=copy_images,
            )
        else:
            raise ValueError(f"Unknown format: {format}")

    success(f"Exported to: {output_dir}")


def get_naming_strategy(strategy_name: str) -> NamingStrategy:
    """Get naming strategy instance by name.

    Args:
        strategy_name: Strategy name

    Returns:
        NamingStrategy instance
    """
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
