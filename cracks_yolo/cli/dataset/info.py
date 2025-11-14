from __future__ import annotations

import argparse
import typing as t

from cracks_yolo.cli.args import BaseArgs
from cracks_yolo.cli.helper import display
from cracks_yolo.dataset import Dataset

if t.TYPE_CHECKING:
    from argparse import _SubParsersAction


class Args(BaseArgs):
    """Arguments for dataset info command."""

    input: str
    format: t.Literal["coco", "yolo"]
    split: str = "train"
    by_source: bool = False
    detailed: bool = False

    def run(self) -> None:
        """Execute the info command."""
        from cracks_yolo.cli.dataset.helper import load_dataset

        try:
            # Load dataset
            dataset = load_dataset(
                input_path=self.input,
                format=self.format,
                yolo_splits=self.split if self.format == "yolo" else None,
            )

            # Print information
            print_dataset_info(
                dataset=dataset,
                by_source=self.by_source,
                detailed=self.detailed,
            )

            display.success("Dataset information displayed successfully")

        except Exception as e:
            display.error(f"Failed to load or display dataset info: {e}")
            raise

    @classmethod
    def build_args(cls, parser: argparse.ArgumentParser) -> None:
        """Build command-line arguments."""
        parser.add_argument(
            "input",
            type=str,
            help="Path to dataset (COCO: annotation file, YOLO: root directory)",
        )

        parser.add_argument(
            "--format",
            "-f",
            type=str,
            choices=["coco", "yolo"],
            required=True,
            help="Dataset format",
        )

        parser.add_argument(
            "--split",
            "-s",
            type=str,
            default="train",
            choices=["train", "val", "test"],
            help="Split to load (for YOLO format only)",
        )

        parser.add_argument(
            "--by-source",
            action="store_true",
            help="Show statistics grouped by source",
        )

        parser.add_argument(
            "--detailed",
            "-d",
            action="store_true",
            help="Show detailed statistics",
        )


def print_dataset_info(
    dataset: Dataset,
    by_source: bool = False,
    detailed: bool = False,
) -> None:
    """Print dataset information in a formatted way.

    Args:
        dataset: Dataset to display
        by_source: Show statistics by source
        detailed: Show detailed information
    """
    from cracks_yolo.cli.helper.display import header
    from cracks_yolo.cli.helper.display import section

    header(f"Dataset: {dataset.name}")

    # Basic information
    with section("Overview"):
        overview = {
            "Name": dataset.name,
            "Images": len(dataset),
            "Annotations": dataset.num_annotations(),
            "Categories": dataset.num_categories(),
        }

        if dataset.get_sources():
            overview["Sources"] = len(dataset.get_sources())

        display.key_value(overview)

    # Statistics
    if by_source:
        stats_by_source = dataset.get_statistics(by_source=True)

        with section("Statistics by Source"):
            for source, stats in stats_by_source.items():
                display.info(f"Source: {source}")
                _print_stats_table(stats, detailed=detailed)
                display.separator()
    else:
        stats = dataset.get_statistics(by_source=False)
        with section("Statistics"):
            _print_stats_table(stats, detailed=detailed)

    # Categories
    with section("Categories"):
        stats = dataset.get_statistics(by_source=False)
        cat_dist = stats["category_distribution"]

        # Sort by count
        sorted_categories = sorted(cat_dist.items(), key=lambda x: x[1], reverse=True)

        headers = ["Category", "Count", "Percentage"]
        rows = []

        total = sum(cat_dist.values())
        for cat_name, count in sorted_categories:
            percentage = f"{count / total * 100:.1f}%"
            rows.append([cat_name, str(count), percentage])

        display.table(headers, rows)


def _print_stats_table(stats: t.Mapping[str, t.Any], detailed: bool = False) -> None:
    """Print statistics in table format.

    Args:
        stats: Statistics dictionary
        detailed: Show detailed statistics
    """
    basic_stats = {
        "Images": stats["num_images"],
        "Annotations": stats["num_annotations"],
        "Categories": stats["num_categories"],
        "Avg Annotations/Image": f"{stats['avg_annotations_per_image']:.2f}",
    }

    display.key_value(basic_stats, indent=1)

    if detailed:
        display.separator()
        detailed_stats = {
            "Std Annotations/Image": f"{stats['std_annotations_per_image']:.2f}",
            "Min Annotations/Image": stats["min_annotations_per_image"],
            "Max Annotations/Image": stats["max_annotations_per_image"],
            "Avg BBox Area": f"{stats['avg_bbox_area']:.2f}",
            "Median BBox Area": f"{stats['median_bbox_area']:.2f}",
        }
        display.key_value(detailed_stats, indent=1)


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the info subcommand."""
    parser = subparser.add_parser(
        "info",
        help="Display dataset information and statistics",
        description="Show detailed information about a dataset including image count, "
        "annotation statistics, and category distribution",
    )
    Args.build_args(parser)
    parser.set_defaults(func=Args.func)
