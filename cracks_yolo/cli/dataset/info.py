from __future__ import annotations

import argparse
import pathlib
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
    splits: list[t.Literal["train", "val", "test"]] | None = None
    by_source: bool = False
    detailed: bool = False
    show_source_dist: bool = False

    def run(self) -> None:
        """Execute the info command."""
        from cracks_yolo.cli.dataset.helper import load_dataset_info

        try:
            # Load dataset(s)
            datasets = load_dataset_info(
                input_path=self.input,
                format=self.format,
                splits=self.splits,
            )

            # Print information
            print_dataset_info(
                datasets=datasets,
                by_source=self.by_source,
                detailed=self.detailed,
                show_source_dist=self.show_source_dist,
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
            help="Path to dataset (COCO: directory with annotations, YOLO: root directory)",
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
            "--splits",
            "-s",
            type=str,
            nargs="+",
            choices=["train", "val", "test"],
            help="Specific splits to load (default: all available)",
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

        parser.add_argument(
            "--show-source-dist",
            action="store_true",
            help="Show source distribution from filename prefixes (e.g., PREFIX_xxx.jpg)",
        )


def print_dataset_info(
    datasets: dict[str, Dataset],
    by_source: bool = False,
    detailed: bool = False,
    show_source_dist: bool = False,
) -> None:
    """Print dataset information in a formatted way.

    Args:
        datasets: Dict mapping split names to Dataset objects
        by_source: Show statistics by source
        detailed: Show detailed information
        show_source_dist: Show source distribution from filename prefixes
    """
    # Calculate total stats
    total_images = sum(len(ds) for ds in datasets.values())
    total_annotations = sum(ds.num_annotations() for ds in datasets.values())

    # Get first dataset for shared info (categories should be same across splits)
    first_dataset = next(iter(datasets.values()))

    display.header(f"Dataset: {first_dataset.name}")

    # Overview with split information
    with display.section("Overview"):
        overview = {
            "Total Images": total_images,
            "Total Annotations": total_annotations,
            "Categories": first_dataset.num_categories(),
            "Splits": len(datasets),
        }

        if first_dataset.get_sources():
            overview["Sources"] = len(first_dataset.get_sources())

        display.key_value(overview)

    # Split breakdown
    with display.section("Split Distribution"):
        headers = ["Split", "Images", "Annotations", "Percentage"]
        rows = []

        for split_name, dataset in datasets.items():
            num_images = len(dataset)
            num_annotations = dataset.num_annotations()
            percentage = f"{num_images / total_images * 100:.1f}%" if total_images > 0 else "0%"

            rows.append([
                split_name.capitalize(),
                str(num_images),
                str(num_annotations),
                percentage,
            ])

        display.table(headers, rows)

    # Combined statistics or per-split
    if len(datasets) == 1:
        # Single split, show detailed stats
        dataset = next(iter(datasets.values()))
        _print_detailed_stats(dataset, by_source, detailed)
    else:
        # Multiple splits, show combined stats
        display.info("\nShowing combined statistics across all splits:")
        _print_combined_stats(datasets, detailed)

    # Categories (combined across all splits)
    with display.section("Categories"):
        _print_category_distribution(datasets)

    # Source distribution (新增)
    if show_source_dist:
        with display.section("Source Distribution (Filename Prefixes)"):
            _print_source_distribution(datasets)


def _print_source_distribution(datasets: dict[str, Dataset]) -> None:
    """Print source distribution extracted from filename prefixes."""
    from collections import Counter

    # Extract sources from all filenames
    source_counts: Counter[str] = Counter()
    source_annotations: Counter[str] = Counter()

    for dataset in datasets.values():
        for img_id, img_info in dataset.images.items():
            # Extract source from filename (prefix before first underscore)
            stem = pathlib.Path(img_info.file_name).stem
            if "_" in stem:
                source = stem.split("_")[0]
                source_counts[source] += 1
                source_annotations[source] += len(dataset.get_annotations(img_id))

    if not source_counts:
        display.info("No source prefixes detected in filenames")
        return

    # Display results
    headers = ["Source", "Images", "Annotations", "Avg Annotations/Image"]
    rows = []

    for source in sorted(source_counts.keys()):
        img_count = source_counts[source]
        ann_count = source_annotations[source]
        avg_ann = ann_count / img_count if img_count > 0 else 0.0

        rows.append([
            source,
            str(img_count),
            str(ann_count),
            f"{avg_ann:.2f}",
        ])

    display.table(headers, rows)


def _print_detailed_stats(
    dataset: Dataset,
    by_source: bool = False,
    detailed: bool = False,
) -> None:
    """Print detailed statistics for a single dataset."""
    if by_source:
        stats_by_source = dataset.get_statistics(by_source=True)

        with display.section("Statistics by Source"):
            for source, stats in stats_by_source.items():
                display.info(f"Source: {source}")
                _print_stats_table(stats, detailed=detailed)
                display.separator()
    else:
        stats = dataset.get_statistics(by_source=False)
        with display.section("Statistics"):
            _print_stats_table(stats, detailed=detailed)


def _print_combined_stats(
    datasets: dict[str, Dataset],
    detailed: bool = False,
) -> None:
    """Print combined statistics across multiple datasets."""
    # Combine all datasets for overall stats
    all_images = sum(len(ds) for ds in datasets.values())
    all_annotations = sum(ds.num_annotations() for ds in datasets.values())

    # Calculate combined metrics
    all_anns_per_image = []
    all_bbox_areas = []

    for dataset in datasets.values():
        for img_id in dataset.images:
            anns = dataset.get_annotations(img_id)
            all_anns_per_image.append(len(anns))
            for ann in anns:
                all_bbox_areas.append(ann.get_area())

    import numpy as np

    combined_stats = {
        "num_images": all_images,
        "num_annotations": all_annotations,
        "avg_annotations_per_image": float(np.mean(all_anns_per_image))
        if all_anns_per_image
        else 0.0,
        "std_annotations_per_image": float(np.std(all_anns_per_image))
        if all_anns_per_image
        else 0.0,
        "min_annotations_per_image": min(all_anns_per_image) if all_anns_per_image else 0,
        "max_annotations_per_image": max(all_anns_per_image) if all_anns_per_image else 0,
        "avg_bbox_area": float(np.mean(all_bbox_areas)) if all_bbox_areas else 0.0,
        "median_bbox_area": float(np.median(all_bbox_areas)) if all_bbox_areas else 0.0,
    }

    with display.section("Combined Statistics"):
        _print_stats_table(combined_stats, detailed=detailed)

    # Per-split statistics
    if detailed:
        for split_name, dataset in datasets.items():
            stats = dataset.get_statistics(by_source=False)
            print()
            print(f"{split_name.capitalize()} Split:")
            print("-" * (len(split_name) + 7))
            _print_stats_table(stats, detailed=False)


def _print_category_distribution(datasets: dict[str, Dataset]) -> None:
    """Print category distribution across all splits."""
    from collections import Counter

    # Combine category counts from all splits
    combined_cat_dist: Counter[str] = Counter()

    for dataset in datasets.values():
        stats = dataset.get_statistics(by_source=False)
        for cat_name, count in stats["category_distribution"].items():
            combined_cat_dist[cat_name] += count

    # Sort by count
    sorted_categories = combined_cat_dist.most_common()

    headers = ["Category", "Count", "Percentage"]
    rows = []

    total = sum(combined_cat_dist.values())
    for cat_name, count in sorted_categories:
        percentage = f"{count / total * 100:.1f}%" if total > 0 else "0%"
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
        "annotation statistics, split distribution, and category distribution",
        epilog="Examples:\n"
        "  # Show info for all splits in COCO dataset\n"
        "  cracks-yolo dataset info data/coco --format coco\n\n"
        "  # Show info for specific splits\n"
        "  cracks-yolo dataset info data/coco --format coco --splits train val\n\n"
        "  # Show detailed info for YOLO dataset\n"
        "  cracks-yolo dataset info data/yolo --format yolo --detailed\n\n"
        "  # Show info with source breakdown\n"
        "  cracks-yolo dataset info data/merged --format yolo --by-source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    Args.build_args(parser)
    parser.set_defaults(func=Args.func)
