from __future__ import annotations

import argparse
import pathlib
import typing as t

from cracks_yolo.cli.args import BaseArgs
from cracks_yolo.cli.helper import display

if t.TYPE_CHECKING:
    from argparse import _SubParsersAction


class Args(BaseArgs):
    input: str
    format: t.Literal["coco", "yolo"]
    output: str
    splits: list[t.Literal["train", "val", "test"]] | None = None
    samples: int = 5
    no_category_dist: bool = False
    show_source_dist: bool = False
    show_heatmap: bool = False
    seed: int | None = None

    def run(self) -> None:
        from cracks_yolo.cli.dataset.helper import load_dataset_info
        from cracks_yolo.cli.dataset.helper.visualizer import DatasetVisualizer

        try:
            display.header("Dataset Visualization")

            # Load dataset(s)
            with display.loading(f"Loading {self.format.upper()} dataset"):
                datasets = load_dataset_info(
                    input_path=self.input,
                    format=self.format,
                    splits=self.splits,
                )

            total_images = sum(len(ds) for ds in datasets.values())
            display.success(f"Loaded {len(datasets)} split(s): {total_images} total images")

            # Create output directory
            output_dir = pathlib.Path(self.output)

            # Initialize visualizer
            visualizer = DatasetVisualizer(
                datasets=datasets,
                output_dir=output_dir,
                seed=self.seed,
            )

            # Generate visualizations
            display.info("Generating visualizations...")

            outputs = visualizer.visualize_all(
                num_samples=self.samples,
                show_distribution=not self.no_category_dist,
                show_source_dist=self.show_source_dist,
                show_heatmap=self.show_heatmap,
            )

            # Display results
            display.separator()
            display.success("Visualization completed!")
            display.info(f"Output directory: {output_dir}")
            print("\nGenerated files:")
            for name, path in outputs.items():
                display.info(f"  • {name}: {path.name}")

        except Exception as e:
            display.error(f"Visualization failed: {e}")
            raise

    @classmethod
    def build_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "input",
            type=str,
            help="Path to dataset (COCO: directory, YOLO: root directory)",
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
            "--output",
            "-o",
            type=str,
            required=True,
            help="Output directory for visualizations",
        )

        parser.add_argument(
            "--splits",
            "-s",
            type=str,
            nargs="+",
            choices=["train", "val", "test"],
            help="Specific splits to visualize (default: all available)",
        )

        parser.add_argument(
            "--samples",
            "-n",
            type=int,
            default=5,
            help="Number of sample images per split (default: 5)",
        )

        parser.add_argument(
            "--no-category-dist",
            action="store_true",
            help="Do not generate category distribution plot",
        )

        parser.add_argument(
            "--show-source-dist",
            action="store_true",
            help="Generate source distribution plot (for merged datasets)",
        )

        parser.add_argument(
            "--show-heatmap",
            action="store_true",
            help="Generate annotation density heatmap",
        )

        parser.add_argument(
            "--seed",
            type=int,
            help="Random seed for sample selection",
        )


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparser.add_parser(
        "visualize",
        help="Generate dataset visualizations",
        description="Create comprehensive visual representations of dataset including "
        "sample images, category distribution, split comparison, and annotation heatmaps",
        epilog="Examples:\n"
        "  # Visualize all splits\n"
        "  cracks-yolo dataset visualize data/coco --format coco -o viz\n\n"
        "  # Visualize with heatmap and source distribution\n"
        "  cracks-yolo dataset visualize data/merged --format yolo -o viz \\\n"
        "    --show-heatmap --show-source-dist\n\n"
        "  # Visualize specific splits with more samples\n"
        "  cracks-yolo dataset visualize data/yolo --format yolo -o viz \\\n"
        "    --splits train val --samples 10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    Args.build_args(parser)
    parser.set_defaults(func=Args.func)
