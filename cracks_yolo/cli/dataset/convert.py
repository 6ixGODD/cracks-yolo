from __future__ import annotations

import argparse
import pathlib
import typing as t

from cracks_yolo.cli.args import BaseArgs
from cracks_yolo.cli.helper import display
from cracks_yolo.dataset import SplitRatio

if t.TYPE_CHECKING:
    from argparse import _SubParsersAction


class Args(BaseArgs):
    """Arguments for dataset convert command."""

    input: str
    input_format: t.Literal["coco", "yolo"]
    output: str
    output_format: t.Literal["coco", "yolo"]
    split: str = "train"
    naming: str = "original"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    no_split: bool = False
    seed: int | None = None
    no_copy: bool = False

    def run(self) -> None:
        """Execute the convert command."""
        from cracks_yolo.cli.dataset.helper import export_dataset
        from cracks_yolo.cli.dataset.helper import get_naming_strategy
        from cracks_yolo.cli.dataset.helper import load_dataset

        try:
            display.header(
                f"Dataset Conversion: {self.input_format.upper()} → {self.output_format.upper()}"
            )

            # Load dataset
            display.step("Loading dataset", step=1)
            with display.loading(f"Loading {self.input_format.upper()} dataset"):
                dataset = load_dataset(
                    input_path=self.input,
                    format=self.input_format,
                    split=self.split if self.input_format == "yolo" else None,
                )

            display.success(
                f"Loaded: {len(dataset)} images, "
                f"{dataset.num_annotations()} annotations, "
                f"{dataset.num_categories()} categories"
            )

            # Prepare split ratio
            if self.no_split:
                split_ratio = None
            else:
                split_ratio = SplitRatio(
                    train=self.train_ratio,
                    val=self.val_ratio,
                    test=self.test_ratio,
                )

            # Get naming strategy
            naming_strategy = get_naming_strategy(self.naming)

            # Export
            display.step("Exporting dataset", step=2)
            output_path = pathlib.Path(self.output)

            export_dataset(
                dataset=dataset,
                output_dir=output_path,
                format=self.output_format,
                split_ratio=split_ratio,
                naming_strategy=naming_strategy,
                seed=self.seed,
                copy_images=not self.no_copy,
            )

            display.separator()
            display.success("Conversion completed successfully!")
            display.info(f"Output: {output_path}")

        except Exception as e:
            display.error(f"Conversion failed: {e}")
            raise

    @classmethod
    def build_args(cls, parser: argparse.ArgumentParser) -> None:
        """Build command-line arguments."""
        parser.add_argument(
            "input",
            type=str,
            help="Input dataset path",
        )

        parser.add_argument(
            "--input-format",
            "-if",
            type=str,
            choices=["coco", "yolo"],
            required=True,
            help="Input dataset format",
        )

        parser.add_argument(
            "output",
            type=str,
            help="Output directory",
        )

        parser.add_argument(
            "--output-format",
            "-of",
            type=str,
            choices=["coco", "yolo"],
            required=True,
            help="Output dataset format",
        )

        parser.add_argument(
            "--split",
            "-s",
            type=str,
            default="train",
            choices=["train", "val", "test"],
            help="Split to load (for YOLO input only)",
        )

        parser.add_argument(
            "--naming",
            "-n",
            type=str,
            default="original",
            choices=[
                "original",
                "prefix",
                "uuid",
                "uuid_prefix",
                "sequential",
                "sequential_prefix",
            ],
            help="File naming strategy",
        )

        parser.add_argument(
            "--train-ratio",
            type=float,
            default=0.8,
            help="Training set ratio (default: 0.8)",
        )

        parser.add_argument(
            "--val-ratio",
            type=float,
            default=0.1,
            help="Validation set ratio (default: 0.1)",
        )

        parser.add_argument(
            "--test-ratio",
            type=float,
            default=0.1,
            help="Test set ratio (default: 0.1)",
        )

        parser.add_argument(
            "--no-split",
            action="store_true",
            help="Do not split dataset (export as single split)",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducible splits",
        )

        parser.add_argument(
            "--no-copy",
            action="store_true",
            help="Do not copy image files (only generate annotations)",
        )


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the convert subcommand."""
    parser = subparser.add_parser(
        "convert",
        help="Convert dataset between formats",
        description="Convert datasets between COCO and YOLO formats with flexible "
        "splitting and naming options",
    )
    Args.build_args(parser)
    parser.set_defaults(func=Args.func)
