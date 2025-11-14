from __future__ import annotations

import argparse
import pathlib
import typing as t

from cracks_yolo.cli.args import BaseArgs
from cracks_yolo.cli.helper import display
from cracks_yolo.dataset import loader
from cracks_yolo.dataset.types import SplitRatio

if t.TYPE_CHECKING:
    from argparse import _SubParsersAction


class DatasetInput(t.NamedTuple):
    """Dataset input specification."""

    path: str
    format: t.Literal["coco", "yolo"]
    name: str | None = None


class Args(BaseArgs):
    inputs: list[list[str]]  # 改成 list[list[str]]
    """Raw input arguments from -i flags."""

    output: str
    """Output directory for merged dataset."""

    output_format: t.Literal["coco", "yolo"] = "coco"
    """Output format for merged dataset."""

    name: str = "merged_dataset"
    """Name for the merged dataset."""

    resolve_conflicts: t.Literal["skip", "rename", "error"] = "skip"
    """How to resolve category name conflicts."""

    preserve_sources: bool = True
    """Whether to preserve source information."""

    fix_duplicates: bool = True
    """Whether to fix duplicate categories."""

    naming: str = "original"
    """File naming strategy."""

    train_ratio: float = 0.8
    """Training set ratio."""

    val_ratio: float = 0.1
    """Validation set ratio."""

    test_ratio: float = 0.1
    """Test set ratio."""

    no_split: bool = False
    """Do not split dataset."""

    seed: int | None = None
    """Random seed for reproducibility."""

    no_copy: bool = False
    """Do not copy image files."""

    yolo_splits: list[str] | None = None
    """Which splits to load for YOLO datasets (train, val, test)."""

    def run(self) -> None:
        from cracks_yolo.cli.dataset.helper import export_dataset
        from cracks_yolo.cli.dataset.helper import get_naming_strategy
        from cracks_yolo.cli.dataset.helper import load_dataset

        try:
            # Parse inputs
            dataset_inputs = self._parse_inputs()

            if len(dataset_inputs) < 2:
                display.error("At least 2 datasets are required for merging")
                return

            display.header("Dataset Merge Operation")

            # Load datasets
            display.info(f"Loading {len(dataset_inputs)} datasets...")
            datasets = []

            for i, ds_input in enumerate(dataset_inputs, 1):
                display.step(f"Loading dataset {i}/{len(dataset_inputs)}: {ds_input.path}", step=i)

                # Determine splits for YOLO
                yolo_splits = None
                if ds_input.format == "yolo" and self.yolo_splits:
                    yolo_splits = self.yolo_splits

                with display.loading(f"Loading {ds_input.format.upper()} dataset"):
                    dataset = load_dataset(
                        input_path=ds_input.path,
                        format=ds_input.format,
                        name=ds_input.name,
                        yolo_splits=yolo_splits,
                    )
                    datasets.append(dataset)

                display.success(
                    f"Loaded '{dataset.name}': "
                    f"{len(dataset)} images, "
                    f"{dataset.num_annotations()} annotations, "
                    f"{dataset.num_categories()} categories"
                )

            display.separator()

            # Show merge summary
            display.info("Merge Summary:")
            total_images = sum(len(d) for d in datasets)
            total_annotations = sum(d.num_annotations() for d in datasets)

            print(f"  • Total images: {total_images}")
            print(f"  • Total annotations: {total_annotations}")
            print(f"  • Conflict resolution: {self.resolve_conflicts}")
            print(f"  • Preserve sources: {self.preserve_sources}")
            print(f"  • Fix duplicates: {self.fix_duplicates}")
            print(f"  • Output format: {self.output_format}")

            if not self.no_split:
                print(
                    f"  • Split ratio: train={self.train_ratio:.2f}, "
                    f"val={self.val_ratio:.2f}, test={self.test_ratio:.2f}"
                )

            display.separator()

            # Confirm merge
            if not display.confirm("Proceed with merge?", default=True):
                display.warning("Merge cancelled by user")
                return

            # Perform merge
            with display.loading("Merging datasets"):
                merged = loader.merge(
                    *datasets,
                    name=self.name,
                    resolve_conflicts=self.resolve_conflicts,
                    preserve_sources=self.preserve_sources,
                    fix_duplicates=self.fix_duplicates,
                )

            display.success(
                f"Merged successfully: "
                f"{len(merged)} images, "
                f"{merged.num_annotations()} annotations, "
                f"{merged.num_categories()} categories"
            )

            if self.preserve_sources:
                sources = merged.get_sources()
                display.info(f"Sources: {', '.join(sources)}")

            # Export merged dataset
            output_path = pathlib.Path(self.output)

            with display.loading(f"Exporting to: {output_path}"):
                naming_strategy = get_naming_strategy(self.naming)

                # Prepare split ratio
                split_ratio = None
                if not self.no_split:
                    split_ratio = SplitRatio(
                        train=self.train_ratio,
                        val=self.val_ratio,
                        test=self.test_ratio,
                    )

                export_dataset(
                    dataset=merged,
                    output_dir=output_path,
                    format=self.output_format,
                    split_ratio=split_ratio,
                    naming_strategy=naming_strategy,
                    seed=self.seed,
                    copy_images=not self.no_copy,
                )

            display.separator()
            display.success("Dataset merge completed successfully!")
            display.info(f"Output: {output_path}")

        except Exception as e:
            display.error(f"Merge failed: {e}")
            raise

    def _parse_inputs(self) -> list[DatasetInput]:
        """Parse input arguments into DatasetInput objects.

        inputs is a list of lists: [['path1', 'format1', 'name1'], ['path2', 'format2'], ...]

        Returns:
            List of DatasetInput objects
        """
        dataset_inputs = []

        for input_args in self.inputs:
            if len(input_args) < 2:
                raise ValueError(
                    f"Invalid input specification: {input_args}. "
                    f"Expected at least 2 arguments (path and format)"
                )

            path = input_args[0]
            format_str = input_args[1]

            # Validate format
            if format_str not in ("coco", "yolo"):
                raise ValueError(f"Invalid format '{format_str}'. Must be 'coco' or 'yolo'")

            format = t.cast(t.Literal["coco", "yolo"], format_str)

            # Check if there's a name (3rd argument)
            name = input_args[2] if len(input_args) >= 3 else None

            dataset_inputs.append(DatasetInput(path=path, format=format, name=name))

        return dataset_inputs

    @classmethod
    def build_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-i",
            "--input",
            dest="inputs",
            action="append",
            nargs="+",
            required=True,
            help="Input dataset: <path> <format> [name]. "
            "Format must be 'coco' or 'yolo'. Name is optional. "
            "Can be specified multiple times. "
            "Example: -i data/ann.json coco my_dataset",
        )

        parser.add_argument(
            "--output",
            "-o",
            type=str,
            required=True,
            help="Output directory for merged dataset",
        )

        parser.add_argument(
            "--output-format",
            "-of",
            type=str,
            choices=["coco", "yolo"],
            default="coco",
            help="Output format (default: coco)",
        )

        parser.add_argument(
            "--name",
            "-n",
            type=str,
            default="merged_dataset",
            help="Name for the merged dataset",
        )

        parser.add_argument(
            "--resolve-conflicts",
            "-r",
            type=str,
            choices=["skip", "rename", "error"],
            default="skip",
            help="How to resolve category name conflicts (default: skip)",
        )

        parser.add_argument(
            "--no-preserve-sources",
            action="store_true",
            help="Do not preserve source information",
        )

        parser.add_argument(
            "--no-fix-duplicates",
            action="store_true",
            help="Do not fix duplicate categories",
        )

        parser.add_argument(
            "--naming",
            type=str,
            choices=[
                "original",
                "prefix",
                "uuid",
                "uuid_prefix",
                "sequential",
                "sequential_prefix",
            ],
            default="original",
            help="File naming strategy (default: original)",
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
            help="Do not split dataset (export as single dataset)",
        )

        parser.add_argument(
            "--seed",
            type=int,
            help="Random seed for reproducibility",
        )

        parser.add_argument(
            "--no-copy",
            action="store_true",
            help="Do not copy image files (only generate annotations)",
        )

        parser.add_argument(
            "--yolo-splits",
            type=str,
            nargs="+",
            choices=["train", "val", "test"],
            help="Which splits to load for YOLO datasets (default: all available)",
        )


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparser.add_parser(
        "merge",
        help="Merge multiple datasets into one",
        description="Merge multiple COCO or YOLO datasets with conflict resolution "
        "and source tracking",
        epilog="Examples:\n"
        "  # Merge with custom names and split\n"
        "  cracks-yolo dataset merge \\\n"
        "    -i data/coco1/ann.json coco dataset1 \\\n"
        "    -i data/coco2/ann.json coco dataset2 \\\n"
        "    -o output/merged \\\n"
        "    --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1\n\n"
        "  # Merge without names\n"
        "  cracks-yolo dataset merge \\\n"
        "    -i data/ann1.json coco \\\n"
        "    -i data/ann2.json coco \\\n"
        "    -o output/merged\n\n"
        "  # Merge YOLO datasets (all splits)\n"
        "  cracks-yolo dataset merge \\\n"
        "    -i data/yolo1 yolo dataset1 \\\n"
        "    -i data/yolo2 yolo dataset2 \\\n"
        "    -o output/merged --output-format yolo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    Args.build_args(parser)
    parser.set_defaults(func=Args.func)
