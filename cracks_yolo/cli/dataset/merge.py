from __future__ import annotations

import argparse
import pathlib
import typing as t

from cracks_yolo.cli.args import BaseArgs
from cracks_yolo.cli.helper import display
from cracks_yolo.dataset import loader

if t.TYPE_CHECKING:
    from argparse import _SubParsersAction


class Args(BaseArgs):
    inputs: list[str]
    """Input dataset paths (COCO: annotation files, YOLO: root directories)."""

    formats: list[t.Literal["coco", "yolo"]]
    """Dataset formats (one for all, or one per input)."""

    output: str
    """Output directory for merged dataset."""

    name: str = "merged_dataset"
    """Name for the merged dataset."""

    resolve_conflicts: t.Literal["skip", "rename", "error"] = "skip"
    """How to resolve category name conflicts."""

    preserve_sources: bool = True
    """Whether to preserve source information."""

    fix_duplicates: bool = True
    """Whether to fix duplicate categories."""

    def run(self) -> None:
        from cracks_yolo.cli.dataset.helper import load_dataset

        try:
            # Validate inputs
            if len(self.inputs) < 2:
                display.error("At least 2 datasets are required for merging")
                return

            if len(self.formats) == 1:
                # Use same format for all inputs
                formats = self.formats * len(self.inputs)
            elif len(self.formats) != len(self.inputs):
                display.error("Number of formats must match number of inputs or be 1")
                return
            else:
                formats = self.formats

            display.header("Dataset Merge Operation")

            # Load datasets
            display.info(f"Loading {len(self.inputs)} datasets...")
            datasets = []

            for i, (input_path, fmt) in enumerate(zip(self.inputs, formats, strict=True), 1):
                display.step(f"Loading dataset {i}/{len(self.inputs)}: {input_path}", step=i)

                with display.loading(f"Loading {fmt.upper()} dataset"):
                    dataset = load_dataset(input_path, fmt)
                    datasets.append(dataset)

                display.success(
                    f"Loaded {dataset.name}: "
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

            # Export merged dataset
            output_path = pathlib.Path(self.output)
            display.info(f"Exporting merged dataset to: {output_path}")

            # Import here to avoid circular dependency
            from cracks_yolo.cli.dataset.helper import export_dataset

            export_dataset(merged, output_path)

            display.separator()
            display.success("Dataset merge completed successfully!")
            display.info(f"Output: {output_path}")

        except Exception as e:
            display.error(f"Merge failed: {e}")
            raise

    @classmethod
    def build_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "inputs",
            type=str,
            nargs="+",
            help="Input dataset paths (COCO: annotation files, YOLO: root directories)",
        )

        parser.add_argument(
            "--formats",
            "-f",
            type=str,
            nargs="+",
            choices=["coco", "yolo"],
            required=True,
            help="Dataset formats (one for all, or one per input)",
        )

        parser.add_argument(
            "--output",
            "-o",
            type=str,
            required=True,
            help="Output directory for merged dataset",
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


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparser.add_parser(
        "merge",
        help="Merge multiple datasets into one",
        description="Merge multiple COCO or YOLO datasets with conflict resolution "
        "and source tracking",
    )
    Args.build_args(parser)
    parser.set_defaults(func=Args.func)
