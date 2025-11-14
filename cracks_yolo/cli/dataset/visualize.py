from __future__ import annotations

import argparse
import pathlib
import random
import typing as t

from cracks_yolo.cli.args import BaseArgs
from cracks_yolo.cli.helper import display
from cracks_yolo.dataset import Dataset

if t.TYPE_CHECKING:
    from argparse import _SubParsersAction


class Args(BaseArgs):
    input: str
    format: t.Literal["coco", "yolo"]
    output: str
    split: str = "train"
    samples: int = 5
    category_dist: bool = True
    seed: int | None = None

    def run(self) -> None:
        from cracks_yolo.cli.dataset.helper import load_dataset

        try:
            display.header("Dataset Visualization")

            # Load dataset
            with display.loading(f"Loading {self.format.upper()} dataset"):
                dataset = load_dataset(
                    input_path=self.input,
                    format=self.format,
                    split=self.split if self.format == "yolo" else None,
                )

            display.success(f"Loaded: {len(dataset)} images, {dataset.num_categories()} categories")

            # Create output directory
            output_dir = pathlib.Path(self.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Visualize
            visualize_dataset(
                dataset=dataset,
                output_dir=output_dir,
                num_samples=self.samples,
                show_distribution=self.category_dist,
                seed=self.seed,
            )

            display.success("Visualization completed!")
            display.info(f"Output directory: {output_dir}")

        except Exception as e:
            display.error(f"Visualization failed: {e}")
            raise

    @classmethod
    def build_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "input",
            type=str,
            help="Path to dataset",
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
            "--split",
            "-s",
            type=str,
            default="train",
            choices=["train", "val", "test"],
            help="Split to visualize (for YOLO format only)",
        )

        parser.add_argument(
            "--samples",
            "-n",
            type=int,
            default=5,
            help="Number of sample images to visualize (default: 5)",
        )

        parser.add_argument(
            "--no-category-dist",
            action="store_true",
            help="Do not generate category distribution plot",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for sample selection",
        )


def visualize_dataset(
    dataset: Dataset,
    output_dir: pathlib.Path,
    num_samples: int = 5,
    show_distribution: bool = True,
    seed: int | None = None,
) -> None:
    """Generate visualizations for a dataset.

    Args:
        dataset: Dataset to visualize
        output_dir: Output directory
        num_samples: Number of sample images to visualize
        show_distribution: Whether to show category distribution
        seed: Random seed for sample selection
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize category distribution
    if show_distribution:
        display.step("Generating category distribution", step=1)
        dist_path = output_dir / "category_distribution.png"

        with display.loading("Creating distribution chart"):
            dataset.visualize_category_distribution(
                figsize=(14, 6),
                save_path=dist_path,
            )

        display.success(f"Saved: {dist_path}")

    # Visualize sample images
    if num_samples > 0:
        display.step(f"Visualizing {num_samples} sample images", step=2)

        # Select random samples
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
                    with display.loading(f"Processing sample {i}/{num_to_sample}"):
                        dataset.visualize_sample(
                            image_id=img_id,
                            figsize=(12, 8),
                            show_labels=True,
                            save_path=sample_path,
                        )

                    display.info(f"  Saved: {sample_path.name}")
                except Exception as e:
                    display.warning(f"  Failed to visualize {img_info.file_name}: {e}")

        display.success(f"Visualized {num_to_sample} samples")


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparser.add_parser(
        "visualize",
        help="Generate dataset visualizations",
        description="Create visual representations of dataset including sample images "
        "and category distribution charts",
    )
    Args.build_args(parser)
    parser.set_defaults(func=Args.func)
