"""CLI entry: dataset diversity analysis + distribution plots.

Example:
    python -m scripts.analyze_dataset --dataset data/Crack --output-dir output/dataset_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cracks_yolo.analysis.dataset import analyze_dataset
from cracks_yolo.analysis.dataset import save_dataset_analysis
from cracks_yolo.dataset.yolo import YOLOSource
from cracks_yolo.viz.dataset import plot_bbox_position_heatmap
from cracks_yolo.viz.dataset import plot_bbox_size_distribution
from cracks_yolo.viz.dataset import plot_class_distribution
from cracks_yolo.viz.dataset import plot_image_size_distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dataset diversity + distributions.")
    parser.add_argument("--dataset", required=True, help="YOLO dataset root (data.yaml)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train", help="Split to analyze (default: train)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    src = YOLOSource(args.dataset)
    records = src.load_split(args.split)
    report = analyze_dataset(records)
    save_dataset_analysis(report, args.output_dir)
    plot_class_distribution(records, args.output_dir / "class_distribution.png")
    plot_bbox_size_distribution(records, args.output_dir / "bbox_size_distribution.png")
    plot_bbox_position_heatmap(records, args.output_dir / "bbox_position_heatmap.png")
    plot_image_size_distribution(records, args.output_dir / "image_size_distribution.png")
    print(f"Done: {args.output_dir}")
    print(f"  n_images={report.n_images} n_annotations={report.n_annotations}")
    print(
        f"  imbalance_ratio={report.imbalance_ratio:.2f} shannon_entropy={report.class_shannon_entropy:.3f}"
    )
    print(f"  spatial_coverage={report.spatial_coverage:.3f}")


if __name__ == "__main__":
    main()
