"""Dataset diversity analysis.

Computes:
- Class distribution + imbalance ratio (max/min count).
- Bbox size distribution split by COCO area buckets (small/medium/large).
- Bbox center spatial density (Shannon-like coverage).
- Image size statistics.
- Diversity metrics: class Shannon entropy, bbox aspect-ratio buckets,
  spatial coverage (fraction of grid cells with ≥1 bbox).
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import json
import math
from pathlib import Path

from cracks_yolo.dataset.types import RawDetection


@dataclass
class DatasetAnalysisReport:
    """Structured result of analyze_dataset."""

    n_images: int = 0
    n_annotations: int = 0
    n_classes: int = 0
    class_counts: dict[int, int] = field(default_factory=dict)
    imbalance_ratio: float = 0.0  # max/min class count
    class_shannon_entropy: float = 0.0
    bbox_area_small: int = 0  # area < 32^2 (COCO small)
    bbox_area_medium: int = 0  # 32^2 <= area < 96^2
    bbox_area_large: int = 0  # area >= 96^2 (in absolute pixels scaled by image size)
    bbox_aspect_ratio_buckets: int = 0  # unique rounded aspect ratios
    spatial_coverage: float = 0.0  # fraction of grid cells with >=1 bbox center
    image_width_mean: float = 0.0
    image_width_std: float = 0.0
    image_height_mean: float = 0.0
    image_height_std: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_dataset(
    records: list[RawDetection],
    grid: int = 16,
) -> DatasetAnalysisReport:
    """Analyze a list of RawDetection records for diversity + balance."""
    report = DatasetAnalysisReport()
    report.n_images = len(records)
    if not records:
        return report

    # Class distribution.
    counts: dict[int, int] = {}
    for rec in records:
        for lab in rec.labels:
            counts[lab] = counts.get(lab, 0) + 1
    report.class_counts = counts
    report.n_classes = len(counts)
    report.n_annotations = sum(counts.values())
    if counts:
        report.imbalance_ratio = max(counts.values()) / max(1, min(counts.values()))
        total = report.n_annotations
        report.class_shannon_entropy = -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )

    # Bbox size buckets (using normalized area as fraction of image; compare
    # against COCO thresholds scaled to image_area).
    small_thr = (32 / 64) ** 2  # COCO 32x32 on a 64x64 image
    large_thr = (96 / 64) ** 2
    aspect_ratios: set[float] = set()
    grid_cells: set[tuple[int, int]] = set()
    for rec in records:
        for box in rec.boxes_norm:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if area < small_thr:
                report.bbox_area_small += 1
            elif area < large_thr:
                report.bbox_area_medium += 1
            else:
                report.bbox_area_large += 1
            if h > 0:
                aspect_ratios.add(round(w / h, 1))
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            gx = min(grid - 1, max(0, int(cx * grid)))
            gy = min(grid - 1, max(0, int(cy * grid)))
            grid_cells.add((gx, gy))
    report.bbox_aspect_ratio_buckets = len(aspect_ratios)
    report.spatial_coverage = len(grid_cells) / (grid * grid)

    # Image size stats.
    widths = [r.width for r in records]
    heights = [r.height for r in records]
    report.image_width_mean = _mean(widths)
    report.image_width_std = _std(widths, report.image_width_mean)
    report.image_height_mean = _mean(heights)
    report.image_height_std = _std(heights, report.image_height_mean)

    return report


def _mean(xs: list[int]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _std(xs: list[int], mean: float) -> float:
    if len(xs) < 2:
        return 0.0
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


def save_dataset_analysis(report: DatasetAnalysisReport, out_dir: Path) -> None:
    """Write ``dataset_analysis.json`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_analysis.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )


__all__ = ["DatasetAnalysisReport", "analyze_dataset", "save_dataset_analysis"]
