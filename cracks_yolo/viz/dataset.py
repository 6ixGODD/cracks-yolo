"""Dataset distribution plots — academic styling, dpi=300.

Generates: class distribution, bbox size histogram (log), bbox area buckets
(COCO small/medium/large), aspect-ratio histogram, objects-per-image
histogram, bbox-center heatmap, image-size scatter, class imbalance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cracks_yolo.dataset.types import RawDetection

_STYLE = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
}
_BLUE = "#1f77b4"
_ORANGE = "#ff7f0e"
_GREEN = "#2ca02c"
_RED = "#d62728"


def _setup() -> None:
    matplotlib.rcParams.update(_STYLE)


def _counts(records: list[RawDetection]) -> dict[int, int]:
    c: dict[int, int] = {}
    for rec in records:
        for lab in rec.labels:
            c[lab] = c.get(lab, 0) + 1
    return c


def plot_class_distribution(records: list[RawDetection], out_png: Path) -> None:
    """Bar plot of per-class instance counts."""
    _setup()
    counts = _counts(records)
    if not counts:
        return
    labels = sorted(counts.keys())
    values = [counts[c] for c in labels]
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(labels)), 4))
    bars = ax.bar([str(c) for c in labels], values, color=_BLUE, width=0.6)
    ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Instance count")
    ax.set_title("Class distribution")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_bbox_size_distribution(records: list[RawDetection], out_png: Path) -> None:
    """Histogram of bbox areas (normalized to image area), log x-axis."""
    _setup()
    areas = [float((b[2] - b[0]) * (b[3] - b[1])) for rec in records for b in rec.boxes_norm]
    if not areas:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(areas, bins=50, color=_ORANGE, edgecolor="black", alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel(r"Bbox area (fraction of image area)")
    ax.set_ylabel("Count")
    ax.set_title("Bounding-box size distribution")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_bbox_area_buckets(records: list[RawDetection], out_png: Path) -> None:
    """Bar/pie of COCO-style small / medium / large bbox counts.

    Thresholds (normalized to image area, assuming 64-grid):
    small  < (32/64)^2 ; medium < (96/64)^2 ; large >= (96/64)^2.
    """
    _setup()
    small_thr = (32 / 64) ** 2
    large_thr = (96 / 64) ** 2
    n_small = n_med = n_large = 0
    for rec in records:
        for b in rec.boxes_norm:
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area < small_thr:
                n_small += 1
            elif area < large_thr:
                n_med += 1
            else:
                n_large += 1
    if n_small + n_med + n_large == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    sizes = [n_small, n_med, n_large]
    labels = [f"Small\n({n_small})", f"Medium\n({n_med})", f"Large\n({n_large})"]
    colors = [_RED, _ORANGE, _GREEN]
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title("Bbox area buckets (COCO scale)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_aspect_ratio(records: list[RawDetection], out_png: Path) -> None:
    """Histogram of bbox aspect ratios (width / height)."""
    _setup()
    ars = []
    for rec in records:
        for b in rec.boxes_norm:
            h = b[3] - b[1]
            if h > 0:
                ars.append(float((b[2] - b[0]) / h))
    if not ars:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(ars, bins=40, color=_GREEN, edgecolor="black", alpha=0.75)
    ax.set_xlabel(r"Aspect ratio $w/h$")
    ax.set_ylabel("Count")
    ax.set_title("Bounding-box aspect-ratio distribution")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_objects_per_image(records: list[RawDetection], out_png: Path) -> None:
    """Histogram of object counts per image."""
    _setup()
    counts = [len(rec.boxes_norm) for rec in records]
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(
        counts,
        bins=range(0, max(counts) + 2),
        color=_BLUE,
        edgecolor="black",
        alpha=0.75,
        align="left",
    )
    ax.set_xlabel("Objects per image")
    ax.set_ylabel("Image count")
    ax.set_title("Object-count distribution per image")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_bbox_position_heatmap(records: list[RawDetection], out_png: Path, grid: int = 32) -> None:
    """Heatmap of bbox centers on a grid×grid spatial grid."""
    _setup()
    heat = np.zeros((grid, grid), dtype=np.float64)
    for rec in records:
        for b in rec.boxes_norm:
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            gx = min(grid - 1, max(0, int(cx * grid)))
            gy = min(grid - 1, max(0, int(cy * grid)))
            heat[gy, gx] += 1
    if heat.sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(heat, cmap="hot", origin="lower")
    ax.set_xlabel(r"$x$ grid")
    ax.set_ylabel(r"$y$ grid")
    ax.set_title("Bbox-center spatial density")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_image_size_distribution(records: list[RawDetection], out_png: Path) -> None:
    """Scatter of image width vs height."""
    _setup()
    if not records:
        return
    widths = [r.width for r in records]
    heights = [r.height for r in records]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(widths, heights, alpha=0.4, color=_GREEN, s=12, edgecolors="none")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image-size distribution")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_class_imbalance(records: list[RawDetection], out_png: Path) -> None:
    """Horizontal bar of per-class counts, sorted, with imbalance ratio."""
    _setup()
    counts = _counts(records)
    if not counts:
        return
    items = sorted(counts.items(), key=lambda kv: kv[1])
    labels = [str(k) for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(items))))
    bars = ax.barh(labels, values, color=_RED, height=0.6)
    ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xlabel("Instance count")
    ax.set_title("Class imbalance")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


__all__ = [
    "plot_aspect_ratio",
    "plot_bbox_area_buckets",
    "plot_bbox_position_heatmap",
    "plot_bbox_size_distribution",
    "plot_class_distribution",
    "plot_class_imbalance",
    "plot_image_size_distribution",
    "plot_objects_per_image",
]
