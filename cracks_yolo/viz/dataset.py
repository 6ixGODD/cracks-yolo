"""Dataset distribution plots — academic styling, sans-serif, dpi=300.

Generates: bbox-size histogram (log-spaced bins), bbox-area buckets
(COCO small/medium/large), aspect-ratio histogram, objects-per-image
histogram, bbox-center heatmap.
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
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "grid.color": "#cccccc",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
_BLUE = "#0072B2"
_ORANGE = "#E69F00"
_GREEN = "#009E73"
_RED = "#D55E00"
_GREY = "#999999"


def _setup() -> None:
    matplotlib.rcParams.update(_STYLE)


def _areas_norm(records: list[RawDetection]) -> list[float]:
    return [float((b[2] - b[0]) * (b[3] - b[1])) for rec in records for b in rec.boxes_norm]


def plot_bbox_size_distribution(records: list[RawDetection], out_png: Path) -> None:
    """Histogram of bbox areas with log-spaced bins (equal width on log axis)."""
    _setup()
    areas = _areas_norm(records)
    if not areas:
        return
    areas = np.asarray(areas)
    lo = max(areas.min(), 1e-5)
    hi = areas.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 30)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(areas, bins=bins, color=_BLUE, edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Bbox area (fraction of image area)")
    ax.set_ylabel("Count")
    ax.set_title("Bounding-box size distribution")
    # format y-axis as integers
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_bbox_area_buckets(records: list[RawDetection], out_png: Path) -> None:
    """Bar chart of COCO-style small / medium / large bbox counts.

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
    total = n_small + n_med + n_large
    if total == 0:
        return
    labels = ["Small", "Medium", "Large"]
    sizes = [n_small, n_med, n_large]
    colors = [_RED, _ORANGE, _GREEN]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(labels, sizes, color=colors, width=0.55, edgecolor="white", linewidth=0.4)
    for b, n in zip(bars, sizes, strict=True):
        pct = 100 * n / total
        ax.text(
            b.get_x() + b.get_width() / 2,
            n,
            f"{n}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_ylabel("Count")
    ax.set_title("Bbox area buckets (COCO scale)")
    ax.set_ylim(0, max(sizes) * 1.18)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
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
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(ars, bins=40, color=_GREEN, edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.set_xlabel("Aspect ratio $w/h$")
    ax.set_ylabel("Count")
    ax.set_title("Bounding-box aspect-ratio distribution")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_objects_per_image(records: list[RawDetection], out_png: Path) -> None:
    """Histogram of object counts per image."""
    _setup()
    counts = [len(rec.boxes_norm) for rec in records]
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(
        counts,
        bins=range(0, max(counts) + 2),
        color=_BLUE,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
        align="left",
    )
    ax.set_xlabel("Objects per image")
    ax.set_ylabel("Image count")
    ax.set_title("Object-count distribution per image")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
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
    with plt.rc_context({"axes.spines.top": True, "axes.spines.right": True}):
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        im = ax.imshow(heat, cmap="magma", origin="lower")
        ax.set_xlabel("$x$ grid")
        ax.set_ylabel("$y$ grid")
        ax.set_title("Bbox-center spatial density")
        ax.tick_params(length=0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Count", fontsize=8.5)
        cbar.ax.tick_params(labelsize=7.5)
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)


__all__ = [
    "plot_aspect_ratio",
    "plot_bbox_area_buckets",
    "plot_bbox_position_heatmap",
    "plot_bbox_size_distribution",
    "plot_objects_per_image",
]
