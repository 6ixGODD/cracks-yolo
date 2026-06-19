"""Dataset distribution plots — class / bbox size / bbox position / image size."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cracks_yolo.dataset.types import RawDetection


def plot_class_distribution(records: list[RawDetection], out_png: Path) -> None:
    """Bar plot of class frequencies across all images."""
    counts: dict[int, int] = {}
    for rec in records:
        for lab in rec.labels:
            counts[lab] = counts.get(lab, 0) + 1
    if not counts:
        return
    labels = sorted(counts.keys())
    values = [counts[cls] for cls in labels]
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(labels)), 4))
    ax.bar([str(cls) for cls in labels], values, color="steelblue")
    ax.set_xlabel("class id")
    ax.set_ylabel("count")
    ax.set_title("Class distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_bbox_size_distribution(records: list[RawDetection], out_png: Path) -> None:
    """Histogram of bbox areas (normalized to image area) on a log scale."""
    areas: list[float] = []
    for rec in records:
        for box in rec.boxes_norm:
            x1, y1, x2, y2 = box
            areas.append(float((x2 - x1) * (y2 - y1)))
    if not areas:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(areas, bins=50, color="darkorange", edgecolor="black", alpha=0.7)
    ax.set_xlabel("bbox area (fraction of image)")
    ax.set_ylabel("count")
    ax.set_title("Bounding box size distribution")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_bbox_position_heatmap(records: list[RawDetection], out_png: Path, grid: int = 32) -> None:
    """Heatmap of bbox centers across a ``grid`` x ``grid`` spatial grid."""
    heat = np.zeros((grid, grid), dtype=np.float64)
    for rec in records:
        for box in rec.boxes_norm:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            gx = min(grid - 1, max(0, int(cx * grid)))
            gy = min(grid - 1, max(0, int(cy * grid)))
            heat[gy, gx] += 1
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heat, cmap="hot", origin="lower")
    ax.set_xlabel("x grid")
    ax.set_ylabel("y grid")
    ax.set_title("Bounding box center heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_image_size_distribution(records: list[RawDetection], out_png: Path) -> None:
    """Scatter plot of image widths vs heights."""
    if not records:
        return
    widths = [r.width for r in records]
    heights = [r.height for r in records]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(widths, heights, alpha=0.5, color="seagreen")
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title("Image size distribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


__all__ = [
    "plot_bbox_position_heatmap",
    "plot_bbox_size_distribution",
    "plot_class_distribution",
    "plot_image_size_distribution",
]
