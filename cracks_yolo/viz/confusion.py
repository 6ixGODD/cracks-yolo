"""Confusion matrix plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    class_names: list[str],
    out_png: Path,
    normalize: bool = True,
) -> None:
    """Plot a confusion matrix with class labels + background row/col."""
    cm = np.asarray(matrix, dtype=np.float64)
    if normalize and cm.sum() > 0:
        cm = cm / cm.sum(axis=1, keepdims=True)
    labels = [*class_names, "background"]
    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels) * 0.8)))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else cm.max())
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("Confusion matrix" + (" (normalized)" if normalize else ""))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            color = "white" if value > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


__all__ = ["plot_confusion_matrix"]
