"""Curve plotting — loss curves, metric curves, PR/ROC curves.

Uses matplotlib with the ``Agg`` backend (no display required). All plot
functions are best-effort: they log a warning and return without raising
if the input data is empty or malformed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_loss_curve(metrics_csv: Path, out_png: Path) -> None:
    """Plot train_loss vs epoch from a metrics.csv."""
    if not metrics_csv.exists():
        logger.warning(f"metrics.csv not found: {metrics_csv}")
        return
    rows = _read_csv(metrics_csv)
    if not rows or "train_loss" not in rows[0]:
        logger.warning("no train_loss column in metrics.csv")
        return
    epochs = [float(r.get("epoch", i)) for i, r in enumerate(rows)]
    losses = [float(r["train_loss"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, marker="o", color="steelblue", label="train_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_metric_curve(metrics_csv: Path, out_png: Path) -> None:
    """Plot val_map50 / val_map5095 vs epoch from a metrics.csv."""
    if not metrics_csv.exists():
        logger.warning(f"metrics.csv not found: {metrics_csv}")
        return
    rows = _read_csv(metrics_csv)
    if not rows or "val_map50" not in rows[0]:
        logger.warning("no val_map50 column in metrics.csv")
        return
    epochs = [float(r.get("epoch", i)) for i, r in enumerate(rows)]
    map50 = [float(r.get("val_map50", 0.0)) for r in rows]
    map5095 = [float(r.get("val_map5095", 0.0)) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, map50, marker="o", color="darkorange", label="mAP@50")
    ax.plot(epochs, map5095, marker="s", color="seagreen", label="mAP@50:95")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Validation mAP")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_pr_curve(
    precision: np.ndarray | list[float],
    recall: np.ndarray | list[float],
    out_png: Path,
) -> None:
    """Plot a Precision-Recall curve."""
    p = np.asarray(precision, dtype=np.float64)
    r = np.asarray(recall, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(p) > 0:
        order = np.argsort(r)
        ax.plot(r[order], p[order], color="steelblue", label="PR")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_roc_curve(
    fpr: np.ndarray | list[float],
    tpr: np.ndarray | list[float],
    out_png: Path,
) -> None:
    """Plot a ROC curve."""
    f = np.asarray(fpr, dtype=np.float64)
    t = np.asarray(tpr, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(f) > 0:
        order = np.argsort(f)
        ax.plot(f[order], t[order], color="darkorange", label="ROC")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


__all__ = [
    "plot_loss_curve",
    "plot_metric_curve",
    "plot_pr_curve",
    "plot_roc_curve",
]
