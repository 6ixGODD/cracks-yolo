"""Visualization: PR curves, ROC curves, confusion matrix from prediction JSONs.

Supports single-model and multi-model comparison plots.
Uses matplotlib with academic styling (dpi=300, muted palette).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

_ACADEMIC_STYLE: dict[str, Any] = {
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
    "lines.linewidth": 1.3,
    "lines.markersize": 0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "grid.color": "#cccccc",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Colorblind-safe, print-friendly palette (Wong 2011 subset)
_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#F0E442",  # yellow
    "#000000",  # black
    "#999999",  # grey
    "#882255",  # wine
]


def _setup_style():
    import matplotlib

    matplotlib.rcParams.update(_ACADEMIC_STYLE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_predictions(json_path: Path) -> list[dict]:
    import json

    return json.loads(json_path.read_text(encoding="utf-8"))


def _load_ground_truths(dataset: str, split: str) -> tuple[list[dict], dict[int, tuple[int, int]]]:
    """Load GT annotations + per-image sizes.  Returns (gts, img_sizes)."""
    from cracks_yolo.dataset.yolo import YOLOSource

    src = YOLOSource(dataset)
    records = src.load_split(split)
    from PIL import Image

    gts: list[dict] = []
    img_sizes: dict[int, tuple[int, int]] = {}
    for img_id, rec in enumerate(records, 1):
        with Image.open(rec.image_path) as im:
            w, h = im.size
        img_sizes[img_id] = (w, h)
        for box, label in zip(rec.boxes_norm, rec.labels, strict=False):
            x1, y1, x2, y2 = box  # norm xyxy
            gts.append({
                "image_id": img_id,
                "category_id": 1 if label == 0 else int(label),
                "bbox": [x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h],  # pixel xywh
            })
    return gts, img_sizes


def _normalize_bbox_xywh(bbox: list[float], size: int) -> list[float]:
    """Convert pixel xywh → normalized xywh by dividing by *size*."""
    return [v / size for v in bbox[:4]]


# ---------------------------------------------------------------------------
# PR / ROC computation
# ---------------------------------------------------------------------------


def _compute_pr_roc(
    preds: list[dict],
    gts: list[dict],
    img_sizes: dict[int, tuple[int, int]],
    model_input_size: int = 640,
    iou_thr: float = 0.5,
) -> dict[str, Any]:
    """Compute precision-recall curve points and ROC data.

    Both preds (pixel xywh) and gts (pixel xywh) are normalized to [0,1]
    before IoU computation.  Predictions are divided by *model_input_size*;
    ground truths are divided by the per-image dimensions from *img_sizes*.
    """
    from collections import defaultdict

    # Build GT index per image, normalized
    gt_per_img: dict[int, list[list[float]]] = defaultdict(list)
    for g in gts:
        w, h = img_sizes.get(g["image_id"], (model_input_size, model_input_size))
        b = g["bbox"]
        gt_per_img[g["image_id"]].append([
            b[0] / w,
            b[1] / h,
            b[2] / w,
            b[3] / h,
        ])

    # Build predictions per image, normalized
    pred_per_img: dict[int, list[tuple[list[float], float]]] = defaultdict(list)
    for p in preds:
        b = p["bbox"]
        nb = [v / model_input_size for v in b[:4]]
        pred_per_img[p["image_id"]].append((nb, p["score"]))

    all_scores: list[float] = []
    all_labels: list[int] = []

    for img_id, gts_list in gt_per_img.items():
        dts = sorted(pred_per_img.get(img_id, []), key=lambda x: -x[1])
        matched = [False] * len(gts_list)
        for bbox, score in dts:
            best_iou = 0.0
            best_idx = -1
            for i, gb in enumerate(gts_list):
                if matched[i]:
                    continue
                iou = _iou_xywh(bbox, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_thr and best_idx >= 0:
                matched[best_idx] = True
                all_scores.append(score)
                all_labels.append(1)
            else:
                all_scores.append(score)
                all_labels.append(0)

    if not all_scores:
        return {
            "precision": [],
            "recall": [],
            "fpr": [],
            "tpr": [],
            "auc_pr": 0.0,
            "auc_roc": 0.5,
            "f1_max": 0.0,
        }

    from sklearn.metrics import auc as _sk_auc
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    try:
        precision, recall, _ = precision_recall_curve(labels_arr, scores_arr)
        fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
        auc_pr = float(_sk_auc(recall, precision))
        auc_roc = float(_sk_auc(fpr, tpr))
    except Exception as exc:
        # sklearn may fail if pandas is broken; compute AUC manually
        precision, recall = _manual_pr_curve(labels_arr, scores_arr)
        fpr, tpr = _manual_roc_curve(labels_arr, scores_arr)
        precision_arr = np.array(precision, dtype=np.float64)
        recall_arr = np.array(recall, dtype=np.float64)
        fpr_arr = np.array(fpr, dtype=np.float64)
        tpr_arr = np.array(tpr, dtype=np.float64)
        auc_pr = float(np.trapezoid(precision_arr, recall_arr)) if len(precision_arr) > 1 else 0.0
        auc_roc = float(np.trapezoid(tpr_arr, fpr_arr)) if len(fpr_arr) > 1 else 0.5
        import warnings

        warnings.warn(f"sklearn unavailable, using manual AUC: {exc}", stacklevel=2)
        # Ensure numpy arrays for downstream processing
        precision = precision_arr  # type: ignore[assignment]
        recall = recall_arr  # type: ignore[assignment]
        fpr = fpr_arr  # type: ignore[assignment]
        tpr = tpr_arr  # type: ignore[assignment]

    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    f1_max = float(np.max(f1_scores))

    return {
        "precision": precision.tolist() if isinstance(precision, np.ndarray) else precision,
        "recall": recall.tolist() if isinstance(recall, np.ndarray) else recall,
        "fpr": fpr.tolist() if isinstance(fpr, np.ndarray) else fpr,
        "tpr": tpr.tolist() if isinstance(tpr, np.ndarray) else tpr,
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "f1_max": f1_max,
    }


def _manual_pr_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[list[float], list[float]]:
    """Compute precision-recall curve without sklearn."""
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    tp_cum = np.cumsum(labels_sorted)
    fp_cum = np.cumsum(1 - labels_sorted)
    n_pos = int(np.sum(labels))
    if n_pos == 0:
        return [1.0], [0.0]
    precision = tp_cum.astype(np.float64) / (tp_cum + fp_cum + 1e-12)
    recall = tp_cum.astype(np.float64) / n_pos
    # Add end points
    prec_list = [1.0, *precision.tolist()]
    rec_list = [0.0, *recall.tolist()]
    return prec_list, rec_list


def _manual_roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[list[float], list[float]]:
    """Compute ROC curve without sklearn."""
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    tp_cum = np.cumsum(labels_sorted)
    fp_cum = np.cumsum(1 - labels_sorted)
    n_pos = int(np.sum(labels))
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return [0.0, 1.0], [0.0, 1.0]
    if n_neg == 0:
        return [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]
    tpr = tp_cum.astype(np.float64) / n_pos
    fpr = fp_cum.astype(np.float64) / n_neg
    return [0.0, *fpr.tolist(), 1.0], [0.0, *tpr.tolist(), 1.0]


def _iou_xywh(b1: list[float], b2: list[float]) -> float:
    """IoU of two xywh boxes (x, y, w, h)."""
    x1, y1, w1, h1 = b1[:4]
    x2, y2, w2, h2 = b2[:4]
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def _compute_confusion(
    preds: list[dict],
    gts: list[dict],
    score_thr: float = 0.25,
    total_imgs: int = 110,
) -> np.ndarray:
    """Return image-level 2x2 confusion matrix [[TN, FP], [FN, TP]].

    Each image is classified as positive (≥1 crack) or negative (no crack).
    A detection ≥ *score_thr* counts as a positive prediction.
    """

    # GT: which images have cracks
    gt_positive_imgs: set[int] = set()
    for g in gts:
        gt_positive_imgs.add(g["image_id"])

    # Predictions: which images have detections above threshold
    dt_positive_imgs: set[int] = set()
    for p in preds:
        if p.get("score", 0) >= score_thr:
            dt_positive_imgs.add(int(p["image_id"]))

    tp_img = len(gt_positive_imgs & dt_positive_imgs)
    fp_img = len(dt_positive_imgs - gt_positive_imgs)
    fn_img = len(gt_positive_imgs - dt_positive_imgs)
    tn_img = total_imgs - tp_img - fp_img - fn_img

    return np.array([[tn_img, fp_img], [fn_img, tp_img]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_pr_curve(
    pr_data: dict[str, dict],
    output_path: Path,
    title: str = "Precision-Recall Curve",
) -> Path:
    """Plot PR curve(s). ``pr_data`` maps label → {precision, recall, auc_pr}."""
    import matplotlib.pyplot as plt

    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, (label, data) in enumerate(pr_data.items()):
        if not data.get("precision") or not data.get("recall"):
            continue
        color = _COLORS[i % len(_COLORS)]
        auc_val = data.get("auc_pr", 0)
        ax.plot(
            data["recall"],
            data["precision"],
            color=color,
            label=f"{label} (AUC={auc_val:.3f})",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower left", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_roc_curve(
    roc_data: dict[str, dict],
    output_path: Path,
    title: str = "ROC Curve",
) -> Path:
    """Plot ROC curve(s). ``roc_data`` maps label → {fpr, tpr, auc_roc}."""
    import matplotlib.pyplot as plt

    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.3)

    for i, (label, data) in enumerate(roc_data.items()):
        if not data.get("fpr") or not data.get("tpr"):
            continue
        color = _COLORS[i % len(_COLORS)]
        auc_val = data.get("auc_roc", 0.5)
        ax.plot(
            data["fpr"],
            data["tpr"],
            color=color,
            label=f"{label} (AUC={auc_val:.3f})",
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
    labels: tuple[str, str] = ("Negative", "Positive"),
) -> Path:
    """Plot an image-level confusion matrix: [[TN, FP], [FN, TP]]."""
    import matplotlib.pyplot as plt

    _setup_style()
    # Confusion matrix keeps all 4 spines for the grid look
    with plt.rc_context({"axes.spines.top": True, "axes.spines.right": True}):
        fig, ax = plt.subplots(figsize=(4.2, 3.8))

        # Normalize by row (recall per true class)
        cm_norm = cm.astype(np.float64).copy()
        for r in range(cm.shape[0]):
            row_sum = cm[r].sum()
            if row_sum > 0:
                cm_norm[r] = cm[r] / row_sum

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

        tick_labels = [f"Pred {labels[0]}", f"Pred {labels[1]}"]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(tick_labels, fontsize=8.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"True {labels[0]}", f"True {labels[1]}"], fontsize=8.5)
        ax.tick_params(length=0)

        # Annotate: count + row-fraction
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                val = int(cm[r, c])
                pct = cm_norm[r, c]
                ax.text(
                    c,
                    r,
                    f"{val}\n({pct:.1%})",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if pct > 0.5 else "#333333",
                )

        ax.set_title(title, fontsize=10)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Row fraction", fontsize=8.5)
        cbar.ax.tick_params(labelsize=7.5)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    return output_path


def plot_metric_bars(
    data: dict[str, dict[str, float]],
    metric: str,
    output_path: Path,
    title: str | None = None,
) -> Path:
    """Horizontal bar chart comparing one metric across models."""
    import matplotlib.pyplot as plt

    _setup_style()
    models = list(data.keys())
    values = [data[m].get(metric, 0) for m in models]
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(8, max(3, len(models) * 0.35)))
    bars = ax.barh(models, values, color=colors, height=0.6)
    ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)
    ax.set_xlabel(metric.upper())
    ax.set_title(title or metric.upper())
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
