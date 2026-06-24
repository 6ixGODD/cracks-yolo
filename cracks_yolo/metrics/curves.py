"""PR / ROC curve computation.

A "detection" is a tuple ``(image_id, class_id, score, bbox_xyxy)``.
A "ground truth" is ``(image_id, class_id, bbox_xyxy)``.

Matching: for each detection, find the highest-IoU ground truth in the
same image with the same class. If IoU ≥ ``iou_thr`` and that GT isn't
already matched, it's a TP; otherwise FP. Unmatched GTs are FNs.

Curve points: sort detections by score (desc), accumulate TPs/FPs, and
emit ``(precision, recall, threshold)`` at each step.
"""

from __future__ import annotations

import numpy as np


def _iou_matrix(
    boxes_a: list[tuple[float, float, float, float]],
    boxes_b: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Pairwise IoU between two lists of xyxy boxes."""
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(min=0) * (a[:, 3] - a[:, 1]).clip(min=0)
    area_b = (b[:, 2] - b[:, 0]).clip(min=0) * (b[:, 3] - b[:, 1]).clip(min=0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, np.zeros_like(union))  # type: ignore[no-any-return]


def _match_detections(
    detections: list[tuple[int, int, float, tuple[float, float, float, float]]],
    ground_truths: list[tuple[int, int, tuple[float, float, float, float]]],
    iou_thr: float = 0.5,
) -> list[tuple[float, int]]:
    """Match each detection to its best-IoU GT; return list of (score, is_tp).

    ``is_tp`` is 1 for true positive, 0 for false positive.
    """
    gts_by_img: dict[int, list[tuple[int, int, tuple[float, float, float, float], int]]] = {}
    for i, (img_id, cls, bbox) in enumerate(ground_truths):
        gts_by_img.setdefault(img_id, []).append((i, cls, bbox, 0))

    det_sorted = sorted(detections, key=lambda d: -d[2])
    out: list[tuple[float, int]] = []
    for img_id, cls, score, bbox in det_sorted:
        gts = gts_by_img.get(img_id, [])
        best_iou = 0.0
        best_gt_idx = -1
        for j, (_gt_i, gt_cls, gt_bbox, matched) in enumerate(gts):
            if gt_cls != cls or matched:
                continue
            iou = float(_iou_matrix([bbox], [gt_bbox])[0, 0])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= iou_thr and best_gt_idx >= 0:
            gi, gc, gb, _ = gts[best_gt_idx]
            gts[best_gt_idx] = (gi, gc, gb, 1)
            out.append((score, 1))
        else:
            out.append((score, 0))
    return out


def compute_pr_curve(
    detections: list[tuple[int, int, float, tuple[float, float, float, float]]],
    ground_truths: list[tuple[int, int, tuple[float, float, float, float]]],
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the global PR curve (all classes pooled).

    Returns:
        (precision, recall, thresholds) — arrays of equal length.
    """
    matched = _match_detections(detections, ground_truths, iou_thr=iou_thr)
    if not matched:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    scores = np.asarray([m[0] for m in matched], dtype=np.float32)
    is_tp = np.asarray([m[1] for m in matched], dtype=np.int32)
    order = np.argsort(-scores)
    scores = scores[order]
    is_tp = is_tp[order]
    tp_cum = np.cumsum(is_tp).astype(np.float32)
    fp_cum = np.cumsum(1 - is_tp).astype(np.float32)
    n_gt = max(len(ground_truths), 1)
    precision = tp_cum / (tp_cum + fp_cum + 1e-12)
    recall = tp_cum / n_gt
    return precision, recall, scores


def compute_roc_curve(
    detections: list[tuple[int, int, float, tuple[float, float, float, float]]],
    ground_truths: list[tuple[int, int, tuple[float, float, float, float]]],
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the global ROC curve (all classes pooled).

    Returns:
        (fpr, tpr, thresholds) — arrays of equal length.
    """
    matched = _match_detections(detections, ground_truths, iou_thr=iou_thr)
    if not matched:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    scores = np.asarray([m[0] for m in matched], dtype=np.float32)
    is_tp = np.asarray([m[1] for m in matched], dtype=np.int32)
    order = np.argsort(-scores)
    scores = scores[order]
    is_tp = is_tp[order]
    # TPR = TP / (TP + FN) = TP / total_GT (well-defined in detection).
    # FPR = FP / (FP + TN). TN is not well-defined in object detection
    # (there are infinitely many possible non-object locations). We
    # normalize by the total number of FPs at the lowest confidence, which
    # makes the curve span [0, 1] for visualization and yields a meaningful
    # AUC-ROC as a ranking-quality metric. IMPORTANT: this FPR should NOT
    # be used to derive specificity (1-FPR). Use the direct matched-
    # detection counts in calculator.py instead.
    tp_cum = np.cumsum(is_tp).astype(np.float32)
    fp_cum = np.cumsum(1 - is_tp).astype(np.float32)
    total_gt = max(len(ground_truths), 1)
    tpr = tp_cum / total_gt
    fpr = fp_cum / max(float(fp_cum[-1]), 1.0)
    return fpr, tpr, scores


def compute_auc(precision: np.ndarray, recall: np.ndarray) -> float:
    """Area under the PR curve via trapezoidal integration."""
    if len(precision) < 2 or len(recall) < 2:
        return 0.0
    # Recall is non-decreasing; integrate precision(recall) via trapezoid.
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    return float(np.trapezoid(p, r))


def compute_auc_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the ROC curve via trapezoidal integration."""
    if len(fpr) < 2 or len(tpr) < 2:
        return 0.0
    order = np.argsort(fpr)
    f = fpr[order]
    t = tpr[order]
    return float(np.trapezoid(t, f))


__all__: list[str] = [
    "compute_auc",
    "compute_auc_roc",
    "compute_pr_curve",
    "compute_roc_curve",
]
