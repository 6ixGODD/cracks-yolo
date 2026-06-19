"""Confusion matrix for detection.

Returns a ``(num_classes+1) x (num_classes+1)`` matrix where:
- Rows = ground-truth class (last row = "background", i.e. unmatched GTs).
- Cols = predicted class (last col = "background", i.e. FP detections).
- ``cm[i, j]`` = count of GTs with class ``i`` matched (IoU ≥ ``iou_thr``)
  to a detection with class ``j``.

A detection below ``conf_thr`` is dropped (treated as not predicted).
A GT not matched to any surviving detection increments ``cm[i, num_classes]``
(background column). A detection not matched to any GT increments
``cm[num_classes, j]`` (background row).
"""

from __future__ import annotations

import numpy as np

from cracks_yolo.metrics.curves import _iou_matrix


def compute_confusion_matrix(
    detections: list[tuple[int, int, float, tuple[float, float, float, float]]],
    ground_truths: list[tuple[int, int, tuple[float, float, float, float]]],
    iou_thr: float = 0.5,
    num_classes: int = 1,
    conf_thr: float = 0.25,
) -> list[list[int]]:
    """Compute the confusion matrix.

    Args:
        detections: ``(image_id, class_id, score, bbox_xyxy)`` per detection.
        ground_truths: ``(image_id, class_id, bbox_xyxy)`` per GT.
        iou_thr: IoU threshold for matching.
        num_classes: Number of foreground classes (background is implicit).
        conf_thr: Confidence threshold; detections below this are dropped.

    Returns:
        ``list[list[int]]`` of shape ``(num_classes+1, num_classes+1)``.
    """
    n = num_classes + 1
    cm = np.zeros((n, n), dtype=np.int64)

    gts_by_img: dict[int, list[tuple[int, int, tuple[float, float, float, float], int]]] = {}
    for i, (img_id, cls, bbox) in enumerate(ground_truths):
        gts_by_img.setdefault(img_id, []).append((i, cls, bbox, 0))

    dets_by_img: dict[int, list[tuple[int, float, tuple[float, float, float, float]]]] = {}
    for img_id, cls, score, bbox in detections:
        if score < conf_thr:
            continue
        dets_by_img.setdefault(img_id, []).append((cls, score, bbox))

    for img_id, dets in dets_by_img.items():
        gts = gts_by_img.get(img_id, [])
        for det_cls, _, det_bbox in dets:
            best_iou = 0.0
            best_gt_idx = -1
            for j, (_gt_i, _gt_cls, gt_bbox, matched) in enumerate(gts):
                if matched:
                    continue
                iou = float(_iou_matrix([det_bbox], [gt_bbox])[0, 0])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            if best_iou >= iou_thr and best_gt_idx >= 0:
                gt_i, gt_cls, gt_bbox, _ = gts[best_gt_idx]
                gts[best_gt_idx] = (gt_i, gt_cls, gt_bbox, 1)
                cm[gt_cls, det_cls] += 1
            else:
                cm[num_classes, det_cls] += 1  # FP — background row

    for gts in gts_by_img.values():
        for _gt_i, gt_cls, _gt_bbox, matched in gts:
            if not matched:
                cm[gt_cls, num_classes] += 1  # FN — background col

    return cm.tolist()


__all__ = ["compute_confusion_matrix"]
