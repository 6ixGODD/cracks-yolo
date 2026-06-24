"""COCOMetricsCalculator — real metrics implementation.

Wraps pycocotools to compute AP@50, AP@75, AP@50:95, AR@1/10/100 +
AR@small/medium/large. Also computes precision/recall/F1 at the
F1-optimal confidence threshold, AUC-PR, AUC-ROC, sensitivity/specificity,
PPV/NPV, and the confusion matrix (via sklearn).

The calculator is fed incrementally via :meth:`update` (per-batch or
per-image) and produces a full :class:`MetricReport` via :meth:`run`.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol
from typing import runtime_checkable

import numpy as np

from cracks_yolo.metrics.confusion import compute_confusion_matrix
from cracks_yolo.metrics.curves import compute_auc
from cracks_yolo.metrics.curves import compute_auc_roc
from cracks_yolo.metrics.curves import compute_pr_curve
from cracks_yolo.metrics.curves import compute_roc_curve
from cracks_yolo.metrics.schemas import MetricReport
from cracks_yolo.metrics.schemas import PerImageDetection


@runtime_checkable
class MetricsCalculator(Protocol):
    """Compute detection metrics from a list of per-image detections.

    Train-side usage (light metrics): the calculator is fed incrementally
    per-batch and emits a small summary at epoch end. Test-side usage (full
    metrics): all per-image detections are collected, then ``run`` produces
    the full :class:`MetricReport` (mAP, AR, per-class AP, performance,
    statistical tests).
    """

    def update(self, batch: list[PerImageDetection]) -> None:
        """Accumulate one batch of per-image detections."""

    def run(self) -> MetricReport:
        """Compute and return the full metric report."""


@dataclass
class _Accumulator:
    """Internal accumulation state — flat lists across all images."""

    det_image_ids: list[int] = field(default_factory=list)
    det_class_ids: list[int] = field(default_factory=list)
    det_scores: list[float] = field(default_factory=list)
    det_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    gt_image_ids: list[int] = field(default_factory=list)
    gt_class_ids: list[int] = field(default_factory=list)
    gt_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    image_ids: set[int] = field(default_factory=set)

    def add_image(self, img: PerImageDetection) -> None:
        self.image_ids.add(img["image_id"])
        for d in img["detections"]:
            self.det_image_ids.append(d["image_id"])
            self.det_class_ids.append(d["class_id"])
            self.det_scores.append(d["score"])
            self.det_boxes.append(d["bbox_xyxy"])
        for g in img["ground_truths"]:
            self.gt_image_ids.append(g["image_id"])
            self.gt_class_ids.append(g["class_id"])
            self.gt_boxes.append(g["bbox_xyxy"])


def _to_coco_format(acc: _Accumulator, num_classes: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build pycocotools-compatible ``COCO`` (GT) + ``COCOres`` (results)."""
    from pycocotools.coco import COCO

    images = [{"id": int(i), "width": 1, "height": 1} for i in sorted(acc.image_ids)]
    annotations = []
    for i, (img_id, cls, bbox) in enumerate(
        zip(acc.gt_image_ids, acc.gt_class_ids, acc.gt_boxes, strict=True)
    ):
        x1, y1, x2, y2 = bbox
        annotations.append({
            "id": i + 1,
            "image_id": int(img_id),
            "category_id": int(cls) + 1,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "area": float((x2 - x1) * (y2 - y1)),
            "iscrowd": 0,
        })
    categories = [
        {"id": c + 1, "name": f"class_{c}", "supercategory": "obj"} for c in range(num_classes)
    ]
    gt_coco = COCO()
    gt_coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    gt_coco.createIndex()

    results = []
    for _i, (img_id, cls, score, bbox) in enumerate(
        zip(
            acc.det_image_ids,
            acc.det_class_ids,
            acc.det_scores,
            acc.det_boxes,
            strict=True,
        )
    ):
        x1, y1, x2, y2 = bbox
        results.append({
            "image_id": int(img_id),
            "category_id": int(cls) + 1,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        })
    if results:
        dt_coco = gt_coco.loadRes(results)
    else:
        # Empty results: loadRes requires at least one detection. Build an
        # empty COCO result set so COCOeval can iterate without crashing.
        from pycocotools.coco import COCO

        dt_coco = COCO()
        dt_coco.dataset["images"] = gt_coco.dataset["images"]
        dt_coco.dataset["categories"] = gt_coco.dataset["categories"]
        dt_coco.dataset["annotations"] = []
        dt_coco.createIndex()
    return gt_coco, dt_coco


def _run_coco_eval(
    gt_coco: Any,
    dt_coco: Any,
    num_classes: int,  # noqa: ARG001 — kept for API symmetry with _to_coco_format
    iou_thr: float | None = None,
) -> dict[str, float]:
    """Run COCOeval and extract the scalar metrics we care about."""
    from pycocotools.cocoeval import COCOeval

    ev = COCOeval(gt_coco, dt_coco, "bbox")
    if iou_thr is not None:
        ev.params.iouThrs = np.array([iou_thr])
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats  # length-12 array
    # COCOeval.stats layout:
    # 0: AP @ IoU=0.50:0.95  area=all  maxDets=100
    # 1: AP @ IoU=0.50        area=all  maxDets=100
    # 2: AP @ IoU=0.75        area=all  maxDets=100
    # 3: AP @ IoU=0.50:0.95   area=small  maxDets=100
    # 4: AP @ IoU=0.50:0.95   area=medium maxDets=100
    # 5: AP @ IoU=0.50:0.95   area=large  maxDets=100
    # 6: AR @ IoU=0.50:0.95   area=all  maxDets=1
    # 7: AR @ IoU=0.50:0.95   area=all  maxDets=10
    # 8: AR @ IoU=0.50:0.95   area=all  maxDets=100
    # 9: AR @ IoU=0.50:0.95   area=small  maxDets=100
    # 10: AR @ IoU=0.50:0.95  area=medium maxDets=100
    # 11: AR @ IoU=0.50:0.95  area=large  maxDets=100
    return {
        "map5095": float(stats[0]),
        "ap50": float(stats[1]),
        "ap75": float(stats[2]),
        "ap_small": float(stats[3]),
        "ap_medium": float(stats[4]),
        "ap_large": float(stats[5]),
        "ar1": float(stats[6]),
        "ar10": float(stats[7]),
        "ar100": float(stats[8]),
        "ar_small": float(stats[9]),
        "ar_medium": float(stats[10]),
        "ar_large": float(stats[11]),
    }


def _compute_pr_curve_from_accumulator(
    acc: _Accumulator,
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a global PR curve (all classes pooled) via IoU matching."""
    det_arr = list(
        zip(acc.det_image_ids, acc.det_class_ids, acc.det_scores, acc.det_boxes, strict=True)
    )
    gt_arr = list(zip(acc.gt_image_ids, acc.gt_class_ids, acc.gt_boxes, strict=True))
    return compute_pr_curve(det_arr, gt_arr, iou_thr=iou_thr)


def _compute_roc_curve_from_accumulator(
    acc: _Accumulator,
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    det_arr = list(
        zip(acc.det_image_ids, acc.det_class_ids, acc.det_scores, acc.det_boxes, strict=True)
    )
    gt_arr = list(zip(acc.gt_image_ids, acc.gt_class_ids, acc.gt_boxes, strict=True))
    return compute_roc_curve(det_arr, gt_arr, iou_thr=iou_thr)


def _compute_classification_metrics_at_threshold(
    acc: _Accumulator,
    iou_thr: float,
    conf_thr: float,
    fallback_recall: float,
) -> tuple[float, float, float]:
    """Compute sensitivity, specificity, NPV at a given confidence threshold.

    Uses per-detection IoU matching (same logic as the PR/ROC curves).
    Sensitivity = TP/(TP+FN), Specificity = TN/(TN+FP), NPV = TN/(TN+FN).

    In object detection, TN is not naturally defined (there are infinitely
    many possible non-object locations). We define TN pragmatically as the
    number of detections *below* the confidence threshold that would have
    been false positives if they were above it — i.e., correctly suppressed
    spurious detections.

    Returns:
        (sensitivity, specificity, npv) tuple.
    """
    from cracks_yolo.metrics.curves import _match_detections

    if not acc.image_ids:
        return fallback_recall, 0.0, 0.0

    detections = list(
        zip(acc.det_image_ids, acc.det_class_ids, acc.det_scores, acc.det_boxes, strict=True)
    )
    ground_truths = list(zip(acc.gt_image_ids, acc.gt_class_ids, acc.gt_boxes, strict=True))
    matched = _match_detections(detections, ground_truths, iou_thr=iou_thr)
    if not matched:
        return fallback_recall, 0.0, 0.0

    # Count TP/FP at the chosen threshold.
    tp = sum(1 for s, is_tp in matched if s >= conf_thr and is_tp == 1)
    fp = sum(1 for s, is_tp in matched if s >= conf_thr and is_tp == 0)
    # TN: detections below threshold that would have been FP if kept.
    tn = sum(1 for s, is_tp in matched if s < conf_thr and is_tp == 0)
    # Total FN: all GTs not detected above threshold.
    total_fn = max(len(ground_truths) - tp, 0)

    sensitivity = tp / max(tp + total_fn, 1)
    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + total_fn, 1)

    return sensitivity, specificity, npv


class COCOMetricsCalculator(MetricsCalculator):
    """Real metrics calculator — pycocotools backend."""

    def __init__(
        self,
        num_classes: int = 1,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
    ) -> None:
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self._acc = _Accumulator()

    def update(self, batch: list[PerImageDetection]) -> None:
        """Accumulate one batch of per-image detections + ground truths."""
        for img in batch:
            self._acc.add_image(img)

    def run(self) -> MetricReport:
        """Compute the full MetricReport from accumulated detections."""
        if not self._acc.image_ids:
            return MetricReport(map50=0.0, map5095=0.0)

        gt_coco, dt_coco = _to_coco_format(self._acc, self.num_classes)
        coco_stats = _run_coco_eval(gt_coco, dt_coco, self.num_classes)

        # Precision/Recall/F1 at IoU=0.5 via PR curve.
        precision, recall, thresholds = _compute_pr_curve_from_accumulator(
            self._acc, iou_thr=self.iou_threshold
        )
        if len(thresholds) > 0:
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            best_idx = int(np.argmax(f1))
            best_p = float(precision[best_idx])
            best_r = float(recall[best_idx])
            best_f1 = float(f1[best_idx])
            best_thr = float(thresholds[best_idx])
            auc_pr = float(compute_auc(precision, recall))
        else:
            best_p = best_r = best_f1 = 0.0
            best_thr = 0.0
            auc_pr = 0.0

        # ROC curve + AUC.
        fpr, tpr, _roc_thr = _compute_roc_curve_from_accumulator(
            self._acc,
            iou_thr=self.iou_threshold,
        )
        auc_roc = float(compute_auc_roc(fpr, tpr)) if len(fpr) > 0 else 0.0

        # Sensitivity, specificity, PPV, NPV — computed directly from
        # matched detections at the F1-optimal confidence threshold.
        # This avoids the ROC FPR normalization issue and the confusion
        # matrix TN ambiguity (TN is not well-defined in detection).
        ppv = best_p
        sensitivity, specificity, npv = _compute_classification_metrics_at_threshold(
            self._acc,
            self.iou_threshold,
            best_thr,
            best_r,
        )

        # Confusion matrix for inspection (not used for metric formulas).
        cm = compute_confusion_matrix(
            detections=list(
                zip(
                    self._acc.det_image_ids,
                    self._acc.det_class_ids,
                    self._acc.det_scores,
                    self._acc.det_boxes,
                    strict=True,
                )
            ),
            ground_truths=list(
                zip(
                    self._acc.gt_image_ids,
                    self._acc.gt_class_ids,
                    self._acc.gt_boxes,
                    strict=True,
                )
            ),
            iou_thr=self.iou_threshold,
            num_classes=self.num_classes,
            conf_thr=best_thr,
        )
        cm_list = [list(row) for row in cm]

        return MetricReport(
            map50=coco_stats["ap50"],
            map5095=coco_stats["map5095"],
            ap50=coco_stats["ap50"],
            ap75=coco_stats["ap75"],
            per_class_ap={},
            precision=best_p,
            recall=best_r,
            f1=best_f1,
            ar1=coco_stats["ar1"],
            ar10=coco_stats["ar10"],
            ar100=coco_stats["ar100"],
            ar300=coco_stats["ar100"],  # pycocotools only does 1/10/100; alias for compat
            ar1000=coco_stats["ar100"],
            ar_small=coco_stats["ar_small"],
            ar_medium=coco_stats["ar_medium"],
            ar_large=coco_stats["ar_large"],
            auc_pr=auc_pr,
            auc_roc=auc_roc,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            confusion_matrix=cm_list,
            iou_threshold=self.iou_threshold,
            conf_threshold=best_thr,
        )
