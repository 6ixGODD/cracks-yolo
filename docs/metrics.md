# Metrics

[English](metrics.md) | [中文](metrics.zh-CN.md)

`cracks_yolo.metrics` provides detection evaluation via `COCOMetricsCalculator` with a
pycocotools backend, plus efficiency profiling and paired statistical tests.

## 1. Architecture

    PerImageDetection[]  -->  COCOMetricsCalculator.update()  -->  .run()  -->  MetricReport

`COCOMetricsCalculator` implements the `MetricsCalculator` Protocol (`update` / `run`). The
pipeline collects per-image predictions and ground truths as `PerImageDetection` records,
accumulates them via `update()`, and calls `run()` to produce a `MetricReport`.

### 1.1 Per-image detection format

`DetectionMetric` (`TypedDict`):

| Field | Type | Description |
| --- | --- | --- |
| `image_id` | `int` | Dataset image index. |
| `class_id` | `int` | Predicted or ground-truth class. |
| `score` | `float` | Confidence in [0,1]; 1.0 for ground truth. |
| `bbox_xyxy` | `tuple[float,float,float,float]` | Box in `(x1, y1, x2, y2)` pixels. |

`PerImageDetection` (`TypedDict`): `image_id`, `detections: list[DetectionMetric]`,
`ground_truths: list[DetectionMetric]`.

## 2. pycocotools backend

`COCOMetricsCalculator.run()` converts accumulated records to COCO JSON via `_to_coco_format`,
instantiates `pycocotools.coco.COCO` for ground truth and `COCO.loadRes` for detections, then
runs `COCOeval` with `"bbox"` iouType. The 12-element `COCOeval.stats` array maps to:

| Index | Field | Description |
| --- | --- | --- |
| 0 | `map5095` | AP at IoU=0.50:0.95, all areas, maxDets=100 |
| 1 | `ap50` | AP at IoU=0.50 |
| 2 | `ap75` | AP at IoU=0.75 |
| 3–5 | `ap_small/medium/large` | Area-stratified AP |
| 6–8 | `ar1/ar10/ar100` | AR at maxDets 1/10/100 |
| 9–11 | `ar_small/medium/large` | Area-stratified AR |

`ar300` and `ar1000` alias `ar100` (pycocotools caps at 100). Per-class AP is not populated
by the stock `COCOeval.summarize` call. When the accumulator is empty, `run()` returns a
`MetricReport` with `map50=0.0, map5095=0.0` and all other fields at their defaults.

## 3. MetricReport

`MetricReport` (`@dataclass`) is the aggregate accuracy payload returned by `run()`:

| Field | Default | Source |
| --- | --- | --- |
| `map50` | (required) | `COCOeval.stats[1]` |
| `map5095` | (required) | `COCOeval.stats[0]` |
| `ap50` | 0.0 | alias for `map50` |
| `ap75` | 0.0 | `COCOeval.stats[2]` |
| `per_class_ap` | `{}` | not populated |
| `precision` | 0.0 | from PR curve at best-F1 threshold |
| `recall` | 0.0 | from PR curve at best-F1 threshold |
| `f1` | 0.0 | max of 2PR/(P+R+eps) over PR thresholds |
| `ar1/ar10/ar100` | 0.0 | `COCOeval.stats[6–8]` |
| `ar300/ar1000` | 0.0 | aliased to `ar100` |
| `ar_small/medium/large` | 0.0 | `COCOeval.stats[9–11]` |
| `auc_pr` | 0.0 | trapezoidal integral of PR curve |
| `auc_roc` | 0.0 | trapezoidal integral of ROC curve |
| `sensitivity` | 0.0 | TP/(TP+FN) at operating point |
| `specificity` | 0.0 | TN/(TN+FP) at operating point |
| `ppv` | 0.0 | positive predictive value = precision |
| `npv` | 0.0 | TN/(TN+FN) at operating point |
| `confusion_matrix` | `[]` | (C+1)x(C+1) including background |
| `iou_threshold` | 0.5 | configured IoU threshold |
| `conf_threshold` | 0.25 | F1-optimal confidence threshold |
| `performance` | `[]` | `list[PerformanceMetric]` (FPS, params, etc.) |
| `statistical_tests` | `[]` | `list[StatisticalTest]` for model comparison |

The F1-optimal operating point is `t* = argmax_t F1(t)` from the PR curve. Precision, recall,
sensitivity, specificity, PPV, NPV, and the confusion matrix are all reported at this threshold.

## 4. PR and ROC curves

### 4.1 Matching rule

A detection `(image_id, class_id, score, bbox)` is matched to the highest-IoU ground truth of
the same class in the same image. If IoU >= threshold and the GT is not yet matched, it is a
TP; otherwise FP. Unmatched GTs are FN. Classes are pooled for global curves.

### 4.2 PR curve

`compute_pr_curve(detections, ground_truths, iou_thr)` returns `(precision, recall, thresholds)`.
Detections sorted by descending score, with cumulative TP and FP:

    P(t) = TP_cum(t) / (TP_cum(t) + FP_cum(t) + eps)
    R(t) = TP_cum(t) / N_gt

AUC-PR is computed via `compute_auc(precision, recall)` as the trapezoidal integral over
points sorted by recall. This is the recommended threshold-independent summary for imbalanced
single-class detection.

### 4.3 ROC curve

`compute_roc_curve(detections, ground_truths, iou_thr)` returns `(fpr, tpr, thresholds)`.
TPR = `TP_cum / N_gt`. FPR = `FP_cum / max(FP_cum)`, normalised by total false positives
at the lowest confidence. This makes the curve span [0,1] for ranking-quality assessment but
the resulting FPR values are unsuitable for deriving specificity (1-FPR).

AUC-ROC is computed via `compute_auc_roc(fpr, tpr)` as the trapezoidal integral over
FPR-sorted points.

Sensitivity, specificity, and NPV are instead computed directly from matched detections in
`_compute_classification_metrics_at_threshold`, where TN is defined as detections below the
threshold that would have been FP if retained -- avoiding both the ROC normalisation issue
and the ill-posed TN definition in object detection.
