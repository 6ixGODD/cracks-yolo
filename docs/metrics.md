# Metrics

[English](metrics.md) | [中文](metrics.zh-CN.md)

`cracks_yolo.metrics` provides metric computation, statistical comparison, and evaluation curves for tongue surface crack detection models. The concrete implementation is `COCOMetricsCalculator`, backed by pycocotools (mAP/AR), sklearn (PR/ROC/confusion matrix), scipy (paired t-test, Wilcoxon), and statsmodels (bootstrap CI).

## Detection-level records

### DetectionMetric (TypedDict)

A single detection after NMS, before metric aggregation.

| Field | Type | Description |
| --- | --- | --- |
| `image_id` | `int` | Image identifier (matches dataset index). |
| `class_id` | `int` | Predicted class id. |
| `score` | `float` | Confidence score in `[0, 1]`. |
| `bbox_xyxy` | `tuple[float, float, float, float]` | `(x1, y1, x2, y2)` in pixels. |

### PerImageDetection (TypedDict)

All detections + ground truths for one image.

| Field | Type |
| --- | --- |
| `image_id` | `int` |
| `detections` | `list[DetectionMetric]` |
| `ground_truths` | `list[DetectionMetric]` |

## COCOMetricsCalculator

`COCOMetricsCalculator` implements the `MetricsCalculator` Protocol:

```python
@runtime_checkable
class MetricsCalculator(Protocol):
    def update(self, batch: list[PerImageDetection]) -> None: ...
    def run(self) -> MetricReport: ...
```

- **Train-side (light metrics):** call `update()` per-batch during training; emit a small summary at epoch end.
- **Test-side (full metrics):** collect all per-image detections, then `run()` produces the full `MetricReport`.

Internally, `run()` delegates to pycocotools for COCO mAP/AR, then computes PR/ROC curves, confusion matrix, and derived metrics (sensitivity, specificity, PPV, NPV) via sklearn.

## Aggregate metrics

### mAP (mean Average Precision)

- **mAP@0.5** (`map50`): mean AP averaged over classes, IoU threshold 0.5.
- **mAP@0.5:0.95** (`map5095`): mean AP averaged over classes and IoU thresholds `[0.5, 0.55, ..., 0.95]` (10 thresholds). The standard COCO primary metric.
- **AP@0.5** (`ap50`): per-class AP at IoU 0.5.
- **AP@0.75** (`ap75`): per-class AP at IoU 0.75.

**Formula (per class, per IoU threshold):**
$$\text{AP}_c^t = \int_0^1 p(r) \, dr$$
where $p(r)$ is the precision-recall curve at IoU threshold $t$ for class $c$. Averaged over thresholds and classes to get mAP.

### Precision / Recall / F1

- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `f1 = 2 * P * R / (P + R)`

All computed at the operating point selected by the F1-optimal confidence threshold.

### AR (Average Recall)

COCO-style AR over increasing max-detection budgets:
- `ar1` — AR with 1 detection per image.
- `ar10` — AR with 10 detections per image.
- `ar100` — AR with 100 detections per image.
- `ar300` — AR with 300 detections per image.
- `ar1000` — AR with 1000 detections per image.
- `ar_small` — AR for small objects (area < 32^2 px).
- `ar_medium` — AR for medium objects (32^2 <= area < 96^2 px).
- `ar_large` — AR for large objects (area >= 96^2 px).

**Formula:** average over IoU thresholds `[0.5, 0.95]` of the maximum recall achievable with the given detection budget.

### AUC-PR / AUC-ROC

- `auc_pr`: area under the precision-recall curve (sklearn).
- `auc_roc`: area under the receiver operating characteristic curve (sklearn).

### Sensitivity / Specificity / PPV / NPV

- `sensitivity = TP / (TP + FN)` — true positive rate (recall).
- `specificity = TN / (TN + FP)` — true negative rate.
- `ppv = TP / (TP + FP)` — positive predictive value (precision).
- `npv = TN / (TN + FN)` — negative predictive value.

These are computed from the confusion matrix at the F1-optimal confidence threshold.

## MetricReport

`MetricReport` (dataclass) — the full report returned by `COCOMetricsCalculator.run()`:

```python
@dataclass
class MetricReport:
    map50: float
    map5095: float
    ap50: dict[int, float]       # per-class AP@0.5
    ap75: dict[int, float]       # per-class AP@0.75
    precision: float
    recall: float
    f1: float
    ar1: float
    ar10: float
    ar100: float
    ar300: float
    ar1000: float
    ar_small: float
    ar_medium: float
    ar_large: float
    auc_pr: float
    auc_roc: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    confusion_matrix: np.ndarray
    iou_threshold: float
    conf_threshold: float
```

## Evaluation curves

The `cracks_yolo.metrics.curves` module produces three plots, saved to `curves/`:

- `pr.png` — precision-recall curve at each IoU threshold.
- `roc.png` — receiver operating characteristic curve.
- `confusion.png` — normalized confusion matrix heatmap.

The `cracks_yolo.metrics.confusion` module computes the raw confusion matrix and derives sensitivity, specificity, PPV, and NPV.

## Statistical tests (model comparison)

`StatisticalTest` (TypedDict) — for comparing two model variants (e.g. baseline vs SAC) on the same test set:

| Field | Type | Description |
| --- | --- | --- |
| `test_name` | `Literal["paired_t", "bootstrap_ci", "wilcoxon"]` | Test type. |
| `statistic` | `float` | Test statistic. |
| `p_value` | `float` | Two-sided p-value. |
| `ci_low` | `float \| None` | Bootstrap lower 95% CI (bootstrap only). |
| `ci_high` | `float \| None` | Bootstrap upper 95% CI (bootstrap only). |
| `n_samples` | `int` | Number of paired samples (e.g. test images). |

### Test selection

- **Paired t-test** — when per-image metric differences are approximately normal. Fast. Reports whether the mean difference is significantly different from 0.
- **Wilcoxon signed-rank** — non-parametric alternative to paired t. Use when differences are non-normal.
- **Bootstrap CI** — resample the test set with replacement (1000+ iterations) to get a 95% CI on the metric difference. Most robust, slowest.

## Interpretation for tongue crack detection

Tongue surface cracks are typically thin, elongated structures. A small spatial offset between prediction and ground truth can significantly reduce IoU, making this task IoU-sensitive. Consider the following when interpreting metrics:

- **Prefer mAP@50 over mAP@50:95** — the stricter IoU thresholds in mAP@50:95 penalize detections that are spatially close but not pixel-perfect. For thin cracks, mAP@50 better reflects clinically useful detections.
- **AUC-PR is threshold-independent** — it summarizes model quality across all confidence thresholds without requiring a fixed operating point.
- **AR@100 and AR@1000** — higher detection budgets help assess whether the model can recall fine crack fragments that may be broken into multiple small segments.
- **Confusion matrix** — with a single class (`crack`), the confusion matrix reduces to TP / FN / FP rates, which map directly to sensitivity and PPV.
