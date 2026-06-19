# Metrics (`cracks_yolo.metrics`)

Implementation (pycocotools + torchvision ops) lands in a later pass.
This document fixes the **shape** of every metric we will emit, so
pipelines and loggers can be written against them today.

## Detection-level records

### `DetectionMetric` (TypedDict)
A single detection after NMS, before metric aggregation.

| Field | Type | Description |
| --- | --- | --- |
| `image_id` | `int` | Image identifier (matches dataset index). |
| `class_id` | `int` | Predicted class id. |
| `score` | `float` | Confidence score in `[0, 1]`. |
| `bbox_xyxy` | `tuple[float, float, float, float]` | `(x1, y1, x2, y2)` in pixels. |

### `PerImageDetection` (TypedDict)
All detections + ground truths for one image.

| Field | Type |
| --- | --- |
| `image_id` | `int` |
| `detections` | `list[DetectionMetric]` |
| `ground_truths` | `list[DetectionMetric]` |

## Aggregate metrics

### mAP (mean Average Precision)

- **mAP@0.5** (`map50`): mean AP averaged over classes, IoU threshold 0.5.
- **mAP@0.5:0.95** (`map5095`): mean AP averaged over classes and IoU
  thresholds `[0.5, 0.55, ..., 0.95]` (10 thresholds). The standard COCO
  primary metric.

**Formula (per class, per IoU threshold):**
$$\text{AP}_c^t = \int_0^1 p(r) \, dr$$
where $p(r)$ is the precision-recall curve at IoU threshold $t$ for class
$c$. Averaged over thresholds and classes to get mAP.

### Per-class AP

`per_class_ap: dict[int, float]` — AP@0.5 for each class id. Empty dict
if `num_classes=1`.

### Precision / Recall / F1

- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `f1 = 2 * P * R / (P + R)`

All computed at the operating point selected by the F1-optimal confidence
threshold.

### AR (Average Recall)

COCO-style AR over `[1, 10, 100]` max detections per image:
- `ar1` — AR with 1 detection per image.
- `ar10` — AR with 10 detections per image.
- `ar100` — AR with 100 detections per image.

**Formula:** average over IoU thresholds `[0.5, 0.95]` of the maximum
recall achievable with the given detection budget.

## Performance metrics

`PerformanceMetric` (TypedDict):

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | e.g. `"fps"`, `"params"`, `"macs"`, `"latency_p50"`, `"latency_p95"` |
| `value` | `float` | Numeric value. |
| `unit` | `str` | e.g. `"fps"`, `"M"` (millions), `"G"` (giga MACs), `"ms"` |

## Statistical tests (model comparison)

`StatisticalTest` (TypedDict) — for comparing two model variants (e.g.
baseline vs SAC) on the same test set:

| Field | Type | Description |
| --- | --- | --- |
| `test_name` | `Literal["paired_t", "bootstrap_ci", "wilcoxon"]` | Test type. |
| `statistic` | `float` | Test statistic. |
| `p_value` | `float` | Two-sided p-value. |
| `ci_low` | `float \| None` | Bootstrap lower 95% CI (bootstrap only). |
| `ci_high` | `float \| None` | Bootstrap upper 95% CI (bootstrap only). |
| `n_samples` | `int` | Number of paired samples (e.g. test images). |

### Test selection

- **Paired t-test** — when per-image metric differences are approximately
  normal. Fast. Reports whether the mean difference is significantly
  different from 0.
- **Wilcoxon signed-rank** — non-parametric alternative to paired t.
  Use when differences are non-normal.
- **Bootstrap CI** — resample the test set with replacement (1000+
  iterations) to get a 95% CI on the metric difference. Most robust,
  slowest.

## MetricReport

`MetricReport` (dataclass) — the full report returned by
`MetricsCalculator.run()`:

```python
@dataclass
class MetricReport:
    map50: float
    map5095: float
    per_class_ap: dict[int, float] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ar1: float = 0.0
    ar10: float = 0.0
    ar100: float = 0.0
    performance: list[PerformanceMetric] = field(default_factory=list)
    statistical_tests: list[StatisticalTest] = field(default_factory=list)
```

## MetricsCalculator Protocol

```python
@runtime_checkable
class MetricsCalculator(Protocol):
    def update(self, batch: list[PerImageDetection]) -> None: ...
    def run(self) -> MetricReport: ...
```

- **Train-side (light metrics):** call `update()` per-batch during
  training; emit a small summary at epoch end.
- **Test-side (full metrics):** collect all per-image detections, then
  `run()` produces the full `MetricReport`.

The concrete implementation (pycocotools backend for mAP/AR + torchvision
ops for IoU/NMS + scipy for statistical tests) lands in a later pass.
