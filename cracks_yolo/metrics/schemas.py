"""TypedDict schemas + dataclasses for detection metrics.

Implementation (pycocotools + torchvision ops) lives in
:mod:`cracks_yolo.metrics.calculator`; this module fixes the *shape* of
metric records so pipelines and loggers can be written against them.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from typing_extensions import TypedDict


class DetectionMetric(TypedDict):
    """A single detection (after NMS, before metric aggregation)."""

    image_id: int
    class_id: int
    score: float
    bbox_xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


class PerImageDetection(TypedDict):
    """All detections + ground truths for one image."""

    image_id: int
    detections: list[DetectionMetric]
    ground_truths: list[DetectionMetric]


class PerformanceMetric(TypedDict):
    """Runtime / efficiency metric (FPS, params, MACs, latency)."""

    name: str
    value: float
    unit: str  # e.g. "fps", "M", "G", "ms"


class StatisticalTest(TypedDict):
    """Result of a paired statistical test between two model variants."""

    test_name: Literal["paired_t", "bootstrap_ci", "wilcoxon"]
    statistic: float
    p_value: float
    ci_low: float | None  # bootstrap only
    ci_high: float | None  # bootstrap only
    n_samples: int


@dataclass
class MetricReport:
    """Aggregated metric report returned by ``MetricsCalculator.run``."""

    # mAP / AR — the COCO primary metrics.
    map50: float
    map5095: float
    ap50: float = 0.0  # alias for map50 when num_classes=1
    ap75: float = 0.0
    per_class_ap: dict[int, float] = field(default_factory=dict)
    # P / R / F1 at the F1-optimal confidence threshold.
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # COCO AR (Average Recall) at varying max-dets.
    ar1: float = 0.0
    ar10: float = 0.0
    ar100: float = 0.0
    ar300: float = 0.0
    ar1000: float = 0.0
    ar_small: float = 0.0
    ar_medium: float = 0.0
    ar_large: float = 0.0
    # Curve-based metrics.
    auc_pr: float = 0.0
    auc_roc: float = 0.0
    # Diagnostic metrics at the chosen operating point.
    sensitivity: float = 0.0  # = recall
    specificity: float = 0.0
    ppv: float = 0.0  # positive predictive value = precision
    npv: float = 0.0  # negative predictive value
    # Confusion matrix (num_classes+1 x num_classes+1) including background.
    confusion_matrix: list[list[int]] = field(default_factory=list)
    # Operational thresholds chosen for the binary metrics.
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    # Free-form lists for downstream consumers.
    performance: list[PerformanceMetric] = field(default_factory=list)
    statistical_tests: list[StatisticalTest] = field(default_factory=list)
