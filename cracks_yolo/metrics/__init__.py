"""Detection metrics — TypedDicts + Protocol + real implementation."""

from __future__ import annotations

from cracks_yolo.metrics.calculator import COCOMetricsCalculator
from cracks_yolo.metrics.confusion import compute_confusion_matrix
from cracks_yolo.metrics.curves import compute_auc
from cracks_yolo.metrics.curves import compute_auc_roc
from cracks_yolo.metrics.curves import compute_pr_curve
from cracks_yolo.metrics.curves import compute_roc_curve
from cracks_yolo.metrics.protocol import MetricsCalculator
from cracks_yolo.metrics.schemas import DetectionMetric
from cracks_yolo.metrics.schemas import MetricReport
from cracks_yolo.metrics.schemas import PerformanceMetric
from cracks_yolo.metrics.schemas import PerImageDetection
from cracks_yolo.metrics.schemas import StatisticalTest
from cracks_yolo.metrics.statistical import bootstrap_ci
from cracks_yolo.metrics.statistical import paired_t_test
from cracks_yolo.metrics.statistical import run_statistical_test
from cracks_yolo.metrics.statistical import wilcoxon

__all__ = [
    "COCOMetricsCalculator",
    "DetectionMetric",
    "MetricReport",
    "MetricsCalculator",
    "PerImageDetection",
    "PerformanceMetric",
    "StatisticalTest",
    "bootstrap_ci",
    "compute_auc",
    "compute_auc_roc",
    "compute_confusion_matrix",
    "compute_pr_curve",
    "compute_roc_curve",
    "paired_t_test",
    "run_statistical_test",
    "wilcoxon",
]
