"""Tests for cracks_yolo.metrics (schemas + Protocol structural check)."""

from __future__ import annotations

from cracks_yolo.metrics import DetectionMetric
from cracks_yolo.metrics import MetricReport
from cracks_yolo.metrics import MetricsCalculator
from cracks_yolo.metrics import PerformanceMetric
from cracks_yolo.metrics import PerImageDetection
from cracks_yolo.metrics import StatisticalTest


def test_detection_metric_is_dict() -> None:
    """DetectionMetric is a TypedDict usable as a plain dict."""
    d: DetectionMetric = {
        "image_id": 0,
        "class_id": 0,
        "score": 0.95,
        "bbox_xyxy": (10.0, 20.0, 30.0, 40.0),
    }
    assert d["image_id"] == 0
    assert d["bbox_xyxy"][2] == 30.0


def test_per_image_detection_shape() -> None:
    """PerImageDetection carries detections + ground_truths lists."""
    img: PerImageDetection = {"image_id": 0, "detections": [], "ground_truths": []}
    assert img["detections"] == []


def test_performance_metric_shape() -> None:
    """PerformanceMetric has name/value/unit."""
    m: PerformanceMetric = {"name": "fps", "value": 120.0, "unit": "fps"}
    assert m["unit"] == "fps"


def test_statistical_test_shape() -> None:
    """StatisticalTest carries test_name, statistic, p_value, CI, n_samples."""
    t: StatisticalTest = {
        "test_name": "paired_t",
        "statistic": 2.1,
        "p_value": 0.04,
        "ci_low": None,
        "ci_high": None,
        "n_samples": 50,
    }
    assert t["p_value"] < 0.05


def test_metric_report_defaults() -> None:
    """MetricReport dataclass has sensible defaults."""
    r = MetricReport(map50=0.6, map5095=0.4)
    assert r.map50 == 0.6
    assert r.per_class_ap == {}
    assert r.performance == []


def test_metrics_calculator_protocol_is_runtime_checkable() -> None:
    """MetricsCalculator is @runtime_checkable — verify it accepts a stub.

    A minimal class with ``update`` and ``run`` satisfies the Protocol.
    """

    class _Stub:
        def update(self, batch: list[PerImageDetection]) -> None:
            pass

        def run(self) -> MetricReport:
            return MetricReport(map50=0.0, map5095=0.0)

    assert isinstance(_Stub(), MetricsCalculator)
