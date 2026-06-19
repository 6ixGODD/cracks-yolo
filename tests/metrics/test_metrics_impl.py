"""Tests for the real metrics implementation (calculator/curves/confusion/statistical)."""

from __future__ import annotations

from cracks_yolo.metrics import COCOMetricsCalculator
from cracks_yolo.metrics import bootstrap_ci
from cracks_yolo.metrics import compute_confusion_matrix
from cracks_yolo.metrics import compute_pr_curve
from cracks_yolo.metrics import paired_t_test
from cracks_yolo.metrics import wilcoxon
from cracks_yolo.metrics.schemas import PerImageDetection


def _make_perfect_image() -> PerImageDetection:
    """Detection that exactly matches its single GT — should give AP=1.0."""
    return {
        "image_id": 0,
        "detections": [
            {
                "image_id": 0,
                "class_id": 0,
                "score": 0.95,
                "bbox_xyxy": (10.0, 10.0, 50.0, 50.0),
            }
        ],
        "ground_truths": [
            {
                "image_id": 0,
                "class_id": 0,
                "score": 1.0,
                "bbox_xyxy": (10.0, 10.0, 50.0, 50.0),
            }
        ],
    }


def _make_missed_image() -> PerImageDetection:
    """Detection that doesn't overlap its GT — should give AP=0.0."""
    return {
        "image_id": 1,
        "detections": [
            {
                "image_id": 1,
                "class_id": 0,
                "score": 0.5,
                "bbox_xyxy": (200.0, 200.0, 240.0, 240.0),
            }
        ],
        "ground_truths": [
            {
                "image_id": 1,
                "class_id": 0,
                "score": 1.0,
                "bbox_xyxy": (10.0, 10.0, 50.0, 50.0),
            }
        ],
    }


def test_calculator_perfect_match() -> None:
    """A perfectly-matched detection gives AP@50 ≈ 1.0, precision=1.0."""
    calc = COCOMetricsCalculator(num_classes=1, iou_threshold=0.5)
    calc.update([_make_perfect_image()])
    report = calc.run()
    assert report.ap50 >= 0.99
    assert report.precision >= 0.99
    assert report.recall >= 0.99
    assert report.f1 >= 0.99


def test_calculator_missed_detection() -> None:
    """A non-overlapping detection is a FP + FN — AP near 0."""
    calc = COCOMetricsCalculator(num_classes=1, iou_threshold=0.5)
    calc.update([_make_missed_image()])
    report = calc.run()
    assert report.ap50 < 0.1
    # The detection was a FP, GT was a FN — confusion matrix should reflect.
    assert len(report.confusion_matrix) == 2
    # FP: background row (index 1), class col (index 0) — 1 count.
    assert report.confusion_matrix[1][0] == 1
    # FN: class row (index 0), background col (index 1) — 1 count.
    assert report.confusion_matrix[0][1] == 1


def test_calculator_empty() -> None:
    """Empty input returns zero metric report (no crash)."""
    calc = COCOMetricsCalculator(num_classes=1)
    report = calc.run()
    assert report.map50 == 0.0
    assert report.map5095 == 0.0


def test_pr_curve_basic() -> None:
    """One TP gives precision=1, recall=1."""
    detections = [(0, 0, 0.9, (10.0, 10.0, 50.0, 50.0))]
    gts = [(0, 0, (10.0, 10.0, 50.0, 50.0))]
    p, r, _t = compute_pr_curve(detections, gts, iou_thr=0.5)
    assert len(p) == 1
    assert p[0] >= 0.99
    assert r[0] >= 0.99


def test_confusion_matrix_perfect() -> None:
    """Perfect match gives cm[0,0] = 1, all else 0."""
    cm = compute_confusion_matrix(
        detections=[(0, 0, 0.9, (10.0, 10.0, 50.0, 50.0))],
        ground_truths=[(0, 0, (10.0, 10.0, 50.0, 50.0))],
        iou_thr=0.5,
        num_classes=1,
        conf_thr=0.0,
    )
    assert cm[0][0] == 1
    assert cm[0][1] == 0
    assert cm[1][0] == 0


def test_paired_t_test_significant() -> None:
    """A clearly larger sample gives a small p-value."""
    a = [0.9, 0.91, 0.92, 0.93, 0.94]
    b = [0.5, 0.51, 0.52, 0.53, 0.54]
    result = paired_t_test(a, b)
    assert result["test_name"] == "paired_t"
    assert result["p_value"] < 0.001
    assert result["n_samples"] == 5


def test_paired_t_test_no_difference() -> None:
    """Identical samples give p_value=1.0."""
    a = [0.8, 0.85, 0.9, 0.95, 1.0]
    result = paired_t_test(a, a)
    assert result["p_value"] > 0.99


def test_wilcoxon_basic() -> None:
    """Wilcoxon returns a statistic + p-value."""
    a = [0.9, 0.91, 0.92, 0.93, 0.94]
    b = [0.5, 0.51, 0.52, 0.53, 0.54]
    result = wilcoxon(a, b)
    assert result["test_name"] == "wilcoxon"
    assert 0.0 <= result["p_value"] <= 1.0


def test_bootstrap_ci_basic() -> None:
    """Bootstrap CI gives low <= mean <= high (CI brackets the point estimate)."""
    a = [0.9, 0.91, 0.92, 0.93, 0.94]
    b = [0.5, 0.51, 0.52, 0.53, 0.54]
    result = bootstrap_ci(a, b, n_boot=200, seed=0)
    assert result["test_name"] == "bootstrap_ci"
    assert result["ci_low"] is not None
    assert result["ci_high"] is not None
    # Diff is constant (0.4), so CI collapses to the point estimate.
    assert result["ci_low"] <= result["statistic"] <= result["ci_high"]
