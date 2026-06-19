"""Smoke tests for cracks_yolo.viz (curves, confusion, dataset, heatmap)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from cracks_yolo.viz.confusion import plot_confusion_matrix
from cracks_yolo.viz.curves import plot_loss_curve
from cracks_yolo.viz.curves import plot_metric_curve
from cracks_yolo.viz.curves import plot_pr_curve
from cracks_yolo.viz.curves import plot_roc_curve
from cracks_yolo.viz.dataset import plot_bbox_position_heatmap
from cracks_yolo.viz.dataset import plot_bbox_size_distribution
from cracks_yolo.viz.dataset import plot_class_distribution
from cracks_yolo.viz.dataset import plot_image_size_distribution
from cracks_yolo.viz.heatmap import GradCAMExtractor
from cracks_yolo.zoo import ZOO


def _write_metrics_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_map50", "val_map5095"])
        writer.writeheader()
        for i in range(3):
            writer.writerow({
                "epoch": i,
                "train_loss": 1.0 - i * 0.2,
                "val_map50": 0.5 + i * 0.1,
                "val_map5095": 0.3 + i * 0.05,
            })


def test_plot_loss_curve(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _write_metrics_csv(csv_path)
    out = tmp_path / "loss.png"
    plot_loss_curve(csv_path, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_metric_curve(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _write_metrics_csv(csv_path)
    out = tmp_path / "metric.png"
    plot_metric_curve(csv_path, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_pr_curve(tmp_path: Path) -> None:
    out = tmp_path / "pr.png"
    plot_pr_curve(
        precision=np.array([1.0, 0.8, 0.5]),
        recall=np.array([0.3, 0.6, 1.0]),
        out_png=out,
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_roc_curve(tmp_path: Path) -> None:
    out = tmp_path / "roc.png"
    plot_roc_curve(
        fpr=np.array([0.0, 0.2, 1.0]),
        tpr=np.array([0.0, 0.7, 1.0]),
        out_png=out,
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_confusion_matrix(tmp_path: Path) -> None:
    out = tmp_path / "cm.png"
    plot_confusion_matrix(
        matrix=[[5, 1], [2, 10]],
        class_names=["crack"],
        out_png=out,
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_class_distribution(tmp_path: Path) -> None:
    from cracks_yolo.dataset.types import RawDetection

    records = [
        RawDetection(
            image_path=Path("x.png"),
            image_id=i,
            width=64,
            height=64,
            boxes_norm=[(0.1, 0.1, 0.4, 0.4)],
            labels=[i % 2],
        )
        for i in range(4)
    ]
    out = tmp_path / "class.png"
    plot_class_distribution(records, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_bbox_size_distribution(tmp_path: Path) -> None:
    from cracks_yolo.dataset.types import RawDetection

    records = [
        RawDetection(
            image_path=Path("x.png"),
            image_id=i,
            width=64,
            height=64,
            boxes_norm=[(0.1, 0.1, 0.4, 0.4)],
            labels=[0],
        )
        for i in range(4)
    ]
    out = tmp_path / "size.png"
    plot_bbox_size_distribution(records, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_bbox_position_heatmap(tmp_path: Path) -> None:
    from cracks_yolo.dataset.types import RawDetection

    records = [
        RawDetection(
            image_path=Path("x.png"),
            image_id=i,
            width=64,
            height=64,
            boxes_norm=[(0.1, 0.1, 0.4, 0.4)],
            labels=[0],
        )
        for i in range(4)
    ]
    out = tmp_path / "heat.png"
    plot_bbox_position_heatmap(records, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_image_size_distribution(tmp_path: Path) -> None:
    from cracks_yolo.dataset.types import RawDetection

    records = [
        RawDetection(
            image_path=Path("x.png"),
            image_id=i,
            width=64 + i,
            height=64,
            boxes_norm=[],
            labels=[],
        )
        for i in range(4)
    ]
    out = tmp_path / "img_size.png"
    plot_image_size_distribution(records, out)
    assert out.exists() and out.stat().st_size > 0


def test_gradcam_extractor_smoke() -> None:
    """GradCAMExtractor produces a heatmap for a YOLOv5 backbone layer."""
    cls = ZOO["yolov5s"]
    model = cls(num_classes=1, input_size=64)
    image = torch.randn(1, 3, 64, 64, requires_grad=True)
    with GradCAMExtractor(model, ["backbone.8"]) as ge:
        heatmaps = ge.generate(image, target_class=0)
    assert "backbone.8" in heatmaps
    cam = heatmaps["backbone.8"]
    assert cam.ndim == 2
    assert cam.shape == (64, 64)
    assert 0.0 <= cam.min() <= cam.max() <= 1.0 + 1e-6
