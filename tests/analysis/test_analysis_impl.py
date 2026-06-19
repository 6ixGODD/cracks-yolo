"""Smoke tests for cracks_yolo.analysis (dataset + model)."""

from __future__ import annotations

import json
from pathlib import Path

from cracks_yolo.analysis.dataset import analyze_dataset
from cracks_yolo.analysis.dataset import save_dataset_analysis
from cracks_yolo.analysis.model import analyze_model
from cracks_yolo.analysis.model import save_model_analysis
from cracks_yolo.dataset.types import RawDetection
from cracks_yolo.zoo import ZOO


def test_analyze_dataset_basic() -> None:
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
    report = analyze_dataset(records)
    assert report.n_images == 4
    assert report.n_annotations == 4
    assert report.n_classes == 1
    assert report.class_counts == {0: 4}
    assert report.imbalance_ratio == 1.0
    assert report.class_shannon_entropy == 0.0  # only one class
    assert report.bbox_aspect_ratio_buckets >= 1
    assert 0.0 <= report.spatial_coverage <= 1.0


def test_save_dataset_analysis(tmp_path: Path) -> None:
    records = [
        RawDetection(
            image_path=Path("x.png"),
            image_id=0,
            width=64,
            height=64,
            boxes_norm=[(0.1, 0.1, 0.4, 0.4)],
            labels=[0],
        )
    ]
    report = analyze_dataset(records)
    save_dataset_analysis(report, tmp_path)
    out = tmp_path / "dataset_analysis.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_images"] == 1


def test_analyze_model_basic() -> None:
    cls = ZOO["yolov5s"]
    model = cls(num_classes=1, input_size=64)
    report = analyze_model(model, input_size=64, device="cpu", n_warmup=1, n_runs=3)
    assert report.model_name.startswith("YOLOv5")
    assert report.n_parameters > 0
    assert report.n_trainable_parameters > 0
    assert report.latency_mean_ms >= 0.0
    assert report.input_size == 64
    assert report.device == "cpu"


def test_save_model_analysis(tmp_path: Path) -> None:
    cls = ZOO["yolov5s"]
    model = cls(num_classes=1, input_size=64)
    report = analyze_model(model, input_size=64, device="cpu", n_warmup=1, n_runs=2)
    save_model_analysis(report, tmp_path)
    out = tmp_path / "model_analysis.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_parameters"] > 0
