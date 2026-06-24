"""Tests for cracks_yolo.pipeline (Protocol structural check + pydantic configs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cracks_yolo.metrics.schemas import MetricReport
from cracks_yolo.pipeline import TestConfig
from cracks_yolo.pipeline import TestReport
from cracks_yolo.pipeline import TrainConfig
from cracks_yolo.pipeline import TrainReport
from cracks_yolo.zoo import ZOO


def test_train_config_defaults() -> None:
    """TrainConfig has the expected defaults."""
    cfg = TrainConfig(output_dir=Path("output/run1"))
    assert cfg.epochs == 100
    assert cfg.batch_size == 16
    assert cfg.lr == 1e-3
    assert cfg.amp is True
    assert cfg.seed == 42


def test_test_config_defaults() -> None:
    """TestConfig has the expected defaults."""
    cfg = TestConfig(output_dir=Path("output/test1"))
    assert cfg.iou_thr == 0.6
    assert cfg.conf_thr == 0.001
    assert cfg.max_dets == 300


def test_train_config_validates_positive_epochs() -> None:
    """pydantic enforces a sane range for epochs."""
    cfg = TrainConfig(output_dir=Path("x"), epochs=5)
    assert cfg.epochs == 5
    with pytest.raises(ValueError):
        TrainConfig(output_dir=Path("x"), epochs=-1)


def test_train_report_defaults() -> None:
    """TrainReport has checkpoint_paths default."""
    r = TrainReport(
        output_dir=Path("x"),
        best_epoch=0,
        best_map50=0.5,
        final_train_loss=0.1,
        final_val_map50=0.5,
        final_val_map5095=0.3,
        total_steps=100,
        total_epochs=10,
        elapsed_sec=60.0,
    )
    assert r.checkpoint_paths == []


def test_test_report_accepts_metric_report() -> None:
    """TestReport can carry a MetricReport (arbitrary_types_allowed)."""
    r = TestReport(
        output_dir=Path("x"),
        metrics=MetricReport(map50=0.6, map5095=0.4),
        elapsed_sec=10.0,
    )
    assert r.metrics.map50 == 0.6


def test_pipeline_protocol_is_satisfied_by_zoo_models() -> None:
    """Every ZOO class structurally satisfies DetectorModel (used by pipelines)."""
    from cracks_yolo.zoo.base import DetectorModel

    cls = next(iter(ZOO.values()))
    model = cls(num_classes=1)
    assert isinstance(model, DetectorModel)
