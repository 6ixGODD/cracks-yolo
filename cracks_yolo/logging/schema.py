"""TypedDict schemas for loguru log records.

Every log call in pipelines passes one of these (or a dict shaped like one)
as ``extra=<record>`` so the JSONL sink writes a structured record. The
``record_type`` discriminator field is mandatory and selects which TypedDict
applies — this lets post-hoc queries filter by record type.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class TrainStepLog(TypedDict):
    """One optimizer step within an epoch."""

    record_type: Literal["train_step"]
    step: int
    epoch: int
    total_loss: float
    box_loss: float
    cls_loss: float
    obj_loss: float | None  # None for v8/v10 (no obj loss)
    dfl_loss: float | None  # None for v5/v7
    lr: float
    timestamp: str  # ISO 8601


class TrainEpochLog(TypedDict):
    """End-of-epoch training summary."""

    record_type: Literal["train_epoch"]
    epoch: int
    mean_total_loss: float
    mean_box_loss: float
    mean_cls_loss: float
    mean_obj_loss: float | None
    mean_dfl_loss: float | None
    lr: float
    elapsed_sec: float
    timestamp: str


class ValLog(TypedDict):
    """Validation pass summary."""

    record_type: Literal["val"]
    epoch: int
    map50: float
    map5095: float
    per_class_ap: list[float]
    elapsed_sec: float
    timestamp: str


class TestLog(TypedDict):
    """Test-set evaluation summary."""

    record_type: Literal["test"]
    map50: float
    map5095: float
    per_class_ap: list[float]
    precision: float
    recall: float
    f1: float
    elapsed_sec: float
    timestamp: str


class MetricLog(TypedDict):
    """A single scalar metric emission (e.g. FPS, params, MACs)."""

    record_type: Literal["metric"]
    name: str
    value: float
    unit: str
    timestamp: str


class PretrainedLoadLog(TypedDict):
    """Pretrained weight load report."""

    record_type: Literal["pretrained_load"]
    key: str
    url: str
    cached: bool
    matched_count: int
    missing_count: int
    unexpected_count: int
    missing_keys: list[str]
    unexpected_keys: list[str]
    timestamp: str


LogRecord = TrainStepLog | TrainEpochLog | ValLog | TestLog | MetricLog | PretrainedLoadLog
"""Union of all log record TypedDicts."""
