"""Structured logging (loguru JSONL sink + TypedDict record schemas)."""

from __future__ import annotations

from cracks_yolo.logging.configure import configure_logger
from cracks_yolo.logging.schema import LogRecord
from cracks_yolo.logging.schema import MetricLog
from cracks_yolo.logging.schema import PretrainedLoadLog
from cracks_yolo.logging.schema import TestLog
from cracks_yolo.logging.schema import TrainEpochLog
from cracks_yolo.logging.schema import TrainStepLog
from cracks_yolo.logging.schema import ValLog

__all__ = [
    "LogRecord",
    "MetricLog",
    "PretrainedLoadLog",
    "TestLog",
    "TrainEpochLog",
    "TrainStepLog",
    "ValLog",
    "configure_logger",
]
