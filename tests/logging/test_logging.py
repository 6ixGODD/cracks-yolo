"""Tests for cracks_yolo.logging (configure_logger + schema TypedDicts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from cracks_yolo.logging.configure import configure_logger
from cracks_yolo.logging.schema import TrainStepLog


def test_configure_logger_writes_jsonl(tmp_path: Path) -> None:
    """configure_logger installs a sink that writes JSON lines to run.log.jsonl."""
    configure_logger(output_dir=tmp_path, level="INFO", stderr=False)
    record: TrainStepLog = {
        "record_type": "train_step",
        "step": 0,
        "epoch": 0,
        "total_loss": 1.23,
        "box_loss": 0.4,
        "cls_loss": 0.5,
        "obj_loss": 0.33,
        "dfl_loss": None,
        "lr": 1e-3,
        "timestamp": "2026-06-18T00:00:00",
    }
    logger.bind(**record).info("step done")
    # Wait for non-enqueue sink to flush (synchronous, so no wait needed).
    log_path = tmp_path / "run.log.jsonl"
    assert log_path.exists(), f"no log file at {log_path}"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    parsed: dict[str, Any] = json.loads(lines[-1])
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "step done"
    assert parsed["step"] == 0
    assert parsed["total_loss"] == 1.23
    assert parsed["dfl_loss"] is None


def test_configure_logger_creates_output_dir(tmp_path: Path) -> None:
    """configure_logger creates the output_dir if missing."""
    nested = tmp_path / "nested" / "dir"
    configure_logger(output_dir=nested, level="INFO", stderr=False)
    assert nested.exists()
    logger.info("hi")
    assert (nested / "run.log.jsonl").exists()


def test_train_step_log_typeddict_is_dict_at_runtime() -> None:
    """TypedDicts are plain dicts at runtime — verify the schema is usable."""
    record: TrainStepLog = {
        "record_type": "train_step",
        "step": 1,
        "epoch": 0,
        "total_loss": 0.5,
        "box_loss": 0.1,
        "cls_loss": 0.2,
        "obj_loss": 0.2,
        "dfl_loss": None,
        "lr": 1e-3,
        "timestamp": "2026-06-18T00:00:00",
    }
    assert record["step"] == 1
    assert record["dfl_loss"] is None
