"""Tests for the YAML experiment scheduler."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import yaml

from scripts import schedule_experiments as scheduler


def test_all_models_direct_config_is_valid() -> None:
    config_path = Path("experiments/all_models_direct.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert len(config["experiments"]) == 52
    assert scheduler._validate_experiments(config["experiments"]) == []


def test_invalid_config_starts_no_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "experiments": [
                {
                    "name": "missing_test_output",
                    "type": "test",
                    "model": "yolov5s",
                    "weights": "best.pt",
                    "dataset": "dataset.yaml",
                }
            ]
        }),
        encoding="utf-8",
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called for invalid config")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    assert scheduler.run_from_yaml(config_path, tmp_path / "output") == 1
    assert not (tmp_path / "output" / "scheduler").exists()


def test_run_one_records_command_generation_error(tmp_path: Path) -> None:
    scheduler_dir = tmp_path / "scheduler"
    scheduler_dir.mkdir()
    errors_path = scheduler_dir / "errors.jsonl"
    results_path = scheduler_dir / "results.jsonl"

    exit_code = scheduler._run_one(
        {"name": "broken", "type": "test"},
        scheduler_dir,
        errors_path,
        results_path,
    )

    assert exit_code == -1
    record = json.loads(errors_path.read_text(encoding="utf-8"))
    assert record["exp_name"] == "broken"
    assert "KeyError" in record["traceback"]
