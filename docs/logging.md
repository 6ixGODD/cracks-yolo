# Logging (`cracks_yolo.logging`)

[English](logging.md) | [中文](logging.zh-CN.md)

## Design

`loguru` configured by `cracks_yolo.logging.configure_logger(output_dir)`. Two sinks are installed:

1. **JSONL file sink** at `{output_dir}/run.log.jsonl` — one JSON object per line, suitable for post-hoc analysis with `jq`, pandas, or any JSONL-aware tool.
2. **Stderr sink** — human-readable, colorized, for live monitoring.

## Usage

```python
from pathlib import Path
from loguru import logger
from cracks_yolo.logging.configure import configure_logger
from cracks_yolo.logging.schema import TrainStepLog

configure_logger(output_dir=Path("output/run1"))

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
```

The `logger.bind(**record)` pattern (canonical loguru) merges the dict into `record["extra"]`. The JSONL sink merges `extra` into the top-level JSON object alongside `level`, `message`, `timestamp`.

## Record schemas (`cracks_yolo.logging.schema`)

All schemas are `TypedDict`s (via `typing_extensions.TypedDict` for Python 3.11 compatibility with pydantic). Each carries a `record_type: Literal[...]` discriminator so post-hoc queries can filter by record type.

### `TrainStepLog` — one optimizer step

| Field | Type | Description |
| --- | --- | --- |
| `record_type` | `Literal["train_step"]` | Discriminator. |
| `step` | `int` | Global step counter. |
| `epoch` | `int` | Current epoch. |
| `total_loss` | `float` | Sum of all loss components. |
| `box_loss` | `float` | Box regression loss. |
| `cls_loss` | `float` | Classification loss. |
| `obj_loss` | `float \| None` | Objectness loss (None for v8/v10). |
| `dfl_loss` | `float \| None` | Distribution Focal Loss (None for v5/v7). |
| `lr` | `float` | Current learning rate. |
| `timestamp` | `str` | ISO 8601 timestamp. |

### `TrainEpochLog` — end-of-epoch summary

Same fields as `TrainStepLog` (minus `step`) plus `mean_*` prefixes and `elapsed_sec`.

### `ValLog` — validation pass

| Field | Type |
| --- | --- |
| `record_type` | `Literal["val"]` |
| `epoch` | `int` |
| `map50` | `float` |
| `map5095` | `float` |
| `per_class_ap` | `list[float]` |
| `elapsed_sec` | `float` |
| `timestamp` | `str` |

### `TestLog` — test-set evaluation

| Field | Type |
| --- | --- |
| `record_type` | `Literal["test"]` |
| `map50` | `float` |
| `map5095` | `float` |
| `per_class_ap` | `list[float]` |
| `precision` | `float` |
| `recall` | `float` |
| `f1` | `float` |
| `elapsed_sec` | `float` |
| `n_images` | `int` |
| `fps_mean` | `float` |
| `latency_mean_ms` | `float` |
| `gflops` | `float` |
| `n_parameters` | `int` |
| `timestamp` | `str` |

The `n_images` / `fps_mean` / `latency_mean_ms` / `gflops` / `n_parameters` fields are the efficiency headline numbers (0 when `measure_efficiency=False`); the full breakdown is in `model_analysis.json`.

### `MetricLog` — a single scalar emission

| Field | Type |
| --- | --- |
| `record_type` | `Literal["metric"]` |
| `name` | `str` |
| `value` | `float` |
| `unit` | `str` |
| `timestamp` | `str` |

Use this for FPS, params, MACs, latency percentiles, etc.

### `PretrainedLoadLog` — pretrained weight load report

| Field | Type |
| --- | --- |
| `record_type` | `Literal["pretrained_load"]` |
| `key` | `str` |
| `url` | `str` |
| `cached` | `bool` |
| `matched_count` | `int` |
| `missing_count` | `int` |
| `unexpected_count` | `int` |
| `missing_keys` | `list[str]` |
| `unexpected_keys` | `list[str]` |
| `timestamp` | `str` |

Emitted by the pipeline after `from_pretrained` so every run records exactly which keys were randomly initialized (SAC/TR layers).

## JSONL format

Each line is a single JSON object. Example:

```json
{"level":"INFO","message":"step done","timestamp":"2026-06-18T10:23:45.123456","record_type":"train_step","step":0,"epoch":0,"total_loss":1.23,"box_loss":0.4,"cls_loss":0.5,"obj_loss":0.33,"dfl_loss":null,"lr":0.001}
```

## Post-hoc queries

Filter by record type with `jq`:

```bash
jq 'select(.record_type == "train_step")' output/run1/run.log.jsonl
```

Compute mean total_loss per epoch:

```bash
jq -s 'group_by(.epoch) | map({epoch: .[0].epoch, mean_loss: (map(.total_loss) | add / length)})' \
  output/run1/run.log.jsonl
```

Or load with pandas:

```python
import pandas as pd
df = pd.read_json("output/run1/run.log.jsonl", lines=True)
train_df = df[df.record_type == "train_step"]
```
