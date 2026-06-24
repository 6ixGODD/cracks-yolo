# Logging (`cracks_yolo.logging`)

## Architecture

The module wraps `loguru` with two sinks installed by `configure_logger(output_dir)`:

1. **JSONL file sink** at `{output_dir}/run.log.jsonl` — one JSON object per line.
2. **Stderr sink** — colourised, human-readable format for live monitoring.

The JSONL sink is built via `_make_jsonl_sink(path)`, which opens the target file in append mode
and serialises each record as a single-line JSON object. The `extra` dict from `logger.bind(**record)`
is merged into the top-level payload alongside `level`, `message`, and `timestamp`.

## Record schemas

All log record types are `TypedDict` subclasses defined in `cracks_yolo.logging.schema`. Each carries
a mandatory `record_type: Literal[...]` discriminator for post-hoc filtering.

| Record type | `record_type` | Purpose |
|---|---|---|
| `TrainStepLog` | `"train_step"` | One optimizer step: `step`, `epoch`, `total_loss`, `box_loss`, `cls_loss`, `obj_loss` (nullable), `dfl_loss` (nullable), `lr`, `timestamp`. |
| `TrainEpochLog` | `"train_epoch"` | End-of-epoch summary: `epoch`, `mean_*` loss fields, `lr`, `elapsed_sec`, `timestamp`. |
| `ValLog` | `"val"` | Validation pass: `epoch`, `map50`, `map5095`, `per_class_ap`, `elapsed_sec`, `timestamp`. |
| `TestLog` | `"test"` | Test-set evaluation: `map50`, `map5095`, `per_class_ap`, `precision`, `recall`, `f1`, `elapsed_sec`, plus efficiency fields (`n_images`, `fps_mean`, `latency_mean_ms`, `gflops`, `n_parameters`), `timestamp`. |
| `MetricLog` | `"metric"` | Arbitrary scalar emission: `name`, `value`, `unit`, `timestamp`. Used for FPS, params, MACs, latency percentiles. |
| `PretrainedLoadLog` | `"pretrained_load"` | Weight-load audit: `key`, `url`, `cached`, `matched_count`, `missing_count`, `unexpected_count`, `missing_keys`, `unexpected_keys`, `timestamp`. Emitted after `from_pretrained`. |

The union type `LogRecord` is defined as
`TrainStepLog | TrainEpochLog | ValLog | TestLog | MetricLog | PretrainedLoadLog`.

## Nullable fields

`obj_loss` is `None` for anchor-free architectures (v8, v10) that omit objectness loss.
`dfl_loss` is `None` for anchor-based architectures (v5, v7) that use IoU loss instead.
Efficiency fields in `TestLog` are zero when `measure_efficiency=False`.

## Output location

`configure_logger(output_dir)` creates `output_dir` if absent and writes `run.log.jsonl` inside it.
Example: `configure_logger(Path("output/run1"))` produces `output/run1/run.log.jsonl`.

## Usage

```python
from pathlib import Path
from loguru import logger
from cracks_yolo.logging import configure_logger, TrainStepLog

configure_logger(Path("output/run1"))

record: TrainStepLog = {
    "record_type": "train_step", "step": 0, "epoch": 0,
    "total_loss": 1.23, "box_loss": 0.4, "cls_loss": 0.5,
    "obj_loss": 0.33, "dfl_loss": None, "lr": 1e-3,
    "timestamp": "2026-06-18T00:00:00",
}
logger.bind(**record).info("step done")
```

The `logger.bind(**record)` pattern merges the dict into `record["extra"]` (canonical loguru idiom).
The JSONL sink then promotes `extra` keys into the top-level JSON object.

## Post-hoc queries

Filter by record type with `jq`:

```bash
jq 'select(.record_type == "train_step")' output/run1/run.log.jsonl
```

Compute per-epoch mean total_loss:

```bash
jq -s 'group_by(.epoch) | map({epoch: .[0].epoch, mean_loss: (map(.total_loss)|add/length)})' \
  output/run1/run.log.jsonl
```

Or with pandas:

```python
import pandas as pd
df = pd.read_json("output/run1/run.log.jsonl", lines=True)
df[df.record_type == "train_step"].groupby("epoch")["total_loss"].mean()
```
