# Experiment scheduler

[English](scheduler.md) | [中文](scheduler.zh-CN.md)

`cracks_yolo.pipeline.compose` executes a YAML-defined batch of experiments via subprocess
isolation. Each experiment resolves to a `cy run` or `cy test` invocation; the scheduler
serialises execution, captures per-experiment logs, and records outcomes in JSONL.

## YAML format

A compose file aggregates individual experiment YAMLs via `$include`. Each included file is a
single experiment dict (an explicit `experiments` list is also accepted). A top-level `scheduler`
block carries metadata but is not consumed at runtime.

```yaml
# experiments/compose_all.yaml
scheduler:
  max_parallel: 1
  seed: 42
$include:
  - models/yolov5s.yaml
  - models/yolov5s_sac.yaml
```

An individual experiment YAML (`experiments/models/yolov5s.yaml`):

```yaml
name: yolov5s
type: train
model: yolov5s
dataset: data/CrackDetection_Augmentation.v1.yolov5pytorch
output_dir: output/yolov5s
epochs: 100
batch_size: 32
lr: 0.001
seed: 42
pretrained: true
device: cuda
```

Supported fields: `name`, `type` (`train`|`test`), `model`, `dataset`, `output_dir`, `epochs`,
`batch_size`, `lr`, `device`, `seed`, `num_workers`, `pretrained`, `weights` (test only),
`optimizer`, `cosine_lr`, `use_ema`, `early_stopping_patience`, `clip_grad_norm`, and `env`
(per-experiment environment dict, e.g. `{CUDA_VISIBLE_DEVICES: "0"}`).

## `$include` resolution (`_load_config`)

`_load_config` resolves `$include` recursively. Included paths are relative to the YAML file
that declares them. Three cases govern contribution:

1. **`$include` present** -- each listed path is loaded recursively; experiments accumulate from
   all children. The parent's `scheduler` block is discarded.
2. **Explicit `experiments` list** -- entries are appended directly. No further interpretation.
3. **Neither `$include` nor `experiments`, but meaningful keys present** -- the file itself is
   treated as a single experiment dict. A `_source` key is injected, recording the file path for
   downstream command construction.

Case (3) is the canonical form for individual model YAMLs under `experiments/models/`.

## Command construction (`_build_cmd`)

`_build_cmd` translates an experiment dict into a `cy` CLI invocation via two strategies:

- **Source-bearing train experiments** (`_source` key + `type: train`). Emits `` cy run -c <_source> ``,
  forwarding `output_dir` and `device` as overrides. `cy run` handles training followed by
  automatic testing on the best checkpoint.

- **All other experiments.** Emits `` cy <type> `` with flags derived from the dict. Eight key
  fields map to `--model`, `--dataset`, `--output-dir`, `--weights`, `--epochs`, `--batch-size`,
  `--lr`, `--device`, `--seed`, `--num-workers`, `--optimizer`. Boolean flags: `--pretrained`,
  `--no-cosine-lr`, `--no-ema`. Scalar flags (`--patience`, `--clip-grad-norm`) emit only when
  the corresponding key is present.

## Execution and log/error tracking

`run_compose` creates `<output_dir>/scheduler/` and iterates over the resolved experiment list.
For each experiment:

1. A log file is opened at `<output_dir>/scheduler/<name>.log`.
2. Per-experiment `env` overrides are merged into a copy of `os.environ`.
3. `subprocess.run` executes the command; stdout and stderr are captured to the log file.
4. On exit code 0, `_record_success` appends to `<output_dir>/scheduler/results.jsonl`:
   `{exp_name, status, log_path, output_dir, timestamp}`.
5. On non-zero exit or exception, `_record_error` appends to `<output_dir>/scheduler/errors.jsonl`:
   `{exp_name, exit_code, log_path, timestamp, [traceback]}`.

The scheduler never aborts on failure -- it continues to the next experiment and reports a
summary of `n_ok` / `n_failed` at completion.

## CLI
```bash
# Serial execution.
cy compose -c experiments/compose_all.yaml -o output/compose_all

# Parallel execution (pin GPUs via per-experiment env in YAML).
cy compose -c experiments/compose_all.yaml -o output/compose_all -p 4
```

`cy compose` is defined in `cracks_yolo/cli.py` and delegates to `compose.run_compose`.
