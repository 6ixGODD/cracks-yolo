# Experiment scheduler

`scripts/schedule_experiments.py` runs a YAML-defined batch of experiments
with subprocess isolation, error capture, and retry mode.

## YAML format

```yaml
# experiments.yaml
scheduler:
  max_parallel: 1              # serial by default; >1 spawns subprocesses
  retry_failed: false          # set true to auto-retry from errors.jsonl
  seed: 42

experiments:
  - name: yolov5s_baseline
    type: train                # train | test | cross_val
    model: yolov5s
    dataset: data/CrackDetection_Augmentation.v1.yolov5pytorch
    epochs: 100
    batch_size: 32
    lr: 0.001
    output_dir: output/yolov5s_baseline
    seed: 42

  - name: yolov5s_sactr_cv
    type: cross_val
    model: yolov5s_sactr
    dataset: data/CrackDetection_Augmentation.v1.yolov5pytorch
    n_folds: 5
    epochs: 100
    batch_size: 32
    output_dir: output/yolov5s_sactr_cv

  - name: yolov5s_sactr_test
    type: test
    model: yolov5s_sactr
    weights: output/yolov5s_sactr_cv/best.pt
    dataset: data/CrackDetection_Augmentation.v1.yolov5pytorch/test
    output_dir: output/yolov5s_sactr_test
```

## Behavior

- **Subprocess isolation**: each experiment runs in a fresh
  `subprocess.Popen` so a crash is contained. Stdout/stderr captured to
  `output/scheduler/<exp_name>.log`.
- **Error capture**: on any exception, append to
  `output/scheduler/errors.jsonl` with `{exp_name, type, config, traceback,
  timestamp}`. Continue to the next experiment — never abort the batch.
- **Success capture**: append to `output/scheduler/results.jsonl` with
  `{exp_name, status, elapsed_sec, output_dir, exit_code}`.
- **Parallel execution**: `max_parallel > 1` spawns up to N subprocesses
  concurrently. Use with `CUDA_VISIBLE_DEVICES` (set per-experiment in the
  YAML via `env: {CUDA_VISIBLE_DEVICES: "0"}`) to distribute across GPUs.
  Single-GPU machines should keep `max_parallel: 1` — multiple experiments
  sharing one GPU OOMs more often than it speeds up the batch.

## Retry mode

Failed experiments can be retried without re-running the whole batch:

```bash
python -m scripts.schedule_experiments \
    --retry-failed output/scheduler/errors.jsonl \
    --output-dir output/scheduler_retry
```

The scheduler:
1. Reads `errors.jsonl`.
2. Generates a new YAML `retry_from_errors.yaml` with only the failed experiments.
3. Reruns them.

This is the intended iterative workflow for the 7-day rented-GPU-server
scenario: launch a large batch → inspect `errors.jsonl` → fix the bad configs
(or library issues) → retry-failed → repeat until `errors.jsonl` is empty.

## CLI

```bash
# Run a fresh batch.
python -m scripts.schedule_experiments --config experiments.yaml --output-dir output/scheduler

# Retry failed experiments.
python -m scripts.schedule_experiments --retry-failed output/scheduler/errors.jsonl --output-dir output/scheduler_retry
```

## Smoke test

A 3-experiment smoke test config is at `experiments_smoke.yaml` (one good
train, one with a bad model key to trigger error capture, one good test).
Verified behavior:
- Good experiments → `results.jsonl` entries + per-experiment logs.
- Bad experiment → `errors.jsonl` entry with config + traceback + timestamp.
- Retry mode picks up the bad one and reruns it (after fixing the config).
