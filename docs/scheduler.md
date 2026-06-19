# Experiment scheduler

[English](scheduler.md) | [中文](scheduler.zh-CN.md)

`scripts/schedule_experiments.py` runs a YAML-defined batch of experiments with subprocess isolation, error capture, and retry mode.

## Configuration directory

Two ready-to-run YAML configs live in [`experiments/`](../experiments/):

- **`experiments/all_models_direct.yaml`** — 26 models × 2 experiments (train + test) = 52 experiments. Train on `train` split, validate on `valid` split during training, then test `best.pt` on the held-out `test` split.
- **`experiments/all_models_cv5.yaml`** — 26 models × 1 cross-validation experiment = 26 experiments. CV mode merges train+valid+test into one pool, runs 5-fold CV (held-out fold = test, remaining 90/10 train/val).

See [`experiments/README.md`](../experiments/README.md) for batch-size tuning and multi-GPU instructions.

## YAML format

```yaml
# experiments/my_sweep.yaml
scheduler:
  max_parallel: 1              # serial by default; >1 spawns subprocesses
  seed: 42

experiments:
  - name: yolov5s_baseline
    type: train                # train | test (cross-val is `type: train` + cross_val: true)
    model: yolov5s
    dataset: data/CrackDetection_Augmentation.v1.yolov5pytorch
    epochs: 100
    batch_size: 32
    lr: 0.001
    output_dir: output/yolov5s_baseline
    seed: 42
    pretrained: true           # load official COCO weights (strict=False)

  - name: yolov5s_sactr_cv
    type: train
    model: yolov5s_sactr
    dataset: data/CrackDetection_Augmentation.v1.yolov5pytorch
    cross_val: true
    n_folds: 5
    val_fraction: 0.1          # 10% of per-fold training pool → val (backprop)
    epochs: 100
    batch_size: 32
    output_dir: output/yolov5s_sactr_cv

  - name: yolov5s_sactr_test
    type: test
    model: yolov5s_sactr
    weights: output/yolov5s_sactr_cv/best.pt
    dataset: data/CrackDetection_Augmentation.v1.yolov5pytorch
    split: test
    output_dir: output/yolov5s_sactr_test
```

### Per-experiment fields

| Field | Type | Applies to | Description |
| --- | --- | --- | --- |
| `name` | str | all | Unique experiment identifier (used in log filenames). |
| `type` | `train` \| `test` | all | `train` runs `scripts.train`; `test` runs `scripts.test`. CV mode is `type: train` with `cross_val: true`. |
| `model` | str | all | ZOO key (e.g. `yolov5s_sactr`). |
| `dataset` | path | all | YOLO dataset root containing `data.yaml`. |
| `output_dir` | path | all | Where artifacts land. |
| `epochs` | int | train | Training epochs. |
| `batch_size` | int | all | Per-GPU batch size. |
| `lr` | float | train | Learning rate. YOLO=0.01, torchvision=0.0001. |
| `input_size` | int | all | Model input size (640 default; 300 for SSD300, 320 for SSDlite320). |
| `device` | str | all | `cuda` or `cpu`. |
| `seed` | int | all | Reproducibility seed. |
| `num_workers` | int | all | Dataloader workers. |
| `pretrained` | bool | train | Load official COCO weights via `from_pretrained` (`strict=False`). |
| `cross_val` | bool | train | Run N-fold CV instead of a single train/val split. |
| `n_folds` | int | train (CV) | Number of CV folds. |
| `val_fraction` | float | train (CV) | Fraction of per-fold training pool carved out as val. Default 0.1. |
| `no_amp` | bool | train | Disable mixed-precision training. |
| `weights` | path | test | Path to `best.pt` for testing. |
| `split` | str | test | Dataset split to evaluate (`test` default). |
| `env` | dict | all | Per-experiment env overrides (e.g. `{CUDA_VISIBLE_DEVICES: "0"}`). |

## Behavior

- **Subprocess isolation**: each experiment runs in a fresh `subprocess.Popen` so a crash is contained. Stdout/stderr captured to `<output-dir>/scheduler/<exp_name>.log`.
- **Error capture**: on any exception (or non-zero exit), append to `<output-dir>/scheduler/errors.jsonl` with `{exp_name, type, config, exit_code, log_path, traceback, timestamp}`. Continue to the next experiment — never abort the batch.
- **Success capture**: append to `<output-dir>/scheduler/results.jsonl` with `{exp_name, status, exit_code, output_dir, elapsed_sec, timestamp}`.
- **Parallel execution**: `max_parallel > 1` uses a `ThreadPoolExecutor` to spawn up to N subprocesses concurrently. Each subprocess is a real OS process (the GIL is released while we wait on it), so true parallelism is achieved. GPU pinning is via per-experiment `env: {CUDA_VISIBLE_DEVICES: "N"}` in the YAML. Single-GPU machines should keep `max_parallel: 1` — multiple experiments sharing one GPU OOMs more often than it speeds up the batch.

## Retry mode

Failed experiments can be retried without re-running the whole batch:

```bash
python -m scripts.schedule_experiments \
    --retry-failed output/all_models_direct/scheduler/errors.jsonl \
    --output-dir output/all_models_direct_retry
```

The scheduler:
1. Reads `errors.jsonl`.
2. Generates `retry_from_errors.yaml` with only the failed experiments.
3. Backs up the original `errors.jsonl` to `errors.bak.jsonl`.
4. Reruns the failed experiments.

The iterative workflow: launch a large batch, inspect `errors.jsonl`, fix the bad configs (or library issues), retry-failed, repeat until `errors.jsonl` is empty.

## CLI

```bash
# Run a fresh batch.
python -m scripts.schedule_experiments \
    --config experiments/all_models_direct.yaml \
    --output-dir output/all_models_direct

# 5-fold CV sweep.
python -m scripts.schedule_experiments \
    --config experiments/all_models_cv5.yaml \
    --output-dir output/all_models_cv5

# Retry failed experiments.
python -m scripts.schedule_experiments \
    --retry-failed output/all_models_direct/scheduler/errors.jsonl \
    --output-dir output/all_models_direct_retry
```
