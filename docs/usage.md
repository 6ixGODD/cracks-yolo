# Usage

[English](usage.md) | [中文](usage.zh-CN.md)

End-to-end guide for training and evaluating tongue surface crack detection models.

## Installation

```bash
git clone <repo>
cd cracks-yolo
uv sync              # library only
uv sync --group dev  # library + CLI + test + typing + linters
```

Requires Python 3.11--3.13, PyTorch >= 2.2. Configured for CUDA 11.8 via
`pyproject.toml`. For CPU-only, comment the `[tool.uv.sources]` block. Two
console scripts are registered: `cracks-yolo` and the shorthand `cy`.

## CLI reference

| Command | Purpose |
|---------|---------|
| `cy train` | Direct training (single run, 5-fold CV). |
| `cy test`  | Standalone evaluation on a trained checkpoint. |
| `cy run`   | Single experiment from YAML; auto-tests after training. |
| `cy compose` | Batch scheduler with subprocess isolation and `$include` resolution. |

### `cy train` / `cy test`

```bash
# Train
cy train -m yolov5s_sactr -d data/CrackDetection_Augmentation.v1.yolov5pytorch \
    -o output/yolov5s_sactr -e 300 -b 64 --pretrained

# 5-fold CV (merges all splits; held-out fold = test; remaining 90/10 train/val)
cy train -m yolov5s_sactr -d data/CrackDetection_Augmentation.v1.yolov5pytorch \
    -o output/yolov5s_sactr_cv -e 300 -b 64 --pretrained \
    --cross-val --n-folds 5 --val-fraction 0.1

# Test
cy test -m yolov5s_sactr --weights output/yolov5s_sactr/weights/best.pt \
    -d data/CrackDetection_Augmentation.v1.yolov5pytorch -o output/yolov5s_sactr/test
```

`cy train` flags: `-m/--model` (ZOO key), `-d/--dataset`, `-o/--output-dir`,
`-e/--epochs` (300), `-b/--batch-size` (64), `--lr` (1e-3),
`--pretrained/--no-pretrained`, `--device` (cuda), `--seed` (42),
`-w/--num-workers` (8), `--optimizer` (adamw/sgd), `--cosine-lr/--no-cosine-lr`,
`--ema/--no-ema`, `--patience` (100), `--clip-grad-norm` (10.0), `--cross-val`,
`--n-folds`, `--val-fraction`.

`cy test` flags: `-m/--model`, `--weights`, `-d/--dataset`, `-o/--output-dir`,
`-b/--batch-size` (32), `--device`, `--seed`.

### `cy run` -- single-experiment YAML

```bash
# Full train + auto-test
cy run -c experiments/models/yolov5s.yaml -o output/yolov5s

# Skip training, evaluate an existing checkpoint
cy run -c experiments/models/yolov5s.yaml -o output/yolov5s_test \
    --test-only --weights output/yolov5s/weights/best.pt
```

When `--test-only` is set, `--weights` is required. For `type: test` YAMLs, a
`weights` key must be present. The config format:

```yaml
name: yolov5s
type: train
model: yolov5s
dataset: data/CrackDetection_Augmentation.v1.yolov5pytorch
output_dir: output/yolov5s
epochs: 300
batch_size: 64
lr: 0.001
pretrained: true
device: cuda
seed: 42
num_workers: 8
optimizer: sgd
cosine_lr: true
use_ema: true
early_stopping_patience: 100
clip_grad_norm: 10.0
```

Supported keys: `name`, `type` (`train`|`test`), `model`, `dataset`,
`output_dir`, `epochs`, `batch_size`, `lr`, `device`, `seed`, `num_workers`,
`pretrained`, `weights` (test only), `optimizer`, `cosine_lr`, `use_ema`,
`early_stopping_patience`, `clip_grad_norm`, `env` (per-experiment env dict,
e.g. `{CUDA_VISIBLE_DEVICES: "0"}`).

### `cy compose` -- batch scheduling

```bash
cy compose -c experiments/compose_all.yaml -o output/compose_all        # serial
cy compose -c experiments/compose_all.yaml -o output/compose_all -p 4   # parallel, 4 workers
```

Compose YAML aggregates experiment YAMLs via `$include`:

```yaml
scheduler:
  max_parallel: 1
  seed: 42
$include:
  - models/yolov5s.yaml
  - models/yolov5s_sac.yaml
```

Each experiment runs in an isolated subprocess. Stdout/stderr captured to
`<output_dir>/scheduler/<name>.log`. Successes in `scheduler/results.jsonl`;
failures in `scheduler/errors.jsonl`. Set per-experiment
`env: {CUDA_VISIBLE_DEVICES: "N"}` for multi-GPU pinning.

## Output directory layout

**`cy train` / `cy run` (type: train):**

```
output/<name>/
  config.yaml          Frozen training configuration
  run.log.jsonl        Structured JSONL log
  metrics.csv          Per-epoch metrics (loss, mAP, precision, recall)
  loss_curve.png       Training loss
  metric_curve.png     Validation metrics
  weights/{best.pt, last.pt}
  test/                Auto-generated (metrics.csv, per_image/*.json,
                         predictions/*.jpg, curves/{pr,roc,confusion}.png)
```

**`cy test` (standalone):** same as `test/` above, plus `run.log.jsonl`.

**`cy compose`:** `<output_dir>/scheduler/{results.jsonl, errors.jsonl, *.log}`
plus one subdirectory per experiment.

**`cy train --cross-val`:** `<output_dir>/{cv_summary.csv, cv_report.json}`
plus `fold_0/`, `fold_1/`, ... per fold.
