# Scripts

[English](scripts.md) | [中文](scripts.zh-CN.md)

## CLI overview

The entry point is a [Typer](https://typer.tiangolo.com/) application registered in
`pyproject.toml` under two names:

```
cracks-yolo <command> [options]
cy <command> [options]
python -m cracks_yolo <command> [options]
```

Four subcommands: `train`, `test`, `run`, `compose`. All write artifacts to
`--output-dir`. `--help` on any subcommand prints the flag reference.

---

## `train`

Trains one ZOO model on a YOLOv5-format dataset.

```
cy train -m yolov5s_sac -d data/CrackDetection_Augmentation.v1.yolov5pytorch -o output/yolov5s_sac
```

| Option | Default |
|---|---|
| `-m, --model` | *(required)* ZOO key |
| `-d, --dataset` | *(required)* dataset root (contains `train/`/`valid/`/`test/`) |
| `-o, --output-dir` | *(required)* |
| `-e, --epochs` | `300` |
| `-b, --batch-size` | `64` |
| `--lr` | `1e-3` |
| `--pretrained`/`--no-pretrained` | `--pretrained` COCO weights |
| `--device` | `cuda` |
| `--seed` | `42` |
| `-w, --num-workers` | `8` |
| `--optimizer` | `adamw` (`adamw`/`sgd`) |
| `--cosine-lr`/`--no-cosine-lr` | `--cosine-lr` |
| `--ema`/`--no-ema` | `--ema` |
| `--patience` | `100` early-stop epochs |
| `--clip-grad-norm` | `10.0` |

Artifacts: `run.log.jsonl`, `metrics.csv`, `loss_curve.png`, `metric_curve.png`,
`config.yaml`, `best.pt`, `analysis.json`.

---

## `test`

Loads a checkpoint, runs inference on `test` + `valid` splits, computes COCO metrics /
PR-ROC curves / confusion matrices.

```
cy test -m yolov5s_sac --weights output/yolov5s_sac/best.pt \
  -d data/CrackDetection_Augmentation.v1.yolov5pytorch -o output/yolov5s_sac_test
```

| Option | Default |
|---|---|
| `-m, --model` | *(required)* ZOO key |
| `--weights` | *(required)* checkpoint `.pt` |
| `-d, --dataset` | *(required)* dataset root |
| `-o, --output-dir` | *(required)* |
| `-b, --batch-size` | `32` |
| `--device` | `cuda` |
| `--seed` | `42` |

Artifacts: `metrics.csv`, `best_predictions_test.json`, `best_predictions_valid.json`.

---

## `run`

Executes one experiment from a YAML file. `type: train` trains then auto-tests on the
resulting checkpoint; `type: test` runs test only. `--test-only` skips training.

```
cy run -c experiments/models/yolov5s.yaml -o output/yolov5s
cy run -c experiments/models/yolov5s.yaml --test-only -w output/yolov5s/best.pt
```

| Option | Default |
|---|---|
| `-c, --config` | *(required)* YAML experiment config |
| `-o, --output-dir` | override YAML `output_dir` |
| `--device` | override YAML `device` |
| `--test-only` | `False` (requires `--weights`) |
| `-w, --weights` | checkpoint for `--test-only` |

### YAML experiment config

```yaml
name: yolov5s
type: train                       # train | test
model: yolov5s
dataset: data/CrackDetection_Augmentation.v1.yolov5pytorch
output_dir: output/yolov5s
epochs: 300; batch_size: 64; lr: 0.001
pretrained: true; device: cuda; seed: 42; num_workers: 8
optimizer: sgd; cosine_lr: true; use_ema: true
early_stopping_patience: 100; clip_grad_norm: 10.0
```

Acceptable fields: `name`, `type` (`train`/`test`), `model`, `dataset`, `output_dir`,
`epochs`, `batch_size`, `lr`, `pretrained`, `device`, `seed`, `num_workers`,
`optimizer`, `cosine_lr`, `use_ema`, `early_stopping_patience`, `clip_grad_norm`,
`weights` (required for `type: test`).

For `type: train` the command auto-tests on `output_dir/weights/best.pt`.

---

## `compose`

Loads a compose YAML with `$include` directives, executes each experiment as a
subprocess. Logs, `results.jsonl`, and `errors.jsonl` land in `{output_dir}/scheduler/`.

```
cy compose -c experiments/compose_all.yaml -o output/compose_all -p 2
```

| Option | Default |
|---|---|
| `-c, --config` | *(required)* compose YAML |
| `-o, --output-dir` | *(required)* |
| `-p, --max-parallel` | `1` max parallel subprocesses |

### Compose YAML format

```yaml
scheduler:
  max_parallel: 1
  seed: 42
$include:
  - models/yolov5s.yaml
  - models/yolov5s_sac.yaml
  - models/yolov8n.yaml
  - models/retinanet_r50.yaml
```

`$include` paths resolve relative to the compose file's directory; included files may
themselves carry `$include`. A file with neither `$include` nor `experiments` is treated
as a single experiment. Each experiment may carry an `env` map (e.g.,
`CUDA_VISIBLE_DEVICES: "0"`) for subprocess environment. `run_compose` returns the
count of failures.
