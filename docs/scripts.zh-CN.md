# 脚本

[English](scripts.md) | [中文](scripts.zh-CN.md)

## CLI 概览

入口点为注册于 `pyproject.toml` 之 [Typer](https://typer.tiangolo.com/) 应用，具三名：

```
cracks-yolo <command> [options]
cy <command> [options]
python -m cracks_yolo <command> [options]
```

四子命令：`train`、`test`、`run`、`compose`。所有产物写入 `--output-dir`。`--help` 于任一子命令列出完整标志参考。

---

## `train`

训练一个 ZOO 模型于 YOLOv5 格式数据集。

```
cy train -m yolov5s_sac -d data/CrackDetection_Augmentation.v1.yolov5pytorch -o output/yolov5s_sac
```

| 选项 | 默认值 |
|---|---|
| `-m, --model` | *(必填)* ZOO 键 |
| `-d, --dataset` | *(必填)* 数据集根目录（含 `train/`/`valid/`/`test/`） |
| `-o, --output-dir` | *(必填)* |
| `-e, --epochs` | `300` |
| `-b, --batch-size` | `64` |
| `--lr` | `1e-3` |
| `--pretrained`/`--no-pretrained` | `--pretrained` 加载 COCO 权重 |
| `--device` | `cuda` |
| `--seed` | `42` |
| `-w, --num-workers` | `8` |
| `--optimizer` | `adamw`（`adamw`/`sgd`） |
| `--cosine-lr`/`--no-cosine-lr` | `--cosine-lr` |
| `--ema`/`--no-ema` | `--ema` |
| `--patience` | `100` 早停 epoch 数 |
| `--clip-grad-norm` | `10.0` |

产物：`run.log.jsonl`、`metrics.csv`、`loss_curve.png`、`metric_curve.png`、
`config.yaml`、`best.pt`、`analysis.json`。

---

## `test`

加载检查点，对 `test` 与 `valid` 划分执行推理，计算 COCO 指标 /
PR-ROC 曲线 / 混淆矩阵。

```
cy test -m yolov5s_sac --weights output/yolov5s_sac/best.pt \
  -d data/CrackDetection_Augmentation.v1.yolov5pytorch -o output/yolov5s_sac_test
```

| 选项 | 默认值 |
|---|---|
| `-m, --model` | *(必填)* ZOO 键 |
| `--weights` | *(必填)* 检查点 `.pt` |
| `-d, --dataset` | *(必填)* 数据集根目录 |
| `-o, --output-dir` | *(必填)* |
| `-b, --batch-size` | `32` |
| `--device` | `cuda` |
| `--seed` | `42` |

产物：`metrics.csv`、`best_predictions_test.json`、`best_predictions_valid.json`。

---

## `run`

执行一个 YAML 文件描述之实验。`type: train` 先训练，再自动对所产生之检查点测试；
`type: test` 仅运行测试。`--test-only` 跳过训练阶段。

```
cy run -c experiments/models/yolov5s.yaml -o output/yolov5s
cy run -c experiments/models/yolov5s.yaml --test-only -w output/yolov5s/best.pt
```

| 选项 | 默认值 |
|---|---|
| `-c, --config` | *(必填)* YAML 实验配置 |
| `-o, --output-dir` | 覆写 YAML `output_dir` |
| `--device` | 覆写 YAML `device` |
| `--test-only` | `False`（需 `--weights`） |
| `-w, --weights` | `--test-only` 之检查点 |

### YAML 实验配置

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

允许字段：`name`、`type`（`train`/`test`）、`model`、`dataset`、`output_dir`、
`epochs`、`batch_size`、`lr`、`pretrained`、`device`、`seed`、`num_workers`、
`optimizer`、`cosine_lr`、`use_ema`、`early_stopping_patience`、`clip_grad_norm`、
`weights`（`type: test` 时必填）。

`type: train` 时，命令自动以 `output_dir/weights/best.pt` 执行测试。

---

## `compose`

加载一个含 `$include` 指令之组合 YAML，以子进程逐个执行实验。
日志、`results.jsonl`、`errors.jsonl` 落入 `{output_dir}/scheduler/`。

```
cy compose -c experiments/compose_all.yaml -o output/compose_all -p 2
```

| 选项 | 默认值 |
|---|---|
| `-c, --config` | *(必填)* 组合 YAML |
| `-o, --output-dir` | *(必填)* |
| `-p, --max-parallel` | `1` 最大并行子进程数 |

### 组合 YAML 格式

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

`$include` 路径基于组合文件所在目录解析；被包含文件自身亦可含有 `$include`。
既无 `$include` 亦无 `experiments` 之文件视为单个实验。每实验可携带一 `env` 映射
（例 `CUDA_VISIBLE_DEVICES: "0"`），用于设置子进程环境变量。`run_compose` 返回失败计数。
