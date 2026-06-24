# 实验调度器

[English](scheduler.md) | [中文](scheduler.zh-CN.md)

`cracks_yolo.pipeline.compose` 以子进程隔离方式执行 YAML 定义的批量实验。每个实验解析为一条
`cy run` 或 `cy test` 调用；调度器串行化执行，捕获逐实验日志，并将结果记录为 JSONL。

## YAML 格式

编排文件通过 `$include` 聚合若干独立实验 YAML。每个被包含的文件是一个实验字典（亦接受显式
`experiments` 列表）。顶层 `scheduler` 块仅承载元数据，运行时不予消费。

```yaml
# experiments/compose_all.yaml
scheduler:
  max_parallel: 1
  seed: 42
$include:
  - models/yolov5s.yaml
  - models/yolov5s_sac.yaml
```

独立实验 YAML（`experiments/models/yolov5s.yaml`）：

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

支持的字段：`name`、`type`（`train`|`test`）、`model`、`dataset`、`output_dir`、`epochs`、
`batch_size`、`lr`、`device`、`seed`、`num_workers`、`pretrained`、`weights`（仅 test）、
`optimizer`、`cosine_lr`、`use_ema`、`early_stopping_patience`、`clip_grad_norm` 及 `env`
（逐实验环境变量字典，如 `{CUDA_VISIBLE_DEVICES: "0"}`）。

## `$include` 解析（`_load_config`）

`_load_config` 递归解析 `$include`。包含路径相对于声明该路径的 YAML 文件。三种情形决定贡献方式：

1. **存在 `$include`** — 递归加载所列路径；所有子项中的实验汇总累积。父级 `scheduler` 块被丢弃。
2. **存在显式 `experiments` 列表** — 条目直接追加，不做进一步解释。
3. **既无 `$include` 亦无 `experiments`，但存在有意义的键** — 文件本身视为单个实验字典。自动注入
   `_source` 键，记录文件路径以供下游命令构建。

情形 (3) 是 `experiments/models/` 下独立模型 YAML 的规范形式。

## 命令构建（`_build_cmd`）

`_build_cmd` 通过两种策略将实验字典翻译为 `cy` CLI 调用：

- **含 `_source` 的训练实验**（`_source` 键 + `type: train`）。生成 `` cy run -c <_source> ``，
  转发 `output_dir` 和 `device` 作为覆盖参数。`cy run` 先完成训练，随后在最佳检查点上自动测试。

- **其他实验。** 生成 `` cy <type> `` 并从字典派生标志。八个关键字段映射到 `--model`、
  `--dataset`、`--output-dir`、`--weights`、`--epochs`、`--batch-size`、`--lr`、`--device`、
  `--seed`、`--num-workers`、`--optimizer`。布尔标志：`--pretrained`、`--no-cosine-lr`、
  `--no-ema`。标量标志（`--patience`、`--clip-grad-norm`）仅在对应键存在时发出。

## 执行与日志/错误追踪

`run_compose` 创建 `<output_dir>/scheduler/` 并遍历已解析的实验列表。对每个实验：

1. 打开日志文件 `<output_dir>/scheduler/<name>.log`。
2. 将逐实验的 `env` 覆盖合并到 `os.environ` 副本中。
3. `subprocess.run` 执行命令；stdout 与 stderr 捕获至日志文件。
4. 退出码为 0 时，`_record_success` 追加至 `<output_dir>/scheduler/results.jsonl`：
   `{exp_name, status, log_path, output_dir, timestamp}`。
5. 非零退出或异常时，`_record_error` 追加至 `<output_dir>/scheduler/errors.jsonl`：
   `{exp_name, exit_code, log_path, timestamp, [traceback]}`。

调度器在失败时绝不中止——继续执行下一实验，并在完成时报告 `n_ok` / `n_failed` 汇总。

## CLI

```bash
# 串行执行。
cy compose -c experiments/compose_all.yaml -o output/compose_all

# 并行执行（通过 YAML 中逐实验 env 绑定 GPU）。
cy compose -c experiments/compose_all.yaml -o output/compose_all -p 4
```

`cy compose` 定义于 `cracks_yolo/cli.py`，委托至 `compose.run_compose`。
