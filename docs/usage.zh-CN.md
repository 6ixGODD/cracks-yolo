# 使用说明

[English](usage.md) | [中文](usage.zh-CN.md)

舌面裂纹检测模型的端到端训练与评估指南。

## 安装

```bash
git clone <repo>
cd cracks-yolo
uv sync              # 仅库
uv sync --group dev  # 库 + CLI + 测试 + 类型检查 + 代码检查工具
```

依赖 Python 3.11--3.13，PyTorch >= 2.2。`pyproject.toml` 已配置 CUDA 11.8
源。若仅需 CPU 环境，注释 `[tool.uv.sources]` 块。注册两个控制台脚本：
`cracks-yolo` 及其简写 `cy`。

## CLI 参考

| 命令 | 用途 |
|---------|---------|
| `cy train` | 直接训练（单次运行、5 折交叉验证）。 |
| `cy test`  | 对已训练检查点进行独立评估。 |
| `cy run`   | 从 YAML 运行单次实验；训练完毕后自动测试。 |
| `cy compose` | 批量调度器，具备子进程隔离与 `$include` 解析能力。 |

### `cy train` / `cy test`

```bash
# 训练
cy train -m yolov5s_sactr -d data/CrackDetection_Augmentation.v1.yolov5pytorch \
    -o output/yolov5s_sactr -e 300 -b 64 --pretrained

# 5 折交叉验证（合并所有划分；留出折 = 测试集；其余 90/10 划分为训练/验证）
cy train -m yolov5s_sactr -d data/CrackDetection_Augmentation.v1.yolov5pytorch \
    -o output/yolov5s_sactr_cv -e 300 -b 64 --pretrained \
    --cross-val --n-folds 5 --val-fraction 0.1

# 测试
cy test -m yolov5s_sactr --weights output/yolov5s_sactr/weights/best.pt \
    -d data/CrackDetection_Augmentation.v1.yolov5pytorch -o output/yolov5s_sactr/test
```

`cy train` 标志：`-m/--model`（ZOO 键），`-d/--dataset`，`-o/--output-dir`，
`-e/--epochs`（默认 300），`-b/--batch-size`（默认 64），`--lr`（默认 1e-3），
`--pretrained/--no-pretrained`，`--device`（默认 cuda），`--seed`（默认 42），
`-w/--num-workers`（默认 8），`--optimizer`（adamw/sgd），
`--cosine-lr/--no-cosine-lr`，`--ema/--no-ema`，`--patience`（默认 100），
`--clip-grad-norm`（默认 10.0），`--cross-val`，`--n-folds`，`--val-fraction`。

`cy test` 标志：`-m/--model`，`--weights`，`-d/--dataset`，`-o/--output-dir`，
`-b/--batch-size`（默认 32），`--device`，`--seed`。

### `cy run` —— 单实验 YAML

```bash
# 完整训练 + 自动测试
cy run -c experiments/models/yolov5s.yaml -o output/yolov5s

# 跳过训练，评估已有检查点
cy run -c experiments/models/yolov5s.yaml -o output/yolov5s_test \
    --test-only --weights output/yolov5s/weights/best.pt
```

设置 `--test-only` 时须同时提供 `--weights`。对于 `type: test` 的 YAML，
`weights` 键为必填项。配置文件格式如下：

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

支持的键：`name`，`type`（`train`|`test`），`model`，`dataset`，`output_dir`，
`epochs`，`batch_size`，`lr`，`device`，`seed`，`num_workers`，`pretrained`，
`weights`（仅测试），`optimizer`，`cosine_lr`，`use_ema`，`early_stopping_patience`，
`clip_grad_norm`，`env`（每实验环境变量字典，例如 `{CUDA_VISIBLE_DEVICES: "0"}`）。

### `cy compose` —— 批量调度

```bash
cy compose -c experiments/compose_all.yaml -o output/compose_all        # 串行
cy compose -c experiments/compose_all.yaml -o output/compose_all -p 4   # 并行，4 个工作进程
```

Compose YAML 通过 `$include` 聚合实验 YAML：

```yaml
scheduler:
  max_parallel: 1
  seed: 42
$include:
  - models/yolov5s.yaml
  - models/yolov5s_sac.yaml
```

每个实验在独立的子进程中运行。stdout/stderr 捕获至
`<output_dir>/scheduler/<name>.log`。成功记录于 `scheduler/results.jsonl`；
失败记录于 `scheduler/errors.jsonl`。设置每实验的
`env: {CUDA_VISIBLE_DEVICES: "N"}` 以支持多 GPU 绑定。

## 输出目录结构

**`cy train` / `cy run` (type: train)：**

```
output/<name>/
  config.yaml          冻结的训练配置
  run.log.jsonl        结构化 JSONL 日志
  metrics.csv          逐 epoch 指标（loss、mAP、precision、recall）
  loss_curve.png       训练损失曲线
  metric_curve.png     验证指标曲线
  weights/{best.pt, last.pt}
  test/                自动生成（metrics.csv、per_image/*.json、
                       predictions/*.jpg、curves/{pr,roc,confusion}.png）
```

**`cy test`（独立评估）：** 与上述 `test/` 目录相同，外加 `run.log.jsonl`。

**`cy compose`：** `<output_dir>/scheduler/{results.jsonl, errors.jsonl, *.log}`
外加每个实验对应一个子目录。

**`cy train --cross-val`：** `<output_dir>/{cv_summary.csv, cv_report.json}`
外加各折的 `fold_0/`、`fold_1/` 等目录。
