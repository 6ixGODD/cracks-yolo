# 日志 (`cracks_yolo.logging`)

[English](logging.md) | [中文](logging.zh-CN.md)

## 设计

`loguru` 由 `cracks_yolo.logging.configure_logger(output_dir)` 配置。安装两个 sink：

1. **JSONL 文件 sink** 位于 `{output_dir}/run.log.jsonl`——每行一个 JSON 对象，适合使用 `jq`、pandas 或任何 JSONL 感知工具进行事后分析。
2. **Stderr sink**——人类可读、带颜色，用于实时监控。

## 用法

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

`logger.bind(**record)` 模式（loguru 的标准做法）将 dict 合并到 `record["extra"]` 中。JSONL sink 将 `extra` 合并到顶层 JSON 对象中，与 `level`、`message`、`timestamp` 并列。

## 记录模式 (`cracks_yolo.logging.schema`)

所有模式都是 `TypedDict`（使用 `typing_extensions.TypedDict` 以兼容 Python 3.11 与 pydantic）。每个都带有一个 `record_type: Literal[...]` 鉴别器，以便事后查询可以按记录类型过滤。

### `TrainStepLog`——一个优化器步骤

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `record_type` | `Literal["train_step"]` | 鉴别器。 |
| `step` | `int` | 全局步数计数器。 |
| `epoch` | `int` | 当前 epoch。 |
| `total_loss` | `float` | 所有损失分量的总和。 |
| `box_loss` | `float` | 边界框回归损失。 |
| `cls_loss` | `float` | 分类损失。 |
| `obj_loss` | `float \| None` | 目标性损失（v8/v10 为 None）。 |
| `dfl_loss` | `float \| None` | 分布聚焦损失（v5/v7 为 None）。 |
| `lr` | `float` | 当前学习率。 |
| `timestamp` | `str` | ISO 8601 时间戳。 |

### `TrainEpochLog`——epoch 结束摘要

字段与 `TrainStepLog` 相同（减去 `step`），加上 `mean_*` 前缀和 `elapsed_sec`。

### `ValLog`——验证阶段

| 字段 | 类型 |
| --- | --- |
| `record_type` | `Literal["val"]` |
| `epoch` | `int` |
| `map50` | `float` |
| `map5095` | `float` |
| `per_class_ap` | `list[float]` |
| `elapsed_sec` | `float` |
| `timestamp` | `str` |

### `TestLog`——测试集评估

| 字段 | 类型 |
| --- | --- |
| `record_type` | `Literal["test"]` |
| `map50` | `float` |
| `map5095` | `float` |
| `per_class_ap` | `list[float]` |
| `precision` | `float` |
| `recall` | `float` |
| `f1` | `float` |
| `elapsed_sec` | `float` |
| `timestamp` | `str` |

### `MetricLog`——单个标量输出

| 字段 | 类型 |
| --- | --- |
| `record_type` | `Literal["metric"]` |
| `name` | `str` |
| `value` | `float` |
| `unit` | `str` |
| `timestamp` | `str` |

用于 FPS、参数量、MACs、延迟百分位数等。

### `PretrainedLoadLog`——预训练权重加载报告

| 字段 | 类型 |
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

由 pipeline 在 `from_pretrained` 之后发出，以便每次运行都记录哪些键被随机初始化（SAC/TR 层）。

## JSONL 格式

每行是一个单独的 JSON 对象。示例：

```json
{"level":"INFO","message":"step done","timestamp":"2026-06-18T10:23:45.123456","record_type":"train_step","step":0,"epoch":0,"total_loss":1.23,"box_loss":0.4,"cls_loss":0.5,"obj_loss":0.33,"dfl_loss":null,"lr":0.001}
```

## 事后查询

使用 `jq` 按记录类型过滤：

```bash
jq 'select(.record_type == "train_step")' output/run1/run.log.jsonl
```

计算每个 epoch 的平均 total_loss：

```bash
jq -s 'group_by(.epoch) | map({epoch: .[0].epoch, mean_loss: (map(.total_loss) | add / length)})' \
  output/run1/run.log.jsonl
```

或使用 pandas 加载：

```python
import pandas as pd
df = pd.read_json("output/run1/run.log.jsonl", lines=True)
train_df = df[df.record_type == "train_step"]
```
