# 日志系统 (`cracks_yolo.logging`)

## 架构

本模块封装 `loguru`，由 `configure_logger(output_dir)` 安装两路输出：

1. **JSONL 文件输出** — 写入 `{output_dir}/run.log.jsonl`，每行一个 JSON 对象。
2. **标准错误输出** — 带色彩、人类可读格式，用于实时监控。

JSONL 输出由 `_make_jsonl_sink(path)` 构造，以追加模式打开目标文件，
将每条记录序列化为单行 JSON。`logger.bind(**record)` 传入的 `extra` 字典
与 `level`、`message`、`timestamp` 一并合并至顶层负载。

## 记录模式

所有日志记录类型均为 `cracks_yolo.logging.schema` 中定义的 `TypedDict` 子类。
每条记录均携带必选的 `record_type: Literal[...]` 鉴别字段，便于事后筛选。

| 记录类型 | `record_type` | 用途 |
|---|---|---|
| `TrainStepLog` | `"train_step"` | 单步优化器记录：`step`、`epoch`、`total_loss`、`box_loss`、`cls_loss`、`obj_loss`（可为空）、`dfl_loss`（可为空）、`lr`、`timestamp`。 |
| `TrainEpochLog` | `"train_epoch"` | 周次结束汇总：`epoch`、各损失均值字段、`lr`、`elapsed_sec`、`timestamp`。 |
| `ValLog` | `"val"` | 验证阶段：`epoch`、`map50`、`map5095`、`per_class_ap`、`elapsed_sec`、`timestamp`。 |
| `TestLog` | `"test"` | 测试集评估：`map50`、`map5095`、`per_class_ap`、`precision`、`recall`、`f1`、`elapsed_sec`，以及效率字段（`n_images`、`fps_mean`、`latency_mean_ms`、`gflops`、`n_parameters`）、`timestamp`。 |
| `MetricLog` | `"metric"` | 任意标量输出：`name`、`value`、`unit`、`timestamp`。用于 FPS、参数量、MACs、延迟百分位数。 |
| `PretrainedLoadLog` | `"pretrained_load"` | 权重加载审计：`key`、`url`、`cached`、`matched_count`、`missing_count`、`unexpected_count`、`missing_keys`、`unexpected_keys`、`timestamp`。由 `from_pretrained` 执行后发出。 |

联合类型 `LogRecord` 定义为
`TrainStepLog | TrainEpochLog | ValLog | TestLog | MetricLog | PretrainedLoadLog`。

## 可空字段

无锚框架构（v8、v10）不含目标置信度损失，故 `obj_loss` 为 `None`。
基于锚框的架构（v5、v7）使用 IoU 损失替代 DFL，故 `dfl_loss` 为 `None`。
`TestLog` 中的效率字段在 `measure_efficiency=False` 时均为零。

## 输出位置

`configure_logger(output_dir)` 若目录不存在则自动创建，并在其中写入 `run.log.jsonl`。
例：`configure_logger(Path("output/run1"))` 生成 `output/run1/run.log.jsonl`。

## 用法

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

`logger.bind(**record)` 模式将字典合并至 `record["extra"]`（loguru 标准惯用法）。
JSONL 输出随即将 `extra` 中的键提升至顶层 JSON 对象。

## 事后查询

按记录类型筛选，使用 `jq`：

```bash
jq 'select(.record_type == "train_step")' output/run1/run.log.jsonl
```

按周次计算 `total_loss` 均值：

```bash
jq -s 'group_by(.epoch) | map({epoch: .[0].epoch, mean_loss: (map(.total_loss)|add/length)})' \
  output/run1/run.log.jsonl
```

或使用 pandas：

```python
import pandas as pd
df = pd.read_json("output/run1/run.log.jsonl", lines=True)
df[df.record_type == "train_step"].groupby("epoch")["total_loss"].mean()
```
