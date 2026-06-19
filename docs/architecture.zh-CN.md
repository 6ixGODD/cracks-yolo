# 架构

[English](architecture.md) | [中文](architecture.zh-CN.md)

## 设计理念

`cracks_yolo` 是一个面向**舌面裂纹检测**的自包含 PyTorch 模型库。三个原则指导着每一个设计决策：

1. **一个模型 = 一个 `nn.Module` 类。** 每个模型库模型拥有自己的层、损失模块、优化器构建器和预训练权重加载器。没有带抽象钩子的抽象基类，没有插件注册表，没有组件解耦。长类名（`YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU`）本身就是文档。
2. **管道合约是结构性的，而非继承性的。** `DetectorModel` 是一个 `typing.Protocol`（见 `cracks_yolo/zoo/base.py`）。管道仅依赖 Protocol，从不依赖 `isinstance(model, ...)`。添加新模型变体永远不会触及管道代码。
3. **YAML 是备忘录，而非运行时产物。** 仓库根目录下的 `yolov5s-sac-tr.yml` 记录目标架构供人类阅读。没有代码会读取它。

本库避免运行时 YAML 解析和外部框架耦合。`deps/{yolov5,yolov7,ultralytics,yolov9}` 中的 vendored 目录树**仅作为移植参考**，运行时从不导入。

## 包布局

```
cracks_yolo/
  ops/         # Conv、CSP、transformer、检测头、SAC/TR、YOLOv9 算子。纯 nn.Modules。
  losses/      # ComputeLoss (v5)、ComputeLossOTA (v7)、v8DetectionLoss、E2ELoss (v10)。
  zoo/         # 26 个模型类（YOLOv5/7/8/9/10 + SAC/TR 变体 + 6 个 torchvision 检测器）。
               # base.py = DetectorModel Protocol + PretrainedSpec。
  weights/     # load_pretrained：下载、键重映射、strict=False 加载 + LoadReport。
  logging/     # loguru JSONL sink + TypedDict 日志记录模式。
  metrics/     # COCOMetricsCalculator + PR/ROC/confusion + 配对 t 检验/Wilcoxon/bootstrap CI。
  pipeline/    # TrainPipelineImpl / TestPipelineImpl / run_cross_validation / compare_models_cross_val。
  dataset/     # YOLOSource、COCOSource、DetectionDataset（torch.utils.data.Dataset）、transforms、yolo<->coco 转换。
  viz/         # loss/metric/PR/ROC 曲线、混淆矩阵、Grad-CAM、数据集分布图。
  analysis/    # DatasetAnalysisReport（多样性指标）、ModelAnalysisReport（params/MACs/latency/VRAM）。
scripts/       # CLI：train、test、convert_dataset、heatmap、analyze_dataset、analyze_model、
               # schedule_experiments（YAML 驱动的子进程调度器 + 重试模式）、compare_models。
```

## 为什么没有抽象基类？

抽象基类将每个模型耦合到共享的继承链上。添加一个需要不同 forward 签名的模型（例如 v10 的双 one2many/one2one 头）要么需要扩宽 ABC（破坏每个其他模型），要么需要重写每个方法（使 ABC 失去意义）。

`typing.Protocol` 将合约与实现解耦。管道依赖 Protocol；模型独立地满足它。结构性子类型意味着模型类甚至不需要声明它满足 Protocol，只需要有正确的方法和属性。`DetectorModel` 上的 `@runtime_checkable` 装饰器允许测试断言 `isinstance(model, DetectorModel)` 而不强制建立子类关系。

## DetectorModel 合约

每个模型库模型满足：

```python
class DetectorModel(Protocol):
    input_size: int
    num_classes: int
    class_names: list[str]
    stride: torch.Tensor
    pretrained_spec: PretrainedSpec | None

    def forward(self, x): ...  # train：原始头输出；eval：解码后
    def compute_loss(self, preds, targets, imgs=None) -> tuple[Tensor, Tensor]: ...
    def decode(self, preds) -> torch.Tensor: ...
    @classmethod
    def from_pretrained(cls, num_classes, weights_dir=None, strict=False) -> DetectorModel: ...
    def build_optimizer(self) -> torch.optim.Optimizer: ...
```

管道只调用这些方法。修改模型的内部结构永远不会触及管道代码。

## 管道合约

```python
class TrainPipeline(Protocol):
    def run(self, model: DetectorModel, train_loader, val_loader, cfg: TrainConfig) -> TrainReport: ...

class TestPipeline(Protocol):
    def run(self, model: DetectorModel, test_loader, cfg: TestConfig) -> TestReport: ...
```

具体的 `TrainPipelineImpl` 和 `TestPipelineImpl` 实现了这些 Protocol，底层使用 pydantic 配置（`TrainConfig`、`TestConfig`）。

## 日志记录

由 `cracks_yolo.logging.configure(output_dir)` 配置的 `loguru`。将 JSONL 写入 `{output_dir}/run.log.jsonl` 以及一个人类可读的 stderr sink。日志记录模式是 `cracks_yolo.logging.schema` 中的 `TypedDict`：

- `TrainStepLog` -- 一个优化器步骤
- `TrainEpochLog` -- 周期结束摘要
- `ValLog` -- 验证过程
- `TestLog` -- 测试集评估
- `MetricLog` -- 单次标量输出
- `PretrainedLoadLog` -- 预训练权重加载报告

每条记录携带一个 `record_type: Literal[...]` 鉴别器，以便事后查询可以按记录类型过滤。用法：

```python
from cracks_yolo.logging.configure import configure_logger
from loguru import logger
configure_logger(output_dir=Path("output/run1"))
logger.bind(**{"record_type": "train_step", "step": 0, ...}).info("step done")
```

## Protocol 自描述

每个模型库类声明两个类属性，管道读取它们而非按类名分支：

- `loss_parts_schema: tuple[str, ...]` -- `compute_loss` 返回的 `parts` 张量中每个条目的名称。v5/v7 = `("box","cls","obj")`；v8/v9/v10 = `("box","cls","dfl")`；torchvision 包装器 = `("total","cls","box_reg","rpn_box_reg")`。
- `decode_format: str` -- `"anchor_free"`（v8/v9/v10：`(B, 4+nc, N)`）或 `"anchor_based"`（v5/v7 + torchvision 包装器：`(B, N, nc+5)` 或 `(B, N_max, 6)`）。

## 损失设备同步约定

模型在 CPU 上构造，然后通过管道中的 `model.to(device)` 移至 CUDA。损失模块持有内部张量（anchors、BCE 正权重、stride），这些不是 `nn.Parameter`，不会随 `.to()` 移动。每个损失的 `__call__` 必须在入口处将 `self.device`、`self.anchors`、`self.stride` 和 BCE 子模块同步到 `preds[0].device`。参见 `cracks_yolo/losses/yolov5.py` 的规范模式。
