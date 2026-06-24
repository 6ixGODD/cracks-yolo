# 模型

## 架构

`cracks-yolo` 中每一模型皆为 `cracks_yolo.zoo.base.BaseModel` 之单一、自包含子类——即一具体 `nn.Module`，持有其层、其损失函数、其优化器构建逻辑及其预训练权重加载器。无抽象钩子，无插件注册表，无运行时 YAML 解析。类本身即其架构之唯一真相源。

`BaseModel` 通过 `ModelState` 强制执行三态生命周期：`UNINITIALIZED`（刚构造）、`PRETRAINED`（已加载 COCO 权重）及 `TRAINED`（已在目标数据集上微调）。其声明五项抽象方法，每子类必须实现：

| 方法 | 用途 |
|---|---|
| `train_model(config: TrainConfig) -> TrainReport` | 执行完整训练循环 |
| `inference(images: Tensor) -> list[InferenceResult]` | 将原始输出解码为 xyxy 边界框、分数及标签 |
| `save(path, torchscript, onnx)` | 将权重持久化至磁盘 |
| `load(path)` | 从检查点恢复权重 |
| `from_pretrained(cls, num_classes, **kwargs) -> BaseModel` | 从 COCO 预训练权重构建 |

两个具体子类覆盖全部模型族：

- **`UltralyticsAdapter`**——包装 ultralytics `DetectionModel` 实例以适配 YOLO 族（v3 至 v26、RT-DETR）。在构造期间于指定骨干网络索引处注入 SAC/TR 算子，并将训练委托至 ultralytics 原生 `DetectionTrainer`（或 `RTDETRTrainer`），同时保留注入后的架构。
- **`TorchvisionBase`**——包装 `torchvision.models.detection` 检测器。将标准 torchvision 训练循环（模型级 `.forward(loss_dict)`）重接为逐图像步骤，并将其异构输出格式解码为统一的 `InferenceResult`。

流水线契约为结构化而非继承式：流水线从不基于 `isinstance(model, ...)` 分支。相反，每类声明两个类级属性供流水线直接消费：`loss_parts_schema` 命名 `compute_loss` 返回之损失张量的各分量，`decode_format` 声明原始输出为基于锚框（`(B, N, nc+5)`）抑或无锚框（`(B, 4+nc, N)`）。

## ZOO 注册表

`cracks_yolo.zoo.ZOO` 为普通 `dict[str, type]`，将简称映射至模型类。其由两个子注册表组合而成：`cracks_yolo.zoo.ultralytics.ZOO` 与 `cracks_yolo.zoo.torchvision.ZOO`。流水线与脚本通过键值解析模型：

```python
from cracks_yolo.zoo import ZOO
model = ZOO["yolov5s_sactr"](num_classes=1).to("cuda")
```

## SAC 与 TR 注入

SAC（Switchable Atrous Convolution，可切换空洞卷积）与 TR（Transformer）系本工作针对舌面裂纹检测所提出的两种架构干预。

**SAC** 将 CSP 瓶颈块内标准 3×3 卷积替换为 `SAConv2d`（定义于 `cracks_yolo.ops.sac`）。每一实例学习逐像素软开关，融合两个并行空洞卷积——一者以基础膨胀率运行，另一者以三倍膨胀率运行——使网络可动态选择其感受野。该算子还在批归一化与激活之前施加前上下文聚合（全局平均池化经 1×1 卷积后广播相加）及一对称的后上下文残差。

**TR** 将整个 C3 块替换为 `C3TR`，后者将一条 CSP 分支路由通过一 `TransformerBlock`（1×1 投影 + 可学习位置嵌入 + 堆叠 `TransformerLayer` 模块，每层施加 QKV 多头自注意力的后接双层 MLP，均带残差连接）。

注入由 `cracks_yolo.zoo.ultralytics.sac_injection` 中的 `apply_sac_tr(model, sac_indices, tr_indices)` 执行。该函数遍历模型之 `model.model` `Sequential`，在给定索引处将块替换为其 SAC/TR 对应体（`C3` 变为 `C3SAC` 或 `C3TR`；`C2f` 变为 `C2fSAC`）。共享卷积权重从原始块复制至替换块；仅 SAC 特有张量（开关卷积、权重差参数、前后上下文卷积）保持随机初始化。路由元数据（`f`、`i`、`type`、`np`）得以保留，使替换后的层透明集成于前向图中。各模型族之 SAC 插入点如下：

- **YOLOv5**：骨干网络第 3、5、7 级 C3 块。TR 位于骨干网络第 9 级（SPPF 颈部输入处）。
- **YOLOv6**：骨干网络第 3、5、7、9 级 C2f 块。
- **YOLOv8/v9/v10**：骨干网络第 3、5、7、9 级 C2f 块。
- **RT-DETR**：骨干网络第 3、5、7、9 级 C2f 块。

以 `strict=False` 加载 COCO 预训练权重时，SAC/TR 层出现于缺失键报告中；其权重为随机初始化。

## 添加新模型变体

1. 在 `cracks_yolo/zoo/ultralytics/__init__.py`（YOLO 族）中添加类，或在 `cracks_yolo/zoo/torchvision/` 下新建文件（torchvision 族）。该类必须继承 `UltralyticsAdapter` 或 `TorchvisionBase`。其 `__init__` 以恰当 YAML 配置名调用 `_build_detection_model`，可选地通过 `apply_sac_tr` 注入 SAC/TR，并将全部元数据传递给 `super().__init__()`。
2. 在对应 `ZOO` 字典中以简短描述性键值注册该类（如 `"yolov8n_sac"`）。
3. 在 `tests/zoo/test_<arch>.py` 中添加测试：`(2,3,640,640)` 上前向形状、`compute_loss` 有限且产生非零梯度、`build_optimizer` 返回 `torch.optim.Optimizer`、`from_pretrained` 部分加载无错误。
4. 更新本文档。

命名约定为 `{基线}_{改进}`，后缀 `_sac` 与 `_tr`。类名中不出现损失函数或优化器缩写——此等由 `TrainConfig` 掌管。

## 可用模型族

### YOLO 族（通过 `UltralyticsAdapter`）

| 族 | 规模 | 变体 |
|---|---|---|
| YOLOv3 | -- | baseline |
| YOLOv5 | n, s, m, l, x | baseline；仅 s：`_sac`、`_tr`、`_sactr` |
| YOLOv6 | n | baseline、`_sac` |
| YOLOv8 | n, s, m, l, x | baseline；n, s：`_sac` |
| YOLOv9 | t, s, m, c, e | baseline；仅 c：`_sac` |
| YOLOv10 | n, s, m, b, l, x | baseline；仅 s：`_sac` |
| RT-DETR | r50 | baseline、`_sac` |
| YOLO11 | n, s | baseline |
| YOLO12 | n, s | baseline |
| YOLO26 | n, s | baseline |

### Torchvision 检测器（通过 `TorchvisionBase`）

| 检测器 | 骨干网络 | 范式 |
|---|---|---|
| RetinaNet | ResNet-50 | 单阶段，基于锚框，Focal Loss |
| Faster R-CNN | ResNet-50 | 双阶段，RPN + RoI 头 |
| Mask R-CNN | ResNet-50 | 双阶段 + 实例分割 |
| FCOS | ResNet-50 | 无锚框，中心度 |
| SSD300 | VGG-16 | 单阶段，多尺度锚框 |
| SSDlite320 | MobileNetV3-Large | 轻量单阶段 |

全部模型接受 `num_classes: int = 1` 与 `input_size: int = 640`。对 torchvision 检测器，预训练骨干网络（COCO、ImageNet）可通过 `from_pretrained` 使用 torchvision 权重 API 获取。对 YOLO 族，预训练 COCO 权重通过 ultralytics 模型中心获取。

## 损失函数设备同步约定

损失模块持有非 `nn.Parameter` 之内部张量（锚框、BCE 正样本权重、步幅），因此不随 `.to()` 移动。每损失函数之 `__call__` 须在入口处将 `self.device`、`self.anchors`、`self.stride` 及 BCE 子模块同步至 `preds[0].device`。参见 `cracks_yolo/losses/yolov5.py` 中的规范模式。
