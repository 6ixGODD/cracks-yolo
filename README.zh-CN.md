# cracks-yolo

[English](README.md) | [中文](README.zh-CN.md)

面向舌面裂纹检测的自包含 PyTorch 检测模型库。涵盖 YOLOv3/v5/v6/v8/v9/v10/v11/v12/v26、
RT-DETR 及六种 torchvision 基线（RetinaNet、Faster R-CNN、Mask R-CNN、FCOS、SSD300、
SSDlite320），共计 45 个模型。每个模型均为显式 `nn.Module` 子类，内聚其层、损失函数、
优化器构建器与预训练权重加载器。

## 快速开始

```bash
pip install -e .
```

```bash
# 单实验：训练后自动在最佳检查点上测试。
cy run -c experiments/models/yolov5s_sactr.yaml

# 或使用完整 CLI 标志：
cy train -m yolov8s -d data/dataset -o output/run1 -e 300 -b 64 --pretrained
cy test -m yolov8s --weights output/run1/weights/best.pt -d data/dataset -o output/run1/test

# 子进程隔离的批量调度。
cy compose -c experiments/compose_all.yaml -o output/all_models -p 2
```

## CLI

| 命令 | 用途 |
| --- | --- |
| `cy train` | 训练单个模型，完整超参数控制。 |
| `cy test` | 在测试集与验证集上评估已训练检查点。 |
| `cy run` | 从 YAML 配置运行一个实验（训练，随后自动测试）。 |
| `cy compose` | 从支持 `$include` 的 compose YAML 批量调度实验。 |

`cy train` 关键标志：

| 标志 | 默认值 | 说明 |
| --- | --- | --- |
| `-m, --model` | （必填） | ZOO 键，如 `yolov8s_sac` |
| `-d, --dataset` | （必填） | 数据集根目录路径 |
| `-o, --output-dir` | （必填） | 输出目录 |
| `-e, --epochs` | 300 | 训练轮数 |
| `-b, --batch-size` | 64 | 批次大小 |
| `--lr` | 1e-3 | 学习率 |
| `--pretrained / --no-pretrained` | `--pretrained` | 加载 COCO 权重 |
| `--optimizer` | adamw | `adamw` 或 `sgd` |
| `--cosine-lr / --no-cosine-lr` | `--cosine-lr` | 余弦学习率调度 |
| `--ema / --no-ema` | `--ema` | 指数移动平均 |
| `--patience` | 100 | 早停耐心值（轮数） |
| `--device` | cuda | `cuda` 或 `cpu` |
| `--seed` | 42 | 随机种子 |

## 模型库

`cracks_yolo.zoo.ZOO` 中 45 个显式类，每个硬编码其架构配置、预训练资产、
SAC/TR 注入索引及解码格式。

### YOLO 系列（39 个模型）

| 系列 | 键 | 数量 | SAC 变体 |
| --- | --- | --- | --- |
| YOLOv3 | `yolov3` | 1 | -- |
| YOLOv5 | `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x` | 5 | `yolov5s_sac`, `yolov5s_tr`, `yolov5s_sactr` |
| YOLOv6 | `yolov6n` | 1 | `yolov6n_sac` |
| YOLOv8 | `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x` | 5 | `yolov8n_sac`, `yolov8s_sac` |
| YOLOv9 | `yolov9t`, `yolov9s`, `yolov9m`, `yolov9c`, `yolov9e` | 5 | `yolov9c_sac` |
| YOLOv10 | `yolov10n`, `yolov10s`, `yolov10m`, `yolov10b`, `yolov10l`, `yolov10x` | 6 | `yolov10s_sac` |
| RT-DETR | `rtdetr_r50` | 1 | `rtdetr_r50_sac` |
| YOLO11 | `yolo11n`, `yolo11s` | 2 | -- |
| YOLO12 | `yolo12n`, `yolo12s` | 2 | -- |
| YOLO26 | `yolo26n`, `yolo26s` | 2 | -- |

### 跨范式基线（6 个模型）

| 键 | 架构 | 范式 |
| --- | --- | --- |
| `retinanet_r50` | RetinaNet, ResNet-50 FPN | 单阶段、anchor-based、Focal Loss |
| `faster_rcnn_r50` | Faster R-CNN, ResNet-50 FPN | 两阶段、RPN + RoI |
| `mask_rcnn_r50` | Mask R-CNN, ResNet-50 FPN | 两阶段 + mask 头 |
| `fcos_r50` | FCOS, ResNet-50 FPN | Anchor-free、centerness |
| `ssd300_vgg16` | SSD300, VGG-16 | 单阶段、300x300 |
| `ssdlite320_mobilenetv3` | SSDlite320, MobileNetV3-Large | 轻量级、320x320 |

## 核心特性

**SAC 与 TR 注入。** 可切换空洞卷积（Switchable Atrous Convolution, SAC）将 backbone
中选定的 C3/C2f 块替换为空洞变体；C3TR 替换为 Transformer 块。注入点为各类常量——
无配置文件，无运行时分发。支持 YOLOv5s、YOLOv6n、YOLOv8n/s、YOLOv9c、YOLOv10s
及 RT-DETR-R50。

**显式模型类。** 每个 ZOO 条目为一具体类（如 `YOLOv8sSAC`），硬编码其 YAML 配置、
预训练资产、SAC/TR 索引及解码格式。无抽象工厂，无注册表间接层，管道中无
`isinstance` 分支。

**预训练权重加载。** `from_pretrained()` 经由 ultralytics 下载 COCO 权重，按键与形状
求交集，以 `strict=False` 加载。SAC/TR 层随机初始化；匹配的 backbone 层获得 COCO
迁移。

**基于基类的管道契约。** `BaseModel` 定义 `train_model`、`inference`、`save`、`load`、
`from_pretrained` 与 `analyze`。三态机（`UNINITIALIZED -> PRETRAINED -> TRAINED`）
在运行时强制生命周期正确性。管道仅依赖此接口。

**`cy compose` 批量调度。** YAML 驱动的实验调度器，支持 `$include` 组合、逐实验环境
覆盖（`CUDA_VISIBLE_DEVICES`）、子进程隔离及用于重试工作流的 `errors.jsonl`。

**模型分析。** `model.analyze()` 返回 `ModelAnalysisReport`，包含参数量、
MACs/GFLOPs（经由 thop）、FPS/延迟百分位、峰值显存及三级结构树。可通过
`cy analyze` 或编程方式调用。

## 输出结构

`cy train` 或 `cy run` 之后：

```
output_dir/
  weights/
    best.pt             # 最佳验证检查点
    last.pt             # 最终轮次检查点
  results.csv           # 逐轮指标（loss, mAP50, mAP50-95）
  metrics.csv           # results.csv 的别名/副本
  args.yaml             # 生效的训练参数
  train_logs/           # Ultralytics 训练日志
  test/                 # 自动测试产物（来自 cy run）
    per_image/          # 逐图像 COCO 格式预测 JSON
    predictions/        # 标注预测图像
    curves/
      pr.png            # 精确率-召回率曲线
      roc.png           # ROC 曲线
      confusion.png     # 混淆矩阵
    metrics_summary.json
```

## 包布局

```
cracks_yolo/
  ops/                  # SAC、C3TR 及共享算子模块
  losses/               # 损失函数（v5, v7 OTA, v8 DFL, v10 E2E）
  zoo/                  # 模型类与 ZOO 注册表
    ultralytics/        # UltralyticsAdapter + 39 个显式 YOLO/RT-DETR 类
    torchvision/        # 6 个 torchvision 包装类
  weights/              # 预训练下载、键重映射、部分加载
  logging/              # loguru JSONL 输出、类型化日志记录模式
  metrics/              # COCO mAP、PR/ROC/混淆矩阵、统计检验
  pipeline/             # 训练、测试、compose（批量调度器）
  dataset/              # YOLO/COCO 加载器、变换、增强
  viz/                  # 曲线、混淆矩阵、Grad-CAM、数据集图
  analysis/             # DatasetAnalysisReport、ModelAnalysisReport
cli.py                  # Typer CLI（train, test, run, compose）
```

## 验证

```bash
ruff check cracks_yolo tests
mypy --strict cracks_yolo tests
pytest -q
```

三项均须零错误通过方可合并。

## 文档

| 文档                     | 内容 |
|------------------------| --- |
| `docs/models.zh-CN.md` | 逐模型架构、损失公式、SAC/TR 插入点 |
| `docs/ops.zh-CN.md`          | 算子参考（SAC, C3TR, Conv, CSP, 检测头） |
| `docs/pipeline.zh-CN.md`     | TrainPipeline、TestPipeline、compose 调度器 |
| `docs/dataset.zh-CN.md`      | 数据格式、转换、变换、目标约定 |
| `docs/metrics.zh-CN.md`      | COCO mAP、PR/ROC、统计检验（t 检验、Wilcoxon、bootstrap） |
| `docs/logging.zh-CN.md`      | JSONL 日志模式、loguru 配置 |
| `docs/usage.zh-CN.md`        | 端到端教程 |
| `docs/heatmap.zh-CN.md`      | Grad-CAM 方法与输出结构 |
| `docs/scripts.zh-CN.md`      | CLI 参考（全部命令与标志） |
| `docs/scheduler.zh-CN.md`    | Compose YAML 格式、`$include`、重试工作流 |
| `docs/models.zh-CN.md`       | 如何添加新模型变体 |

## 许可证

参见 `LICENSE`。
