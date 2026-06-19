# cracks-yolo

[English](README.md) | [中文](README.zh-CN.md)

面向**舌面裂纹检测**的自包含 PyTorch 模型库，包含 SAC（Switchable Atrous Convolution）和 TR（Transformer）增强的 YOLOv5 / v7 / v8 / v9 / v10 变体，以及 torchvision RetinaNet、Faster R-CNN、Mask R-CNN、FCOS、SSD300 和 SSDlite320 基线模型，用于跨范式比较。每个模型都是一个独立的 `nn.Module` 类，拥有自己的层、损失函数、优化器构建器和预训练权重加载器。无需运行时 YAML 解析，无外部框架耦合。

## 模型（26 个 ZOO 条目）

### YOLO 系列（anchor-based）

| 键 | 类 | 说明 |
| --- | --- | --- |
| `yolov5s` | `YOLOv5s` | 基线 |
| `yolov5s_sac` | `YOLOv5sSAC` | backbone 中添加 SAC |
| `yolov5s_tr` | `YOLOv5sTR` | 添加 TR（Transformer） |
| `yolov5s_sactr` | `YOLOv5sSACTR` | 同时添加 SAC + TR |
| `yolov7w` | `YOLOv7w` | OTA 损失、RepConv |
| `yolov7w_sac` | `YOLOv7wSAC` | 添加 SAC |

### YOLO 系列（anchor-free）

| 键 | 类 | 说明 |
| --- | --- | --- |
| `yolov8n` / `yolov8s` / `yolov8m` / `yolov8l` / `yolov8x` | `YOLOv8{n,s,m,l,x}` | n/s/m/l/x 尺寸、DFL、CIoU |
| `yolov8n_sac` ... `yolov8x_sac` | `YOLOv8{n,s,m,l,x}SAC` | 在 C2f 中添加 SAC |
| `yolov9c` | `YOLOv9c` | GELAN backbone、SPPELAN neck（无 PGI） |
| `yolov9c_sac` | `YOLOv9cSAC` | 添加 SAC（C2fSAC 回退） |
| `yolov10s` | `YOLOv10s` | 无需 NMS，双 one2many/one2one 检测头 |
| `yolov10s_sac` | `YOLOv10sSAC` | 添加 SAC |

### 跨范式基线（torchvision）

| 键 | 类 | 说明 |
| --- | --- | --- |
| `retinanet_r50` | `RetinaNetR50` | 单阶段 anchor-based、FocalLoss |
| `faster_rcnn_r50` | `FasterRCNNR50` | 两阶段、RPN + ROI |
| `mask_rcnn_r50` | `MaskRCNNR50` | 两阶段 + mask 头（bbox 填充掩码） |
| `fcos_r50` | `FCOSR50` | anchor-free、centerness 分支 |
| `ssd300_vgg16` | `SSD300VGG16` | 单阶段经典模型、300x300 输入 |
| `ssdlite320_mobilenetv3` | `SSDlite320MobileNetV3` | 轻量级、320x320 输入 |

## 快速开始

```bash
uv sync          # 或：pip install -e .

# 如需 CUDA 11.8 支持：
uv pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

```python
import torch
from cracks_yolo.zoo import ZOO

model = ZOO["yolov5s_sactr"](num_classes=1)
model.train()

x = torch.randn(2, 3, 640, 640)
preds = model(x)

targets = torch.tensor(
    [[0, 0, 0.50, 0.50, 0.20, 0.20],
     [1, 0, 0.40, 0.40, 0.15, 0.25]],
    dtype=torch.float32,
)
loss, parts = model.compute_loss(preds, targets, imgs=x)
loss.backward()
```

加载 COCO 预训练权重（仅基线变体可用 -- SAC/TR 层没有 COCO 权重，因此使用 `strict=False` 部分加载）：

```python
from cracks_yolo.zoo import YOLOv5s
model = YOLOv5s.from_pretrained(num_classes=1)  # 下载 + strict=False 加载
```

### 训练 / 测试 / 交叉验证 / 比较

```bash
# 使用 COCO 预训练初始化的单次训练。
python -m scripts.train --model yolov5s_sactr --pretrained \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --epochs 100 --batch-size 32 --output-dir output/yolov5s_sactr

# 5 折交叉验证。将 train+valid+test 合并为一个池，
# 留出折 = test，剩余数据按 90/10 划分为 train/val。
python -m scripts.train --model yolov5s_sactr --cross-val --n-folds 5 \
    --val-fraction 0.1 --pretrained \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --epochs 100 --batch-size 32 --output-dir output/yolov5s_sactr_cv

# 带配对 t 检验的多模型比较。
python -m scripts.compare_models \
    --models yolov5s,yolov5s_sactr,yolov8s,yolov9c,retinanet_r50,faster_rcnn_r50 \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --n-folds 5 --epochs 100 --output-dir output/comparison

# 带子进程隔离 + 错误捕获 + 重试的批量调度。
python -m scripts.schedule_experiments \
    --config experiments/all_models_direct.yaml \
    --output-dir output/all_models_direct
```

### 完整的 26 模型扫荡

两个即用型配置位于 [`experiments/`](experiments/)：

- **`experiments/all_models_direct.yaml`** -- 26 个模型 x 2（训练 + 测试）= 52 个实验。在原始划分上直接训练到测试。
- **`experiments/all_models_cv5.yaml`** -- 26 个模型 x 1 个 CV 实验 = 26 个实验。在合并池上进行 5 折 CV（留出折 = test）。

完成 `git clone` + `uv sync` + cu118 torch 安装后：

```bash
# 直接扫荡（52 个实验）。
python -m scripts.schedule_experiments \
    --config experiments/all_models_direct.yaml \
    --output-dir output/all_models_direct

# 5 折 CV 扫荡（26 个实验）。
python -m scripts.schedule_experiments \
    --config experiments/all_models_cv5.yaml \
    --output-dir output/all_models_cv5
```

对于多 GPU 服务器，可增加 `scheduler.max_parallel` 并为每个实验添加 `env: {CUDA_VISIBLE_DEVICES: "N"}`（参见 [`experiments/README.md`](experiments/README.md) 了解批次大小调优）。

## 项目结构

```
cracks_yolo/
  ops/         # Conv、CSP、transformer、检测头、SAC/TR、YOLOv9 算子。
  losses/      # ComputeLoss (v5)、ComputeLossOTA (v7)、v8DetectionLoss、E2ELoss (v10)。
  zoo/         # 26 个模型类。base.py = DetectorModel Protocol + PretrainedSpec。
  weights/     # load_pretrained：下载、键重映射、strict=False + LoadReport。
  logging/     # loguru JSONL 输出 + TypedDict 日志记录模式。
  metrics/     # COCOMetricsCalculator + PR/ROC/confusion + 配对 t 检验/Wilcoxon/bootstrap CI。
  pipeline/    # TrainPipelineImpl / TestPipelineImpl / crossval / compare。
  dataset/     # YOLOSource、COCOSource、DetectionDataset、变换、yolo<->coco 转换。
  viz/         # loss/metric/PR/ROC 曲线、混淆矩阵、Grad-CAM、数据集分布图。
  analysis/    # DatasetAnalysisReport、ModelAnalysisReport。
scripts/       # train、test、convert_dataset、heatmap、analyze_dataset、analyze_model、
               # schedule_experiments、compare_models。
```

## 文档

**英文**：

- [`docs/architecture.md`](docs/architecture.md) -- 设计哲学、包结构、Protocol 契约。
- [`docs/ops.md`](docs/ops.md) -- 算子数学公式、构造参数、SAC/TR 详解。
- [`docs/models.md`](docs/models.md) -- 各模型架构、损失公式、SAC/TR 插入点。
- [`docs/metrics.md`](docs/metrics.md) -- 所有指标（mAP、AR、精确率/召回率、统计检验）。
- [`docs/pretrained.md`](docs/pretrained.md) -- `from_pretrained` 语义、键重映射、SAC/TR 部分加载。
- [`docs/logging.md`](docs/logging.md) -- 日志记录模式、JSONL 格式、事后查询。
- [`docs/usage.md`](docs/usage.md) -- 端到端教程。
- [`docs/development.md`](docs/development.md) -- 如何添加新模型变体。
- [`docs/dataset.md`](docs/dataset.md) -- 数据格式、转换、变换、目标张量约定。
- [`docs/pipeline.md`](docs/pipeline.md) -- TrainPipeline/TestPipeline 用法、5 折 CV、多模型比较。
- [`docs/scheduler.md`](docs/scheduler.md) -- YAML 格式、重试工作流、并行执行。
- [`docs/scripts.md`](docs/scripts.md) -- 各脚本的用途、参数、输入/输出。
- [`docs/heatmap.md`](docs/heatmap.md) -- Grad-CAM 方法、层选择、输出结构。
- [`docs/cross_validation.md`](docs/cross_validation.md) -- 5 折机制、配对 t 检验、结果解读。
- [`docs/cuda_setup.md`](docs/cuda_setup.md) -- cu118 安装、显存缩放、AMP、多 GPU。
- [`experiments/README.md`](experiments/README.md) -- 即用型扫荡配置。

**中文**：

- [`docs/architecture.zh-CN.md`](docs/architecture.zh-CN.md)
- [`docs/ops.zh-CN.md`](docs/ops.zh-CN.md)
- [`docs/models.zh-CN.md`](docs/models.zh-CN.md)
- [`docs/metrics.zh-CN.md`](docs/metrics.zh-CN.md)
- [`docs/pretrained.zh-CN.md`](docs/pretrained.zh-CN.md)
- [`docs/logging.zh-CN.md`](docs/logging.zh-CN.md)
- [`docs/usage.zh-CN.md`](docs/usage.zh-CN.md)
- [`docs/development.zh-CN.md`](docs/development.zh-CN.md)
- [`docs/dataset.zh-CN.md`](docs/dataset.zh-CN.md)
- [`docs/pipeline.zh-CN.md`](docs/pipeline.zh-CN.md)
- [`docs/scheduler.zh-CN.md`](docs/scheduler.zh-CN.md)
- [`docs/scripts.zh-CN.md`](docs/scripts.zh-CN.md)
- [`docs/heatmap.zh-CN.md`](docs/heatmap.zh-CN.md)
- [`docs/cross_validation.zh-CN.md`](docs/cross_validation.zh-CN.md)
- [`docs/cuda_setup.zh-CN.md`](docs/cuda_setup.zh-CN.md)

## 验证

```bash
uv run ruff check cracks_yolo tests scripts
uv run mypy --strict cracks_yolo tests scripts
uv run pytest -q
```

以上三项在合并前必须全部通过。

## 许可证

参见 `LICENSE`（或您项目的许可文件）。
