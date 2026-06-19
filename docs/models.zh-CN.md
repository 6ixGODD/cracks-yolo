# 模型

[English](models.md) | [中文](models.zh-CN.md)

`cracks_yolo.zoo` 中的每个模型都是一个自包含的 `nn.Module`，拥有自己的层、损失模块、优化器构建器和 `from_pretrained` 类方法。长类名编码了每一个选择——`YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU`——而短别名位于 `cracks_yolo.zoo.__init__` 中以方便使用。`ZOO` 注册表共有 26 个条目，覆盖 7 个模型家族及其基线和 SAC/TR 变体。

每个 zoo 类声明两个类属性，供 pipeline 读取而不是通过类名分支：

- `loss_parts_schema: tuple[str, ...]` —— `compute_loss` 返回的 `parts` 张量中每个条目的名称。v5/v7 = `("box","cls","obj")`；v8/v9/v10 = `("box","cls","dfl")`；torchvision 封装 = `("total","cls","box_reg","rpn_box_reg")`。
- `decode_format: str` —— `"anchor_free"`（v8/v9/v10：`(B, 4+nc, N)`）或 `"anchor_based"`（v5/v7 + torchvision 封装：`(B, N, nc+5)` 或 `(B, N_max, 6)`）。

所有 `compute_loss` 签名都接受 `imgs: torch.Tensor | None = None`（v7 使用它进行 OTA 分配；其他则忽略）。pipeline 始终传递 `imgs=images`。

## 注册表

`cracks_yolo.zoo.ZOO: dict[str, type[nn.Module]]` 将短名称映射到类：

| 键 | 类 | 家族 |
| --- | --- | --- |
| `yolov5s` | `YOLOv5s` | YOLOv5 |
| `yolov5s_sac` | `YOLOv5sSAC` | YOLOv5 + SAC |
| `yolov5s_tr` | `YOLOv5sTR` | YOLOv5 + TR |
| `yolov5s_sactr` | `YOLOv5sSACTR` | YOLOv5 + SAC + TR |
| `yolov7w` | `YOLOv7w` | YOLOv7 |
| `yolov7w_sac` | `YOLOv7wSAC` | YOLOv7 + SAC |
| `yolov8n` | `YOLOv8n` | YOLOv8 nano |
| `yolov8n_sac` | `YOLOv8nSAC` | YOLOv8 nano + SAC |
| `yolov8s` | `YOLOv8s` | YOLOv8 small |
| `yolov8s_sac` | `YOLOv8sSAC` | YOLOv8 small + SAC |
| `yolov8m` | `YOLOv8m` | YOLOv8 medium |
| `yolov8m_sac` | `YOLOv8mSAC` | YOLOv8 medium + SAC |
| `yolov8l` | `YOLOv8l` | YOLOv8 large |
| `yolov8l_sac` | `YOLOv8lSAC` | YOLOv8 large + SAC |
| `yolov8x` | `YOLOv8x` | YOLOv8 xlarge |
| `yolov8x_sac` | `YOLOv8xSAC` | YOLOv8 xlarge + SAC |
| `yolov9c` | `YOLOv9c` | YOLOv9 compact |
| `yolov9c_sac` | `YOLOv9cSAC` | YOLOv9 compact + SAC |
| `yolov10s` | `YOLOv10s` | YOLOv10 small |
| `yolov10s_sac` | `YOLOv10sSAC` | YOLOv10 small + SAC |
| `retinanet_r50` | `RetinaNetR50` | RetinaNet R50 |
| `faster_rcnn_r50` | `FasterRCNNR50` | Faster R-CNN R50 |
| `mask_rcnn_r50` | `MaskRCNNR50` | Mask R-CNN R50 |
| `fcos_r50` | `FCOSR50` | FCOS R50 |
| `ssd300_vgg16` | `SSD300VGG16` | SSD300 VGG16 |
| `ssdlite320_mobilenetv3` | `SSDlite320MobileNetV3` | SSDlite320 MobileNetV3 |

---

## YOLOv5s（`cracks_yolo/zoo/yolov5.py`）

### 架构

```
输入 (B, 3, 640, 640)
  |
  +- 主干网络 (stride 8/16/32)
  |   Conv-P1/2 -> Conv-P2/4 -> C3 -> Conv-P3/8 -> C3 -> Conv-P4/16 -> C3
  |   -> Conv-P5/32 -> C3 -> SPPF
  |
  +- 颈部 (FPN + PAN)
  |   Conv -> Upsample -> Concat(P4) -> C3 -> Conv -> Upsample
  |   -> Concat(P3) -> C3 (P3/8-小)
  |   -> Conv -> Concat(neck-P4) -> C3 (P4/16-中)
  |   -> Conv -> Concat(neck-P5) -> C3 (P5/32-大)
  |
  +- 检测头: DetectAnchorBased (3 个尺度 x 3 个锚点，硬编码 COCO 锚点)
       -> (B, 25200, nc+5)  [25200 = (80^2 + 40^2 + 20^2) x 3]
```

### 变体

- **SAC：** 在 P2/P3/P4/P5 阶段将主干 C3 替换为 `C3SAC`。加载 COCO 权重时 SAC 层随机初始化（不存在 COCO SAC 权重）。
- **TR：** 将一个主干 C3 替换为 `C3TR`（TransformerBlock）。TransformerBlock 在 COCO 加载时随机初始化。
- **SACTR：** 同时应用 SAC 和 TR。

### 损失（`ComputeLoss`）

CIoU + BCEobj + BCEcls，使用 IoU 感知的目标置信度目标和 anchor_t=4.0 网格偏移匹配。从上游 YOLOv5 损失移植。

$$L = \lambda_\text{box} \cdot L_\text{CIoU} + \lambda_\text{obj} \cdot L_\text{BCEobj} + \lambda_\text{cls} \cdot L_\text{BCEcls}$$

增益：$\lambda_\text{box}=0.05$、$\lambda_\text{obj}=0.7$、$\lambda_\text{cls}=0.3$。目标置信度目标由预测与匹配的真实边界框之间的 IoU 缩放，因此目标置信度分支学习的是"这个锚点与 GT 匹配得有多好"，而非二元的 0/1。

### 优化器

`build_optimizer()` 返回 `torch.optim.AdamW(model.parameters(), lr=1e-3)`。

### 锚点（COCO，硬编码）

```
P3/8:  [10,13], [16,30],   [33,23]
P4/16: [30,61], [62,45],   [59,119]
P5/32: [116,90], [156,198], [373,326]
```

初始化时除以 stride：`head.anchors /= head.stride.view(-1, 1, 1)`。

### Stride 初始化

在 `s=256` 处进行一次虚拟训练 forward 产生特征图；stride 计算为 `[s / f.shape[-2] for f in feats]` 并存储在检测头上。

### 论文

- YOLOv5：[Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- SAC：[Switchable Atrous Convolution (ICCV 2021)](https://arxiv.org/abs/1908.07698)
- TR：来自 [Attention Is All You Need (NeurIPS 2017)](https://arxiv.org/abs/1706.03762) 的 Transformer 块

---

## YOLOv7w（`cracks_yolo/zoo/yolov7.py`）

### 架构

YOLOv7-w（宽）变体。使用 `RepConv`（训练时多分支，推理时融合）、`SPPCSPC` 和 `IDetect`（带 `ImplicitA`/`ImplicitM`）。第一个卷积为 `Conv(3, 32, 3, 2)`，用于设置 P1/2 的 stride=2。

```
输入 (B, 3, 640, 640)
  |
  +- 主干网络
  |   Conv-P1/2 -> Conv-P2/4 -> C3 -> Conv-P3/8 -> C3
  |   -> Conv-P4/16 -> C3 -> Conv-P5/32 -> C3 -> SPPCSPC
  |
  +- 颈部 (带有 RepConv 的 FPN + PAN)
  |
  +- 检测头: IDetect (ImplicitA + ImplicitM, COCO 锚点)
       -> 每个尺度 (B, 3, H, W, nc+5) -> 解码为 (B, N, nc+5)
```

SAC 变体将主干 C3 阶段替换为 `C3SAC`。

### 损失（`ComputeLossOTA`）

最优传输分配（SimOTA）——动态 k 匹配。代价矩阵 = `cls_loss + 3 * iou_loss`；对于每个 GT，选择 top-k 锚点（k 由 IoU 之和计算得出）作为正样本。**需要 `imgs` 参数**传递给 `compute_loss`，因为 OTA 使用图像尺寸进行动态 k 计算。

### 部署

在 eval 前调用 `model.fuse()` 以将 `RepConv` 多分支融合为单个卷积，并将 `ImplicitA`/`ImplicitM` 融合到前面的卷积中。

### 论文

- YOLOv7：[YOLOv7: Trainable bag-of-freebies sets new state-of-the-art (CVPR 2023)](https://arxiv.org/abs/2207.02696)
- SimOTA：来自 [YOLOX (2021)](https://arxiv.org/abs/2107.08430)

---

## YOLOv8 {n, s, m, l, x}（`cracks_yolo/zoo/yolov8.py`）

### 架构

无锚点。主干网络中使用 C2f 阶段（C3 + 分割 + 重用），P5 处使用 SPPF，FPN+PAN 颈部，`DetectAnchorFree` 检测头，带有独立的 `cv2`（边界框，reg_max=16）和 `cv3`（类别）分支。通过 `width_mult` 和 `depth_mult` 缩放提供五种尺寸变体。

```
输入 (B, 3, 640, 640)
  |
  +- 主干网络 (stride 8/16/32)
  |   Conv-P1/2 -> Conv-P2/4 -> C2f -> Conv-P3/8 -> C2f
  |   -> Conv-P4/16 -> C2f -> Conv-P5/32 -> C2f -> SPPF
  |
  +- 颈部 (FPN + PAN)
  |   Upsample -> Concat(P4) -> C2f -> Upsample
  |   -> Concat(P3) -> C2f (P3/8-小)
  |   -> Conv -> Concat(neck-P4) -> C2f (P4/16-中)
  |   -> Conv -> Concat(neck-P5) -> C2f (P5/32-大)
  |
  +- 检测头: DetectAnchorFree (DFL box + cls 分支)
       训练: {"boxes": (B, 4*reg_max, N), "scores": (B, nc, N), "feats": [...]}
       Eval:     (B, 4+nc, N)  其中 N = 8400，在 640x640 时
```

### 尺寸变体

| 变体 | width_mult | depth_mult | P3 通道 | P4 通道 | P5 通道 |
| ------- | ---------- | ---------- | ------- | ------- | ------- |
| nano    | 0.25       | 0.33       | 64      | 128     | 256     |
| small   | 0.50       | 0.33       | 128     | 256     | 512     |
| medium  | 0.75       | 0.67       | 192     | 384     | 768     |
| large   | 1.00       | 1.00       | 256     | 512     | 1024    |
| xlarge  | 1.25       | 1.00       | 320     | 640     | 1280    |

### 损失（`v8DetectionLoss`）

- `TaskAlignedAssigner`（topk=10，alpha=0.5，beta=6.0）用于正样本分配。
- `BboxLoss`，带 CIoU + DFL（Distribution Focal Loss）。
- **没有目标置信度损失**（v8 去掉了 objectness 分支）。
- 返回 `(3,)` 张量 `[box, cls, dfl]`。`compute_loss` 在反向传播前求和为标量。

### SAC 变体

在四个主干 C2f 阶段（P2/P3/P4/P5）中将 `C2f` 替换为 `C2fSAC`。五种尺寸均有 SAC 变体。

### 论文

- YOLOv8：[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- DFL：来自 [Generalized Focal Loss (NeurIPS 2020)](https://arxiv.org/abs/2006.04388)

---

## YOLOv9c（`cracks_yolo/zoo/yolov9.py`）

### 架构

YOLOv9-c（紧凑版），简化版——无 PGI 辅助分支。主干网络使用 GELAN 风格的 `RepNCSPELAN4` 阶段，配合 `ADown` 下采样。颈部：`SPPELAN` 底部 + 带有 `RepNCSPELAN4` 阶段的 FPN/PAN。检测头：v8 风格的 `DetectAnchorFree`（独立的 cv2/cv3 分支 + DFL）。损失：`v8DetectionLoss`。

上游 PGI（Programmable Gradient Information）辅助监督分支（`DualDDetect` 头 + `CBFuse` 融合）被省略。这使得 YOLOv9 条目成为一个公平的同类 Protocol 比较基线，而非完美复现。

```
输入 (B, 3, 640, 640)
  |
  +- 主干网络
  |   Conv-P1/2 -> Conv-P2/4 -> RepNCSPELAN4 -> ADown
  |   -> RepNCSPELAN4 (P3) -> ADown -> RepNCSPELAN4 (P4)
  |   -> ADown -> RepNCSPELAN4 (P5)
  |
  +- 颈部 (SPPELAN + 带有 RepNCSPELAN4 的 FPN/PAN)
  |
  +- 检测头: DetectAnchorFree (v8 风格, DFL)
```

SAC 变体将 `RepNCSPELAN4` 替换为相同主干位置的 `C2fSAC`（P2/P3/P4/P5）。由于 `RepConvN` 在结构上与 SAC 不兼容（重参数化），SAC 变体退回到 `C2fSAC` 阶段。COCO 权重以 `strict=False` 加载；SAC 层随机初始化。

### 论文

- YOLOv9：[YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information (2024)](https://arxiv.org/abs/2402.13616)

---

## YOLOv10s（`cracks_yolo/zoo/yolov10.py`）

### 架构

YOLOv10 主干：C2f -> C2fCIB + SCDown 在 P3/P4/P5，`PSA`（部分自注意力）在 P5。颈部：与 v8 相同的 FPN+PAN，但使用 `SCDown` 进行下采样。检测头：`v10Detect`（双头）。

```
输入 (B, 3, 640, 640)
  |
  +- 主干网络
  |   Conv-P1/2 -> Conv-P2/4 -> C2f -> Conv-P3/8 -> C2fCIB
  |   -> SCDown -> C2fCIB (P4) -> SCDown -> C2fCIB (P5) -> PSA
  |
  +- 颈部 (带有 SCDown 的 FPN+PAN)
  |
  +- 检测头: v10Detect (双头)
       训练: {"one2many": {...}, "one2one": {...}}
       Eval:     (B, 4+nc, N)  [仅 one2one 头 — 无 NMS]
```

### 损失（`E2ELoss`）

两个 `v8DetectionLoss` 实例：

- **one2many**（topk=10）：标准训练，一个 GT 匹配多个预测。
- **one2one**（topk=7，衰减调度）：一个 GT 匹配一个预测——这实现了无 NMS 推理。

one2one 损失权重遵循衰减调度，使其在训练过程中逐渐增大。返回 `(3,)` 张量；`compute_loss` 求和为标量。

### 无 NMS 推理

Eval 模式 forward 只运行 one2one 头，每个 GT 产生一个检测——无需 NMS。Top-K 过滤在解码步骤中应用。

### 论文

- YOLOv10：[YOLOv10: Real-Time End-to-End Object Detection (NeurIPS 2024)](https://arxiv.org/abs/2405.14458)

---

## Torchvision 检测器（`cracks_yolo/zoo/torchvision_detectors.py`）

### 架构

六个 torchvision 检测模型被封装以满足 `DetectorModel` Protocol，提供跨范式的比较基线。

| 键 | 内部模型 | 范式 | 主干网络 |
| --- | ----------- | -------- | -------- |
| `retinanet_r50` | `retinanet_resnet50_fpn` | 单阶段，基于锚点 | ResNet-50 FPN |
| `faster_rcnn_r50` | `fasterrcnn_resnet50_fpn` | 两阶段，基于锚点 | ResNet-50 FPN |
| `mask_rcnn_r50` | `maskrcnn_resnet50_fpn` | 两阶段，基于锚点 + 掩码 | ResNet-50 FPN |
| `fcos_r50` | `fcos_resnet50_fpn` | 单阶段，无锚点 | ResNet-50 FPN |
| `ssd300_vgg16` | `ssd300_vgg16` | 单阶段，基于锚点 | VGG-16 |
| `ssdlite320_mobilenetv3` | `ssdlite320_mobilenet_v3_large` | 单阶段，基于锚点 | MobileNetV3-Large |

### 适配策略

- `forward(x)` 在训练模式下暂存图像并返回 `{"_tv_images": x}`。在 eval 模式下调用内部模型并返回 `list[dict]`。
- `compute_loss(preds, targets, imgs=None)` 将 `(N, 6)` YOLO 目标转换为 torchvision 的 `list[dict]` 格式（带 `boxes`/`labels`），调用内部模型，对损失字典求和，返回 `(total_loss, parts_tensor)`，其中 `parts_tensor` 为 `(total, cls, box_reg, rpn_box_reg)`。
- `decode(preds)` 将 eval 模式的 `list[dict]` 转换为 `(B, N_max, 6)` 格式的 `(x1, y1, x2, y2, score, class_id)`——基于锚点的格式。较短的检测结果用零填充；pipeline 的 NMS 步骤会忽略零分值的行。

### 损失

Torchvision 模型使用各自内置的损失头：Focal Loss（RetinaNet、FCOS、SSD）、Cross-Entropy + Smooth L1（Faster/Mask R-CNN）。COCO 预训练权重通过 torchvision 自身的权重机制（`weights="DEFAULT"`）加载，而非通过 `cracks_yolo` 的权重注册表。

### 论文

- RetinaNet：[Focal Loss for Dense Object Detection (ICCV 2017)](https://arxiv.org/abs/1708.02002)
- Faster R-CNN：[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (NeurIPS 2015)](https://arxiv.org/abs/1506.01497)
- Mask R-CNN：[Mask R-CNN (ICCV 2017)](https://arxiv.org/abs/1703.06870)
- FCOS：[FCOS: Fully Convolutional One-Stage Object Detection (ICCV 2019)](https://arxiv.org/abs/1904.01355)
- SSD：[SSD: Single Shot MultiBox Detector (ECCV 2016)](https://arxiv.org/abs/1512.02325)

---

## SAC 插入点（总结）

- v5：在 P2/P3/P4/P5 处将主干 `C3` 替换为 `C3SAC`（标准 v5s 主干中的索引 2、4、6、8）。
- v8/v10：在相同位置将主干 `C2f` 替换为 `C2fSAC`。
- v7：将 `RepConv`/Bottleneck 序列替换为基于 `BottleneckSAC` 的等价结构。
- v9：将 `RepNCSPELAN4` 替换为 P2/P3/P4/P5 位置的 `C2fSAC`（结构折衷——`RepConvN` 无法承载 SAC）。

加载 COCO 权重时，SAC 层出现在 `LoadReport.missing` 中——它们是随机初始化的。

## TR 插入点

- v5：将一个主干 `C3`（通常是 P5/32 阶段）替换为 `C3TR`。`TransformerBlock` 在 COCO 加载时随机初始化。

## 预训练权重语义

- 基线变体声明一个 `PretrainedSpec` 类属性，指向官方 COCO 发行版 URL。
- SAC/TR 变体声明 `pretrained_spec = None`（增强层不存在 COCO 权重）。
- `from_pretrained` 使用 `strict=False`，返回包含 `matched`、`missing` 和 `unexpected` 键列表的 `LoadReport`。
- Torchvision 封装使用 torchvision 内置的权重 API（`weights="DEFAULT"`），而非 `cracks_yolo` 的权重注册表。

## 添加新变体

参见 `docs/development.md`。

## 当前状态

- 26 个 ZOO 条目，覆盖 7 个模型家族。
- 318 个测试通过；`ruff` 和 `mypy --strict` 在 97 个源文件上干净通过。
- 损失设备同步约定：损失模块在入口处将内部张量（锚点、stride、BCE 正样本权重）同步到 `preds[0].device`。参见 `cracks_yolo/losses/yolov5.py` 中的规范模式。
