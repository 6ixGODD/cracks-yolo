# 热力图生成

[English](heatmap.md) | [中文](heatmap.zh-CN.md)

`cracks_yolo.viz.heatmap.GradCAMExtractor` 对任意 `nn.Module` 子图生成梯度加权类激活映射
（Grad-CAM；Selvaraju et al., 2017）。`scripts/heatmap.py` 批量处理图像并输出叠加结果。

## 方法

对于目标层输出 `A ∈ R^{C×H×W}` 与标量分数 `y`，前向钩子捕获 `A`，反向钩子捕获 ∂y/∂A。
通道权重为梯度的全局平均池化：

```
α_k = (1 / (H·W)) · Σ_i Σ_j  ∂y / ∂A_{k,i,j}
```

原始热力图为经 ReLU 阈值化的加权和，上采样至输入尺寸后归一化：

```
M = ReLU( Σ_k α_k · A_k )   ∈ R^{H×W}
```

### 分数定义

| 范式           | 分数 `y`                                                              |
|----------------|-----------------------------------------------------------------------|
| Anchor-free    | 网格维度上对 σ(cls_logits) 的类间极大值取全局极大值                    |
| Anchor-based   | 锚框维度上对 σ(obj) · σ(cls) 的类间极大值取全局极大值                  |
| Torchvision    | 置信度最高的检测框 `score`                                             |

## 层指定

以模型根为起点的点分隔路径：`backbone.0`..`backbone.9`（YOLOv5/v8/v9/v10）、
`neck.0`、`neck.4` 等。Torchvision 模型：`_inner.backbone.body.layer1`..`layer4`。
写入脚本前以 `len(model.backbone)` 确认深度——YOLOv5s 的 backbone 恰含 10 个子模块。

## CLI

```bash
python -m scripts.heatmap \
    --model yolov5s_sactr \
    --weights output/yolov5s_sactr/best.pt \
    --input data/CrackDetection_Augmentation.v1.yolov5pytorch/test \
    --layers backbone.8,backbone.9 \
    --output-dir output/heatmaps \
    --input-size 640
```

输出：`heatmaps/`（每张图像每层的 jet 叠加 PNG）与 `feature_maps/`（原始 `.npy`）。

## API

```python
from cracks_yolo.viz.heatmap import GradCAMExtractor

extractor = GradCAMExtractor(model, target_layers=["backbone.8", "backbone.9"])
heatmaps = extractor(image_tensor)  # dict[str, np.ndarray]
```

## 可视化工具

- `overlay_heatmap(image, heatmap, alpha=0.4, cmap="jet")` — 将归一化热力图叠加至 PIL/numpy
  图像上，返回 RGB 数组。
- `save_heatmap_grid(images, heatmaps, ncols=4)` — 跨样本或层的平铺对比 PNG。
- `plot_activation_profile(heatmap, axis=0)` — 水平/垂直均值投影，用于沿裂缝走向量化空间集中度。

## 裂缝检测的解读

1. **SAC/TR 消融。** 在同一输入上对比 YOLOv5s、YOLOv5sSAC、YOLOv5sTR 和 YOLOv5sSACTR
   的热力图。SACTR 变体沿细裂缝走向呈现更紧致的激活包络，向周围组织的弥散渗漏显著降低。
2. **跨范式对比。** 密集网格 YOLO 热力图提供与裂缝拓扑对齐的空间连续覆盖；基于 ROI 的
   检测器（Faster-RCNN）产生更锐利的逐实例映射，但缺乏同等的空间密度。
