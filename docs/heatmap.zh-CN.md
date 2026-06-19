# Grad-CAM 热力图

[English](heatmap.md) | [中文](heatmap.zh-CN.md)

`cracks_yolo.viz.heatmap.GradCAMExtractor` 为任意 `nn.Module` 和目标层生成 Grad-CAM 显著性图。由 `scripts/heatmap.py` 使用，用于可视化训练好的模型在每张图片上关注了骨干网络的哪些区域。

## 方法

Grad-CAM（Gradient-weighted Class Activation Mapping，Selvaraju 等人，2017）：

1. 在目标层注册前向钩子，捕获其激活输出 `A`（形状为 `(C, H, W)` 的特征图）。
2. 注册反向钩子，捕获标量分数（所有锚点/网格位置上的最大类别分数）相对于 `A` 的梯度。
3. 对输入图像进行一次前向 + 反向传播后：
   - 计算通道级权重 `α_k = 在 (H, W) 上对 ∂score/∂A_k 求平均`（梯度的全局平均池化）。
   - 计算热力图 `M = ReLU(Σ_k α_k · A_k)` — 形状为 `(H, W)`。
4. 将 `M` 上采样至输入图像尺寸，归一化到 `[0, 1]`，以 jet 颜色映射叠加到原图上。

**检测模型的分数选择**：对于 anchor-free 的 v8/v9/v10，标量分数为 `所有网格单元上的最大值 × 所有类别上的 sigmoid(cls_logits) 最大值`。对于 anchor-based 的 v5/v7，则为 `所有锚点上的最大值 × 所有类别上的 sigmoid(obj_logit) × sigmoid(cls_logits) 最大值`。对于 torchvision 检测器，直接使用得分最高的检测结果的 `score` 字段。

## 层选择

使用相对于模型顶层属性的点号表示法。`cracks_yolo.zoo` 中的 YOLO 模型暴露了 `backbone`（一个 `nn.Sequential`）和 `neck`（另一个 `nn.Sequential`）。有效的层指定方式：

- `backbone.0` 到 `backbone.9`（v5s：10 个子模块，索引 0-9）。
- `backbone.0` 到 `backbone.9`（v8/v9/v10：索引方案相同）。
- `neck.0`、`neck.4`、`neck.7` 等 — neck 中的任意层索引。

对于 torchvision 检测器，使用 ResNet 子模块：`_inner.backbone.body.layer1`、`_inner.backbone.body.layer4` 等。

**常见陷阱**：YOLOv5s 的 backbone 有 10 个子模块（索引 0-9），因此 `backbone.10` 会抛出 `IndexError`。在指定 `--layers` 之前务必检查 `len(model.backbone)`。

## 命令行

```bash
python -m scripts.heatmap \
    --model yolov5s_sactr \
    --weights output/yolov5s_sactr/best.pt \
    --input data/CrackDetection_Augmentation.v1.yolov5pytorch/test \
    --layers backbone.8,backbone.9 \
    --output-dir output/heatmaps \
    --input-size 640
```

输出结构：

```
output/heatmaps/
  heatmaps/
    000000/
      backbone.8.png      # 在原图上叠加 jet 颜色映射
      backbone.9.png
    000001/
      ...
  feature_maps/
    000000/
      backbone.8.npy      # 原始 (C, H, W) 特征图
      backbone.9.npy
    000001/
      ...
```

## 舌面裂纹检测中的解读

Grad-CAM 在舌面裂纹检测中有两个用途：

1. **SAC/TR 消融实验证据**：在相同的测试图像上比较 `YOLOv5s`、`YOLOv5sSAC`、`YOLOv5sTR` 和 `YOLOv5sSACTR` 的热力图。SACTR 倾向于沿细裂纹线产生更集中的激活（更少的上下文扩散）——从视觉上可以量化为热力图包络更紧密地贴合真实裂纹像素。

2. **跨范式比较**：YOLO 的热力图能很好地定位到裂纹像素，因为其密集网格预测在空间上是对齐的。Faster-RCNN 基于 ROI 的热力图在视觉效果上会有所不同（每个检测结果更锐利，但缺少密集覆盖）。
