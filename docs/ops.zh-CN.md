# 算子（`cracks_yolo.ops`）

[English](ops.md) | [中文](ops.zh-CN.md)

纯 `nn.Module` / `nn.Conv2d` 子类。没有抽象基类，没有插件注册表——每个算子都是自包含的。构造函数签名完全保留自上游，以便 COCO 预训练 state_dict 可以干净地加载（state_dict 键名很重要）。

## 卷积 — `conv.py`

### `autopad(k, p=None, d=1) -> int`
自动计算 `'same'` 形状卷积输出的填充。

### `Conv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)`
Conv2d + BatchNorm2d + 激活函数（默认 SiLU）。这是每个 YOLO 家族的主力算子。

### `DWConv(c2, k=1, s=1, d=1, act=True)`
深度可分离变体：groups = 输入通道数。

### `ConvAWS2d(c1, c2, k=1, s=1, p=None, g=1, d=1, act=None)`
权重标准化 Conv2d（用于 SAC）。在 forward 卷积之前将权重乘以 `(weight - weight.mean()) / (weight.std() + eps)`。

**数学公式：**
$$W' = \gamma \cdot \frac{W - \mu_W}{\sigma_W + \epsilon}, \quad y = W' * x + b$$

### `SAConv2d(c1, c2, k=1, s=1, p=None, g=1, d=1, act=None)`
**可切换空洞卷积（SAC）**——核心增强。包装一个 `ConvAWS2d`，学习一个逐通道开关，在三种空洞率（默认 `a=[1, 2, 3]`）之间切换。一个类似注意力的模块产生逐通道、逐空间位置的权重，用于在三个膨胀卷积之间进行插值。第二个全局标量乘以输出（初始化为使 SAC 近似恒等映射）。

**为什么在舌面裂纹检测中使用 SAC？** 裂缝在同一张图像内的尺度变化极大。单一的膨胀率无法同时捕捉细微的发丝裂缝和更宽的剥落。SAC 让网络为每个像素选择合适的感受野。

**偏置处理：** `bias=False`，以便偏置折叠到后续的 BatchNorm 中。这曾是一个真实 bug 的原因——当 `bias=True` 时，SAC 偏置参数由于未被使用而接收不到梯度。

## CSP — `csp.py`

### `Bottleneck(c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5)`
标准残差瓶颈：1x1 卷积 -> 3x3 卷积 -> 可选残差相加。

### `BottleneckSAC(c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0)`
第二个卷积为 `SAConv2d`（而不是 `Conv`）的瓶颈。这是 SAC 构建块——将其插入任何 C3/C2f 阶段即可使该阶段具备 SAC 能力。

### `C3(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
CSP 风格阶段，包含 3 个卷积 + N 个 bottleneck。用于 YOLOv5 和 YOLOv7。

### `C3SAC(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
`C3` 使用 `BottleneckSAC` 代替 `Bottleneck`。

### `C3TR(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
`C3` 的 bottleneck 序列被替换为 `TransformerBlock`。这是 **TR 增强**：一个融合了局部 CNN 特征与全局自注意力的 CSP 阶段。

### `C2f(c1, c2, n=1, shortcut=False, g=1, e=0.5)`
YOLOv8 的 CSP 阶段。比 `C3` 更快，因为它分割通道并在分割间复用 bottleneck 输出。

### `C2fSAC(c1, c2, n=1, shortcut=False, g=1, e=0.5)`
在其内部 bottleneck 序列中使用 `BottleneckSAC` 的 `C2f`。

### `C2fCIB(c1, c2, n=1, shortcut=False, ev=0.5, lk=False)`
YOLOv10 的 CIB（带植入分支的跨阶段部分瓶颈）C2f 变体。用于 v10 主干网络。

### `CIB(c1, c2, shortcut=False, e=0.5, lk=False)`
带植入分支的跨阶段部分瓶颈——v10 构建块。

### `SPPF(c1, c2, k=5)`
空间金字塔池化 - 快速版（单个 5x5 MaxPool 连续应用 3 次）。标准 YOLOv5/v8 头部前置处理。

### `SPPCSPC(c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13))`
YOLOv7 使用的 SPP-CSP 变体。

### `RepConv(c1, c2, k=3, s=1, p=1, g=1, d=1, act=True)`
RepVGG 风格的可重参数化卷积。训练时 = 多分支（3x3 卷积 + 1x1 卷积 + 恒等 BN），推理时 = 单个融合卷积。由 YOLOv7 使用。在部署前调用 `fuse_repvgg` 以融合 `RepConv`。

### `SCDown(c1, c2, k=3, s=2, d=1, act=True)`
YOLOv10 的空间-通道解耦下采样。将空间下采样（DW 卷积 stride 2）与通道混合（PW 卷积）解耦。

### `PSA(c1, c2, e=0.5)`
YOLOv10 的部分自注意力。分割通道，对一半应用轻量级多头注意力，再拼接回去。位于主干 P5 阶段。

### `Concat(dimension=1)`
拼接层（用于颈部合并跳跃连接）。在 `_forward_impl` 中显式跟踪，因为其输入是一个列表，而非张量。

## YOLOv9 专用算子 — `yolov9.py`

### `Silence()`
空操作层（v9 主干索引 0 的占位符）。

### `SP(k=3, s=1)`
空间池化——使用核 `k` 和步幅 `s` 的最大池化。

### `ADown(c1, c2)`
平均池化 + 分割 + (conv3x3/s2, maxpool+conv1x1) 下采样器。将空间分辨率减半；通道数近似加倍。用于 v9 主干中的 GELAN 阶段之间。

### `RepConvN(c1, c2, n=1, se=False, act=True)`
YOLOv9 的可重参数化卷积块。一个由 `n` 个卷积（3x3）组成的序列，带有可选的 Squeeze-and-Excitation。在推理时，`fuse_repvgg` 将多分支合并为单个卷积。与 SAC 不兼容（重参数化会破坏 SAC 开关）。

### `RepNCSPELAN4(c1, c2, c3, c4, c5=1)`
YOLOv9 的 GELAN 构建块：带有 `RepConvN` 的 CSP-ELAN。分割通道，通过 `RepConvN` 序列处理一个分支，拼接并投影。v9 主干在 P2/P3/P4/P5 处使用四个这样的块。

### `SPPELAN(c1, c2, c3, k=(5, 9, 13))`
带有 ELAN 风格拼接的 SPP。用于 FPN/PAN 之前的 v9 颈部底部。

## Transformer — `transformer.py`

### `TransformerLayer(c, num_heads=8, dropout=0.0, ff_mult=4)`
一个 transformer 块：QKV 投影 -> `nn.MultiheadAttention` -> 残差 -> MLP（Linear -> 激活 -> Linear）。**没有 LayerNorm**——与 ultralytics 的 v5/v8 transformer 一致（省略 LayerNorm 以保持 state_dict 键最少）。

### `TransformerBlock(c1, c2, num_heads=4, num_layers=2, dropout=0.0)`
卷积投影 -> 学习的位置嵌入 -> N x `TransformerLayer`。由 `C3TR` 使用。

## 隐式知识 — `implicit.py`（YOLOv7）

### `ImplicitA(channel, mean=0.0, std=0.02)`
加性隐式知识——一个学习的逐通道偏置，加到前一个卷积的输出上。

### `ImplicitM(channel, mean=1.0, std=0.02)`
乘性隐式知识——一个学习的逐通道缩放因子。

两者都有一个 `fuse(into_conv)` 方法，用于在部署时将隐式知识折叠到前一个卷积的权重/偏置中。由 YOLOv7 的 `IDetect` 和 `IAuxDetect` 使用。

## 检测头 — `detect_heads.py`

### 基于锚点（v5/v7）

- `DetectAnchorBased(nc=80, anchors=(), ch=())` — YOLOv5 检测头。三个尺度 x 每个尺度三个锚点。训练 forward 返回一个 `(B, 3, H, W, nc+5)` 原始张量列表；eval forward 解码为 `(B, N, nc+5)`，其中在 640x640 时 N = 25200。
- `IDetect(nc=80, anchors=(), ch=())` — 带 `ImplicitA`/`ImplicitM` 的 YOLOv7 检测头。
- `IAuxDetect(nc=80, anchors=(), ch=())` — YOLOv7 辅助检测头（仅训练——同时输出最终和辅助输出）。

### 无锚点（v8/v9/v10）

- `DetectAnchorFree(nc=80, reg_max=16, ch=())` — YOLOv8/v9 检测头。每个尺度两个分支：`cv2`（边界框，输出 4x`reg_max` 通道）和 `cv3`（类别，`nc` 通道）。训练 forward 返回 `{"boxes": ..., "scores": ..., "feats": [...]}`；eval forward 解码为 `(B, 4+nc, N)`，其中在 640x640 时 N = 8400。
- `DFL(reg_max=16)` — Distribution Focal Loss 层：在 `reg_max` 个 bins 上做 softmax，加权求和，为每个坐标生成一个可微标量。
- `v10Detect(nc=80, reg_max=16, ch=())` — YOLOv10 双头：`one2many`（top-10 匹配，用于训练）+ `one2one`（top-1，无 NMS 推理）。

### 辅助函数

- `make_anchors(feats, strides, grid_cell_offset=0.5)` — 从特征图构建锚点网格。
- `dist2bbox(distance, anchor_points, xywh=True, dim=-1)` — 将 `dist`（左/上/右/下）解码为 xywh 或 xyxy。
- `bbox2dist(anchor_points, bbox, xywh=True)` — `dist2bbox` 的逆操作。

## 激活函数 — `activation.py`

- `SiLU`、`Mish`、`ReLU` 从 torch 重新导出。
- `parse_activation(name_or_module)` — 工厂函数：接受一个字符串（`"silu"`、`"mish"`、`"relu"`）或一个 `nn.Module` 实例，返回激活函数。允许模型库类在其构造函数中暴露 `act: str | nn.Module = "silu"`。
