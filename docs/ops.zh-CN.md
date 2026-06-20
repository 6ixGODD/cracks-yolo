# 算子（`cracks_yolo.ops`）

[English](ops.md) | [中文](ops.zh-CN.md)

均为 `nn.Module` / `nn.Conv2d` 子类。无抽象基类、无插件注册表——每个算子自包含。构造签名与上游逐字一致，以便 COCO 预训练 `state_dict` 干净加载（键名重要）。下文中 $H,W$ 为空间维度，$C_{\text{in}},C_{\text{out}}$ 为通道数，$\sigma(\cdot)$ 为 SiLU 激活 $\sigma(x)=x\,\mathrm{sigmoid}(x)$。

## 1. 卷积 —— `conv.py`

### `autopad(k, p=None, d=1) -> int`
返回 `'same'` 空间输出的填充：$p_{\text{auto}} = \lfloor k/2 \rfloor$（按膨胀 $d$ 调整）。

### `Conv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)`
Conv2d + BatchNorm2d + 激活（SiLU）。所有 YOLO 家族的主力：
$$
\mathbf{y} = \sigma\!\big(\mathrm{BN}(\mathrm{Conv}_{k,s}(\mathbf{x}))\big).
$$

### `DWConv(c2, k=1, s=1, d=1, act=True)`
深度可分变体：分组卷积 $g = C_{\text{in}}$，每个输入通道用各自核卷积。

### `ConvAWS2d(c1, c2, k=1, s=1, p=None, g=1, d=1, act=None)`
权重标准化 Conv2d（SAC 使用）。卷积核 $W$ 在前向之前沿空间轴标准化：
$$
\mu_W = \frac{1}{|\mathcal{S}|}\sum_{i\in\mathcal{S}} W_i, \qquad
\sigma_W = \sqrt{\frac{1}{|\mathcal{S}|}\sum_{i\in\mathcal{S}} (W_i - \mu_W)^2}, \qquad
W' = \gamma \cdot \frac{W - \mu_W}{\sigma_W + \epsilon},
$$
$$
\mathbf{y} = W' * \mathbf{x} + b,
$$
其中 $\mathcal{S}$ 索引空间核位置，$\gamma$ 为可学习的逐通道缩放，$\epsilon$ 为小常数。权重标准化稳定小 batch 检测头的训练。

### `SAConv2d(c1, c2, k=1, s=1, p=None, g=1, d=1, act=None)`
**可切换空洞卷积（SAC）** —— 核心增强。SAC 包裹一个 `ConvAWS2d`，对一组空洞（膨胀）率 $\mathcal{A}=\{a_1,a_2,a_3\}$（默认 $\{1,2,3\}$）学习逐像素、逐通道的切换。三个并行膨胀卷积产生特征 $\{\mathbf{F}_{a}\}_{a\in\mathcal{A}}$，类注意力模块在每个位置 $\mathbf{u}$ 输出空间权重 $\{w_a(\mathbf{u})\}$，输出为凸组合：
$$
\mathbf{F}_{\text{SAC}}(\mathbf{u}) = \sum_{a\in\mathcal{A}} w_a(\mathbf{u}) \cdot \mathrm{ConvAWS}_{a}(\mathbf{x})(\mathbf{u}), \qquad \sum_{a\in\mathcal{A}} w_a(\mathbf{u}) = 1.
$$
第二个全局标量 $\alpha$（初始化使 SAC 起步近似恒等）缩放输出：$\mathbf{y} = \alpha \cdot \mathbf{F}_{\text{SAC}}$。

**为何对舌面裂纹检测用 SAC？** 裂纹在同一图中尺度变化剧烈——细发状裂纹与较宽裂缝并存。单一膨胀率无法兼顾。SAC 让网络逐像素选择感受野。

**偏置处理：** `bias=False`，使偏置折叠进后续 BatchNorm。若 `bias=True`，SAC 偏置参数因未被使用而收不到梯度——此约定避免了一个真实 bug。

## 2. CSP 构建块 —— `csp.py`

### `Bottleneck(c1, c2, shortcut=True, g=1, k=(3,3), e=0.5)`
标准残差瓶颈：$1{\times}1$ 卷积 $\to$ $3{\times}3$ 卷积 $\to$ 可选残差相加，
$$
\mathbf{y} = \mathrm{Conv}_{3\times3}(\mathrm{Conv}_{1\times1}(\mathbf{x})) + \mathbf{x} \quad (\text{若 } \texttt{shortcut} \wedge\, c_1{=}c_2).
$$

### `BottleneckSAC(c1, c2, shortcut=True, g=1, k=(3,3), e=1.0)`
第二个卷积为 `SAConv2d` 的瓶颈。SAC 构建块——插入任意 C3/C2f 阶段即令该阶段具备 SAC 感知。

### `C3(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
CSP 阶段，三个卷积 + $n$ 个瓶颈（YOLOv5/v7）。切分通道，一半经 $n$ 个 `Bottleneck` 处理，与跳过的另一半拼接后投影。

### `C3SAC(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
`C3` 以 `BottleneckSAC` 替换 `Bottleneck`。

### `C3TR(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
瓶颈序列被 `TransformerBlock` 替换的 `C3`。**TR 增强**：融合局部 CNN 特征与全局自注意力的 CSP 阶段。

### `C2f(c1, c2, n=1, shortcut=False, g=1, e=0.5)`
YOLOv8 的 CSP 阶段。比 `C3` 更快，因其将通道切分为更多分区并跨分区复用瓶颈输出。

### `C2fSAC(c1, c2, n=1, shortcut=False, g=1, e=0.5)`
内部瓶颈序列使用 `BottleneckSAC` 的 `C2f`。

### `C2fCIB(c1, c2, n=1, shortcut=False, ev=0.5, lk=False)` / `CIB(...)`
YOLOv10 的 CIB（带植入分支的跨阶段部分瓶颈）C2f 变体——用于 v10 主干。

### `SPPF(c1, c2, k=5)`
快速空间金字塔池化：单个 $5{\times}5$ MaxPool 串行应用三次后拼接并投影。YOLOv5/v8 头部标准准备：
$$
\mathbf{y} = \mathrm{Conv}\big(\mathrm{cat}[\mathbf{x},\, \mathrm{MP}_5(\mathbf{x}),\, \mathrm{MP}_5^2(\mathbf{x}),\, \mathrm{MP}_5^3(\mathbf{x})]\big).
$$

### `SPPCSPC(c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5,9,13))`
YOLOv7 的 SPP-CSP 变体（多核池化 $\{5,9,13\}$ + CSP 融合）。

### `RepConv(c1, c2, k=3, s=1, p=1, g=1, d=1, act=True)`
RepVGG 风格可重参数化卷积。训练时为多分支（$3{\times}3$ 卷积 $+$ $1{\times}1$ 卷积 $+$ 恒等 BN），评估时为单一融合卷积。对 $3{\times}3$ 分支核 $W^{(3)}$、$1{\times}1$ 分支 $W^{(1)}$、BN 参数 $(\gamma,\beta,\mu,\sigma)$，融合将 BN 折入卷积后逐分支求和：
$$
W_{\text{fused}} = \frac{\gamma}{\sigma}(W \cdot \mathbf{1}_{k\times k}), \qquad b_{\text{fused}} = \beta - \frac{\gamma\,\mu}{\sigma},
$$
其中 $W\cdot\mathbf{1}_{k\times k}$ 将 $1{\times}1$ 核零填充为 $3{\times}3$。部署前调用 `fuse_repvgg`。YOLOv7 使用。

### `SCDown(c1, c2, k=3, s=2, d=1, act=True)`
YOLOv10 的空间-通道解耦下采样：将空间下采样（深度卷积，stride 2）与通道混合（点卷积）解耦。

### `PSA(c1, c2, e=0.5)`
YOLOv10 的部分自注意力。切分通道，对一半施加轻量多头注意力后拼接回。位于主干 P5 阶段。

### `Concat(dimension=1)`
拼接层（在颈部合并跳跃连接）。

## 3. YOLOv9 专用算子 —— `yolov9.py`

### `Silence()`
空操作层（v9 主干索引 0 占位）。

### `SP(k=3, s=1)`
空间池化——核 $k$、stride $s$ 的最大池化。

### `ADown(c1, c2)`
平均池化 $\to$ 切分 $\to$（conv $3{\times}3$/s2, maxpool+conv$1{\times}1$）下采样器。空间分辨率减半；通道约翻倍。用于 GELAN 阶段之间。

### `RepConvN(c1, c2, n=1, se=False, act=True)`
YOLOv9 可重参数化卷积块：$n$ 个 $3{\times}3$ 卷积序列，可选 Squeeze-and-Excitation。评估时由 `fuse_repvgg` 融合。**与 SAC 不兼容**（重参数化会破坏 SAC 切换）。

### `RepNCSPELAN4(c1, c2, c3, c4, c5=1)`
YOLOv9 GELAN 构建块：带 `RepConvN` 的 CSP-ELAN。切分通道，一支经 `RepConvN` 序列处理后拼接并投影。P2/P3/P4/P5 各一。

### `SPPELAN(c1, c2, c3, k=(5,9,13))`
ELAN 风格拼接的 SPP，位于 v9 颈部底部。

## 4. Transformer —— `transformer.py`

### `TransformerLayer(c, num_heads=8, dropout=0.0, ff_mult=4)`
单个 transformer 块。输入 $\mathbf{X}\in\mathbb{R}^{L\times c}$（序列长 $L$）。$h$ 头自注意力，头维 $d_h=c/h$：
$$
\mathrm{MHA}(\mathbf{X}) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O, \qquad \mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right)V_i,
$$
$$
Q_i = \mathbf{X}W_i^Q,\quad K_i = \mathbf{X}W_i^K,\quad V_i = \mathbf{X}W_i^V.
$$
随后残差与 MLP（Linear $\to$ 激活 $\to$ Linear）：
$$
\mathbf{X}' = \mathbf{X} + \mathrm{MHA}(\mathbf{X}), \qquad
\mathbf{Y} = \mathbf{X}' + \mathrm{MLP}(\mathbf{X}').
$$
**无 LayerNorm** —— 与 ultralytics 的 v5/v8 transformer 一致（省略以保持 `state_dict` 键最小）。

### `TransformerBlock(c1, c2, num_heads=4, num_layers=2, dropout=0.0)`
卷积投影 $\to$ 可学习位置嵌入 $\to$ $N{\times}$ `TransformerLayer`。`C3TR` 使用。

## 5. 隐式知识 —— `implicit.py`（YOLOv7）

### `ImplicitA(channel, mean=0.0, std=0.02)`
加性隐式知识——可学习的逐通道偏置 $\mathbf{a}\in\mathbb{R}^{C}$，加到前驱卷积输出：$\mathbf{y} = \mathrm{Conv}(\mathbf{x}) + \mathbf{a}$。

### `ImplicitM(channel, mean=1.0, std=0.02)`
乘性隐式知识——可学习的逐通道缩放 $\mathbf{m}\in\mathbb{R}^{C}$：$\mathbf{y} = \mathbf{m} \odot \mathrm{Conv}(\mathbf{x})$。

两者均有 `fuse(into_conv)`，将隐式折叠进前驱卷积的权重/偏置以供部署。YOLOv7 的 `IDetect` / `IAuxDetect` 使用。

## 6. 检测头 —— `detect_heads.py`

### 6.1 基于锚框（v5/v7）

- `DetectAnchorBased(nc=80, anchors=(), ch=())` —— YOLOv5 头。三个尺度 $\times$ 每尺度三锚。训练前向返回 $(B, 3, H, W, n_c{+}5)$ 原始张量列表；评估前向解码为 $(B, N, n_c{+}5)$，$640{\times}640$ 时 $N{=}25200$。每锚预测 $=(\Delta x, \Delta y, w, h, \text{obj}, \mathbf{p})$ 解码为
  $$
  x = (2\sigma(\Delta x)-1) + c_x,\quad y = (2\sigma(\Delta y)-1) + c_y,\quad w = p_w\,e^{\Delta w},\quad h = p_h\,e^{\Delta h},
  $$
  其中 $c_x,c_y$ 为网格偏移，$p_w,p_h$ 为锚先验，$\mathbf{p}$ 为类别概率。
- `IDetect` —— 带 `ImplicitA`/`ImplicitM` 的 YOLOv7 头。
- `IAuxDetect` —— YOLOv7 辅助头（仅训练；输出最终 + 辅助）。

### 6.2 无锚框（v8/v9/v10）

- `DetectAnchorFree(nc=80, reg_max=16, ch=())` —— YOLOv8/v9 头。每尺度两支：`cv2`（框，$4{\times}r_{\max}$ 通道）与 `cv3`（类，$n_c$ 通道）。训练前向返回 `{"boxes", "scores", "feats"}`；评估前向解码为 $(B, 4{+}n_c, N)$，$640{\times}640$ 时 $N{=}8400$。框距 $(l,t,r,b)$ 由 DFL 回归。
- `DFL(reg_max=16)` —— **分布焦点损失**层：对 $r_{\max}$ 个 bin 的 softmax 后接期望加权和，给出每坐标可微标量。对分布 logits $\{p_i\}_{i=0}^{r_{\max}-1}$：
  $$
  \hat{d} = \sum_{i=0}^{r_{\max}-1} \frac{i}{r_{\max}-1}\cdot \mathrm{softmax}(p_i).
  $$
- `v10Detect(nc=80, reg_max=16, ch=())` —— YOLOv10 双头：`one2many`（top-10 匹配，训练）+ `one2one`（top-1，无 NMS 推理）。

### 6.3 辅助函数

- `make_anchors(feats, strides, grid_cell_offset=0.5)` —— 由特征图构建锚点网格。
- `dist2bbox(distance, anchor_points, xywh=True, dim=-1)` —— 将 $(l,t,r,b)$ 距离解码为 xywh/xyxy。
- `bbox2dist(anchor_points, bbox, xywh=True)` —— `dist2bbox` 的逆。

## 7. 激活 —— `activation.py`

- `SiLU`（$x\,\mathrm{sigmoid}(x)$）、`Mish`（$x\,\mathrm{tanh}(\mathrm{softplus}(x))$）、`ReLU`（$\max(0,x)$）自 torch 再导出。
- `parse_activation(name_or_module)` —— 工厂，接受字符串（`"silu"`、`"mish"`、`"relu"`）或 `nn.Module`；使 zoo 类可在构造器暴露 `act: str | nn.Module = "silu"`。
