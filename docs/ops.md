# Operators (`cracks_yolo.ops`)

[English](ops.md) | [中文](ops.zh-CN.md)

Plain `nn.Module` / `nn.Conv2d` subclasses. No abstract base, no plugin registry — each operator is self-contained. Constructor signatures are preserved verbatim from upstream so COCO pretrained `state_dict`s load cleanly (key names matter). Below, $H,W$ denote spatial dimensions, $C_{\text{in}},C_{\text{out}}$ channel counts, and $\sigma(\cdot)$ the SiLU activation $\sigma(x)=x\,\mathrm{sigmoid}(x)$.

## 1. Convolution — `conv.py`

### `autopad(k, p=None, d=1) -> int`
Returns the padding for `'same'`-spatial output: $p_{\text{auto}} = \lfloor k/2 \rfloor$ (adjusted for dilation $d$).

### `Conv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)`
Conv2d + BatchNorm2d + activation (SiLU). The workhorse of every YOLO family:
$$
\mathbf{y} = \sigma\!\big(\mathrm{BN}(\mathrm{Conv}_{k,s}(\mathbf{x}))\big).
$$

### `DWConv(c2, k=1, s=1, d=1, act=True)`
Depth-wise variant: grouped convolution with $g = C_{\text{in}}$, each input channel convolved with its own kernel.

### `ConvAWS2d(c1, c2, k=1, s=1, p=None, g=1, d=1, act=None)`
Weight-standardized Conv2d (used by SAC). The convolution weight $W$ is normalized along its spatial axes before the forward pass:
$$
\mu_W = \frac{1}{|\mathcal{S}|}\sum_{i\in\mathcal{S}} W_i, \qquad
\sigma_W = \sqrt{\frac{1}{|\mathcal{S}|}\sum_{i\in\mathcal{S}} (W_i - \mu_W)^2}, \qquad
W' = \gamma \cdot \frac{W - \mu_W}{\sigma_W + \epsilon},
$$
$$
\mathbf{y} = W' * \mathbf{x} + b,
$$
where $\mathcal{S}$ indexes the spatial kernel positions, $\gamma$ a learned per-channel scale, $\epsilon$ a small constant. Weight standardization stabilizes training of small-batch detection heads.

### `SAConv2d(c1, c2, k=1, s=1, p=None, g=1, d=1, act=None)`
**Switchable Atrous Convolution (SAC)** — the core enhancement. SAC wraps a `ConvAWS2d` and learns a per-pixel, per-channel switch over a set of atrous (dilation) rates $\mathcal{A}=\{a_1,a_2,a_3\}$ (default $\{1,2,3\}$). Three parallel dilated convolutions produce feature maps $\{\mathbf{F}_{a}\}_{a\in\mathcal{A}}$, and an attention-like module emits spatial weights $\{w_a(\mathbf{u})\}$ at each location $\mathbf{u}$; the output is the convex combination
$$
\mathbf{F}_{\text{SAC}}(\mathbf{u}) = \sum_{a\in\mathcal{A}} w_a(\mathbf{u}) \cdot \mathrm{ConvAWS}_{a}(\mathbf{x})(\mathbf{u}), \qquad \sum_{a\in\mathcal{A}} w_a(\mathbf{u}) = 1.
$$
A second global scalar $\alpha$ (initialized so SAC starts near identity) scales the output: $\mathbf{y} = \alpha \cdot \mathbf{F}_{\text{SAC}}$.

**Why SAC for tongue surface crack detection?** Cracks vary wildly in scale within a single image — thin hairline cracks and wider spalls coexist. A single dilation rate cannot capture both. SAC lets the network select the receptive field per pixel.

**Bias handling:** `bias=False` so the bias folds into the following BatchNorm. With `bias=True` the SAC bias params receive no gradient (they are unused) — a real bug this convention avoids.

## 2. CSP building blocks — `csp.py`

### `Bottleneck(c1, c2, shortcut=True, g=1, k=(3,3), e=0.5)`
Standard residual bottleneck: $1{\times}1$ conv $\to$ $3{\times}3$ conv $\to$ optional residual add,
$$
\mathbf{y} = \mathrm{Conv}_{3\times3}(\mathrm{Conv}_{1\times1}(\mathbf{x})) + \mathbf{x} \quad (\text{if } \texttt{shortcut} \wedge\, c_1{=}c_2).
$$

### `BottleneckSAC(c1, c2, shortcut=True, g=1, k=(3,3), e=1.0)`
Bottleneck whose second conv is `SAConv2d` instead of `Conv`. The SAC building block — plug it into any C3/C2f stage to make that stage SAC-aware.

### `C3(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
CSP stage with three convs + $n$ bottlenecks (YOLOv5/v7). Splits channels, processes one half through $n$ `Bottleneck`s, concatenates with the skipped half, and projects.

### `C3SAC(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
`C3` with `BottleneckSAC` replacing `Bottleneck`.

### `C3TR(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
`C3` whose bottleneck sequence is replaced by a `TransformerBlock`. The **TR enhancement**: a CSP stage fusing local CNN features with global self-attention.

### `C2f(c1, c2, n=1, shortcut=False, g=1, e=0.5)`
YOLOv8's CSP stage. Faster than `C3` because it splits channels into more partitions and reuses the bottleneck output across splits.

### `C2fSAC(c1, c2, n=1, shortcut=False, g=1, e=0.5)`
`C2f` with `BottleneckSAC` in its inner bottleneck sequence.

### `C2fCIB(c1, c2, n=1, shortcut=False, ev=0.5, lk=False)` / `CIB(...)`
YOLOv10's CIB (Cross-Stage Partial Bottleneck with Implant Branches) variant of C2f — used in the v10 backbone.

### `SPPF(c1, c2, k=5)`
Spatial Pyramid Pooling — Fast: a single $5{\times}5$ MaxPool applied three times in sequence, then concatenated and projected. Standard YOLOv5/v8 head prep:
$$
\mathbf{y} = \mathrm{Conv}\big(\mathrm{cat}[\mathbf{x},\, \mathrm{MP}_5(\mathbf{x}),\, \mathrm{MP}_5^2(\mathbf{x}),\, \mathrm{MP}_5^3(\mathbf{x})]\big).
$$

### `SPPCSPC(c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5,9,13))`
SPP-CSP variant used by YOLOv7 (multi-kernel pooling $\{5,9,13\}$ + CSP fusion).

### `RepConv(c1, c2, k=3, s=1, p=1, g=1, d=1, act=True)`
RepVGG-style reparameterizable conv. Training time is multi-branch ($3{\times}3$ conv $+$ $1{\times}1$ conv $+$ identity BN); eval time is a single fused conv. For a $3{\times}3$ branch weight $W^{(3)}$, $1{\times}1$ branch $W^{(1)}$, and BN parameters $(\gamma,\beta,\mu,\sigma)$, fusion folds BN into the conv then sums branches:
$$
W_{\text{fused}} = \frac{\gamma}{\sigma}(W \cdot \mathbf{1}_{k\times k}), \qquad b_{\text{fused}} = \beta - \frac{\gamma\,\mu}{\sigma},
$$
where $W\cdot\mathbf{1}_{k\times k}$ zero-pads a $1{\times}1$ kernel to $3{\times}3$. Call `fuse_repvgg` before deployment. Used by YOLOv7.

### `SCDown(c1, c2, k=3, s=2, d=1, act=True)`
YOLOv10's Spatial-Channel Decoupled Downsampling: decouples spatial downsampling (depth-wise conv, stride 2) from channel mixing (point-wise conv).

### `PSA(c1, c2, e=0.5)`
YOLOv10's Partial Self-Attention. Splits channels, applies lightweight multi-head attention to one half, concatenates back. Located at the backbone P5 stage.

### `Concat(dimension=1)`
Concatenation layer (merges skip connections in necks).

## 3. YOLOv9-specific operators — `yolov9.py`

### `Silence()`
No-op layer (v9 backbone index-0 placeholder).

### `SP(k=3, s=1)`
Spatial pooling — max-pool with kernel $k$, stride $s$.

### `ADown(c1, c2)`
Avg-pool $\to$ chunk $\to$ (conv $3{\times}3$/s2, maxpool+conv$1{\times}1$) downsampler. Halves spatial resolution; ~doubles channels. Used between GELAN stages.

### `RepConvN(c1, c2, n=1, se=False, act=True)`
YOLOv9 reparameterizable conv block: a sequence of $n$ $3{\times}3$ convs with optional Squeeze-and-Excitation. Fused via `fuse_repvgg` at eval. **Not compatible with SAC** (re-parameterization would break the SAC switches).

### `RepNCSPELAN4(c1, c2, c3, c4, c5=1)`
YOLOv9 GELAN building block: CSP-ELAN with `RepConvN`. Splits channels, processes one branch through a `RepConvN` sequence, concatenates, and projects. Four of these at P2/P3/P4/P5.

### `SPPELAN(c1, c2, c3, k=(5,9,13))`
SPP with ELAN-style concatenation, at the bottom of the v9 neck.

## 4. Transformer — `transformer.py`

### `TransformerLayer(c, num_heads=8, dropout=0.0, ff_mult=4)`
One transformer block. Input $\mathbf{X}\in\mathbb{R}^{L\times c}$ (sequence length $L$). Multi-head self-attention with $h$ heads, head dimension $d_h=c/h$:
$$
\mathrm{MHA}(\mathbf{X}) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O, \qquad \mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right)V_i,
$$
$$
Q_i = \mathbf{X}W_i^Q,\quad K_i = \mathbf{X}W_i^K,\quad V_i = \mathbf{X}W_i^V.
$$
Followed by a residual and an MLP (Linear $\to$ activation $\to$ Linear):
$$
\mathbf{X}' = \mathbf{X} + \mathrm{MHA}(\mathbf{X}), \qquad
\mathbf{Y} = \mathbf{X}' + \mathrm{MLP}(\mathbf{X}').
$$
**No LayerNorm** — matches ultralytics' v5/v8 transformer (omitted to keep `state_dict` keys minimal).

### `TransformerBlock(c1, c2, num_heads=4, num_layers=2, dropout=0.0)`
Conv projection $\to$ learned positional embedding $\to$ $N{\times}$ `TransformerLayer`. Used by `C3TR`.

## 5. Implicit knowledge — `implicit.py` (YOLOv7)

### `ImplicitA(channel, mean=0.0, std=0.02)`
Additive implicit knowledge — a learned per-channel bias $\mathbf{a}\in\mathbb{R}^{C}$ added to the preceding conv output: $\mathbf{y} = \mathrm{Conv}(\mathbf{x}) + \mathbf{a}$.

### `ImplicitM(channel, mean=1.0, std=0.02)`
Multiplicative implicit knowledge — a learned per-channel scale $\mathbf{m}\in\mathbb{R}^{C}$: $\mathbf{y} = \mathbf{m} \odot \mathrm{Conv}(\mathbf{x})$.

Both have `fuse(into_conv)` that folds the implicit into a preceding conv's weights/bias for deployment. Used by `IDetect` / `IAuxDetect` in YOLOv7.

## 6. Detect heads — `detect_heads.py`

### 6.1 Anchor-based (v5/v7)

- `DetectAnchorBased(nc=80, anchors=(), ch=())` — YOLOv5 head. Three scales $\times$ three anchors per scale. Training forward returns a list of $(B, 3, H, W, n_c{+}5)$ raw tensors; eval forward decodes to $(B, N, n_c{+}5)$ with $N{=}25200$ at $640{\times}640$. Per anchor, prediction $=(\Delta x, \Delta y, w, h, \text{obj}, \mathbf{p})$ decoded by
  $$
  x = (2\sigma(\Delta x)-1) + c_x,\quad y = (2\sigma(\Delta y)-1) + c_y,\quad w = p_w\,e^{\Delta w},\quad h = p_h\,e^{\Delta h},
  $$
  with grid offset $c_x,c_y$, anchor priors $p_w,p_h$, and $\mathbf{p}$ class probabilities.
- `IDetect` — YOLOv7 head with `ImplicitA`/`ImplicitM`.
- `IAuxDetect` — YOLOv7 auxiliary head (training only; emits final + auxiliary outputs).

### 6.2 Anchor-free (v8/v9/v10)

- `DetectAnchorFree(nc=80, reg_max=16, ch=())` — YOLOv8/v9 head. Two branches per scale: `cv2` (box, $4{\times}r_{\max}$ channels) and `cv3` (cls, $n_c$ channels). Training forward returns `{"boxes", "scores", "feats"}`; eval forward decodes to $(B, 4{+}n_c, N)$ with $N{=}8400$ at $640{\times}640$. Box distances $(l,t,r,b)$ from anchor points are regressed via DFL.
- `DFL(reg_max=16)` — **Distribution Focal Loss** layer: a softmax over $r_{\max}$ bins followed by an expectation-weighted sum, giving a differentiable scalar per coordinate. For distribution logits $\{p_i\}_{i=0}^{r_{\max}-1}$:
  $$
  \hat{d} = \sum_{i=0}^{r_{\max}-1} \frac{i}{r_{\max}-1}\cdot \mathrm{softmax}(p_i).
  $$
- `v10Detect(nc=80, reg_max=16, ch=())` — YOLOv10 dual head: `one2many` (top-10 matching, training) + `one2one` (top-1, NMS-free inference).

### 6.3 Helpers

- `make_anchors(feats, strides, grid_cell_offset=0.5)` — build anchor-point grids from feature maps.
- `dist2bbox(distance, anchor_points, xywh=True, dim=-1)` — decode $(l,t,r,b)$ distances to xywh/xyxy.
- `bbox2dist(anchor_points, bbox, xywh=True)` — inverse of `dist2bbox`.

## 7. Activation — `activation.py`

- `SiLU` ($x\,\mathrm{sigmoid}(x)$), `Mish` ($x\,\mathrm{tanh}(\mathrm{softplus}(x))$), `ReLU` ($\max(0,x)$) re-exported from torch.
- `parse_activation(name_or_module)` — factory accepting a string (`"silu"`, `"mish"`, `"relu"`) or an `nn.Module`; lets zoo classes expose `act: str | nn.Module = "silu"`.
