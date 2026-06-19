# Operators (`cracks_yolo.ops`)

[English](ops.md) | [中文](ops.zh-CN.md)

Plain `nn.Module` / `nn.Conv2d` subclasses. No abstract base, no plugin registry -- each operator is self-contained. Constructor signatures are preserved verbatim from upstream so COCO pretrained state_dicts load cleanly (state_dict key names matter).

## Convolution -- `conv.py`

### `autopad(k, p=None, d=1) -> int`
Auto-calculate padding for `'same'`-shape conv output.

### `Conv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)`
Conv2d + BatchNorm2d + activation (default SiLU). This is the workhorse of every YOLO family.

### `DWConv(c2, k=1, s=1, d=1, act=True)`
Depth-wise variant: groups = input channels.

### `ConvAWS2d(c1, c2, k=1, s=1, p=None, g=1, d=1, act=None)`
Weight-standardized Conv2d (used by SAC). Multiplies weights by `(weight - weight.mean()) / (weight.std() + eps)` before the forward convolution.

**Math:**
$$W' = \gamma \cdot \frac{W - \mu_W}{\sigma_W + \epsilon}, \quad y = W' * x + b$$

### `SAConv2d(c1, c2, k=1, s=1, p=None, g=1, d=1, act=None)`
**Switchable Atrous Convolution (SAC)** -- the core enhancement. Wraps a `ConvAWS2d` and learns a per-channel switch over three atrous rates (`{1, 2, 3}` by default via `a=[1, 2, 3]`). An attention-like module produces per-channel, per-spatial-location weights that interpolate between the three dilated convolutions. A second global scalar multiplies the output (initialized so SAC starts approx. identity).

**Why SAC for tongue surface crack detection?** Cracks vary wildly in scale within a single image. A single dilation rate cannot capture both thin hairline cracks and wider spalls. SAC lets the network pick the right receptive field per-pixel.

**Bias handling:** `bias=False` so the bias folds into the following BatchNorm. This was the cause of a real bug -- with `bias=True` the SAC bias params received no gradient because they were unused.

## CSP -- `csp.py`

### `Bottleneck(c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5)`
Standard residual bottleneck: 1x1 conv -> 3x3 conv -> optional residual add.

### `BottleneckSAC(c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0)`
Bottleneck whose second conv is `SAConv2d` (instead of `Conv`). This is the SAC building block -- plug it into any C3/C2f stage to make that stage SAC-aware.

### `C3(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
CSP-style stage with 3 convs + N bottlenecks. Used by YOLOv5 and YOLOv7.

### `C3SAC(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
`C3` with `BottleneckSAC` instead of `Bottleneck`.

### `C3TR(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
`C3` whose bottleneck sequence is replaced by a `TransformerBlock`. This is the **TR enhancement**: a CSP stage that fuses local CNN features with global self-attention.

### `C2f(c1, c2, n=1, shortcut=False, g=1, e=0.5)`
YOLOv8's CSP stage. Faster than `C3` because it splits channels and reuses the bottleneck output across the split.

### `C2fSAC(c1, c2, n=1, shortcut=False, g=1, e=0.5)`
`C2f` with `BottleneckSAC` in its inner bottleneck sequence.

### `C2fCIB(c1, c2, n=1, shortcut=False, ev=0.5, lk=False)`
YOLOv10's CIB (Cross-Stage Partial Bottleneck with Implant Branches) variant of C2f. Used in the v10 backbone.

### `CIB(c1, c2, shortcut=False, e=0.5, lk=False)`
Cross-Stage Partial Bottleneck with an implant branch -- v10 building block.

### `SPPF(c1, c2, k=5)`
Spatial Pyramid Pooling - Fast (single 5x5 MaxPool applied 3x in sequence). Standard YOLOv5/v8 head prep.

### `SPPCSPC(c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13))`
SPP-CSP variant used by YOLOv7.

### `RepConv(c1, c2, k=3, s=1, p=1, g=1, d=1, act=True)`
RepVGG-style reparameterizable conv. Train-time = multi-branch (3x3 conv + 1x1 conv + identity BN), eval-time = single fused conv. Used by YOLOv7. Call `fuse_repvgg` on the model before deployment.

### `SCDown(c1, c2, k=3, s=2, d=1, act=True)`
YOLOv10's Spatial-Channel Decoupled Downsampling. Decouples spatial downsampling (DW conv stride 2) from channel mixing (PW conv).

### `PSA(c1, c2, e=0.5)`
YOLOv10's Partial Self-Attention. Splits channels, applies lightweight multi-head attention to one half, concatenates back. Located at the backbone P5 stage.

### `Concat(dimension=1)`
Concatenation layer (used in necks to merge skip connections). Tracked explicitly in `_forward_impl` because its input is a list, not a tensor.

## YOLOv9-specific operators -- `yolov9.py`

### `Silence()`
No-op layer (v9 backbone index 0 placeholder).

### `SP(k=3, s=1)`
Spatial pooling -- max-pool with kernel `k` and stride `s`.

### `ADown(c1, c2)`
Avg-pool + chunk + (conv3x3/s2, maxpool+conv1x1) downsampler. Halves spatial resolution; doubles-ish channels. Used in the v9 backbone between GELAN stages.

### `RepConvN(c1, c2, n=1, se=False, act=True)`
YOLOv9's reparameterizable conv block. A sequence of `n` convs (3x3) with optional Squeeze-and-Excitation. During eval, `fuse_repvgg` merges the multi-branch into a single conv. Not compatible with SAC (re-parameterization would break the SAC switches).

### `RepNCSPELAN4(c1, c2, c3, c4, c5=1)`
YOLOv9's GELAN building block: CSP-ELAN with `RepConvN`. Splits channels, processes one branch through `RepConvN` sequence, concatenates, and projects. The v9 backbone uses four of these at P2/P3/P4/P5.

### `SPPELAN(c1, c2, c3, k=(5, 9, 13))`
SPP with ELAN-style concatenation. Used at the bottom of the v9 neck before the FPN/PAN.

## Transformer -- `transformer.py`

### `TransformerLayer(c, num_heads=8, dropout=0.0, ff_mult=4)`
One transformer block: QKV projection -> `nn.MultiheadAttention` -> residual -> MLP (Linear -> activation -> Linear). **No LayerNorm** -- matches ultralytics' v5/v8 transformer (LayerNorm is omitted to keep state_dict keys minimal).

### `TransformerBlock(c1, c2, num_heads=4, num_layers=2, dropout=0.0)`
Conv projection -> learned positional embedding -> N x `TransformerLayer`. Used by `C3TR`.

## Implicit -- `implicit.py` (YOLOv7)

### `ImplicitA(channel, mean=0.0, std=0.02)`
Additive implicit knowledge -- a learned per-channel bias added to the preceding conv's output.

### `ImplicitM(channel, mean=1.0, std=0.02)`
Multiplicative implicit knowledge -- a learned per-channel scale.

Both have a `fuse(into_conv)` method that folds the implicit into a preceding conv's weights/bias for deployment. Used by `IDetect` and `IAuxDetect` in YOLOv7.

## Detect heads -- `detect_heads.py`

### Anchor-based (v5/v7)

- `DetectAnchorBased(nc=80, anchors=(), ch=())` -- YOLOv5 head. Three scales x three anchors per scale. Training forward returns a list of `(B, 3, H, W, nc+5)` raw tensors; eval forward decodes to `(B, N, nc+5)` where N = 25200 at 640x640.
- `IDetect(nc=80, anchors=(), ch=())` -- YOLOv7 head with `ImplicitA`/`ImplicitM`.
- `IAuxDetect(nc=80, anchors=(), ch=())` -- YOLOv7 auxiliary head (training only -- emits both final and auxiliary outputs).

### Anchor-free (v8/v9/v10)

- `DetectAnchorFree(nc=80, reg_max=16, ch=())` -- YOLOv8/v9 head. Two branches per scale: `cv2` (box, output 4x`reg_max` channels) and `cv3` (cls, `nc` channels). Training forward returns `{"boxes": ..., "scores": ..., "feats": [...]}`; eval forward decodes to `(B, 4+nc, N)` where N = 8400 at 640x640.
- `DFL(reg_max=16)` -- Distribution Focal Loss layer: softmax over `reg_max` bins, weighted sum, gives a differentiable scalar per coordinate.
- `v10Detect(nc=80, reg_max=16, ch=())` -- YOLOv10 dual head: `one2many` (top-10 matching, for training) + `one2one` (top-1, NMS-free inference).

### Helpers

- `make_anchors(feats, strides, grid_cell_offset=0.5)` -- build anchor point grids from feature maps.
- `dist2bbox(distance, anchor_points, xywh=True, dim=-1)` -- decode `dist` (left/top/right/bottom) to xywh or xyxy.
- `bbox2dist(anchor_points, bbox, xywh=True)` -- inverse of `dist2bbox`.

## Activation -- `activation.py`

- `SiLU`, `Mish`, `ReLU` re-exported from torch.
- `parse_activation(name_or_module)` -- factory: accept a string (`"silu"`, `"mish"`, `"relu"`) or an `nn.Module` instance, return the activation. Lets zoo classes expose `act: str | nn.Module = "silu"` in their constructors.
