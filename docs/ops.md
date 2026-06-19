# Operators (`cracks_yolo.ops`)

Plain `nn.Module` / `nn.Conv2d` subclasses. No abstract base, no plugin
registry — each operator is self-contained. Constructor signatures are
preserved verbatim from upstream so COCO pretrained state_dicts load
cleanly (state_dict key names matter).

## Convolution — `conv.py`

### `autopad(k, p=None, d=1) -> int`
Auto-calculate padding for `'same'`-shape conv output.

### `Conv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)`
Conv2d + BatchNorm2d + activation (default SiLU). This is the workhorse
of every YOLO family.

### `DWConv(c2, k=1, s=1, d=1, act=True)`
Depth-wise variant: groups = input channels.

### `ConvAWS2d(c1, c2, k=1, s=1, p=None, g=1, d=1, act=None)`
Weight-standardized Conv2d (used by SAC). Multiplies weights by
`(weight - weight.mean()) / (weight.std() + eps)` before the forward
convolution. Ported verbatim from `cracks_yolo/compat/ultralytics/conv.py`
(now deleted).

**Math:**
$$W' = \gamma \cdot \frac{W - \mu_W}{\sigma_W + \epsilon}, \quad y = W' * x + b$$

### `SAConv2d(c1, c2, k=1, s=1, p=None, g=1, d=1, act=None)`
**Switchable Atrous Convolution (SAC)** — the core enhancement. Wraps a
`ConvAWS2d` and learns a per-channel switch over three atrous rates
(`{1, 2, 3}` by default via `a=[1, 2, 3]`). An attention-like module
produces per-channel, per-spatial-location weights that interpolate between
the three dilated convolutions. A second global scalar multiplies the
output (initialized so SAC starts ≈ identity).

**Why SAC?** Cracks vary wildly in scale within a single image. A single
dilation rate can't capture both thin hairline cracks and wider spalls.
SAC lets the network pick the right receptive field per-pixel.

**Bias handling:** `bias=False` so the bias folds into the following
BatchNorm. This was the cause of a real bug — with `bias=True` the SAC
bias params received no gradient because they were unused.

## CSP — `csp.py`

### `Bottleneck(c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5)`
Standard residual bottleneck: 1×1 conv → 3×3 conv → optional residual add.

### `BottleneckSAC(c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0)`
Bottleneck whose second conv is `SAConv2d` (instead of `Conv`). This is
the SAC building block — plug it into any C3/C2f stage to make that stage
SAC-aware.

### `C3(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
CSP-style stage with 3 convs + N bottlenecks. Used by YOLOv5.

### `C3SAC(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
`C3` with `BottleneckSAC` instead of `Bottleneck`.

### `C3TR(c1, c2, n=1, shortcut=True, g=1, e=0.5)`
`C3` whose bottleneck sequence is replaced by a `TransformerBlock`. This
is the **TR enhancement**: a CSP stage that fuses local CNN features with
global self-attention.

### `C2f(c1, c2, n=1, shortcut=False, g=1, e=0.5)`
YOLOv8's CSP stage. Faster than `C3` because it splits channels and
reuses the bottleneck output across the split.

### `C2fSAC(c1, c2, n=1, shortcut=False, g=1, e=0.5)`
`C2f` with `BottleneckSAC` in its inner bottleneck sequence.

### `C2fCIB(c1, c2, n=1, shortcut=False, ev=0.5, lk=False)`
YOLOv10's CIB (Cross-Stage Partial Bottleneck with Implant Branches)
variant of C2f. Used in the v10 backbone.

### `CIB(c1, c2, shortcut=False, e=0.5, lk=False)`
Cross-Stage Partial Bottleneck with an implant branch — v10 building block.

### `SPPF(c1, c2, k=5)`
Spatial Pyramid Pooling - Fast (single 5×5 MaxPool applied 3× in
sequence). Standard YOLOv5/v8 head prep.

### `SPPCSPC(c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13))`
SPP-CSP variant used by YOLOv7.

### `RepConv(c1, c2, k=3, s=1, p=1, g=1, d=1, act=True)`
RepVGG-style reparameterizable conv. Train-time = multi-branch
(3×3 conv + 1×1 conv + identity BN), eval-time = single fused conv.
Used by YOLOv7. Call `fuse_repvgg` on the model before deployment.

### `SCDown(c1, c2, k=3, s=2, d=1, act=True)`
YOLOv10's Spatial-Channel Decoupled Downsampling. Decouples spatial
downsampling (DW conv stride 2) from channel mixing (PW conv).

### `PSA(c1, c2, e=0.5)`
YOLOv10's Partial Self-Attention. Splits channels, applies lightweight
multi-head attention to one half, concatenates back. Located at the
backbone P5 stage.

### `Concat(dimension=1)`
Concatenation layer (used in necks to merge skip connections). Tracked
explicitly in `_forward_impl` because its input is a list, not a tensor.

## Transformer — `transformer.py`

### `TransformerLayer(c, num_heads=8, dropout=0.0, ff_mult=4)`
One transformer block: QKV projection → `nn.MultiheadAttention` →
residual → MLP (Linear → activation → Linear). **No LayerNorm** — matches
ultralytics' v5/v8 transformer (LayerNorm is omitted to keep state_dict
keys minimal).

### `TransformerBlock(c1, c2, num_heads=4, num_layers=2, dropout=0.0)`
Conv projection → learned positional embedding → N × `TransformerLayer`.
Used by `C3TR`.

## Implicit — `implicit.py` (YOLOv7)

### `ImplicitA(channel, mean=0.0, std=0.02)`
Additive implicit knowledge — a learned per-channel bias added to the
preceding conv's output.

### `ImplicitM(channel, mean=1.0, std=0.02)`
Multiplicative implicit knowledge — a learned per-channel scale.

Both have a `fuse(into_conv)` method that folds the implicit into a
preceding conv's weights/bias for deployment. Used by `IDetect` and
`IAuxDetect` in YOLOv7.

## Detect heads — `detect_heads.py`

### Anchor-based (v5/v7)

- `DetectAnchorBased(nc=80, anchors=(), ch=())` — YOLOv5 head. Three
  scales × three anchors per scale. Training forward returns a list of
  `(B, 3, H, W, nc+5)` raw tensors; eval forward decodes to
  `(B, N, nc+5)` where N = 25200 at 640×640.
- `IDetect(nc=80, anchors=(), ch=())` — YOLOv7 head with `ImplicitA`/`ImplicitM`.
- `IAuxDetect(nc=80, anchors=(), ch=())` — YOLOv7 auxiliary head
  (training only — emits both final and auxiliary outputs).

### Anchor-free (v8/v10)

- `DetectAnchorFree(nc=80, reg_max=16, ch=())` — YOLOv8 head. Two
  branches per scale: `cv2` (box, output 4×`reg_max` channels) and `cv3`
  (cls, `nc` channels). Training forward returns `{"boxes": ..., "scores": ..., "feats": [...]}`;
  eval forward decodes to `(B, 4+nc, N)` where N = 8400 at 640×640.
- `DFL(reg_max=16)` — Distribution Focal Loss layer: softmax over
  `reg_max` bins, weighted sum, gives a differentiable scalar per
  coordinate.
- `v10Detect(nc=80, reg_max=16, ch=())` — YOLOv10 dual head:
  `one2many` (top-10 matching, for training) + `one2one` (top-1, NMS-free
  inference).

### Helpers

- `make_anchors(feats, strides, grid_cell_offset=0.5)` — build anchor
  point grids from feature maps.
- `dist2bbox(distance, anchor_points, xywh=True, dim=-1)` — decode
  `dist` (left/top/right/bottom) to xywh or xyxy.
- `bbox2dist(anchor_points, bbox, xywh=True)` — inverse of `dist2bbox`.

## Activation — `activation.py`

- `SiLU`, `Mish`, `ReLU` re-exported from torch.
- `parse_activation(name_or_module)` — factory: accept a string
  (`"silu"`, `"mish"`, `"relu"`) or an `nn.Module` instance, return the
  activation. Lets zoo classes expose `act: str | nn.Module = "silu"` in
  their constructors.
