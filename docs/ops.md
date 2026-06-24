# Operators

## Convolution and CSP blocks

**Conv.** `Conv2d -> BatchNorm2d -> SiLU`. Defined in ultralytics, imported by all
downstream modules. Signature: `Conv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)`.

**Bottleneck.** Two-stage residual: 1x1 `Conv` reduces channels by `e` (default 0.5),
3x3 `Conv` restores to `c2`. Shortcut when `c1 == c2`. Inside SAC variants, `e` is
forced to 1.0.

**C3.** YOLOv5/v7 CSP block. Two 1x1 `Conv` projections process the input in parallel.
`cv1` feeds an `nn.Sequential` of `n` Bottlenecks; `cv2` bypasses unchanged. The two
halves are concatenated and fused by `cv3` (1x1).

**C2f.** YOLOv8/v9/v10 CSP variant. `cv1` (1x1) projects to `2c` channels, split via
`chunk(2, dim=1)`. The first half passes through; the second enters a chain of `n`
Bottlenecks in `nn.ModuleList`, each receiving the previous output — accumulating
features iteratively. All `n+1` tensors are concatenated and fused by `cv2` (1x1).

---

## SAC: Switchable Atrous Convolution

SAC replaces the inner 3x3 convolution of selected backbone Bottlenecks with a
learned per-pixel switch blending two atrous convolutions at dilations `d` and `3d`.

**ConvAWS2d.** Weight-standardised `Conv2d` subclass. Before each forward, the kernel
is normalised per output channel: mean over spatial dims subtracted, divided by
std-dev (+ 1e-5), rescaled by learned buffers `gamma` (init 1) and `beta` (init 0).
On `load_state_dict`, if `gamma.mean() <= 0`, both buffers are recomputed from the
incoming weight, enabling loading from non-AWS checkpoints.

**SAConv2d.** Pipeline: (1) *Pre-context* — global average pool, 1x1 conv, broadcast
residual. (2) *Switch* — reflect-pad (2px), 5x5 avg pool, 1x1 conv with sigmoid,
producing a single-channel soft gate. (3) *Dual atrous* — two `_conv_forward` calls
on the same weight-standardised kernel `w` (dilation `d`) and `w + weight_diff`
(dilation `3d`), where `weight_diff` is a learned zero-initialised `nn.Parameter`.
The switch fuses them: `switch * out_s + (1-switch) * out_l`. (4) *Post-context* —
global avg pool, 1x1 conv, residual. BatchNorm + SiLU complete the block. Switch
bias is initialised to 1, defaulting to the small-dilation branch.

**BottleneckSAC.** Drop-in replacement for Bottleneck. First 1x1 `Conv` unchanged;
second 3x3 `Conv` replaced by `SAConv2d` with `e=1.0`.

**C3SAC.** `C3` with `n` `BottleneckSAC(e=1.0)` in the inner `nn.Sequential`. For
YOLOv5/v7 backbones.

**C2fSAC.** `C2f` with `BottleneckSAC(e=1.0)` in the `nn.ModuleList` chain. For
YOLOv8/v9/v10 backbones.

---

## TR: Transformer

**TransformerLayer.** QKV self-attention without layer normalisation: three linear
projections `(q, k, v)`, `MultiheadAttention(batch_first=True)`, residual, two-layer
MLP (`Linear -> Linear`), second residual.

**TransformerBlock.** A 1x1 `Conv2d` projects input channels; a learned positional
embedding `(1, C, 1, 1)` is broadcast-added after flattening to `(B, H*W, C)`.
`nn.Sequential` of `num_layers` `TransformerLayer`s; output reshaped to `(B, C, H, W)`.

**C3TR.** `C3` where the inner Bottleneck sequence is replaced by
`TransformerBlock(c_, c_, num_heads=4, num_layers=n)`. The CSP split-transform-merge
structure is preserved (transformer processes only the `cv1` branch).

---

## Detect heads

Heads consume the multi-scale feature pyramid `[P3, P4, P5]` (stride 8, 16, 32).
All use decoupled branches: separate conv stacks for box regression and class
confidence.

**Detect (YOLOv5/v8).** For each layer, `cv2` predicts `4 * reg_max` box distribution
parameters; `cv3` predicts `nc` class logits. Training returns raw per-scale tensors.
Inference applies DFL, converts per-edge distributions to `(l,t,r,b)` distances, and
decodes to absolute coordinates via stride anchors.

**IDetect (YOLOv7).** Extends `Detect` with `ImplicitA` (additive bias) and
`ImplicitM` (multiplicative scale) per scale, both fusable into the preceding conv.

**v10Detect (YOLOv10).** Dual-head: `cv2/cv3` for one-to-many assignment (training),
`cv4/cv5` for one-to-one assignment (inference). The one-to-one head eliminates NMS.

---

## SAC/TR injection

`apply_sac_tr(model, sac_indices, tr_indices)` in
`cracks_yolo.zoo.ultralytics.sac_injection` mutates `model.model` — the backbone
`nn.Sequential` — in-place, by index.

For each `i` in `sac_indices`: if `model.model[i]` is a `C3`, replace with
`C3SAC(c1, c2, n, shortcut=True)`; if a `C2f`, replace with `C2fSAC(c1, c2, n)`.
`c1` and `c2` are read from the original block's first and final conv layers; `n`
from `len(old.m)`. For each `i` in `tr_indices`: `C3` blocks become
`C3TR(c1, c2, n, shortcut=True)`.

`_copy_shared_weights(new, old)` matches `state_dict` keys by name and shape, copying
pretrained `cv1`/`cv2`/`cv3` conv weights. SAC-specific parameters (switch, context
convs, weight_diff, AWS buffers) and TR-specific parameters (QKV projections,
positional embedding) remain randomly initialised. `_copy_layer_meta(new, old)`
propagates ultralytics routing attributes (`f`, `i`, `type`, `np`) so the replacement
integrates with the model's parse-tree forward graph.

Invoked once during `__init__`, before moving to device:
```python
apply_sac_tr(model, sac_indices=(4, 6, 8), tr_indices=(9,))
```
Indices depend on the specific YAML architecture definition.
