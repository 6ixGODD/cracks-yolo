# Grad-CAM heatmaps

`cracks_yolo.viz.heatmap.GradCAMExtractor` generates Grad-CAM saliency maps
for any `nn.Module` and any target layer. Used by `scripts/heatmap.py` to
visualize which backbone regions a trained model attends to per image.

## Methodology

Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al. 2017):

1. Register a forward hook on the target layer to capture its activation output `A` (a `(C, H, W)` feature map).
2. Register a backward hook to capture the gradient of the scalar score
   (max class score over all anchor/grid positions) with respect to `A`.
3. After one forward + backward pass on an input image:
   - Compute channel-wise weights `α_k = mean over (H, W) of ∂score/∂A_k`
     (global average pooling of the gradient).
   - Compute the heatmap `M = ReLU(Σ_k α_k · A_k)` — shape `(H, W)`.
4. Upsample `M` to the input image size, normalize to `[0, 1]`, overlay as a
   jet colormap on the original image.

**Score choice for detection models**: for anchor-free v8/v9/v10, the scalar
score is `max over grid cells of max over classes of sigmoid(cls_logits)`.
For anchor-based v5/v7, it's `max over anchors of max over classes of
sigmoid(obj_logit) * sigmoid(cls_logits)`. For torchvision detectors, the
highest-scoring detection's `score` field is used directly.

## Layer selection

Use dot-notation relative to the model's top-level attributes. The YOLO
models in `cracks_yolo.zoo` expose `backbone` (an `nn.Sequential`) and
`neck` (another `nn.Sequential`). Valid layer specs:

- `backbone.0` through `backbone.9` (v5s: 10 children, indices 0-9).
- `backbone.0` through `backbone.9` (v8/v9/v10: same indexing scheme).
- `neck.0`, `neck.4`, `neck.7`, etc. — any layer index in the neck.

For torchvision detectors, use the ResNet submodules:
`_inner.backbone.body.layer1`, `_inner.backbone.body.layer4`, etc.

**Common pitfall**: YOLOv5s backbone has 10 children (indices 0-9), so
`backbone.10` raises `IndexError`. Always check
`len(model.backbone)` before specifying `--layers`.

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

Output structure:

```
output/heatmaps/
  heatmaps/
    000000/
      backbone.8.png      # jet-overlay on original image
      backbone.9.png
    000001/
      ...
  feature_maps/
    000000/
      backbone.8.npy      # raw (C, H, W) feature map
      backbone.9.npy
    000001/
      ...
```

## Interpretation for cracks detection

For the cracks-detection thesis, Grad-CAM serves two purposes:

1. **SAC/TR ablation evidence**: compare heatmaps for `YOLOv5s` vs
   `YOLOv5sSAC` vs `YOLOv5sTR` vs `YOLOv5sSACTR` on the same test images.
   The hypothesis is that SACTR produces more focused activation along thin
   crack lines (less diffuse context bleed) — visually quantifiable as a
   tighter heatmap envelope around ground-truth crack pixels.

2. **Cross-paradigm comparison**: YOLO heatmaps localize well to crack
   pixels because the dense grid predictions are spatially aligned.
   Faster-RCNN's ROI-based heatmaps will look qualitatively different
   (sharper per-detection, but missing the dense coverage).
