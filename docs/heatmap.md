# Heatmap generation

[English](heatmap.md) | [中文](heatmap.zh-CN.md)

`cracks_yolo.viz.heatmap.GradCAMExtractor` produces gradient-weighted class activation maps
(Grad-CAM; Selvaraju et al., 2017) for arbitrary `nn.Module` subgraphs. `scripts/heatmap.py`
batch-processes images and saves overlays.

## Method

For a target layer output `A ∈ R^{C×H×W}` and a scalar score `y`, a forward hook captures `A`
and a backward hook captures ∂y/∂A. Channel weights are global-average-pooled gradients:

```
α_k = (1 / (H·W)) · Σ_i Σ_j  ∂y / ∂A_{k,i,j}
```

The raw heatmap is the ReLU-thresholded weighted sum, upsampled to input size and normalized:

```
M = ReLU( Σ_k α_k · A_k )   ∈ R^{H×W}
```

### Score definition

| Paradigm       | Score `y`                                                              |
|----------------|------------------------------------------------------------------------|
| Anchor-free    | max over grid of max over classes of σ(cls_logits)                     |
| Anchor-based   | max over anchors of max over classes of σ(obj) · σ(cls)                |
| Torchvision    | highest-scoring detection `score`                                      |

## Layer specification

Dot-separated paths relative to model root: `backbone.0`..`backbone.9` (YOLOv5/v8/v9/v10),
`neck.0`, `neck.4`, etc. Torchvision: `_inner.backbone.body.layer1`..`layer4`. Verify depth
with `len(model.backbone)` before scripting — YOLOv5s backbone has exactly 10 children.

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

Output: `heatmaps/` (jet-overlay PNG per image per layer) and `feature_maps/` (raw `.npy`).

## API

```python
from cracks_yolo.viz.heatmap import GradCAMExtractor

extractor = GradCAMExtractor(model, target_layers=["backbone.8", "backbone.9"])
heatmaps = extractor(image_tensor)  # dict[str, np.ndarray]
```

## Visualization utilities

- `overlay_heatmap(image, heatmap, alpha=0.4, cmap="jet")` — superimpose normalized heatmap on
  a PIL/numpy image, return RGB array.
- `save_heatmap_grid(images, heatmaps, ncols=4)` — tiled comparison PNG across samples or layers.
- `plot_activation_profile(heatmap, axis=0)` — horizontal/vertical mean projection for
  quantifying spatial concentration along crack traces.

## Interpretation for crack detection

1. **SAC/TR ablation.** Compare heatmaps across YOLOv5s, YOLOv5sSAC, YOLOv5sTR, and
   YOLOv5sSACTR on identical inputs. SACTR variants exhibit tighter activation envelopes along
   thin crack traces with reduced diffuse bleed into surrounding tissue.
2. **Cross-paradigm comparison.** Dense-grid YOLO heatmaps provide spatially continuous coverage
   aligned to crack topology; ROI-based detectors (Faster-RCNN) produce sharper per-instance
   maps but lack the same spatial density.
