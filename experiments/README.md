# Experiments — full-model sweep

[English](README.md) | [中文](README.zh-CN.md)

Unified sweep over all detector families for tongue surface crack detection. All YOLO models (v3/v5/v8/v9/v10) load via **Ultralytics** with SAC/TR injected by runtime module replacement (`cracks_yolo.zoo.ultralytics_sac.apply_sac_tr`); torchvision detectors (RetinaNet/Faster-RCNN/Mask-RCNN/FCOS/SSD) use the cracks_yolo wrappers. No `_official` suffix, no per-arch reimplementation.

## Files

- `all_models_compose.yaml` — all 24 active + 27 deferred models (train+test pairs). Self-contained.
- `compose_4/group_{1..4}.yaml` — 4-way split of the **active** set, balanced by GPU-hour cost (~44 h each). One file per server.
- `compose_4/group_deferred.yaml` — the 27 models commented out of the first sweep (v5 n/m/l/x SAC/TR variants, v8 m/l/x SAC, v8x, v9 t/s/m/e, v10 n/m/b/l/x, v3, DETR). Run if the active sweep finishes with time to spare.
- `all_models_cv5.yaml`, `all_models_direct.yaml` — older single-config sweeps (reference).

## Active model set (24 models)

| Family | Models | Epochs |
| --- | --- | --- |
| YOLOv5 | s{baseline,sac,tr,sactr}, n/m/l baseline (no x) | 1200 (early-stop patience 30) |
| YOLOv7 | yolov7w (cracks_yolo reimpl — not ultralytics) | 300 |
| YOLOv8 | n/s{baseline,sac}, m/l baseline | 300 |
| YOLOv9 | c{baseline,sac} | 300 |
| YOLOv10 | s{baseline,sac} | 300 |
| torchvision | retinanet/faster/mask/fcos/ssd300/ssdlite | 150 |

All YOLO: lr 1e-3, SGD (v5) / AdamW (others), cosine annealing, EMA, mosaic+HSV aug, no-AMP (fp32 — AMP NaNs v5 CIoU), clip_grad_norm 10, pretrained COCO (strict=False; SAC/TR random init). Torchvision: lr 1e-4, AdamW, AMP on. Seed = 42.

## Running on 4–6 servers (parallel)

Each server runs ONE `compose_4/group_N.yaml` independently. Groups are cost-balanced.

### Server setup (autodl, per server)

```bash
# 1. Clone (ghfast.top mirror for speed in China)
git clone https://ghfast.top/https://github.com/6ixGODD/cracks-yolo.git
cd cracks-yolo

# 2. Copy dataset to fast local disk for IO
cp -r /root/autodl-fs/CrackDetection_Augmentation.v1.yolov5pytorch /root/autodl-tmp/
ln -s /root/autodl-tmp/CrackDetection_Augmentation.v1.yolov5pytorch data/CrackDetection_Augmentation.v1.yolov5pytorch

# 3. Install deps
pip install ultralytics thop torchsummary pycocotools opencv-python

# 4. Run one group (server k runs group_k)
python -m scripts.schedule_experiments \
    --config experiments/compose_4/group_1.yaml \
    --output-dir output/group_1
```

### After the sweep

```bash
cd output/group_1 && zip -r /root/autodl-fs/group_1.zip . && cd ../..
```

## Time estimate

**Basis**: per-model GPU-hours on a single A100 40GB (fp32, batch sizes in YAML):
- v5 × 1200 epochs (early-stop): ~13 h each × 7 = 91 h
- v8/v9/v10/v7 × 300 epochs: ~5 h each × 11 = 55 h
- torchvision × 150 epochs: ~5 h each × 6 = 30 h
- **Total active: ~176 GPU-hours**

Wall-clock = 176 / N_servers:

| Servers | GPU | Wall-clock | Cost (~7¥/h) |
| --- | --- | --- | --- |
| 4 | A100 40GB | ~44 h (≈2 days) | ~4900¥ |
| 6 | A100 40GB | ~29 h (≈1.2 days) | ~7400¥ |
| 8 | A100 40GB | ~22 h (<1 day) | ~9800¥ |

**To finish in 1 day: rent 8× A100 40GB** (re-split into 8 groups — run `output/_gen_compose.py` with 8 groups, or run 2 groups per server sequentially). The 4-way split targets a 2-day budget on 4 servers. Costs scale with total GPU-hours (fixed ~176 h); more servers = faster, same total cost.

Estimate assumes early-stopping fires near patience for v5 (1200 ep is the cap; real runs may stop at 600–900).

## Notes

- Output dirs: `output/{model}/` (scheduler passes `--output-dir`). Standalone `python -m scripts.train` auto-timestamps to `output/{ISO_ts}/{model}/`.
- v7 (`yolov7w`) is the only family NOT on Ultralytics — it uses the cracks_yolo reimplementation.
- GFLOPs via thop; FPS measured end-to-end on the real test loader; model structure via `model.info()` / torchsummary.
