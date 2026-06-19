# 示例 — YOLOv8 直接训练 + 测试

[English](README.md) | [中文](README.zh-CN.md)

冒烟测试：在原始 train/valid 划分上训练 YOLOv8 基线（无 SAC、无交叉验证），再用 `best.pt` 在留出的测试集上评估。

## 命令

```bash
# 1. 训练（训练期间在 valid 划分上验证）。
uv run python -m scripts.train \
    --model yolov8s \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --output-dir examples/yolov8_direct/output/train \
    --epochs 10 \
    --batch-size 8 \
    --lr 1e-3 \
    --num-workers 0 \
    --device cuda

# 2. 测试（用得到的 best.pt 在测试集上评估）。
uv run python -m scripts.test \
    --model yolov8s \
    --weights examples/yolov8_direct/output/train/best.pt \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --split test \
    --output-dir examples/yolov8_direct/output/test \
    --batch-size 8 \
    --num-workers 0 \
    --device cuda
```

## 配置

- **模型**：`yolov8s`（YOLOv8s 基线，anchor-free，无 SAC）。
- **数据集**：`CrackDetection_Augmentation.v1.yolov5pytorch` — 尊重原始划分（train 770 / valid 220 / test 110），1 类（`cracks`）。
- **训练**：10 epoch，batch 8，lr 1e-3，开启 AMP，seed 42。每 epoch 在 `valid` 划分上验证。
- Windows 下需用 `--num-workers 0`（多 worker 的 DataLoader 会触及共享内存上限）。

## 产物

全部产物已 gitignore，位于 `examples/yolov8_direct/output/`：

**训练**（`output/train/`）：
- `metrics.csv` — 每 epoch 损失 + 验证指标。
- `loss_curve.png`、`metric_curve.png`、`config.yaml`、`best.pt`。
- `run.log.jsonl`。

**测试**（`output/test/`）：
- `metrics.csv` — 精度 + 效率（FPS、延迟、参数量、GFLOPs、峰值显存）。
- `model_analysis.json` — 完整效率报告。
- `per_image/<id>.json`、`predictions/<id>.jpg`、`curves/{pr,roc,confusion}.png`。
- `run.log.jsonl`。
