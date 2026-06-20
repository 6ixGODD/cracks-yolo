# 示例 — YOLOv5s-SACTR 5 折交叉验证

[English](README.md) | [中文](README.zh-CN.md)

冒烟测试：在舌面裂纹数据集上对 YOLOv5s-SACTR（SAC + TR）跑 5 折交叉验证，验证完整的 CV 产物集。

## 命令

```bash
uv run python -m scripts.train \
    --model yolov5s_sactr \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --output-dir examples/yolov5s_sactr_cv/output/cv \
    --epochs 10 \
    --batch-size 8 \
    --lr 1e-3 \
    --num-workers 0 \
    --cross-val --n-folds 5 --val-fraction 0.1 \
    --device cuda
```

## 配置

- **模型**：`yolov5s_sactr`（YOLOv5s 主干 + 可切换空洞卷积 SAC + Transformer 模块）。
- **数据集**：`CrackDetection_Augmentation.v1.yolov5pytorch` — 770 / 220 / 110（train / valid / test），1 类（`cracks`）。CV 模式将三者合并为一个池（1100 条）。
- **每折划分**：留出折 = **测试集**（220 条）；剩余 880 条按 `train_test_split(random_state=seed+fold, stratify=...)` 切为训练（90% = 792）+ 验证（10% = 88）。
- **训练**：10 epoch，batch 8，lr 1e-3，开启 AMP，seed 42。
- Windows 下需用 `--num-workers 0`（多 worker 的 DataLoader 会触及共享内存上限）。

## 产物

全部产物写入 `examples/yolov5s_sactr_cv/output/cv/`（已 gitignore）：

- `cv_summary.csv` — 每折测试指标（精度 + 效率）。
- `cv_report.json` — 每折训练/测试摘要 + 聚合后的均值 ± 标准差。
- `fold_0` … `fold_4/` — 每折完整的训练 + 测试运行（见 `docs/pipeline.md`）。
- `fold_<i>/test/metrics.csv`、`model_analysis.json`、`per_image/`、`predictions/`、`curves/`。
- `run.log.jsonl` — 结构化日志记录。
