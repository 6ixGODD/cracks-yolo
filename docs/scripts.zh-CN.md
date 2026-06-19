# 脚本

[English](scripts.md) | [中文](scripts.zh-CN.md)

`scripts/` 包含所有 CLI 入口点。每个脚本是 `cracks_yolo.*` 模块的薄包装。所有脚本均接受 `--config <yaml>` 和单独的 `--flags`，并写入 `--output-dir`。

## train.py

```bash
python -m scripts.train \
    --model yolov5s_sactr \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --epochs 100 --batch-size 32 --input-size 640 \
    --output-dir output/yolov5s_sactr \
    --pretrained \
    --seed 42
```

标志：
- `--model` — ZOO 键（参见 `cracks_yolo.zoo.ZOO`）。
- `--dataset` — YOLOv5 格式的数据集根目录（包含 `data.yaml` + `train/`、`valid/`、`test/`）。
- `--epochs`、`--batch-size`、`--lr`、`--weight-decay`、`--input-size` — 训练超参数。
- `--amp` / `--no-amp` — 切换 AMP（默认开启）。注意：AMP + `lr=0.01` 在长时间运行时可能发散——请使用 `lr=1e-3` 或 `--no-amp` 以保证稳定性。
- `--num-workers` — DataLoader 工作进程数（默认 0）。
- `--device` — `cuda` 或 `cpu`（默认 `cuda`）。
- `--seed` — 可复现性种子（默认 42）。
- `--val-interval` — 每 N 个 epoch 验证一次（默认 1）。
- `--log-every-n-steps` — 训练步骤日志频率（默认 50）。
- `--pretrained` — 通过 `from_pretrained(strict=False)` 加载官方 COCO 预训练权重（SAC/TR 层保持随机初始化）。
- `--weights-dir` — 下载的 `.pt` 文件的缓存目录（默认 `weights/`）。
- `--cross-val` — 切换到 N 折 CV 模式。将 train+valid+test 划分合并为一个池，划分为 N 折。每折：保留折 = 测试，剩余记录划分为训练（90%）+ 验证（10%）。使用 `--n-folds`、`--val-fraction`。
- `--n-folds` — CV 折数（默认 5）。
- `--val-fraction` — 每折训练池中划出作为反向传播验证的比例（默认 0.1）。设为 0.0 则禁用验证。
- `--train-split`、`--val-split` — 单次运行模式下的划分名称（默认 `train`、`valid`）。在 CV 模式下忽略。

输出到 `output-dir`：`run.log.jsonl`、`metrics.csv`、`loss_curve.png`、`metric_curve.png`、`config.yaml`、`best.pt`。在 CV 模式下：每折一个 `fold_<i>/` 目录 + `cv_summary.csv` + `cv_report.json`。

## test.py

```bash
python -m scripts.test \
    --model yolov5s_sactr \
    --weights output/yolov5s_sactr/best.pt \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --input-size 640 \
    --output-dir output/yolov5s_sactr_test
```

输出：`metrics.csv`、`per_image/<id>.json`、`predictions/<id>.jpg`、`curves/{pr,roc,confusion}.png`、`run.log.jsonl` 中的 `TestLog`。

## convert_dataset.py

```bash
python -m scripts.convert_dataset \
    --input data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --from yolo --to coco \
    --output data/Crack_coco
```

## heatmap.py

```bash
python -m scripts.heatmap \
    --model yolov5s_sactr \
    --weights output/yolov5s_sactr/best.pt \
    --input data/CrackDetection_Augmentation.v1.yolov5pytorch/test \
    --layers backbone.8,backbone.9 \
    --output-dir output/heatmaps
```

为指定的 backbone 层生成 Grad-CAM 热力图。每张图片每层：`heatmaps/<image_id>/<layer>.png` + `feature_maps/<image_id>/<layer>.npy`。

**层命名**：使用相对于模型顶层属性的点号表示法。YOLOv5s backbone 有 10 个子层（索引 0-9），因此有效层为 `backbone.0` 到 `backbone.9`。无效索引会引发 `IndexError: index N is out of range`——请先检查 `len(model.backbone)`。

参见 `docs/heatmap.md` 了解方法说明。

## analyze_dataset.py

```bash
python -m scripts.analyze_dataset \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --output-dir output/dataset_analysis
```

输出：`class_distribution.png`、`bbox_size_distribution.png`、`bbox_position_heatmap.png`、`image_size_distribution.png`、`diversity_metrics.json`（Shannon 熵、唯一边界框宽高比桶数、空间覆盖度）。

## analyze_model.py

```bash
python -m scripts.analyze_model \
    --model yolov5s_sactr \
    --input-size 640 \
    --output-dir output/model_analysis
```

输出：`params.csv`、`macs.csv`（通过 `fvcore.nn.FlopCountAnalysis`）、`latency.csv`（100 次运行的 p50/p95，CPU + CUDA）、`vram.csv`（峰值 `torch.cuda.max_memory_allocated`）、`comparison_plot.png`。

使用 `--all` 运行所有 ZOO 条目：

```bash
python -m scripts.analyze_model --all --output-dir output/model_analysis_all
```

## schedule_experiments.py

YAML 驱动的批量调度器。参见 `docs/scheduler.md` 了解完整的 YAML 格式，以及 `experiments/README.md` 了解开箱即用的扫描配置。

```bash
# 直接 26 模型扫描（每个模型训练 + 测试）。
python -m scripts.schedule_experiments \
    --config experiments/all_models_direct.yaml \
    --output-dir output/all_models_direct

# 5 折 CV 26 模型扫描。
python -m scripts.schedule_experiments \
    --config experiments/all_models_cv5.yaml \
    --output-dir output/all_models_cv5

# 重试任何失败的实验。
python -m scripts.schedule_experiments \
    --retry-failed output/all_models_direct/scheduler/errors.jsonl \
    --output-dir output/all_models_direct_retry
```

## compare_models.py

```bash
python -m scripts.compare_models \
    --models yolov5s,yolov5s_sactr,yolov8s,yolov10s,yolov9c,retinanet_r50,faster_rcnn_r50 \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --n-folds 5 --epochs 100 \
    --metric map50 \
    --output-dir output/comparison
```

为每个模型运行 5 折 CV，然后在选定指标上进行逐折配对 t 检验。输出 `comparison.csv`、`paired_t_test.csv`、`comparison_plot.png`。
