# 实验配置

[English](README.md) | [中文](README.zh-CN.md)

面向舌面裂纹检测的 26 模型批量扫描由两个 YAML 配置驱动。二者均由 `scripts/schedule_experiments.py` 读取，并以子进程方式调用 `python -m scripts.train` / `python -m scripts.test` 执行（详见 `docs/scheduler.zh-CN.md`）。

## 文件

- **`all_models_direct.yaml`** — 26 个模型 × 2 个实验（训练 + 测试）= 52 个实验。每个模型在 `train` 划分上训练（训练期间在 `valid` 划分上验证），随后在留出的 `test` 划分上评估 `best.pt`。每个模型产出一组直接的训练→测试指标。
- **`all_models_cv5.yaml`** — 26 个模型 × 1 个交叉验证实验 = 26 个实验。每个实验将原始 train+valid+test 划分合并为一个数据池，运行 `StratifiedKFold(n_splits=5)`。每一折：留出折 = **测试集**，其余记录再按 9:1 拆分为训练集和验证集（`val_fraction=0.1`，用于反向传播期间的验证）。产出每折指标 + 每个模型的均值 ± 标准差。

## 用法

```bash
# 直接扫描（52 个实验）。
python -m scripts.schedule_experiments \
    --config experiments/all_models_direct.yaml \
    --output-dir output/all_models_direct

# 5 折交叉验证扫描（26 个实验）。
python -m scripts.schedule_experiments \
    --config experiments/all_models_cv5.yaml \
    --output-dir output/all_models_cv5

# 重试失败的实验。
python -m scripts.schedule_experiments \
    --retry-failed output/all_models_direct/scheduler/errors.jsonl \
    --output-dir output/all_models_direct_retry
```

## 多 GPU

将 `scheduler.max_parallel` 设为 GPU 数量，并为每个实验添加 `env: {CUDA_VISIBLE_DEVICES: "N"}`（N = 0/1/2/...）。此时 batch size 可下调至约 1/N。详见 YAML 文件头部注释。

## batch size（单卡大显存 GPU）

| 模型族 | batch size |
| --- | --- |
| YOLOv8n | 128 |
| YOLOv5s / YOLOv8s / YOLOv10s | 64 |
| YOLOv7w / YOLOv8m / YOLOv9c | 32 |
| YOLOv8l | 16 |
| YOLOv8x / Faster-RCNN / Mask-RCNN | 8 |
| RetinaNet / FCOS | 16 |
| SSD300 | 32 |
| SSDlite320 | 64 |

若发生 OOM，将 batch size 减半后重试。YOLO 系列使用 `lr=0.01`；torchvision 检测器使用 `lr=0.0001`（更低——封装层默认使用 SGD，`lr=1e-4`）。
