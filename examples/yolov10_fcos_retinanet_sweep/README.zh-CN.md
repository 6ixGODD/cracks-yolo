# 示例 — 调度器扫描：YOLOv10 / FCOS / RetinaNet

[English](README.md) | [中文](README.zh-CN.md)

冒烟测试：用统一调度器在一次扫描中跑三种检测范式 —— 每个模型训练 + 测试，各 5 epoch。

## 命令

```bash
uv run python -m scripts.schedule_experiments \
    --config examples/yolov10_fcos_retinanet_sweep/sweep.yaml \
    --output-dir examples/yolov10_fcos_retinanet_sweep/output
```

## 配置

定义在 [`sweep.yaml`](sweep.yaml)。一次调度，6 个实验（3 个模型 × 训练 + 测试），串行执行（`max_parallel=1`）：

| 模型 | 范式 | Batch | LR |
| --- | --- | --- | --- |
| `yolov10s` | anchor-free、无 NMS | 8 | 1e-3 |
| `fcos_r50` | anchor-free、center-ness | 4 | 1e-4 |
| `retinanet_r50` | anchor-based、focal loss | 4 | 1e-4 |

- **数据集**：`CrackDetection_Augmentation.v1.yolov5pytorch` — 原始划分（train 770 / valid 220 / test 110），1 类（`cracks`）。
- **训练**：每个 5 epoch，开启 AMP，seed 42。两个 R50 检测器用 batch 4 + lr 1e-4 以适配 4 GB 显存。
- 全程 `num_workers: 0`（Windows 共享内存约束）。

## 产物

全部产物已 gitignore，位于 `examples/yolov10_fcos_retinanet_sweep/output/`：

- `scheduler/results.jsonl` — 每个成功实验一条记录（耗时、output_dir）。
- `scheduler/errors.jsonl` — 失败记录（退出码 + 日志路径），仅出错时生成。
- `scheduler/<exp_name>.log` — 每个实验的完整 stdout/stderr。
- `<model>/` — 每个模型的训练运行（`metrics.csv`、`loss_curve.png`、`best.pt` 等）及 `test/` 子目录（含效率指标的 `metrics.csv`、`model_analysis.json`、`per_image/`、`predictions/`、`curves/`）。
