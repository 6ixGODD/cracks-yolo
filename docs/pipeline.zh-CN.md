# 流水线

[English](pipeline.md) | [中文](pipeline.zh-CN.md)

`cracks_yolo.pipeline` 提供训练循环、测试循环、5 折交叉验证以及多模型比较，专为舌面裂纹检测模型设计。所有组件仅依赖 `DetectorModel` Protocol，不存在模型特定的分支。

## TrainPipelineImpl

`TrainPipelineImpl.run(model, train_loader, val_loader, cfg) -> TrainReport`：

1. `configure_logger(cfg.output_dir)` — JSONL 输出位于 `run.log.jsonl`。
2. `optimizer = model.build_optimizer()`（lr/weight_decay 从 cfg 覆盖）。
3. 可选的 AMP `torch.amp.GradScaler`（通过 `--amp` 启用，默认关闭）。
4. 每个 epoch：
   - 训练一次：`preds = model(images)` → `loss, parts = model.compute_loss(preds, yolo_targets, imgs=images)`。
   - 反向传播 + 参数更新（可选的梯度裁剪通过 `cfg.clip_grad_norm`）。
   - 每 N 步记录 `TrainStepLog`；每个 epoch 记录 `TrainEpochLog`。
   - 验证（如果提供了 `val_loader` 且 `(epoch+1) % cfg.val_interval == 0`）：前向 → `decode` → NMS → COCO mAP。
   - 如果验证 mAP@50 有提升，保存 `best.pt`。
5. 最终：写入 `metrics.csv`（每个 epoch 的损失 + 验证指标）、`loss_curve.png`、`metric_curve.png`、`config.yaml`。
6. 返回 `TrainReport(output_dir, best_weights_path, best_epoch, best_map50, history)`。

### 各部分含义

每个模型声明 `loss_parts_schema: tuple[str, ...]`：
- v5/v7：`("box", "cls", "obj")`
- v8/v9/v10：`("box", "cls", "dfl")`
- torchvision RetinaNet/Faster-RCNN：`("total", "cls", "box_reg", "rpn_box_reg")`

流水线从模型读取 schema（不通过类名分支），并将每个条目映射到相应的 `TrainStepLog` 字段。未知条目（如 `total`）会累加到总损失中，但不会作为命名部分单独显示。

### TrainConfig (pydantic BaseModel)

```python
class TrainConfig(BaseModel):
    output_dir: Path
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 16
    input_size: int = 640
    amp: bool = False
    clip_grad_norm: float | None = None
    val_interval: int = 1
    log_every_n_steps: int = 10
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
```

## TestPipelineImpl

`TestPipelineImpl.run(model, test_loader, cfg) -> TestReport`：

1. `configure_logger(cfg.output_dir)`。
2. 每个 batch：`model.eval()` → `preds = model(images)` → `decoded = model.decode(preds)` → NMS（通过 `detections_to_per_image`）。
3. `COCOMetricsCalculator.update(all_per_image)` → `report = calculator.run()`。
4. 计算 PR/ROC 曲线、混淆矩阵、AUC（通过 `cracks_yolo.metrics.curves` + `confusion`）。
5. 写入 `metrics.csv`、`per_image/<id>.json`、`predictions/<id>.jpg`、`curves/{pr,roc,confusion}.png`、`TestLog`。
6. 返回 `TestReport(output_dir, metrics, elapsed_sec)`。

### 解码格式

每个模型声明 `decode_format: str`：
- `"anchor_free"`（v8/v9/v10）：输出 `(B, 4+nc, N)`。流水线置换为 `(B, N, 4+nc)` 并应用 NMS。
- `"anchor_based"`（v5/v7、torchvision 包装器）：输出 `(B, N, nc+5)`，已采用 xyxy-score-cls 布局。

流水线从模型读取 `decode_format` 来决定置换步骤（不通过类名分支）。

## 产物

每次训练或测试运行产生以下产物：

| 产物 | 说明 |
| --- | --- |
| `run.log.jsonl` | 结构化 JSONL 日志（loguru）。 |
| `metrics.csv` | 逐 epoch 的损失和验证指标。 |
| `loss_curve.png` | 训练损失曲线。 |
| `metric_curve.png` | 验证指标曲线。 |
| `config.yaml` | 冻结的训练配置。 |
| `best.pt` | 验证 mAP@50 最优的检查点。 |
| `per_image/<id>.json` | 逐图像检测结果。 |
| `predictions/<id>.jpg` | 带边界框的可视化预测。 |
| `curves/{pr,roc,confusion}.png` | 评估曲线。 |

交叉验证运行额外产生：

| 产物 | 说明 |
| --- | --- |
| `cv_summary.csv` | 每折指标及均值与标准差。 |
| `cv_report.json` | 完整交叉验证报告。 |
| `fold_<i>/` | 每折输出目录（包含所有运行产物）。 |
| `comparison*.csv` | 多模型比较表。 |
| `paired_t_test.csv` | 模型间两两 p 值。 |

## run_cross_validation

`run_cross_validation(model_cls, dataset, cfg, n_folds=5, seed=42, val_fraction=0.1) -> CrossValReport`：

1. 在 `--cross-val` 模式下，先将 train + valid + test 三个 split 合并为单一池，再重新划分（原始 split 分配被忽略）。
2. 通过 `sklearn.model_selection.StratifiedKFold(random_state=seed)` 进行分层 5 折划分。分层依据是**图像级别的类别组成**（每张图像被分配其包含的类别 ID 集合；以最频繁的类别作为分层依据，以保持各折的类别分布大致稳定）。
3. 留出的一折作为**测试**集。剩余 N-1 折进一步划分为训练集（`1 - val_fraction`，默认 0.9）和验证集（`val_fraction`，默认 0.1），使用 `sklearn.model_selection.train_test_split(random_state=seed+fold_idx, stratify=...)`。
4. 每折：实例化一个新模型，在折划分上构建训练/验证加载器，运行 `TrainPipelineImpl.run`，保存到 `output/cv/fold_<i>/`。
5. 汇总：所有指标在各折上的均值 +- 标准差；写入 `cv_summary.csv` + `cv_report.json`。
6. 返回 `CrossValReport(folds, mean_metrics, std_metrics, output_dir)`。

详见 `docs/cross_validation.md`。

## compare_models_cross_val

`compare_models_cross_val(model_keys, dataset, cfg, n_folds=5, seed=42, metric="map50", val_fraction=0.1) -> ComparisonReport`：

1. 为 `model_keys` 中的每个模型运行 5 折 CV。
2. 对所有模型对，在选定指标（默认 `map50`）上进行逐折配对 t 检验：`scipy.stats.ttest_rel`。同时计算 Wilcoxon 符号秩检验（适用于小样本 N）和逐折差异的 bootstrap 置信区间。
3. 写入 `comparison.csv`（每个模型的均值 +- 标准差）、`paired_t_test.csv`（两两 p 值）、`comparison_plot.png`。
4. 返回 `ComparisonReport(per_model, pairwise_tests, output_dir)`。

详见 `docs/cross_validation.md` 了解统计方法。

## 命令行

- `scripts/train.py` — 单次训练或通过 `--cross-val` 标志触发 `run_cross_validation`。
- `scripts/test.py` — 单次测试运行。
- `scripts/compare_models.py` — 包装 `compare_models_cross_val`。
