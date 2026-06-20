# 交叉验证

[English](cross_validation.md) | [中文](cross_validation.zh-CN.md)

`cracks_yolo.pipeline.crossval.run_cross_validation` 和 `cracks_yolo.pipeline.compare.compare_models_cross_val` 实现了带统计比较的 N 折交叉验证。

## 数据合并

**CV 模式会忽略原始的 train/valid/test 划分。** 调用方在调用 `run_cross_validation` 之前，需要自行将所有数据集记录合并为一个列表。`scripts/train.py --cross-val` 会自动完成此操作：

```python
records = (
    src.load_split("train")
    + src.load_split("valid")
    + src.load_split("test")
)
```

合并后的池子由 `StratifiedKFold` 进行划分。这保证了每一折中的测试集与训练数据来自相同的分布——这对配对 t 检验的有效性至关重要。

## N 折机制

1. **外层划分**：`sklearn.model_selection.StratifiedKFold(n_splits=N, shuffle=True, random_state=seed)`。
   - 分层依据：每张图片中出现频率最高的类别 ID。这保持了每折中类别分布的相对稳定。对于单类别数据集（如舌面裂纹），这退化为普通随机打乱——`StratifiedKFold` 会优雅地降级。
2. **每折处理**：
   - **留出的一折 (1/N)** → **TEST** 记录（没有任何训练过程看到它们）。
   - **剩余记录 (N-1/N)** → 训练池，进一步划分为：
     - **train**（`1 - val_fraction`，默认 90%）— 用于反向传播。
     - **val**（`val_fraction`，默认 10%）— 用于反向传播验证（最佳检查点选择、学习率调度）。
   - 内部划分使用 `sklearn.model_selection.train_test_split`，参数为 `random_state=seed + fold_idx`，并使用相同的每张图片标签进行分层（如果某个类别的样本太少无法分层，则退化为普通随机划分）。
   - 从 `model_factory` 实例化一个全新的模型。
   - 运行 `TrainPipelineImpl.run(model, train_loader, val_loader, cfg)`。
   - 在留出的一折上运行 `TestPipelineImpl.run(model, test_loader, test_cfg)`。
   - 保存到 `output/cv/fold_<i>/`（每折的完整产物集）+ `output/cv/fold_<i>/test/`（测试产物）。
3. **聚合**：
   - 计算 N 折上每个指标的均值 ± 标准差（基于每折的 TEST 指标）。
   - 写入 `cv_summary.csv`（每折指标）+ `cv_report.json`（每折训练 + 测试摘要 + 聚合后的均值/标准差）。

给定 `seed`，划分是确定性的。相同的 `seed` + 相同的数据集 = 相同的 N 折划分，因此多模型比较是配对的（每折索引对所有模型都相同——配对 t 检验的要求）。

## 为什么留出折 = test（而非 val）

原始实现将留出的折同时用于训练期间的验证和测试——这混淆了两者的角色，并使测试指标产生偏差（模型根据用于评估的相同数据选择了其最佳检查点）。当前实现将两者分离：

- **val**（来自训练池）→ 驱动训练期间的检查点选择。模型通过调度器间接看到了这部分数据。
- **test**（留出的折）→ 从不影响训练。测试指标是诚实的泛化能力估计。

`val_fraction=0.0` 可完全禁用验证（在完整的 N-1/N 池上训练，在留出的折上测试）。适用于数据集非常小的情况，此时划分出验证集会饿死训练过程。

## 多模型比较

`compare_models_cross_val(model_keys, dataset, cfg, n_folds=5, seed=42, metric="map50")`：

1. 对 `model_keys` 中的每个模型：运行 5 折 CV → 存储每折的 `metric` 值。
2. 对所有模型对进行每折配对 t 检验：`scipy.stats.ttest_rel(per_fold_A, per_fold_B)`。
3. Wilcoxon 符号秩检验（非参数替代方法，适用于小样本 N=5）：`scipy.stats.wilcoxon(per_fold_A, per_fold_B)`。
4. 对每折差异 `A - B` 进行 Bootstrap 置信区间估计：
   - 有放回地重采样 1000 次，计算每次重采样的均值。
   - 95% CI = Bootstrap 分布的第 2.5 和第 97.5 百分位数。
5. 写入：
   - `comparison.csv` — 每个模型的均值 ± 标准差。
   - `paired_t_test.csv` — 成对结果（model_A, model_B, t_stat, p_value, wilcoxon_p, bootstrap_ci_low, bootstrap_ci_high）。
   - `comparison_plot.png` — 每个模型每折 `metric` 的箱线图。

## 统计结果解读

在舌面裂纹检测任务中比较两个模型时，应报告为：

> 在 5 个分层折上，模型 A 达到 mAP@50 = X.XX ± Y.YY，优于模型 B (A.AA ± B.BB)，每折平均差异为 D.DD（配对 t(4) = t.ttt，p = 0.0PPP；Wilcoxon p = 0.0WWW；95% bootstrap CI [Lo.Lo, Hi.Hi]）。

**注意事项**：
- N=5 是小样本——t 检验假设每折差异服从正态分布。Wilcoxon 和 bootstrap CI 是不假设正态性的非参数替代方法；请同时报告三者。
- N=5 时 p < 0.05 应连同效应量（均值差异）+ CI 一起报告，而非作为二元的"显著/不显著"结论。
- 如果 bootstrap CI 包含 0，则即使在 t 检验 p < 0.05 的情况下，该差异在 0.05 水平上仍不具统计显著性。

## 指标选择

默认指标为 `map50`（IoU 为 0.5 时的 COCO AP）。对于舌面裂纹检测，`map50` 是主要指标——裂纹框很薄，IoU 对小定位误差很敏感，因此 `map5095`（在 IoU 0.5-0.95 上取平均）对薄裂纹检测器的惩罚比合理程度更严厉。

要在不同指标上进行比较：

```bash
python -m scripts.compare_models \
    --models yolov5s,yolov5s_sactr \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --n-folds 5 --epochs 100 \
    --metric map5095 \
    --output-dir output/comparison_map5095
```

支持的指标键：`map50`、`map5095`、`map75`、`precision`、`recall`、`f1`、`auc_pr`、`auc_roc`。

`cv_summary.csv` 与 `comparison.csv` 还会以均值 ± 标准差聚合每折的**效率**指标（`fps_mean`、`latency_mean_ms`、`gflops`、`n_parameters`、`peak_vram_bytes` 等），因此精度对比表自带速度/算力对比。

## 每折产物

每折的 `output/cv/fold_<i>/` 目录包含单次训练运行的完整产物集（参见 `docs/pipeline.md`），因此可以事后检查单个折——例如确认某一折的训练曲线没有发散。
