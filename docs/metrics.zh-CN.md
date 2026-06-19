# 指标

[English](metrics.md) | [中文](metrics.zh-CN.md)

`cracks_yolo.metrics` 提供舌面裂纹检测模型的指标计算、统计比较和评估曲线。具体实现为 `COCOMetricsCalculator`，后端基于 pycocotools（mAP/AR）、sklearn（PR/ROC/混淆矩阵）、scipy（配对 t 检验、Wilcoxon）和 statsmodels（bootstrap CI）。

## 检测级记录

### DetectionMetric（TypedDict）

NMS 之后、指标聚合之前的单个检测。

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `image_id` | `int` | 图像标识符（匹配数据集索引）。 |
| `class_id` | `int` | 预测的类别 id。 |
| `score` | `float` | 置信度分数，范围 `[0, 1]`。 |
| `bbox_xyxy` | `tuple[float, float, float, float]` | `(x1, y1, x2, y2)`，以像素为单位。 |

### PerImageDetection（TypedDict）

一张图像的所有检测结果 + 真实标注。

| 字段 | 类型 |
| --- | --- |
| `image_id` | `int` |
| `detections` | `list[DetectionMetric]` |
| `ground_truths` | `list[DetectionMetric]` |

## COCOMetricsCalculator

`COCOMetricsCalculator` 实现了 `MetricsCalculator` Protocol：

```python
@runtime_checkable
class MetricsCalculator(Protocol):
    def update(self, batch: list[PerImageDetection]) -> None: ...
    def run(self) -> MetricReport: ...
```

- **训练侧（轻量指标）：** 训练期间每批调用 `update()`；在周期结束时输出小摘要。
- **测试侧（完整指标）：** 收集所有逐图像检测结果，然后 `run()` 生成完整的 `MetricReport`。

内部实现中，`run()` 委托 pycocotools 计算 COCO mAP/AR，再通过 sklearn 计算 PR/ROC 曲线、混淆矩阵及衍生指标（sensitivity、specificity、PPV、NPV）。

## 聚合指标

### mAP（平均精度均值）

- **mAP@0.5**（`map50`）：按类别平均的 AP，IoU 阈值为 0.5。
- **mAP@0.5:0.95**（`map5095`）：按类别和 IoU 阈值 `[0.5, 0.55, ..., 0.95]`（10 个阈值）平均的 AP。标准 COCO 主要指标。
- **AP@0.5**（`ap50`）：每类 AP，IoU 0.5。
- **AP@0.75**（`ap75`）：每类 AP，IoU 0.75。

**公式（每类、每 IoU 阈值）：**
$$\text{AP}_c^t = \int_0^1 p(r) \, dr$$
其中 $p(r)$ 是在 IoU 阈值 $t$ 下类别 $c$ 的精确率-召回率曲线。跨阈值和类别平均得到 mAP。

### 精确率 / 召回率 / F1

- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `f1 = 2 * P * R / (P + R)`

所有值均在由 F1 最优置信度阈值选择的工作点处计算。

### AR（平均召回率）

COCO 风格的 AR，按递增的最大检测数预算计算：
- `ar1` — 每张图像 1 个检测的 AR。
- `ar10` — 每张图像 10 个检测的 AR。
- `ar100` — 每张图像 100 个检测的 AR。
- `ar300` — 每张图像 300 个检测的 AR。
- `ar1000` — 每张图像 1000 个检测的 AR。
- `ar_small` — 小目标（面积 < 32^2 像素）的 AR。
- `ar_medium` — 中等目标（32^2 <= 面积 < 96^2 像素）的 AR。
- `ar_large` — 大目标（面积 >= 96^2 像素）的 AR。

**公式：** 在 IoU 阈值 `[0.5, 0.95]` 上平均，在给定检测预算下可实现的最大召回率。

### AUC-PR / AUC-ROC

- `auc_pr`：精确率-召回率曲线下面积（sklearn）。
- `auc_roc`：受试者工作特征曲线下面积（sklearn）。

### Sensitivity / Specificity / PPV / NPV

- `sensitivity = TP / (TP + FN)` — 真阳性率（召回率）。
- `specificity = TN / (TN + FP)` — 真阴性率。
- `ppv = TP / (TP + FP)` — 阳性预测值（精确率）。
- `npv = TN / (TN + FN)` — 阴性预测值。

这些指标由 F1 最优置信度阈值处的混淆矩阵计算得出。

## MetricReport

`MetricReport`（dataclass）——由 `COCOMetricsCalculator.run()` 返回的完整报告：

```python
@dataclass
class MetricReport:
    map50: float
    map5095: float
    ap50: dict[int, float]       # 每类 AP@0.5
    ap75: dict[int, float]       # 每类 AP@0.75
    precision: float
    recall: float
    f1: float
    ar1: float
    ar10: float
    ar100: float
    ar300: float
    ar1000: float
    ar_small: float
    ar_medium: float
    ar_large: float
    auc_pr: float
    auc_roc: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    confusion_matrix: np.ndarray
    iou_threshold: float
    conf_threshold: float
```

`MetricReport` 只含检测精度。效率（速度与算力）由测试流程单独产出。

## 效率指标

每次测试（以及每一折交叉验证）都会在精度之外同步测量效率，一次扫描即可得到完整的「精度—速度—算力」对比表。设 `TestConfig.measure_efficiency=False` 可跳过。

`EfficiencyReport`（pydantic）由 `TestPipelineImpl` 产出，写入 `metrics.csv`、`model_analysis.json` 与 `TestLog` 记录：

| 字段 | 含义 |
| --- | --- |
| `fps_mean` / `fps_p50` / `fps_p95` | 在真实测试集上的端到端推理吞吐（张/秒），含前向 + 解码 + NMS，batch 为 `TestConfig.batch_size`。 |
| `latency_mean_ms` / `latency_p50_ms` / `latency_p95_ms` | 单张延迟（批耗时 / 批大小）。 |
| `n_parameters` / `n_trainable_parameters` | 参数量。 |
| `macs` / `gflops` | 单张输入的 MACs 与 GFLOPs（= 2 × MACs），由 `fvcore.FlopCountAnalysis` 计算。 |
| `peak_vram_bytes` | 真实推理循环中的 CUDA 峰值显存（CPU 下取单张合成的值）。 |

独立的 `scripts/analyze_model.py` 用合成输入报告同样的结构指标（参数量/MACs/延迟/显存），无需跑测试；测试流程的数值才是权威的端到端结果，因为它在真实 batch 下包含解码与 NMS。

## 评估曲线

`cracks_yolo.metrics.curves` 模块生成三种图表，保存到 `curves/` 目录：

- `pr.png` — 各 IoU 阈值下的精确率-召回率曲线。
- `roc.png` — 受试者工作特征曲线。
- `confusion.png` — 归一化混淆矩阵热力图。

`cracks_yolo.metrics.confusion` 模块计算原始混淆矩阵并导出 sensitivity、specificity、PPV 和 NPV。

## 统计检验（模型比较）

`StatisticalTest`（TypedDict）——用于在同一个测试集上比较两个模型变体（例如基线 vs SAC）：

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `test_name` | `Literal["paired_t", "bootstrap_ci", "wilcoxon"]` | 检验类型。 |
| `statistic` | `float` | 检验统计量。 |
| `p_value` | `float` | 双尾 p 值。 |
| `ci_low` | `float \| None` | Bootstrap 下限 95% CI（仅 bootstrap）。 |
| `ci_high` | `float \| None` | Bootstrap 上限 95% CI（仅 bootstrap）。 |
| `n_samples` | `int` | 配对样本数（例如测试图像数）。 |

### 检验选择

- **配对 t 检验**——当每张图像的指标差异近似正态分布时使用。速度快。报告均值差异是否显著不同于 0。
- **Wilcoxon 符号秩检验**——配对 t 的非参数替代方法。当差异非正态时使用。
- **Bootstrap CI**——从测试集中有放回地重采样（1000 次以上迭代），获取指标差异的 95% CI。最稳健，最慢。

## 舌面裂纹检测的指标解读

舌面裂纹通常为细长结构。预测框与真实标注之间的微小空间偏移即可显著降低 IoU，因此该任务对 IoU 高度敏感。解读指标时请注意以下几点：

- **优先关注 mAP@50 而非 mAP@50:95**——mAP@50:95 中的严格 IoU 阈值会惩罚空间接近但未达到像素级精度的检测。对于细裂纹，mAP@50 更能反映临床可用的检测质量。
- **AUC-PR 是阈值无关的**——它在所有置信度阈值上总结模型质量，无需固定工作点。
- **AR@100 和 AR@1000**——更高的检测预算有助于评估模型是否能够召回可能断裂为多个小片段的细裂纹。
- **混淆矩阵**——单类别（`crack`）场景下，混淆矩阵简化为 TP / FN / FP 比率，直接对应 sensitivity 和 PPV。
