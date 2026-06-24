# 评估指标

[English](metrics.md) | [中文](metrics.zh-CN.md)

`cracks_yolo.metrics` 基于 pycocotools 后端提供 `COCOMetricsCalculator` 检测评估，
并包含效率剖析与配对统计检验。

## 1. 架构

    PerImageDetection[]  -->  COCOMetricsCalculator.update()  -->  .run()  -->  MetricReport

`COCOMetricsCalculator` 实现 `MetricsCalculator` Protocol（`update`/`run`）。
流水线逐帧收集预测与真值，构为 `PerImageDetection` 记录，经 `update()` 累积后
调用 `run()` 生成 `MetricReport`。

### 1.1 逐帧检测格式

`DetectionMetric` (`TypedDict`)：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `image_id` | `int` | 数据集图像索引 |
| `class_id` | `int` | 预测或真值类别 |
| `score` | `float` | 置信度，$\in [0,1]$；真值恒为 1.0 |
| `bbox_xyxy` | `tuple[float,float,float,float]` | $(x_1,y_1,x_2,y_2)$，单位像素 |

`PerImageDetection` (`TypedDict`)：`image_id`、`detections: list[DetectionMetric]`、
`ground_truths: list[DetectionMetric]`。

## 2. pycocotools 后端

`COCOMetricsCalculator.run()` 通过 `_to_coco_format` 将累积记录转换为 COCO JSON，
实例化 `pycocotools.coco.COCO` 承载真值，以 `COCO.loadRes` 加载检测结果，
再以 `"bbox"` 为 iouType 执行 `COCOeval`。`COCOeval.stats` 的 12 个元素映射如下：

| 索引 | 字段 | 说明 |
| --- | --- | --- |
| 0 | `map5095` | AP@IoU=0.50:0.95，全区域，maxDets=100 |
| 1 | `ap50` | AP@IoU=0.50 |
| 2 | `ap75` | AP@IoU=0.75 |
| 3–5 | `ap_small/medium/large` | 按区域分层的 AP |
| 6–8 | `ar1/ar10/ar100` | AR，maxDets 分别取 1/10/100 |
| 9–11 | `ar_small/medium/large` | 按区域分层的 AR |

`ar300` 与 `ar1000` 均取 alias 指向 `ar100`（pycocotools 上限为 100）。stock
`COCOeval.summarize` 不输出逐类 AP。累加器为空时，`run()` 返回 `map50=0.0, map5095=0.0`，
其余字段为默认值。

## 3. MetricReport

`MetricReport`（`@dataclass`）为 `run()` 返回的聚合精度载荷：

| 字段 | 默认值 | 来源 |
| --- | --- | --- |
| `map50` | (必填) | `COCOeval.stats[1]` |
| `map5095` | (必填) | `COCOeval.stats[0]` |
| `ap50` | 0.0 | `map50` 的别名 |
| `ap75` | 0.0 | `COCOeval.stats[2]` |
| `per_class_ap` | `{}` | 未填充 |
| `precision` | 0.0 | PR 曲线最佳 F1 阈值处取值 |
| `recall` | 0.0 | PR 曲线最佳 F1 阈值处取值 |
| `f1` | 0.0 | $\max_t 2PR/(P+R+\epsilon)$，遍历 PR 阈值 |
| `ar1/ar10/ar100` | 0.0 | `COCOeval.stats[6–8]` |
| `ar300/ar1000` | 0.0 | 指向 `ar100` 的别名 |
| `ar_small/medium/large` | 0.0 | `COCOeval.stats[9–11]` |
| `auc_pr` | 0.0 | PR 曲线梯形积分 |
| `auc_roc` | 0.0 | ROC 曲线梯形积分 |
| `sensitivity` | 0.0 | 工作点处 TP/(TP+FN) |
| `specificity` | 0.0 | 工作点处 TN/(TN+FP) |
| `ppv` | 0.0 | 阳性预测值，即 precision |
| `npv` | 0.0 | 工作点处 TN/(TN+FN) |
| `confusion_matrix` | `[]` | $(C+1)\times(C+1)$，含背景类 |
| `iou_threshold` | 0.5 | 配置的 IoU 阈值 |
| `conf_threshold` | 0.25 | F1 最优置信度阈值 |
| `performance` | `[]` | `list[PerformanceMetric]`（FPS、参数量等） |
| `statistical_tests` | `[]` | `list[StatisticalTest]`，用于模型比较 |

F1 最优工作点为 $t^* = \arg\max_t F1(t)$，其中 $t$ 遍历 PR 曲线阈值。precision、
recall、sensitivity、specificity、PPV、NPV 及混淆矩阵均在该阈值处报告。

## 4. PR 与 ROC 曲线

### 4.1 匹配规则

检测结果 $(image\_id, class\_id, score, bbox)$ 与同图同类真值中 IoU 最大者匹配。
若 $\text{IoU}\geq\text{threshold}$ 且该真值尚未被匹配，记为 TP；否则记为 FP。
未被匹配的真值为 FN。所有类别汇总后计算全局曲线。

### 4.2 PR 曲线

`compute_pr_curve(detections, ground_truths, iou_thr)` 返回 `(precision, recall, thresholds)`。
检测结果按置信度降序排列，累积 TP 与 FP：

$$P(t) = \frac{\text{TP}_{\text{cum}}(t)}{\text{TP}_{\text{cum}}(t) + \text{FP}_{\text{cum}}(t) + \epsilon}, \quad
R(t) = \frac{\text{TP}_{\text{cum}}(t)}{N_{gt}}$$

`compute_auc(precision, recall)` 对 recall 排序后的点做梯形积分，得 AUC-PR。
此为类别不平衡单类检测任务首选的阈值无关汇总指标。

### 4.3 ROC 曲线

`compute_roc_curve(detections, ground_truths, iou_thr)` 返回 `(fpr, tpr, thresholds)`。
$\text{TPR} = \text{TP}_{\text{cum}} / N_{gt}$。$\text{FPR} = \text{FP}_{\text{cum}} / \max(\text{FP}_{\text{cum}})$，
按最低置信度处的总假阳性数归一化。该归一化使曲线跨满 $[0,1]$ 以便评估排序质量，
但所得 FPR 值不适用于推导 specificity（$1-\text{FPR}$）。

`compute_auc_roc(fpr, tpr)` 对 FPR 排序后的点做梯形积分，得 AUC-ROC。

sensitivity、specificity、NPV 改为在 `_compute_classification_metrics_at_threshold`
中直接从已匹配的检测结果计算：TN 定义为低于阈值且若保留则构成 FP 的检测数量——
以此规避 ROC 归一化问题及目标检测中 TN 定义的不适定性。
