# Visualization

[English](visualization.md) | [中文](visualization.zh-CN.md)

## Overview

The `cy visualize` command produces a suite of publication-grade diagnostic figures that summarize the comparative detection performance of all models evaluated on a given split. All figures are rendered at 300 dpi with a serif typeface and a muted academic palette, ensuring legibility in print and consistency across manuscripts. Each figure type encodes a distinct facet of model behavior: ranking quality, discriminability, decision-level agreement, and cross-model ranking. The figures are generated from per-image detection records persisted during evaluation and are deterministic functions of the matched-detection tensors.

## Precision–Recall Curves

**File:** `pr_curve_{split}.png`

The Precision–Recall (PR) curve characterizes the ranking quality of a detector as the confidence threshold $\tau$ is swept from $0$ to $1$. For each model, precision $p(\tau)$ and recall $r(\tau)$ are computed by matching detections to ground-truth boxes at Intersection-over-Union (IoU) $\geq 0.5$; unmatched detections contribute to false positives, unmatched ground-truth boxes to false negatives. The curve is parametrized in $\tau$ and rendered monotonically by interpolation. Each model is annotated with the area under the PR curve,

$$
\mathrm{AUC_{PR}} = \int_0^1 p(r)\, dr,
$$

computed by trapezoidal integration. The PR curve is the recommended summary for detection tasks on imbalanced datasets, where the positive class (crack pixels) occupies a small fraction of the image. A model whose curve dominates another's at all recall levels possesses superior ranking quality; the gap between curves at high recall reveals robustness to threshold selection in deployment.

## Receiver Operating Characteristic Curves

**File:** `roc_curve_{split}.png`

The ROC curve plots the true positive rate (TPR, sensitivity) against the false positive rate (FPR, $1 - \mathrm{specificity}$) as $\tau$ varies, and summarizes the discriminability of the detector independent of class prevalence. The diagonal $y = x$ is rendered as a reference for chance performance. The area under the ROC curve,

$$
\mathrm{AUC_{ROC}} = \int_0^1 \mathrm{TPR}(f)\, d\,\mathrm{FPR}(f),
$$

is reported as a scalar. $\mathrm{AUC_{ROC}} = 0.5$ corresponds to a random classifier; $\mathrm{AUC_{ROC}} = 1.0$ to a perfect one. Whereas the PR curve emphasizes performance on the positive class, the ROC curve is invariant to the base rate of negatives, making it complementary when comparing models across datasets with differing crack prevalence. A model with high $\mathrm{AUC_{ROC}}$ but modest $\mathrm{AUC_{PR}}$ typically indicates a high false-positive burden at low confidence, a regime common to crack-like textures in non-crack imagery.

## Image-Level Confusion Matrix

**File:** `confusion_{model}_{split}.png`

For each model and split, a $2 \times 2$ image-level confusion matrix is rendered. An image is defined as *truly positive* if at least one crack instance appears in the ground-truth annotation, and as *predicted positive* if at least one detection with score $\geq 0.25$ is emitted. The matrix layout is

$$
\begin{array}{c|cc}
 & \text{Predicted Negative} & \text{Predicted Positive} \\ \hline
\text{True Negative} & \mathrm{TN} & \mathrm{FP} \\
\text{True Positive} & \mathrm{FN} & \mathrm{TP}
\end{array}
$$

with $\mathrm{TN} + \mathrm{FP} + \mathrm{FN} + \mathrm{TP} = N_{\mathrm{images}}$. Each cell displays both the raw count and the row-normalized fraction (normalized by true class), the latter driving the color intensity. From this matrix we derive the clinical decision metrics

$$
\mathrm{Sensitivity} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}, \qquad
\mathrm{Specificity} = \frac{\mathrm{TN}}{\mathrm{TN} + \mathrm{FP}}.
$$

**Why image-level, not box-level.** A box-level confusion matrix conflates localization accuracy with clinical decision quality and is sensitive to annotation granularity—cracks are amorphous and annotators disagree on extents. In clinical crack screening, the operational question is binary: *does this image warrant human review?* The image-level matrix answers this directly, is robust to annotation variance, and aligns with the deployment workflow in which a flagged image is routed to an inspector. The score threshold of $0.25$ is fixed to reflect a deployment-typical operating point rather than a per-model optimum, ensuring fair cross-model comparison.

## Cross-Model Bar Charts

**Files:** `bar_auc_pr_{split}.png`, `bar_auc_roc_{split}.png`, `bar_f1_max_{split}.png`

Three horizontal bar charts rank models by scalar summary statistics: $\mathrm{AUC_{PR}}$, $\mathrm{AUC_{ROC}}$, and the maximum F1 score. The F1 score is the harmonic mean of precision and recall,

$$
F_1(\tau) = \frac{2\, p(\tau)\, r(\tau)}{p(\tau) + r(\tau)}, \qquad
F_1^{\max} = \max_\tau F_1(\tau),
$$

and $F_1^{\max}$ identifies the best achievable trade-off between precision and recall for each model. The bar charts encode cross-model ranking directly: the model ordering along the horizontal axis is the ordering of the reported statistic, and the bar length is proportional to the statistic value. Bar charts complement the curve overlays by collapsing each model's performance to a single scalar, enabling rapid identification of the best-performing architecture and quantifying the margin between competitors. Where curve crossings occur—indicating regime-dependent dominance—the bar charts disambiguate by reporting the integral or extremal statistic that the deployment criterion will ultimately weigh.

## Comparative Insight Summary

| Figure | Encodes | Comparative Insight |
|---|---|---|
| PR curve | Ranking quality on positive class | Threshold-robustness at high recall |
| ROC curve | Discriminability across thresholds | Class-prevalence-invariant separability |
| Confusion matrix | Image-level decision agreement | Clinical sensitivity / specificity at $\tau = 0.25$ |
| Bar (AUC-PR / AUC-ROC / F1-max) | Scalar ranking | Cross-model ordering and margin |

All figures are generated from the same matched-detection tensors, ensuring internal consistency across the diagnostic suite.
