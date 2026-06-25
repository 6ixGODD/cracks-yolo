# Evaluation Metrics

[English](metrics.md) | [中文](metrics.zh-CN.md)

This document formalises every metric reported in `test_metrics.csv`. All
definitions follow the MS COCO evaluation protocol and standard binary
classification theory, adapted to single-class object detection.

## Notation

Let $\mathcal{D}=\{(I_i,\mathcal{G}_i)\}_{i=1}^{N}$ denote the evaluation set
of $N$ images, where $I_i$ is the $i$-th image and $\mathcal{G}_i$ its set of
ground-truth boxes. Let $\mathcal{P}_i=\{(b_j,s_j)\}$ be the set of predicted
boxes on $I_i$ with confidence scores $s_j$. A prediction $b_j$ is deemed a
**true positive** (TP) if it matches an unmatched ground-truth box with
Intersection-over-Union (IoU) at least $\tau$; otherwise it is a **false
positive** (FP). Unmatched ground-truth boxes are **false negatives** (FN).

$$
\mathrm{IoU}(b,g)=\frac{|b\cap g|}{|b\cup g|}
$$

## mAP@0.5 (map50)

Average Precision at IoU threshold $\tau=0.5$. The precision–recall curve is
swept by varying the confidence threshold; AP is its area.

$$
\mathrm{AP}_{0.5}=\int_0^1 p(r)\,dr,\qquad
p(r)=\frac{\mathrm{TP}(r)}{\mathrm{TP}(r)+\mathrm{FP}(r)},\quad
r=\frac{\mathrm{TP}(r)}{\mathrm{TP}(r)+\mathrm{FN}}
$$

For single-class detection, $\mathrm{mAP}_{0.5}=\mathrm{AP}_{0.5}$. It rewards
the ranking quality of detections: a model that places true boxes at the top
of its confidence-ordered list attains a high AP.

## mAP@[0.5:0.95] (map5095)

The primary COCO metric. AP is averaged over ten IoU thresholds
$\tau\in\{0.50,0.55,\dots,0.95\}$:

$$
\mathrm{mAP}_{[0.5:0.95]}=\frac{1}{10}\sum_{k=0}^{9}\mathrm{AP}_{0.50+0.05k}
$$

This penalises loose localisation: a box overlapping a ground truth at IoU
0.5 but not 0.75 contributes to $\mathrm{AP}_{0.5}$ yet not to
$\mathrm{AP}_{0.75}$. The metric therefore jointly measures classification
and regression accuracy.

## AP@0.75 (ap75)

Average Precision at the strict IoU threshold $\tau=0.75$. It isolates
localisation precision and is far more sensitive to box-fit quality than
$\mathrm{AP}_{0.5}$.

## Precision (precision)

At the operating point maximising the F1 score,

$$
\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}
$$

It is the fraction of predicted boxes that are correct — the model's
false-alarm rate is $1-\mathrm{Precision}$.

## Recall / Sensitivity (recall, sensitivity)

$$
\mathrm{Recall}=\mathrm{Sensitivity}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}=\frac{\mathrm{TP}}{|\mathcal{G}|}
$$

The fraction of ground-truth cracks that are recovered. The two columns are
identical by definition; the duplicate naming reflects the machine-learning
(`recall`) and medical-statistics (`sensitivity`) conventions.

## F1 (f1)

The harmonic mean of precision and recall at the F1-optimal threshold:

$$
F_1=\frac{2\,p\,r}{p+r}
$$

It is invariant to the precision/recall trade-off imbalance and is preferred
when a single operating point is required.

## AR@1 / AR@10 / AR@100 (ar1, ar10, ar100)

Average Recall capped at $K$ detections per image:

$$
\mathrm{AR}_K=\frac{1}{N}\sum_{i=1}^{N}\frac{\mathrm{TP}_i^{(K)}}{|\mathcal{G}_i|}
$$

where $\mathrm{TP}_i^{(K)}$ counts true positives among the top-$K$ highest-
confidence predictions on image $i$. $\mathrm{AR}_1$ measures the model's
single-best-guess accuracy; $\mathrm{AR}_{100}$ approximates saturated recall.
All AR values are averaged over IoU $\in[0.5:0.95]$.

## AUC-PR (auc_pr)

Area under the precision–recall curve, computed over **all** detections (not
thresholded to a single operating point). For detection,

$$
\mathrm{AUC}_{PR}=\int_0^1 p(r)\,dr
$$

where the curve is built by ranking all detections by confidence. Unlike AP,
which uses 101-point interpolation, AUC-PR here uses the trapezoidal rule on
the empirical curve. It is the recommended summary statistic under severe
class imbalance.

## AUC-ROC (auc_roc)

Area under the Receiver Operating Characteristic curve, plotting true-positive
rate against false-positive rate:

$$
\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\qquad
\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}},\qquad
\mathrm{AUC}_{ROC}=\int_0^1 \mathrm{TPR}(\mathrm{FPR})\,d(\mathrm{FPR})
$$

Here detections are matched to ground-truth boxes at IoU $0.5$ and treated as
binary outcomes; unmatched predictions are negatives. $\mathrm{AUC}_{ROC}=0.5$
is chance level.

## Specificity (specificity)

Image-level true-negative rate. An image is *positive* if it contains $\geq 1$
crack in the ground truth, *negative* otherwise; it is *predicted positive* if
the model emits $\geq 1$ detection above the score threshold $\sigma=0.25$.

$$
\mathrm{Specificity}=\frac{\mathrm{TN}_{\text{img}}}{\mathrm{TN}_{\text{img}}+\mathrm{FP}_{\text{img}}}
$$

where $\mathrm{FP}_{\text{img}}$ counts crack-free images falsely flagged and
$\mathrm{TN}_{\text{img}}$ counts crack-free images correctly left alone. This
is the clinically meaningful "specificity" — the probability that a normal
tongue surface is not alarmed.

## PPV / NPV (ppv, npv)

Positive and Negative Predictive Values, image-level:

$$
\mathrm{PPV}=\frac{\mathrm{TP}_{\text{img}}}{\mathrm{TP}_{\text{img}}+\mathrm{FP}_{\text{img}}},\qquad
\mathrm{NPV}=\frac{\mathrm{TN}_{\text{img}}}{\mathrm{TN}_{\text{img}}+\mathrm{FN}_{\text{img}}}
$$

PPV equals precision computed at the image level; NPV is the probability that
a non-alarm is truly crack-free. Both depend on disease prevalence and are
reported alongside specificity for clinical interpretation.
