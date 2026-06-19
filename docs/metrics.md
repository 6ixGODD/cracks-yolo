# Metrics

[English](metrics.md) | [中文](metrics.zh-CN.md)

`cracks_yolo.metrics` implements detection evaluation, efficiency profiling, and statistical model comparison for tongue surface crack detection. Accuracy metrics are computed by `COCOMetricsCalculator` (pycocotools for mAP/AR, scikit-learn for PR/ROC/confusion), efficiency metrics by the test pipeline reusing `analyze_model` (fvcore for MACs), and significance tests by `cracks_yolo.metrics.statistical` (scipy + statsmodels). All scalar outputs are stored at full float64 precision in `metrics.csv` / `model_analysis.json` / `cv_report.json`; round only at typesetting time.

## 1. Preliminaries

### 1.1 Intersection-over-Union (IoU)

For a predicted box $b_p=(x_1,y_1,x_2,y_2)$ and a ground-truth box $b_g$, the Jaccard IoU is

$$
\operatorname{IoU}(b_p, b_g) = \frac{|b_p \cap b_g|}{|b_p \cup b_g|} = \frac{|b_p \cap b_g|}{|b_p| + |b_g| - |b_p \cap b_g|} \in [0,1].
$$

A prediction is a **true positive (TP)** at IoU threshold $\tau$ if it is the highest-scoring prediction matched to a so-far-unmatched ground truth with $\operatorname{IoU} \ge \tau$. Unmatched predictions are **false positives (FP)**; unmatched ground truths are **false negatives (FN)**. For the single-class tongue-crack task, background is treated as the implicit negative class, giving **true negatives (TN)** from image regions producing no detection.

### 1.2 Precision–Recall curve

Ranking all detections by confidence score $s$ and sweeping the operating threshold, precision and recall at threshold $t$ are

$$
P(t) = \frac{\mathrm{TP}(t)}{\mathrm{TP}(t) + \mathrm{FP}(t)}, \qquad
R(t) = \frac{\mathrm{TP}(t)}{\mathrm{TP}(t) + \mathrm{FN}(t)} = \frac{\mathrm{TP}(t)}{N_g},
$$

where $N_g$ is the total number of ground truths. The precision–recall (PR) curve is the locus $\{(R(t), P(t))\}$.

## 2. Aggregate accuracy metrics

### 2.1 Average Precision (AP)

COCO computes AP by 101-point interpolation over the recall axis. Let $p_{\text{interp}}(r) = \max_{\tilde r \ge r} P(\tilde r)$ be the monotonically-decreasing envelope of the measured PR curve. Then for class $c$ at IoU threshold $\tau$,

$$
\mathrm{AP}_{c}^{\tau} = \frac{1}{101}\sum_{i=0}^{100} p_{\text{interp}}\!\left(\frac{i}{100}\right) \in [0,1].
$$

The integral form $\mathrm{AP} = \int_0^1 p_{\text{interp}}(r)\,dr$ is the continuous equivalent.

### 2.2 mAP

Averaging over the $C$ classes gives mean AP at a single IoU threshold:

$$
\mathrm{mAP}^{\tau} = \frac{1}{C}\sum_{c=1}^{C} \mathrm{AP}_{c}^{\tau}.
$$

- **mAP@0.5** (`map50`): $\mathrm{mAP}^{\tau=0.50}$ — single-threshold AP, IoU $\tau=0.5$.
- **mAP@0.5:0.95** (`map5095`): the COCO primary metric, averaged over ten evenly-spaced thresholds $\mathcal{T}=\{0.50,0.55,\ldots,0.95\}$:
  $$
  \mathrm{mAP}_{[.50:.95]} = \frac{1}{|\mathcal{T}|}\sum_{\tau\in\mathcal{T}} \mathrm{mAP}^{\tau} = \frac{1}{|\mathcal{T}|\,C}\sum_{\tau\in\mathcal{T}}\sum_{c=1}^{C}\mathrm{AP}_{c}^{\tau}.
  $$
- **AP@0.5** (`ap50`), **AP@0.75** (`ap75`): per-class single-threshold AP; `ap50` aliases `map50` for the single-class case.

### 2.3 Precision / Recall / F1

Reported at the F1-optimal confidence threshold $t^*=\arg\max_t F_1(t)$:

$$
P = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, \qquad
R = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \qquad
F_1 = \frac{2\,P\,R}{P+R} = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}.
$$

### 2.4 Average Recall (AR)

COCO AR is the maximum recall achievable subject to a per-image detection budget $K$, averaged over IoU thresholds $\mathcal{T}$ and classes:

$$
\mathrm{AR}_{K} = \frac{1}{|\mathcal{T}|\,C}\sum_{\tau\in\mathcal{T}}\sum_{c=1}^{C} \max_{\mathcal{D}\subseteq\mathcal{D}_c,\,|\mathcal{D}|\le K} R_{c}^{\tau}(\mathcal{D}),
$$

where $\mathcal{D}_c$ is the set of class-$c$ detections and $R_{c}^{\tau}(\mathcal{D})$ the recall of subset $\mathcal{D}$ at $\tau$. Reported budgets: `ar1` ($K{=}1$), `ar10` ($K{=}10$), `ar100` ($K{=}100$), with `ar300`/`ar1000` aliased to `ar100` (pycocotools caps at 100). Object-size-stratified variants use COCO area bands:

$$
\text{small: } a < 32^2,\quad \text{medium: } 32^2 \le a < 96^2,\quad \text{large: } a \ge 96^2 \text{ px}^2,
$$

yielding `ar_small`, `ar_medium`, `ar_large`.

## 3. Curve and operating-point metrics

### 3.1 AUC-PR / AUC-ROC

- **AUC-PR** (`auc_pr`): trapezoidal area under the PR curve, $\int_0^1 P(r)\,dr$. Threshold-independent summary; the recommended primary figure for imbalanced single-class detection.
- **AUC-ROC** (`auc_roc`): area under the ROC curve $\{(1-\mathrm{Specificity}(t),\,R(t))\}$, i.e. $\int_0^1 \mathrm{TPR}(t)\,d\mathrm{FPR}(t)$ where $\mathrm{FPR}=\mathrm{FP}/(\mathrm{FP}+\mathrm{TN})$.

### 3.2 Sensitivity / Specificity / PPV / NPV

From the $2{\times}2$ confusion matrix at $t^*$:

$$
\text{Sensitivity (Recall, TPR)} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \qquad
\text{Specificity (TNR)} = \frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}},
$$
$$
\text{PPV (Precision)} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, \qquad
\text{NPV} = \frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FN}}.
$$

For the single-class crack task the $2{\times}2$ matrix collapses TP/FN/FP/TN directly onto sensitivity and PPV.

## 4. Efficiency metrics

Every test run (and every CV fold) profiles efficiency alongside accuracy, so one sweep yields a full accuracy–speed–cost comparison table. Set `TestConfig.measure_efficiency=False` to skip. `EfficiencyReport` (pydantic) is written into `metrics.csv`, `model_analysis.json`, and the `TestLog` record.

### 4.1 Throughput and latency

Let $T_b$ be the wall-clock time (CUDA-synchronized) of one inference batch covering $n_b$ images, including forward $\to$ decode $\to$ NMS but excluding file I/O and visualization. With $B$ batches over $N=\sum_b n_b$ test images:

$$
\mathrm{FPS}_{\text{mean}} = \frac{N}{\sum_b T_b}, \qquad
\ell_b = \frac{T_b}{n_b}\ \text{(per-image latency, ms)}, \qquad
\mathrm{FPS}_{p} = \frac{1000}{\ell_{p}}\ \text{for } p\in\{50, 95\},
$$

where $\ell_{p}$ is the $p$-th percentile of $\{\ell_b\}$ via linear interpolation. `fps_mean`/`fps_p50`/`fps_p95` and `latency_mean_ms`/`latency_p50_ms`/`latency_p95_ms` are reported. These are **end-to-end** figures on the real test loader at `TestConfig.batch_size` — not synthetic single-image forwards.

### 4.2 Compute and memory budget

- **Parameters** (`n_parameters`, `n_trainable_parameters`): $\sum_{\theta\in\Theta}|\theta|$ over all / trainable tensors.
- **MACs** (`macs`): multiply–accumulate operations on a single $(1,3,H,W)$ input, measured by `fvcore.nn.FlopCountAnalysis` and divided by 2 (fvcore reports FLOPs; $\mathrm{MACs}=\mathrm{FLOPs}/2$ by the convention one MAC $=$ two FLOPs).
- **GFLOPs** (`gflops`): $2\times\mathrm{MACs}$ in giga-units, $\mathrm{GFLOPs} = \mathrm{MACs}\times 2 / 10^{9}$.
- **Peak VRAM** (`peak_vram_bytes`): $\max_t \mathrm{torch.cuda.max\_memory\_allocated}$ over the real inference loop (synthetic single-image value on CPU).

> **Known limitation.** `fvcore` traces standard ops but cannot fully trace the custom YOLO detect heads and torchvision wrapper forwards, so `macs`/`gflops` may report `0.0` for those families. Params, FPS, latency, and VRAM are unaffected and authoritative. For a portable compute-cost proxy, use parameter count.

The standalone `scripts/analyze_model.py` reports the same structural metrics on a dummy input without running a test; the test-pipeline figures are authoritative because they include decode + NMS at the real batch size.

## 5. Statistical tests (model comparison)

`StatisticalTest` compares two model variants (e.g. baseline vs. SAC) on matched per-fold or per-image metrics. Let $d_i = x_i - y_i$ be the paired differences, $i=1,\dots,n$, $\bar d = \frac1n\sum_i d_i$, $s_d = \sqrt{\frac1{n-1}\sum_i(d_i-\bar d)^2}$.

### 5.1 Paired t-test (`paired_t`)

$$
t = \frac{\bar d}{s_d / \sqrt{n}}, \qquad \mathrm{df} = n-1,
$$

with a two-sided $p$-value from Student's $t$ distribution. Assumes approximately normal differences; fast.

### 5.2 Wilcoxon signed-rank (`wilcoxon`)

Non-parametric. Rank $|d_i|$ ascending, assign signed ranks $r_i = \mathrm{sgn}(d_i)\,\mathrm{rank}(|d_i|)$, then

$$
W = \sum_{i:d_i > 0} |r_i| \quad (\text{or } W = \min(W^+, W^-) \text{ per convention}).
$$

Robust to non-normal differences and outliers.

### 5.3 Bootstrap confidence interval (`bootstrap_ci`)

Resample the $n$ differences with replacement $B$ times (default $B{=}1000$), computing $\bar d^{(b)}$ each time; the 95% percentile CI is the $[2.5\%, 97.5\%]$ empirical quantiles of $\{\bar d^{(b)}\}_{b=1}^{B}$:

$$
\mathrm{CI}_{0.95} = \big[\, Q_{0.025}(\{\bar d^{(b)}\}),\ Q_{0.975}(\{\bar d^{(b)}\})\,\big].
$$

Most robust; slowest. `ci_low`/`ci_high` are populated only for this test.

### 5.4 Test selection

- **Paired t** — fast, normality assumption.
- **Wilcoxon** — non-normal differences.
- **Bootstrap CI** — robust effect-size interval; report alongside the $p$-value rather than as a binary significant/not-significant verdict.

## 6. Detection-level records

`DetectionMetric` and `PerImageDetection` (`TypedDict`) are the per-image units fed to the calculator.

| Field | Type | Description |
| --- | --- | --- |
| `image_id` | `int` | Image identifier (dataset index). |
| `class_id` | `int` | Predicted / ground-truth class id. |
| `score` | `float` | Confidence score in $[0,1]$ (`1.0` for ground truth). |
| `bbox_xyxy` | `tuple[float,float,float,float]` | $(x_1,y_1,x_2,y_2)$ in pixels. |

## 7. MetricReport

`MetricReport` (dataclass) is the accuracy payload returned by `COCOMetricsCalculator.run()`:

```python
@dataclass
class MetricReport:
    map50: float
    map5095: float
    ap50: dict[int, float]       # per-class AP@0.5
    ap75: dict[int, float]       # per-class AP@0.75
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

`MetricReport` is detection accuracy only; efficiency is reported separately (§4).

## 8. Evaluation curves

`cracks_yolo.metrics.curves` / `.confusion` produce, saved to `curves/`:

- `pr.png` — precision–recall curve at IoU $0.5$.
- `roc.png` — ROC curve.
- `confusion.png` — normalized confusion-matrix heatmap.

## 9. Interpretation for tongue crack detection

Tongue surface cracks are thin, elongated structures; a small spatial offset between prediction and ground truth sharply reduces IoU, making the task IoU-sensitive.

- **Prefer mAP@0.5 over mAP@0.5:0.95** — the stricter thresholds in $[.50:.95]$ penalize spatially-close but not pixel-perfect detections; for thin cracks, mAP@0.5 better reflects clinically useful detections.
- **AUC-PR is threshold-independent** — summarizes quality across all confidence thresholds without a fixed operating point; preferred under class imbalance.
- **AR@100 / AR@1000** — higher budgets assess whether the model recalls fine crack fragments split into multiple small segments.
- **Confusion matrix** — with a single class, it reduces to TP/FN/FP rates mapping onto sensitivity and PPV.
