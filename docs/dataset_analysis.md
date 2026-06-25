# Dataset Analysis

[English](dataset_analysis.md) | [中文](dataset_analysis.zh-CN.md)

## 1. Overview

The `cy analyze-dataset` command computes a suite of diversity and balance
metrics for an object-detection dataset and emits eight publication-grade
figures (300 dpi, serif typography). The computation is implemented in
`cracks_yolo/analysis/dataset.py` and the figure suite in
`cracks_yolo/viz/dataset.py`. The structured result is serialized to
`dataset_analysis.json`; the figures are written as PNG files into the analysis
output directory.

The metrics quantify three orthogonal axes of dataset bias that are known to
influence detector training:

- **Class balance** — the distribution of instances across semantic categories.
- **Scale balance** — the distribution of bounding-box areas and aspect ratios.
- **Spatial balance** — the spatial density of annotation centers and the
  heterogeneity of image dimensions.

For crack-detection corpora, where a single class ("crack") typically dominates
and where elongated, thin structures at multiple scales are the rule rather
than the exception, these diagnostics are prerequisite to choosing anchor
configurations, loss pos-weights, and augmentation policies.

## 2. Diversity Metrics

Let $\mathcal{D} = \{(I_i, \mathcal{B}_i)\}_{i=1}^{N}$ denote the dataset of $N$
images, where each image $I_i$ is annotated with a set
$\mathcal{B}_i = \{b_{ij}\}_{j=1}^{n_i}$ of bounding boxes. Each box
$b_{ij} = (x_1, y_1, x_2, y_2)$ is stored in normalized image coordinates, i.e.
$x_k, y_k \in [0, 1]$.

### 2.1 Cardinalities

- **`n_images`** — $N = |\mathcal{D}|$, the number of image records.
- **`n_annotations`** — $M = \sum_{i=1}^{N} n_i$, the total number of boxes.
- **`n_classes`** — $K = |\{c_{ij}\}|$, the number of distinct labels present.

### 2.2 Class Distribution

`class_counts` is the histogram $n_k$ over class labels
$k \in \{1, \dots, K\}$, where $n_k = \sum_{i,j} \mathbb{1}[c_{ij} = k]$.

### 2.3 Imbalance Ratio

$$
\rho \;=\; \frac{\max_k n_k}{\max\!\left(1,\; \min_k n_k\right)}
$$

$\rho = 1$ indicates a perfectly balanced dataset; $\rho \gg 1$ signals severe
class imbalance. For single-class crack datasets $\rho \equiv 1$ by
construction, so the metric is only informative for multi-class corpora (e.g.
longitudinal versus transverse cracks).

### 2.4 Class Shannon Entropy

$$
H \;=\; -\sum_{k=1}^{K} p_k \log_2 p_k, \qquad p_k \;=\; \frac{n_k}{M}
$$

$H$ is measured in bits and bounded by $0 \le H \le \log_2 K$. $H = 0$
corresponds to a degenerate single-class dataset; $H = \log_2 K$ to a uniform
distribution. A normalized evenness index may be recovered as
$J = H / \log_2 K$ (Pielou's $J$).

### 2.5 COCO Scale Buckets

Each box has normalized area $a_{ij} = (x_2 - x_1)(y_2 - y_1) \in [0, 1]$.
Adapting the COCO convention (small $< 32 \times 32$ px, medium
$< 96 \times 96$ px, large $\ge 96 \times 96$ px) to a normalized coordinate
frame referenced to a nominal $64$-pixel grid cell, the area thresholds become

$$
\tau_s \;=\; \left(\frac{32}{64}\right)^{\!2} \;=\; 0.25,
\qquad
\tau_l \;=\; \left(\frac{96}{64}\right)^{\!2} \;=\; 2.25.
$$

The three counters are

$$
\texttt{bbox\_area\_small} \;=\; \sum_{i,j} \mathbb{1}\!\left[a_{ij} < \tau_s\right],
\quad
\texttt{bbox\_area\_medium} \;=\; \sum_{i,j} \mathbb{1}\!\left[\tau_s \le a_{ij} < \tau_l\right],
\quad
\texttt{bbox\_area\_large} \;=\; \sum_{i,j} \mathbb{1}\!\left[a_{ij} \ge \tau_l\right].
$$

This normalization renders the buckets scale-invariant with respect to the
original image resolution, which is essential when the corpus aggregates images
of heterogeneous dimensions.

### 2.6 Aspect-Ratio Diversity

For each box with $h_{ij} = y_2 - y_1 > 0$, the aspect ratio is
$r_{ij} = (x_2 - x_1) / h_{ij}$. Ratios are quantized to one decimal place,
$\tilde{r}_{ij} = \mathrm{round}(r_{ij}, 1)$, and the metric counts the
cardinality of distinct quantized values:

$$
\texttt{bbox\_aspect\_ratio\_buckets} \;=\; \left|\left\{ \tilde{r}_{ij} \right\}\right|.
$$

A large value denotes a heterogeneous population of box shapes; for crack
corpora this is typically high because cracks span orientations from
near-horizontal to near-vertical.

### 2.7 Spatial Coverage

The unit image square $[0,1]^2$ is partitioned into a $G \times G$ uniform grid
(default $G = 16$). Each box center
$(c_x, c_y) = \bigl((x_1+x_2)/2,\, (y_1+y_2)/2\bigr)$ is mapped to a cell index

$$
g_x \;=\; \mathrm{clip}\!\left(\lfloor c_x\, G \rfloor,\, 0,\, G-1\right),
\qquad
g_y \;=\; \mathrm{clip}\!\left(\lfloor c_y\, G \rfloor,\, 0,\, G-1\right).
$$

Let $\mathcal{C} \subseteq \{0, \dots, G-1\}^2$ be the set of occupied cells.
The spatial coverage is

$$
\texttt{spatial\_coverage} \;=\; \frac{|\mathcal{C}|}{G^2} \;\in\; [0, 1].
$$

A value close to $0$ indicates that annotations are concentrated in a few image
regions (a spatial sampling bias); a value close to $1$ indicates near-uniform
spatial coverage.

### 2.8 Image-Size Statistics

For the sequences of widths $\{W_i\}_{i=1}^{N}$ and heights $\{H_i\}_{i=1}^{N}$
in pixels, the report records the population mean and population standard
deviation:

$$
\mu_W \;=\; \frac{1}{N}\sum_{i=1}^{N} W_i,
\qquad
\sigma_W \;=\; \sqrt{\frac{1}{N}\sum_{i=1}^{N} (W_i - \mu_W)^2},
$$

and analogously $\mu_H, \sigma_H$ for heights. These four fields
(`image_width_mean`, `image_width_std`, `image_height_mean`,
`image_height_std`) diagnose whether the corpus is dimensionally homogeneous —
a prerequisite for choosing a fixed inference resolution without distortion.

## 3. Visualization Suite

All figures are rendered at 300 dpi with serif typography (`matplotlib`
rcParams `font.family = "serif"`) and tight bounding boxes, suitable for direct
inclusion in a manuscript.

### 3.1 `class_distribution.png`

A vertical bar plot of $n_k$ versus class ID $k$. Bar labels report the integer
count. **Encoding**: $x$-axis = class ID, $y$-axis = instance count.
**Insight**: directly visualizes the per-class prior $p_k$ and complements the
scalar imbalance ratio $\rho$.

### 3.2 `bbox_size.png`

A 50-bin histogram of normalized areas $a_{ij}$ on a logarithmic $x$-axis.
**Encoding**: $x$-axis = $\log_{10} a_{ij}$, $y$-axis = count. **Insight**:
reveals whether the corpus is dominated by small (fine crack) or large (long
crack) instances, informing anchor-area priors.

### 3.3 `area_buckets.png`

A pie chart of the three COCO scale buckets (small / medium / large) with
percentage labels and absolute counts. **Encoding**: slice area $\propto$ bucket
count. **Insight**: a single-glance summary of the scale distribution; a
corpus skewed toward "small" foreshadows poor recall for thin hairline cracks
at low inference resolution.

### 3.4 `aspect_ratio.png`

A 40-bin histogram of aspect ratios $r_{ij} = w/h$. **Encoding**: $x$-axis =
$r$, $y$-axis = count. **Insight**: a multimodal distribution suggests that
the anchor design should include both flat and tall aspect ratios; for crack
detection the distribution is typically heavy-tailed around $r \approx 1$ with
substantial mass at $r \gg 1$ and $r \ll 1$.

### 3.5 `objects_per_image.png`

A histogram of per-image object counts $n_i$ with unit-width bins aligned to
integer counts. **Encoding**: $x$-axis = $n_i$, $y$-axis = image count.
**Insight**: characterizes the density of annotations per image and informs
batching and padding strategy; a long right tail benefits from grouping-based
batching.

### 3.6 `bbox_heatmap.png`

A $32 \times 32$ heatmap of bounding-box center counts, rendered with the
`"hot"` colormap and an origin-lower layout. **Encoding**: cell intensity
$\propto$ number of centers falling in the cell. **Insight**: exposes spatial
sampling bias (e.g. cracks concentrated along image diagonals or centers) and
informs crop and mosaic augmentation policy.

### 3.7 `image_size.png`

A scatter plot of $(W_i, H_i)$ pairs. **Encoding**: $x$-axis = width (px),
$y$-axis = height (px). **Insight**: identifies resolution clusters and
outlier dimensions; a tight cluster justifies a single fixed inference
resolution, whereas a diffuse cloud motivates letterbox resizing or
multi-scale training.

### 3.8 `class_imbalance.png`

A horizontal bar plot of $n_k$ sorted in ascending order, with bar labels.
**Encoding**: $y$-axis = class ID (sorted by count), $x$-axis = instance
count. **Insight**: the sorted ordering makes the head-tail structure of the
class distribution immediately visible and is the canonical figure for arguing
the need for class-balanced sampling or focal loss.

## 4. Practical Use

Run the analysis via:

```bash
cy analyze-dataset --dataset <dataset_root> --output-dir <output_dir> \
                   [--splits train,valid,test]
```

The command writes `dataset_analysis.json` and the eight PNG figures into
`<output_dir>`, both per-split (under `<output_dir>/<split>/`) and combined.
The JSON record can be diffed across dataset versions to track the effect of
augmentation or re-annotation on the diversity axes defined above.
