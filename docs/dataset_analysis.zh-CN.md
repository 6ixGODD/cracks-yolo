# 数据集分析

[English](dataset_analysis.md) | [中文](dataset_analysis.zh-CN.md)

## 1. 概述

`cy analyze-dataset` 命令对目标检测数据集计算一组多样性与均衡性指标，并生成八幅出版级图件（300 dpi，衬线字体）。计算实现位于 `cracks_yolo/analysis/dataset.py`，图件生成位于 `cracks_yolo/viz/dataset.py`。结构化结果序列化为 `dataset_analysis.json`；八幅图件以 PNG 格式写入分析输出目录。

所计算之指标从三个正交维度刻画数据集偏差，此三者于检测器训练中均具显著影响：

- **类别均衡性** —— 实例在语义类别上的分布。
- **尺度均衡性** —— 边界框面积与长宽比的分布。
- **空间均衡性** —— 标注中心的空间密度与图像尺寸的异质性。

对于裂缝检测语料而言，单一类别（"crack"）通常占据主导，且细长、薄壁结构在多尺度上普遍存在，故上述诊断指标乃确定锚框配置、损失正权重及数据增强策略之前置条件。

## 2. 多样性指标

记数据集 $\mathcal{D} = \{(I_i, \mathcal{B}_i)\}_{i=1}^{N}$，含 $N$ 幅图像，每幅图像 $I_i$ 标注有边界框集合 $\mathcal{B}_i = \{b_{ij}\}_{j=1}^{n_i}$。每个框 $b_{ij} = (x_1, y_1, x_2, y_2)$ 以归一化图像坐标存储，即 $x_k, y_k \in [0, 1]$。

### 2.1 基数

- **`n_images`** —— $N = |\mathcal{D}|$，图像记录数。
- **`n_annotations`** —— $M = \sum_{i=1}^{N} n_i$，边界框总数。
- **`n_classes`** —— $K = |\{c_{ij}\}|$，出现过的类别标签数。

### 2.2 类别分布

`class_counts` 为类别标签 $k \in \{1, \dots, K\}$ 上的直方图 $n_k$，其中 $n_k = \sum_{i,j} \mathbb{1}[c_{ij} = k]$。

### 2.3 不平衡比

$$
\rho \;=\; \frac{\max_k n_k}{\max\!\left(1,\; \min_k n_k\right)}
$$

$\rho = 1$ 表示数据集完全均衡；$\rho \gg 1$ 表示严重的类别不平衡。对于单类别裂缝数据集，$\rho \equiv 1$ 恒成立，该指标仅对多类别语料（如纵向裂缝与横向裂缝并存）具有判别力。

### 2.4 类别香农熵

$$
H \;=\; -\sum_{k=1}^{K} p_k \log_2 p_k, \qquad p_k \;=\; \frac{n_k}{M}
$$

$H$ 以比特为单位，满足 $0 \le H \le \log_2 K$。$H = 0$ 对应退化之单类别数据集；$H = \log_2 K$ 对应均匀分布。规范化均匀度指数可由 $J = H / \log_2 K$（Pielou $J$）恢复。

### 2.5 COCO 尺度区间

每个框的归一化面积为 $a_{ij} = (x_2 - x_1)(y_2 - y_1) \in [0, 1]$。将 COCO 约定（小 $< 32 \times 32$ 像素，中 $< 96 \times 96$ 像素，大 $\ge 96 \times 96$ 像素）适配至以 $64$ 像素名义网格单元为参照的归一化坐标框架，面积阈值化为

$$
\tau_s \;=\; \left(\frac{32}{64}\right)^{\!2} \;=\; 0.25,
\qquad
\tau_l \;=\; \left(\frac{96}{64}\right)^{\!2} \;=\; 2.25.
$$

三个计数器为

$$
\texttt{bbox\_area\_small} \;=\; \sum_{i,j} \mathbb{1}\!\left[a_{ij} < \tau_s\right],
\quad
\texttt{bbox\_area\_medium} \;=\; \sum_{i,j} \mathbb{1}\!\left[\tau_s \le a_{ij} < \tau_l\right],
\quad
\texttt{bbox\_area\_large} \;=\; \sum_{i,j} \mathbb{1}\!\left[a_{ij} \ge \tau_l\right].
$$

此归一化使尺度区间对原始图像分辨率保持尺度不变，当语料汇集不同尺寸之图像时尤为重要。

### 2.6 长宽比多样性

对每一满足 $h_{ij} = y_2 - y_1 > 0$ 的框，定义长宽比 $r_{ij} = (x_2 - x_1) / h_{ij}$。将比值量化至小数点后一位，$\tilde{r}_{ij} = \mathrm{round}(r_{ij}, 1)$，该指标计数不同量化值之基数：

$$
\texttt{bbox\_aspect\_ratio\_buckets} \;=\; \left|\left\{ \tilde{r}_{ij} \right\}\right|.
$$

取值越大表明框形状总体越异质；对裂缝语料该值通常较高，因裂缝走向覆盖从近水平到近竖直的连续区间。

### 2.7 空间覆盖率

将单位图像正方形 $[0,1]^2$ 均匀划分为 $G \times G$ 网格（默认 $G = 16$）。每个框中心 $(c_x, c_y) = \bigl((x_1+x_2)/2,\, (y_1+y_2)/2\bigr)$ 映射至单元索引

$$
g_x \;=\; \mathrm{clip}\!\left(\lfloor c_x\, G \rfloor,\, 0,\, G-1\right),
\qquad
g_y \;=\; \mathrm{clip}\!\left(\lfloor c_y\, G \rfloor,\, 0,\, G-1\right).
$$

设 $\mathcal{C} \subseteq \{0, \dots, G-1\}^2$ 为被占据单元之集合，则空间覆盖率为

$$
\texttt{spatial\_coverage} \;=\; \frac{|\mathcal{C}|}{G^2} \;\in\; [0, 1].
$$

取值接近 $0$ 表明标注集中于少数图像区域（空间采样偏差）；取值接近 $1$ 表明空间覆盖近乎均匀。

### 2.8 图像尺寸统计

对于以像素为单位的宽度序列 $\{W_i\}_{i=1}^{N}$ 与高度序列 $\{H_i\}_{i=1}^{N}$，报告记录总体均值与总体标准差：

$$
\mu_W \;=\; \frac{1}{N}\sum_{i=1}^{N} W_i,
\qquad
\sigma_W \;=\; \sqrt{\frac{1}{N}\sum_{i=1}^{N} (W_i - \mu_W)^2},
$$

高度同理有 $\mu_H, \sigma_H$。此四项字段（`image_width_mean`、`image_width_std`、`image_height_mean`、`image_height_std`）用以诊断语料在尺寸上的同质性——此为选择固定推理分辨率而不引入畸变的前置条件。

## 3. 图件套件

全部图件以 300 dpi 渲染，采用衬线字体（`matplotlib` rcParams `font.family = "serif"`）与紧凑边框，可直接嵌入稿件。

### 3.1 `class_distribution.png`

类别 ID $k$ 上的实例计数 $n_k$ 竖直条形图，条形上标注整数计数值。**编码**：$x$ 轴为类别 ID，$y$ 轴为实例计数。**判读**：直观呈现类别先验 $p_k$，对标量不平衡比 $\rho$ 形成补充。

### 3.2 `bbox_size.png`

归一化面积 $a_{ij}$ 的 50 区间直方图，$x$ 轴取对数刻度。**编码**：$x$ 轴为 $\log_{10} a_{ij}$，$y$ 轴为计数。**判读**：揭示语料是否以小尺度（细微裂缝）或大尺度（长裂缝）实例为主，用以校准锚框面积先验。

### 3.3 `area_buckets.png`

COCO 三尺度区间（小 / 中 / 大）的饼图，标注百分比与绝对计数。**编码**：扇形面积正比于区间计数。**判读**：尺度分布的单览摘要；若语料严重偏向"小"区间，预示在低推理分辨率下对细如发丝的裂缝召回率不足。

### 3.4 `aspect_ratio.png`

长宽比 $r_{ij} = w/h$ 的 40 区间直方图。**编码**：$x$ 轴为 $r$，$y$ 轴为计数。**判读**：多峰分布提示锚框设计须同时纳入扁平与高瘦两种长宽比；裂缝数据集的长宽比分布通常在 $r \approx 1$ 处重尾，且在 $r \gg 1$ 与 $r \ll 1$ 处均有显著质量。

### 3.5 `objects_per_image.png`

每幅图像目标计数 $n_i$ 的直方图，区间宽度为 1，对齐至整数计数。**编码**：$x$ 轴为 $n_i$，$y$ 轴为图像计数。**判读**：刻画每图标注密度，指导批组织与填充策略；右尾较长者宜采用基于分组（grouping）的批构建。

### 3.6 `bbox_heatmap.png`

边界框中心计数的 $32 \times 32$ 热力图，采用 `"hot"` 配色，原点置于左下。**编码**：单元亮度正比于落入该单元的中心数。**判读**：暴露空间采样偏差（如裂缝集中于图像对角线或中心），指导裁剪与马赛克（mosaic）增强策略。

### 3.7 `image_size.png`

$(W_i, H_i)$ 对的散点图。**编码**：$x$ 轴为宽度（像素），$y$ 轴为高度（像素）。**判读**：辨识分辨率簇与离群尺寸；紧密簇证明可采用单一固定推理分辨率，弥散云状则需 letterbox 缩放或多尺度训练。

### 3.8 `class_imbalance.png`

$n_k$ 按升序排列的水平条形图，条形上标注计数值。**编码**：$y$ 轴为类别 ID（按计数排序），$x$ 轴为实例计数。**判读**：排序后的呈现使类别分布的"头-尾"结构一目了然，乃论证采用类别均衡采样或 focal loss 之标准图件。

## 4. 实际使用

经如下命令执行分析：

```bash
cy analyze-dataset --dataset <dataset_root> --output-dir <output_dir> \
                   [--splits train,valid,test]
```

该命令将 `dataset_analysis.json` 与八幅 PNG 图件写入 `<output_dir>`，分别按划分（位于 `<output_dir>/<split>/`）与合并整体两层次输出。该 JSON 记录可跨数据集版本进行差异比对，以追踪数据增强或重新标注对上述各多样性维度的影响。
