# 指标

[English](metrics.md) | [中文](metrics.zh-CN.md)

`cracks_yolo.metrics` 为舌面裂纹检测实现检测评估、效率分析与统计模型比较。精度指标由 `COCOMetricsCalculator` 计算（mAP/AR 用 pycocotools，PR/ROC/混淆矩阵用 scikit-learn），效率指标由测试流程复用 `analyze_model` 计算（MACs 用 fvcore），显著性检验由 `cracks_yolo.metrics.statistical` 提供（scipy + statsmodels）。所有标量输出以完整 float64 精度存入 `metrics.csv` / `model_analysis.json` / `cv_report.json`，仅在排版时取整。

## 1. 预备定义

### 1.1 交并比（IoU）

对预测框 $b_p=(x_1,y_1,x_2,y_2)$ 与真值框 $b_g$，Jaccard 交并比为

$$
\operatorname{IoU}(b_p, b_g) = \frac{|b_p \cap b_g|}{|b_p \cup b_g|} = \frac{|b_p \cap b_g|}{|b_p| + |b_g| - |b_p \cap b_g|} \in [0,1].
$$

在 IoU 阈值 $\tau$ 下，若某预测是与某尚未匹配真值中 IoU 最高且 $\operatorname{IoU} \ge \tau$ 的匹配，则记为**真正例（TP）**；未匹配预测为**假正例（FP）**；未匹配真值为**假负例（FN）**。对单类裂纹任务，背景视为隐式负类，由此得到**真负例（TN）**。

### 1.2 精确率–召回率曲线

按置信度 $s$ 排序所有检测，扫描操作阈值，阈值 $t$ 处的精确率与召回率为

$$
P(t) = \frac{\mathrm{TP}(t)}{\mathrm{TP}(t) + \mathrm{FP}(t)}, \qquad
R(t) = \frac{\mathrm{TP}(t)}{\mathrm{TP}(t) + \mathrm{FN}(t)} = \frac{\mathrm{TP}(t)}{N_g},
$$

其中 $N_g$ 为真值总数。精确率–召回率（PR）曲线为点集 $\{(R(t), P(t))\}$。

## 2. 聚合精度指标

### 2.1 平均精度（AP）

COCO 对召回率轴做 101 点插值计算 AP。令 $p_{\text{interp}}(r) = \max_{\tilde r \ge r} P(\tilde r)$ 为实测 PR 曲线的单调递减包络。则类 $c$、IoU 阈值 $\tau$ 下的 AP 为

$$
\mathrm{AP}_{c}^{\tau} = \frac{1}{101}\sum_{i=0}^{100} p_{\text{interp}}\!\left(\frac{i}{100}\right) \in [0,1].
$$

其连续等价形式为 $\mathrm{AP} = \int_0^1 p_{\text{interp}}(r)\,dr$。

### 2.2 mAP

对 $C$ 个类取平均，得到单一 IoU 阈值的均值 AP：

$$
\mathrm{mAP}^{\tau} = \frac{1}{C}\sum_{c=1}^{C} \mathrm{AP}_{c}^{\tau}.
$$

- **mAP@0.5**（`map50`）：$\mathrm{mAP}^{\tau=0.50}$ —— 单阈值 AP，IoU $\tau=0.5$。
- **mAP@0.5:0.95**（`map5095`）：COCO 主指标，在十个等距阈值 $\mathcal{T}=\{0.50,0.55,\ldots,0.95\}$ 上取平均：
  $$
  \mathrm{mAP}_{[.50:.95]} = \frac{1}{|\mathcal{T}|}\sum_{\tau\in\mathcal{T}} \mathrm{mAP}^{\tau} = \frac{1}{|\mathcal{T}|\,C}\sum_{\tau\in\mathcal{T}}\sum_{c=1}^{C}\mathrm{AP}_{c}^{\tau}.
  $$
- **AP@0.5**（`ap50`）、**AP@0.75**（`ap75`）：单类单阈值 AP；单类时 `ap50` 与 `map50` 同义。

### 2.3 精确率 / 召回率 / F1

在 F1 最优置信阈值 $t^*=\arg\max_t F_1(t)$ 处报告：

$$
P = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, \qquad
R = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \qquad
F_1 = \frac{2\,P\,R}{P+R} = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}.
$$

### 2.4 平均召回率（AR）

COCO 的 AR 为在每图检测预算 $K$ 约束下可达的最大召回，对 IoU 阈值 $\mathcal{T}$ 与类取平均：

$$
\mathrm{AR}_{K} = \frac{1}{|\mathcal{T}|\,C}\sum_{\tau\in\mathcal{T}}\sum_{c=1}^{C} \max_{\mathcal{D}\subseteq\mathcal{D}_c,\,|\mathcal{D}|\le K} R_{c}^{\tau}(\mathcal{D}),
$$

其中 $\mathcal{D}_c$ 为类 $c$ 的检测集合，$R_{c}^{\tau}(\mathcal{D})$ 为子集 $\mathcal{D}$ 在 $\tau$ 下的召回。报告的预算：`ar1`（$K{=}1$）、`ar10`（$K{=}10$）、`ar100`（$K{=}100$），`ar300`/`ar1000` 与 `ar100` 同义（pycocotools 上限为 100）。按目标尺寸分层的变体采用 COCO 面积区间：

$$
\text{小： } a < 32^2,\quad \text{中： } 32^2 \le a < 96^2,\quad \text{大： } a \ge 96^2 \text{ 像素}^2,
$$

对应 `ar_small`、`ar_medium`、`ar_large`。

## 3. 曲线与操作点指标

### 3.1 AUC-PR / AUC-ROC

- **AUC-PR**（`auc_pr`）：PR 曲线下梯形面积，$\int_0^1 P(r)\,dr$。与阈值无关的汇总指标，是类别不平衡的单类检测的首选主指标。
- **AUC-ROC**（`auc_roc`）：ROC 曲线 $\{(1-\mathrm{Specificity}(t),\,R(t))\}$ 下面积，即 $\int_0^1 \mathrm{TPR}(t)\,d\mathrm{FPR}(t)$，其中 $\mathrm{FPR}=\mathrm{FP}/(\mathrm{FP}+\mathrm{TN})$。

### 3.2 灵敏度 / 特异度 / PPV / NPV

由 $t^*$ 处的 $2{\times}2$ 混淆矩阵给出：

$$
\text{灵敏度（召回率, TPR）} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \qquad
\text{特异度（TNR）} = \frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}},
$$
$$
\text{PPV（精确率）} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, \qquad
\text{NPV} = \frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FN}}.
$$

对单类裂纹任务，$2{\times}2$ 矩阵的 TP/FN/FP/TN 直接对应灵敏度与 PPV。

## 4. 效率指标

每次测试（及每一折）都会在精度之外同步测量效率，一次扫描即可得到完整的精度–速度–算力对比表。设 `TestConfig.measure_efficiency=False` 可跳过。`EfficiencyReport`（pydantic）写入 `metrics.csv`、`model_analysis.json` 与 `TestLog` 记录。

### 4.1 吞吐与延迟

令 $T_b$ 为一个推理批的墙钟耗时（CUDA 同步），覆盖 $n_b$ 张图，包含前向 $\to$ 解码 $\to$ NMS 但不含文件 I/O 与可视化。设共 $B$ 批、$N=\sum_b n_b$ 张测试图：

$$
\mathrm{FPS}_{\text{mean}} = \frac{N}{\sum_b T_b}, \qquad
\ell_b = \frac{T_b}{n_b}\ \text{（单图延迟, ms）}, \qquad
\mathrm{FPS}_{p} = \frac{1000}{\ell_{p}}\ \text{, } p\in\{50, 95\},
$$

其中 $\ell_{p}$ 为 $\{\ell_b\}$ 经线性插值的第 $p$ 百分位。报告 `fps_mean`/`fps_p50`/`fps_p95` 与 `latency_mean_ms`/`latency_p50_ms`/`latency_p95_ms`。这些是在真实测试集、`TestConfig.batch_size` 下的**端到端**数值，而非合成的单图前向。

### 4.2 算力与显存预算

- **参数量**（`n_parameters`、`n_trainable_parameters`）：所有 / 可训练张量上 $\sum_{\theta\in\Theta}|\theta|$。
- **MACs**（`macs`）：单张 $(1,3,H,W)$ 输入上的乘加运算数，由 `fvcore.nn.FlopCountAnalysis` 测量后除以 2（fvcore 报告 FLOPs，按一次 MAC $=$ 两次 FLOPs 的约定 $\mathrm{MACs}=\mathrm{FLOPs}/2$）。
- **GFLOPs**（`gflops`）：$\mathrm{GFLOPs} = \mathrm{MACs}\times 2 / 10^{9}$。
- **峰值显存**（`peak_vram_bytes`）：真实推理循环内 $\max_t \mathrm{torch.cuda.max\_memory\_allocated}$（CPU 下取合成单图值）。

> **已知局限**：`fvcore` 仅能 trace 标准算子，无法完整 trace 自定义 YOLO 检测头与 torchvision 包装器前向，故上述家族的 `macs`/`gflops` 可能为 `0.0`。参数量、FPS、延迟、显存不受影响，为权威值。若需可移植的算力代理，使用参数量。

独立的 `scripts/analyze_model.py` 在合成输入上报告相同的结构指标而无需运行测试；测试流程的数值才是权威的，因为它们在真实 batch 下包含解码与 NMS。

## 5. 统计检验（模型比较）

`StatisticalTest` 在匹配的逐折或逐图指标上比较两个模型变体（如基线 vs SAC）。令 $d_i = x_i - y_i$ 为配对差，$i=1,\dots,n$，$\bar d = \frac1n\sum_i d_i$，$s_d = \sqrt{\frac1{n-1}\sum_i(d_i-\bar d)^2}$。

### 5.1 配对 t 检验（`paired_t`）

$$
t = \frac{\bar d}{s_d / \sqrt{n}}, \qquad \mathrm{df} = n-1,
$$

双尾 $p$ 值取自 Student $t$ 分布。假设差近似正态；速度快。

### 5.2 Wilcoxon 符号秩（`wilcoxon`）

非参数。将 $|d_i|$ 升序排名，赋带符号秩 $r_i = \mathrm{sgn}(d_i)\,\mathrm{rank}(|d_i|)$，则

$$
W = \sum_{i:d_i > 0} |r_i| \quad (\text{或按约定 } W = \min(W^+, W^-)).
$$

对非正态差与离群点稳健。

### 5.3 自助置信区间（`bootstrap_ci`）

对 $n$ 个差有放回重采样 $B$ 次（默认 $B{=}1000$），每次算 $\bar d^{(b)}$；95% 百分位 CI 为 $\{\bar d^{(b)}\}_{b=1}^{B}$ 的 $[2.5\%, 97.5\%]$ 经验分位：

$$
\mathrm{CI}_{0.95} = \big[\, Q_{0.025}(\{\bar d^{(b)}\}),\ Q_{0.975}(\{\bar d^{(b)}\})\,\big].
$$

最稳健；最慢。`ci_low`/`ci_high` 仅此检验填充。

### 5.4 检验选择

- **配对 t** —— 快，假设正态。
- **Wilcoxon** —— 非正态差。
- **自助 CI** —— 稳健的效应量区间；应与 $p$ 值一并报告，而非作为二元的"显著/不显著"结论。

## 6. 检测级记录

`DetectionMetric` 与 `PerImageDetection`（`TypedDict`）为喂给计算器的逐图单元。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `image_id` | `int` | 图像标识（数据集索引）。 |
| `class_id` | `int` | 预测 / 真值类别 id。 |
| `score` | `float` | 置信分数，$[0,1]$（真值为 $1.0$）。 |
| `bbox_xyxy` | `tuple[float,float,float,float]` | $(x_1,y_1,x_2,y_2)$，像素。 |

## 7. MetricReport

`MetricReport`（dataclass）为 `COCOMetricsCalculator.run()` 返回的精度载荷：

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

`MetricReport` 仅含检测精度；效率另报（见 §4）。

## 8. 评估曲线

`cracks_yolo.metrics.curves` / `.confusion` 生成并保存到 `curves/`：

- `pr.png` —— IoU $0.5$ 下的精确率–召回率曲线。
- `roc.png` —— ROC 曲线。
- `confusion.png` —— 归一化混淆矩阵热力图。

## 9. 舌面裂纹检测的解读

舌面裂纹为细长结构；预测与真值的微小空间偏移即会显著降低 IoU，使本任务对 IoU 敏感。

- **优先 mAP@0.5 而非 mAP@0.5:0.95** —— $[.50:.95]$ 的更严阈值会惩罚空间相近但非像素级精确的检测；对细裂纹，mAP@0.5 更能反映临床有用的检测。
- **AUC-PR 与阈值无关** —— 在所有置信阈值上汇总质量而无需固定操作点；类别不平衡下首选。
- **AR@100 / AR@1000** —— 更高预算评估模型能否召回被拆成多个小段的细裂纹碎片。
- **混淆矩阵** —— 单类时退化为 TP/FN/FP 率，直接对应灵敏度与 PPV。
