# 评估指标

[English](metrics.md) | [中文](metrics.zh-CN.md)

本文档对 `test_metrics.csv` 中所报告的每一项指标进行严格的形式化定义。
全部定义遵循 MS COCO 评测协议与经典二分类理论，并适配于单类目标检测任务。

## 符号约定

设评测集 $\mathcal{D}=\{(I_i,\mathcal{G}_i)\}_{i=1}^{N}$，其中 $I_i$ 为第 $i$ 张
图像，$\mathcal{G}_i$ 为其真值框集合。设 $\mathcal{P}_i=\{(b_j,s_j)\}$ 为模型在
$I_i$ 上输出的预测框及其置信度。若预测框 $b_j$ 与某一未匹配真值框的交并比
（IoU）不低于阈值 $\tau$，则计为**真正例**（TP）；否则计为**假正例**（FP）。
未匹配的真值框计为**假负例**（FN）。

$$
\mathrm{IoU}(b,g)=\frac{|b\cap g|}{|b\cup g|}
$$

## mAP@0.5（map50）

IoU 阈值 $\tau=0.5$ 下的平均精度。通过变化置信度阈值扫描精确率—召回率曲线，
其下面积即为 AP。

$$
\mathrm{AP}_{0.5}=\int_0^1 p(r)\,dr,\qquad
p(r)=\frac{\mathrm{TP}(r)}{\mathrm{TP}(r)+\mathrm{FP}(r)},\quad
r=\frac{\mathrm{TP}(r)}{\mathrm{TP}(r)+\mathrm{FN}}
$$

单类检测下 $\mathrm{mAP}_{0.5}=\mathrm{AP}_{0.5}$。该指标衡量检测结果的排序
质量：若模型将真实框置于置信度排序的前列，则可获得较高的 AP。

## mAP@[0.5:0.95]（map5095）

COCO 主指标。在十个 IoU 阈值 $\tau\in\{0.50,0.55,\dots,0.95\}$ 上对 AP 取平均：

$$
\mathrm{mAP}_{[0.5:0.95]}=\frac{1}{10}\sum_{k=0}^{9}\mathrm{AP}_{0.50+0.05k}
$$

该指标对定位偏差施加惩罚：若某框与真值的 IoU 为 0.5 但不足 0.75，则计入
$\mathrm{AP}_{0.5}$ 却不计入 $\mathrm{AP}_{0.75}$。故该指标同时度量分类与回归精度。

## AP@0.75（ap75）

严格 IoU 阈值 $\tau=0.75$ 下的平均精度。它单独反映定位精度，对框贴合质量的敏感
程度远高于 $\mathrm{AP}_{0.5}$。

## 精确率（precision）

在使 F1 最大的操作点处，

$$
\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}
$$

即预测框中正确的比例，模型的误报率为 $1-\mathrm{Precision}$。

## 召回率 / 灵敏度（recall, sensitivity）

$$
\mathrm{Recall}=\mathrm{Sensitivity}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}=\frac{\mathrm{TP}}{|\mathcal{G}|}
$$

即被检出的真值裂纹比例。两列数值恒等，命名差异仅源于机器学习（`recall`）与
医学统计（`sensitivity`）的约定。

## F1（f1）

精确率与召回率在 F1 最优阈值处的调和平均：

$$
F_1=\frac{2\,p\,r}{p+r}
$$

该指标对精确率—召回率的失衡不敏感，适用于需要单一操作点的场景。

## AR@1 / AR@10 / AR@100（ar1, ar10, ar100）

每图限取 $K$ 个检测下的平均召回率：

$$
\mathrm{AR}_K=\frac{1}{N}\sum_{i=1}^{N}\frac{\mathrm{TP}_i^{(K)}}{|\mathcal{G}_i|}
$$

其中 $\mathrm{TP}_i^{(K)}$ 为图像 $I_i$ 上置信度最高的前 $K$ 个预测中的真正例
数。$\mathrm{AR}_1$ 度量单次最佳猜测的准确率，$\mathrm{AR}_{100}$ 近似饱和召回。
所有 AR 值均在 IoU $\in[0.5:0.95]$ 上取平均。

## AUC-PR（auc_pr）

精确率—召回率曲线下面积，基于**全部**检测（不限定单一操作点）计算。对检测任务，

$$
\mathrm{AUC}_{PR}=\int_0^1 p(r)\,dr
$$

曲线由按置信度排序的全部检测构建。与采用 101 点插值的 AP 不同，本文 AUC-PR 采用
经验曲线的梯形积分。在类别严重不平衡时，该指标为推荐的综合统计量。

## AUC-ROC（auc_roc）

受试者工作特征曲线下面积，以真正例率对假正例率作图：

$$
\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\qquad
\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}},\qquad
\mathrm{AUC}_{ROC}=\int_0^1 \mathrm{TPR}(\mathrm{FPR})\,d(\mathrm{FPR})
$$

此处检测在 IoU $0.5$ 下与真值框匹配并视作二分类结果，未匹配预测作为负例。
$\mathrm{AUC}_{ROC}=0.5$ 为随机水平。

## 特异度（specificity）

图像级真负例率。若图像真值含 $\geq 1$ 个裂纹则为阳性，否则为阴性；若模型在置信度
阈值 $\sigma=0.25$ 以上输出 $\geq 1$ 个检测，则预测为阳性。

$$
\mathrm{Specificity}=\frac{\mathrm{TN}_{\text{img}}}{\mathrm{TN}_{\text{img}}+\mathrm{FP}_{\text{img}}}
$$

其中 $\mathrm{FP}_{\text{img}}$ 为被误报的无裂纹图像数，$\mathrm{TN}_{\text{img}}$
为被正确放过的无裂纹图像数。此即临床意义上的"特异度"——正常舌面不被报警的概率。

## PPV / NPV（ppv, npv）

图像级阳性/阴性预测值：

$$
\mathrm{PPV}=\frac{\mathrm{TP}_{\text{img}}}{\mathrm{TP}_{\text{img}}+\mathrm{FP}_{\text{img}}},\qquad
\mathrm{NPV}=\frac{\mathrm{TN}_{\text{img}}}{\mathrm{TN}_{\text{img}}+\mathrm{FN}_{\text{img}}}
$$

PPV 等价于图像级精确率；NPV 为"未报警即确无裂纹"的概率。二者均依赖疾病患病率，
故与特异度并列报告以供临床解读。
