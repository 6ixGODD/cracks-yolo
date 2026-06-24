# 算子

## 卷积与CSP块

**Conv.** `Conv2d -> BatchNorm2d -> SiLU`。出自ultralytics，为下游模块所共用。签名为`Conv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)`。

**Bottleneck.** 两阶段残差结构：1×1 `Conv`以比例$e$（默认0.5）缩减通道，3×3 `Conv`恢复至$c_2$。当$c_1=c_2$时添加捷径连接。SAC变体中$e$强制为1.0。

**C3.** YOLOv5/v7所用的CSP块。两个1×1 `Conv`投影并行处理输入：`cv1`分支馈入含$n$个Bottleneck的`nn.Sequential`，`cv2`旁路直通。两半经`cv3`（1×1）拼接融合。

**C2f.** YOLOv8/v9/v10的CSP变体。`cv1`（1×1）投影至$2c$通道，经`chunk(2, dim=1)`拆分。前一半直通；后一半依次通过`nn.ModuleList`中$n$个Bottleneck，每层接收前层输出，逐层累积特征。全部$n+1$个张量经`cv2`（1×1）拼接融合。

---

## SAC：可切换空洞卷积

SAC将选定主干Bottleneck的内层3×3卷积替换为一个可学习的逐像素切换器，该切换器融合膨胀率分别为$d$与$3d$的两路空洞卷积。

**ConvAWS2d.** 权重标准化的`Conv2d`子类。每次前向传播前，卷积核按输出通道标准化：空间维度均值减去，除以标准差（+1e-5），由可学习缓冲区`gamma`（初值1）与`beta`（初值0）重缩放。`load_state_dict`时若`gamma.mean() <= 0`，则从传入权重重新计算两个缓冲区，从而支持从非AWS检查点加载。

**SAConv2d.** 流水线：(1) *预上下文*——全局平均池化，1×1卷积，广播残差。(2) *切换器*——反射填充（2px），5×5平均池化，1×1卷积含sigmoid，产生单通道软门控。(3) *双路空洞*——在同一权重标准化核$w$（膨胀$d$）与$w+\text{weight\_diff}$（膨胀$3d$）上分别执行`_conv_forward`，其中`weight_diff`是学得的零初始化`nn.Parameter`。切换器融合两路：$\text{switch}\cdot\text{out}_s+(1-\text{switch})\cdot\text{out}_l$。(4) *后上下文*——全局平均池化，1×1卷积，残差。BatchNorm+SiLU收尾。切换器偏置初始化为1，默认倾向小膨胀分支。

**BottleneckSAC.** Bottleneck的即插即用替换。首层1×1 `Conv`不变；次层3×3 `Conv`替换为$e=1.0$的`SAConv2d`。

**C3SAC.** 内部`nn.Sequential`中$n$个`BottleneckSAC(e=1.0)`的`C3`。适用于YOLOv5/v7主干。

**C2fSAC.** `nn.ModuleList`链中`BottleneckSAC(e=1.0)`的`C2f`。适用于YOLOv8/v9/v10主干。

---

## TR：Transformer

**TransformerLayer.** 无层归一化的QKV自注意力：三个线性投影$(q,k,v)$，`MultiheadAttention(batch_first=True)`，残差连接，两层MLP（`Linear -> Linear`），第二次残差。

**TransformerBlock.** 1×1 `Conv2d`投影输入通道；可学习位置嵌入$(1,C,1,1)$在展平为$(B,HW,C)$后广播相加。`nn.Sequential`含`num_layers`个`TransformerLayer`；输出重塑为$(B,C,H,W)$。

**C3TR.** 内部Bottleneck序列替换为`TransformerBlock(c_, c_, num_heads=4, num_layers=n)`的`C3`。CSP拆分-变换-合并结构得以保留（transformer仅处理`cv1`分支）。

---

## 检测头

检测头消费多尺度特征金字塔`[P3, P4, P5]`（步长8, 16, 32）。均采用解耦分支：框回归与类别置信度各用独立卷积栈。

**Detect（YOLOv5/v8）.** 每层`cv2`预测$4\cdot r_{\max}$个框分布参数；`cv3`预测$n_c$个类别logits。训练时返回各尺度原始张量；推理时施加DFL，将逐边分布转换为$(l,t,r,b)$距离，并通过步长锚点解码为绝对坐标。

**IDetect（YOLOv7）.** 扩展`Detect`，每尺度附加`ImplicitA`（加性偏置）与`ImplicitM`（乘性缩放），二者均可融合入前置卷积。

**v10Detect（YOLOv10）.** 双头结构：`cv2/cv3`用于一对多分配（训练），`cv4/cv5`用于一对一分配（推理）。一对一检测头消除NMS。

---

## SAC/TR注入

`cracks_yolo.zoo.ultralytics.sac_injection`中的`apply_sac_tr(model, sac_indices, tr_indices)`按索引就地修改`model.model`——即主干`nn.Sequential`。

对于`sac_indices`中每个$i$：若`model.model[i]`为`C3`则替换为`C3SAC(c1, c2, n, shortcut=True)`；若为`C2f`则替换为`C2fSAC(c1, c2, n)`。$c_1$与$c_2$从原始块的首尾卷积层读取，$n$取自`len(old.m)`。对于`tr_indices`中每个$i$：`C3`块变为`C3TR(c1, c2, n, shortcut=True)`。

`_copy_shared_weights(new, old)`按名称与形状匹配`state_dict`键，复制预训练的`cv1`/`cv2`/`cv3`卷积权重。SAC专用参数（切换器、上下文卷积、weight_diff、AWS缓冲区）与TR专用参数（QKV投影、位置嵌入）保持随机初始化。`_copy_layer_meta(new, old)`传播ultralytics路由属性（`f`, `i`, `type`, `np`），使替换后的块融入模型的解析树前向图。

在`__init__`中调用一次，随后移至设备：

```python
apply_sac_tr(model, sac_indices=(4, 6, 8), tr_indices=(9,))
```

索引取决于具体的YAML架构定义。
