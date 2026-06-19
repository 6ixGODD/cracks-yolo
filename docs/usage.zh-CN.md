# 用法

[English](usage.md) | [中文](usage.zh-CN.md)

使用 `cracks_yolo` 训练和评估舌面裂纹检测模型的端到端教程。

## 安装

```bash
git clone <repo>
cd cracks-yolo
uv sync          # 或：pip install -e .
```

需要 Python 3.11 或 3.12，PyTorch >= 2.2。如需 CUDA 11.8 支持：

```bash
uv pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

## 快速开始：前向传播 + 损失计算 + 反向传播

```python
import torch
from cracks_yolo.zoo import ZOO

# 实例化任意已注册的模型。
model = ZOO["yolov5s_sactr"](num_classes=1)
model.train()

# 前向传播（训练模式返回原始检测头输出）。
x = torch.randn(2, 3, 640, 640)
preds = model(x)

# 构建 YOLO 格式的目标：(N, 6) = (img_idx, cls, x, y, w, h) 归一化。
targets = torch.tensor(
    [
        [0, 0, 0.50, 0.50, 0.20, 0.20],
        [0, 0, 0.30, 0.70, 0.10, 0.10],
        [1, 0, 0.40, 0.40, 0.15, 0.25],
    ],
    dtype=torch.float32,
)

# 计算损失。
# v7 需要图像批次（OTA 分配使用图像尺寸）。
if model.__class__.__name__.startswith("YOLOv7"):
    loss, parts = model.compute_loss(preds, targets, imgs=x)
else:
    loss, parts = model.compute_loss(preds, targets)

loss.backward()
assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
```

## 评估模式前向传播 + 解码

```python
model.eval()
with torch.no_grad():
    out = model(x)               # eval 前向传播在内部解码
    decoded = model.decode(out)  # 返回 (B, N, nc+5) 或 (B, 4+nc, N)

print(decoded.shape)
# v5/v7：torch.Size([2, 25200, 6])  -- (B, anchors, nc+5)
# v8/v10：torch.Size([2, 5, 8400])  -- (B, 4+nc, grid_cells)
```

## 加载 COCO 预训练权重

```python
from cracks_yolo.zoo import YOLOv5s

# 基线（有 pretrained_spec）-- 下载并使用 strict=False 加载。
model = YOLOv5s.from_pretrained(num_classes=1)

# SAC/TR 变体返回随机初始化（pretrained_spec 为 None）。
from cracks_yolo.zoo import YOLOv5sSACTR
model = YOLOv5sSACTR(num_classes=1)
```

检查加载报告：

```python
from cracks_yolo.weights.loader import load_pretrained
from cracks_yolo.zoo import YOLOv5s

model = YOLOv5s(num_classes=1)
report = load_pretrained(
    model=model,
    spec=YOLOv5s.pretrained_spec,
    weights_dir=None,  # 默认为 ./weights
    strict=False,
)
print(f"匹配：{len(report.matched)}")
print(f"缺失：{report.missing[:5]} ...（共 {len(report.missing)} 个）")
print(f"意外：{len(report.unexpected)}")
```

## 构建优化器

```python
model = ZOO["yolov8s_sac"](num_classes=1)
optimizer = model.build_optimizer()
# torch.optim.AdamW(model.parameters(), lr=1e-3)
```

## 列出所有可用模型

```python
from cracks_yolo.zoo import ZOO

for key, cls in ZOO.items():
    print(f"{key:18s} -> {cls.__name__}")
```

输出（共 26 个条目）：

```
yolov5s            -> YOLOv5s_CIoU_BCEObj_BCECls_AdamW_SILU
yolov5s_sac        -> YOLOv5sSAC_CIoU_BCEObj_BCECls_AdamW_SILU
yolov5s_tr         -> YOLOv5sTR_CIoU_BCEObj_BCECls_AdamW_SILU
yolov5s_sactr      -> YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU
yolov7w            -> YOLOv7w_CIoU_BCEObj_BCECls_AdamW_SILU
yolov7w_sac        -> YOLOv7wSAC_CIoU_BCEObj_BCECls_AdamW_SILU
yolov8s            -> YOLOv8s_CIoU_DFL_AdamW_SILU
yolov8s_sac        -> YOLOv8sSAC_CIoU_DFL_AdamW_SILU
yolov10s           -> YOLOv10s_CIoU_DFL_AdamW_SILU
yolov10s_sac       -> YOLOv10sSAC_CIoU_DFL_AdamW_SILU
...
```

## 运行训练

训练脚本需要 YOLOv5 PyTorch 格式的数据集：

```bash
python -m scripts.train \
    --model yolov5s_sactr \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --epochs 100 --batch-size 32 \
    --output-dir output/yolov5s_sactr \
    --pretrained
```

`output/yolov5s_sactr/` 下生成的训练产物：

- `run.log.jsonl` -- 结构化 JSONL 日志
- `metrics.csv` -- 每周期指标
- `loss_curve.png`, `metric_curve.png` -- 训练曲线
- `config.yaml` -- 冻结的训练配置
- `best.pt` -- 最佳检查点
- `per_image/*.json` -- 每张图像的预测结果
- `predictions/*.jpg` -- 预测可视化
- `curves/{pr,roc,confusion}.png` -- 评估曲线

## 对训练好的模型运行测试

```bash
python -m scripts.test \
    --model yolov5s_sactr \
    --weights output/yolov5s_sactr/best.pt \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --output-dir output/yolov5s_sactr/test
```

## 5 折交叉验证

将 train+valid+test 合并为一个池，留出折 = test，剩余数据按 90/10 划分为 train/val：

```bash
python -m scripts.train \
    --model yolov5s_sactr \
    --cross-val --n-folds 5 --val-fraction 0.1 \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --epochs 100 --batch-size 32 \
    --output-dir output/yolov5s_sactr_cv \
    --pretrained
```

CV 输出包括 `cv_summary.csv`、`cv_report.json` 以及各个 `fold_*/` 目录。

## 带统计检验的多模型比较

```bash
python -m scripts.compare_models \
    --models yolov5s,yolov5s_sactr,yolov8s,yolov9c,retinanet_r50,faster_rcnn_r50 \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --n-folds 5 --epochs 100 \
    --output-dir output/comparison
```

生成 `comparison*.csv`、`paired_t_test.csv`，以及 Wilcoxon 和 bootstrap CI 结果。

## 带子进程隔离的批量调度

```bash
python -m scripts.schedule_experiments \
    --config experiments/all_models_direct.yaml \
    --output-dir output/all_models_direct
```

调度器在隔离的子进程中运行每个实验，将错误捕获到 `errors.jsonl`，并支持 `--retry` 模式重新运行仅失败的实验。设置 `max_parallel > 1` 和每个实验的 `env: {CUDA_VISIBLE_DEVICES: "N"}` 以用于多 GPU 服务器。

## 结构化日志

```python
from pathlib import Path
from loguru import logger
from cracks_yolo.logging.configure import configure_logger
from cracks_yolo.logging.schema import TrainStepLog

configure_logger(output_dir=Path("output/run1"))

record: TrainStepLog = {
    "record_type": "train_step",
    "step": 0, "epoch": 0,
    "total_loss": 1.23, "box_loss": 0.4, "cls_loss": 0.5,
    "obj_loss": 0.33, "dfl_loss": None,
    "lr": 1e-3, "timestamp": "2026-06-18T00:00:00",
}
logger.bind(**record).info("step done")
# 写入一行 JSON 到 output/run1/run.log.jsonl
```

## 其他脚本

| 脚本 | 用途 |
| --- | --- |
| `convert_dataset` | 在 COCO 和 YOLOv5 格式之间转换 |
| `heatmap` | 训练模型的 Grad-CAM 可视化 |
| `analyze_dataset` | 数据集多样性指标和分布图 |
| `analyze_model` | 模型 params/MACs/latency/VRAM 分析 |
