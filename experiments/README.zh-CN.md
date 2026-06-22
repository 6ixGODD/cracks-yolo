# 实验 — 全模型扫描

[English](README.md) | [中文](README.zh-CN.md)

舌面裂纹检测全检测器家族统一扫描。所有 YOLO 模型（v3/v5/v8/v9/v10）通过 **Ultralytics** 加载，SAC/TR 由运行时模块替换注入（`cracks_yolo.zoo.ultralytics_sac.apply_sac_tr`）；torchvision 检测器（RetinaNet/Faster-RCNN/Mask-RCNN/FCOS/SSD）用 cracks_yolo 封装。无 `_official` 后缀，无逐架构重实现。

## 文件

- `all_models_compose.yaml` — 全部 24 个主力 + 27 个延后模型（训练+测试对）。自包含。
- `compose_4/group_{1..4}.yaml` — **主力**集的 4 路拆分，按 GPU 时成本均衡（每组约 44 h）。每台服务器跑一个文件。
- `compose_4/group_deferred.yaml` — 第一期注释掉的 27 个模型（v5 n/m/l/x 的 SAC/TR 变体、v8 m/l/x SAC、v8x、v9 t/s/m/e、v10 n/m/b/l/x、v3、DETR）。主力跑完有时间再跑。
- `all_models_cv5.yaml`、`all_models_direct.yaml` — 旧的单配置扫描（参考）。

## 主力模型集（24 个）

| 家族 | 模型 | Epoch |
| --- | --- | --- |
| YOLOv5 | s{baseline,sac,tr,sactr}、n/m/l baseline（不跑 x） | 1200（早停 patience 30） |
| YOLOv7 | yolov7w（cracks_yolo 重实现，非 ultralytics） | 300 |
| YOLOv8 | n/s{baseline,sac}、m/l baseline | 300 |
| YOLOv9 | c{baseline,sac} | 300 |
| YOLOv10 | s{baseline,sac} | 300 |
| torchvision | retinanet/faster/mask/fcos/ssd300/ssdlite | 150 |

所有 YOLO：lr 1e-3、SGD（v5）/ AdamW（其他）、余弦退火、EMA、mosaic+HSV 增强、关闭 AMP（fp32——AMP 会让 v5 CIoU NaN）、clip_grad_norm 10、COCO 预训练（strict=False；SAC/TR 随机初始化）。torchvision：lr 1e-4、AdamW、开 AMP。种子 = 42。

## 在 4–6 台服务器上并行跑

每台服务器独立跑一个 `compose_4/group_N.yaml`。各组成本均衡，同时完工。

### 服务器准备（autodl，每台）

```bash
# 1. 克隆（国内用 ghfast.top 镜像加速）
git clone https://ghfast.top/https://github.com/6ixGODD/cracks-yolo.git
cd cracks-yolo

# 2. 数据集拷到快速本地盘提升 IO
cp -r /root/autodl-fs/CrackDetection_Augmentation.v1.yolov5pytorch /root/autodl-tmp/
ln -s /root/autodl-tmp/CrackDetection_Augmentation.v1.yolov5pytorch data/CrackDetection_Augmentation.v1.yolov5pytorch

# 3. 装依赖
pip install ultralytics thop torchsummary pycocotools opencv-python

# 4. 跑一个组（第 k 台服务器跑 group_k）
python -m scripts.schedule_experiments \
    --config experiments/compose_4/group_1.yaml \
    --output-dir output/group_1
```

### 跑完之后

```bash
cd output/group_1 && zip -r /root/autodl-fs/group_1.zip . && cd ../..
```

## 时间预估

**依据**：单张 A100 40GB（fp32，YAML 里的 batch size）每模型 GPU 时粗估：
- v5 × 1200 epoch（早停）：每个约 13 h × 7 = 91 h
- v8/v9/v10/v7 × 300 epoch：每个约 5 h × 11 = 55 h
- torchvision × 150 epoch：每个约 5 h × 6 = 30 h
- **主力合计：约 176 GPU 时**

墙钟 = 176 / 服务器数：

| 服务器数 | GPU | 墙钟 | 成本（约 7¥/h） |
| --- | --- | --- | --- |
| 4 | A100 40GB | ~44 h（约 2 天） | ~4900¥ |
| 6 | A100 40GB | ~29 h（约 1.2 天） | ~7400¥ |
| 8 | A100 40GB | ~22 h（<1 天） | ~9800¥ |

**要 1 天跑完：租 8× A100 40GB**（重新拆成 8 组——改 `output/_gen_compose.py` 的组数为 8，或每台服务器顺序跑 2 组）。4 路拆分对应 4 服务器 2 天预算。成本随总 GPU 时（固定约 176 h）线性，服务器越多越快但总成本不变。

预估假设 v5 早停在 patience 附近触发（1200 是上限，实际可能 600–900 停）。

## 注意

- 输出目录：`output/{model}/`（调度器传 `--output-dir`）。单独跑 `python -m scripts.train` 自动加时间戳到 `output/{ISO时间戳}/{model}/`。
- v7（`yolov7w`）是唯一不在 Ultralytics 上的家族——用 cracks_yolo 重实现。
- GFLOPs 用 thop；FPS 在真实测试集上端到端测；模型结构用 `model.info()` / torchsummary。
