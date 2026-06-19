# 数据集

[English](dataset.md) | [中文](dataset.zh-CN.md)

`cracks_yolo.dataset` 提供格式无关的数据集加载、YOLOv5-PyTorch 与 COCO 格式之间的转换、一个 torchvision 兼容的 `DetectionDataset`，以及一套保守的变换流水线。默认数据集（`data/CrackDetection_Augmentation.v1.yolov5pytorch`）是舌面裂纹图像集合，单类别（`crack`），采用 YOLOv5 PyTorch 格式。

## 支持的格式

| 格式 | 布局 | 加载器 |
| --- | --- | --- |
| YOLOv5 PyTorch | `data.yaml` + `train/`, `valid/`, `test/` 目录含 `images/` + `labels/*.txt` | `cracks_yolo.dataset.yolo.YOLOSource` |
| COCO | `instances_{train,val}.json` + `train2017/`, `val2017/`（图片目录） | `cracks_yolo.dataset.coco.COCOSource` |

两种数据源均暴露统一的 `load_split(split: str) -> list[RawDetection]` API。`RawDetection` 是格式无关的记录类型：

```python
@dataclass
class RawDetection:
    image_id: int
    image_path: Path
    boxes_norm: np.ndarray   # (N, 4) xyxy 归一化到 [0, 1]
    labels: np.ndarray       # (N,) int64
```

## DetectionDataset (torch.utils.data.Dataset)

`DetectionDataset` 包装一组 `RawDetection` 列表 + 一个 `DetectionTransform`，每次返回 `DetectionSample(image, boxes, labels, image_id)`。通过类方法构造：

```python
from cracks_yolo.dataset import DetectionDataset

# 从 YOLOv5 格式的根目录创建。
ds = DetectionDataset.from_yolo(
    root="data/CrackDetection_Augmentation.v1.yolov5pytorch",
    split="train",
    input_size=640,
    train=True,
)

# 从 COCO 格式的根目录创建。
ds = DetectionDataset.from_coco(
    root="data/coco",
    split="train",
    input_size=640,
    train=True,
)

# 从预构建的记录创建（用于 5 折交叉验证划分）。
ds = DetectionDataset.from_records(records=subset, input_size=640, train=True)
```

## DataLoader

`build_dataloader` 返回一个 `torch.utils.data.DataLoader`，使用自定义的 `detection_collate` 处理每张图片中数量不等的边界框：

```python
from cracks_yolo.dataset import build_dataloader

loader = build_dataloader(ds, batch_size=32, num_workers=4, shuffle=True, pin_memory=True)
for images, targets in loader:
    # images: (B, 3, H, W) float 类型，范围 [0, 1]
    # targets: list[dict]，包含 {"boxes": (N,4) xyxy 绝对坐标, "labels": (N,), "image_id": Tensor}
    ...
```

## 变换

`DetectionTransform` 是一个小型可调用类（非 torchvision v2）。默认的训练流水线有意保持保守——仅使用水平翻转——以避免数据增强过强干扰基准对比。

- **Resize**：双线性插值至 `input_size x input_size`。
- **Normalize**：像素缩放到 `[0, 1]`（不进行均值/标准差减法——YOLO 惯例）。
- **训练增强**：随机水平翻转（p=0.5）。边界框相应翻转。
- **评估**：仅 resize + normalize。

`build_transforms(input_size, train, augment)` 是 `DetectionDataset.from_*` 使用的工厂函数。

## 格式转换

`cracks_yolo.dataset.convert` 提供：

- `yolo_to_coco(yolo_root, out_json)` — 为每个 split 生成一个 COCO `instances.json`。
- `coco_to_yolo(coco_json, image_dir, out_labels_dir)` — 为每张图片生成一个 `.txt` 文件。

使用 `scripts/convert_dataset.py` 通过命令行调用：

```bash
python -m scripts.convert_dataset \
    --input data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --from yolo --to coco \
    --output data/Crack_coco
```

## 目标张量约定

流水线中的 `targets_to_yolo` 辅助函数将 dataloader 的 `list[dict]` 格式转换为所有 YOLO 损失函数使用的 `(N, 6)` YOLO 目标张量：

| 列 | 含义 |
| --- | --- |
| 0 | 图片索引（0 起始，batch 内） |
| 1 | 类别 ID（0 起始） |
| 2 | x_center（归一化） |
| 3 | y_center（归一化） |
| 4 | 宽度（归一化） |
| 5 | 高度（归一化） |

torchvision 检测器包装器（`cracks_yolo/zoo/torchvision_detectors.py`）将此 `(N, 6)` 格式转换回 torchvision 的 `list[dict]` 格式，使用 xyxy 绝对像素坐标和 1 起始标签（背景 = 0）。
