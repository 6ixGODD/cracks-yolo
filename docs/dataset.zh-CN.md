# 数据集

`cracks_yolo.dataset` 模块提供 YOLOv5-PyTorch 与 COCO 检测数据集的格式无关摄入。所有数据源统一产出
`RawDetection` 记录——一种共享的中间表示，由单一的 `DetectionDataset` 包装器消费，后者为训练与推理
产出 `(image_tensor, targets)` 批次。

## YOLO 磁盘格式

模块预期采用 Roboflow 导出布局：

```
<root>/
  data.yaml            # nc、names、train/val/test 路径提示
  train/{images,labels}/
  valid/{images,labels}/
  test/{images,labels}/
```

`data.yaml` 声明 `nc`（整数）、`names`（类别标签字符串）以及 `train`/`val`/`test` 路径提示。路径提示
**不被采用**；模块按惯例解析各划分（`<root>/<split>/images/`），从而对 Roboflow 导出中的绝对路径
污染具有鲁棒性。

每个标签文件（`<split>/labels/<stem>.txt`）每行编码一个标注：

```
<cls> <xc> <yc> <w> <h>
```

所有字段均为浮点数。`cls` 从零开始索引；`xc`、`yc`、`w`、`h` 为中心归一化值，范围 `[0, 1]`。读取器
内部将坐标转换为 `xyxy` 归一化框。接受的图像格式：`.jpg`、`.jpeg`、`.png`。

## YOLOSource

`YOLOSource(root)` 解析 `data.yaml`。两个方法：

- `list_splits()` 返回 `("train", "valid", "test")` 中存在 `<split>/images/` 目录的子集。`"val"` 被
  规范化为 `"valid"`。
- `load_split(split)` 读取标签目录和 JPEG 头部（仅 PIL 解码；像素由 `DetectionDataset` 后续加载）。
  返回 `list[RawDetection]`，其中 `boxes_norm` 为 `xyxy` 格式且归一化至 `[0, 1]`，`image_id` 设为
  文件枚举索引。

`YOLODataset` 是向后兼容的别名。

## COCOSource

`COCOSource(root, image_dir=None)` 读取标准 COCO `instances_*.json` 布局。`image_dir` 参数覆写
默认推断（`<root>/<split>/` 或 `<root>/images/`），以适配具有非标准图像嵌套的 Roboflow COCO
导出。`list_splits` 扫描 `instances_{split}.json`、`{split}.json` 或 `instances_{split}2017.json`。
`load_split` 返回 `RawDetection` 记录，其中 COCO 的 `xywh` 绝对像素框被转换为归一化 `xyxy`。
`COCODataset` 是别名。

## RawDetection

冻结数据类（`cracks_yolo.dataset.types`）：

| 字段         | 类型                                       | 描述                             |
|-------------|-------------------------------------------|---------------------------------|
| `image_path` | `Path`                                    | 图像文件路径                        |
| `image_id`   | `int`                                     | 划分内唯一标识符                      |
| `width`      | `int`                                     | 原始像素宽度（来自 JPEG 头部）           |
| `height`     | `int`                                     | 原始像素高度                        |
| `boxes_norm` | `list[tuple[float, float, float, float]]` | `xyxy` 框，归一化至 `[0, 1]`         |
| `labels`     | `list[int]`                               | 类别索引，从零开始                     |

此举将标注解析与像素加载解耦：数据源仅读取图像头部，将完整解码推迟至 `DetectionDataset.__getitem__`。

## DetectionDataset

`DetectionDataset(records, transform)` 是 `torch.utils.data.Dataset[DetectionSample]` 的子类。
`__getitem__` 打开 JPEG、转换为 RGB、应用变换，并返回
`DetectionSample(image: Tensor (3,H,W), boxes: Tensor (N,4) xyxy abs, labels: Tensor (N,),
image_id: int)`。

三个类方法：

| 类方法          | 签名                                                        | 用途                     |
|---------------|-------------------------------------------------------------|-------------------------|
| `from_yolo`    | `(root, split, input_size, train, augment)`                | 单个 YOLO 划分             |
| `from_coco`    | `(root, split, input_size, train, augment, image_dir)`     | 单个 COCO 划分             |
| `from_records` | `(records, input_size, train, augment)`                    | 任意记录（如交叉验证折）         |

三者内部均通过 `build_transforms` 构建 `DetectionTransform`。

## build_transforms

`build_transforms(input_size, train=False, augment=True) -> DetectionTransform` 返回一个可调用对象，
该对象 (a) 双线性缩放至 `(input_size, input_size)`；(b) 转换为 `Tensor` 并归一化至 `[0, 1]`；
(c) 当 `train` 且 `augment` 时，施加随机水平翻转（p = 0.5）。不做均值/标准差减法——YOLO 惯例。

`DetectionTransform.__call__` 签名：
`(image: PIL.Image, boxes_norm, labels) -> (tensor (3,H,W), boxes_xyxy_abs (N,4), labels (N,))`。

## build_dataloader

`build_dataloader(dataset, batch_size, num_workers=0, shuffle=False, pin_memory=False) -> DataLoader`
以 `detection_collate` 作为 `collate_fn` 将数据集包装为 `torch.utils.data.DataLoader`。该整理函数将图像
堆叠为 `(B, 3, H, W)`，并将目标返回为 `list[dict]`，格式为
`{"boxes": (N,4), "labels": (N,), "image_id": int}`，避免了跨可变 `N` 框的有损填充。

## COCO 转换工具

`cracks_yolo.dataset.convert` 中两个独立函数：

| 函数           | 签名                                              | 描述                                 |
|---------------|---------------------------------------------------|-------------------------------------|
| `yolo_to_coco` | `(yolo_root, out_dir) -> dict[str, Path]`         | 按 YOLO 划分写出 `instances_<split>.json` |
| `coco_to_yolo` | `(coco_json, image_dir, out_labels_dir) -> int`   | 按图像写出 YOLO `.txt`                   |

两个方向均**不复制**图像；仅重新生成标注。`coco_to_yolo` 为负样本写出空 `.txt` 文件（YOLO 惯例）。
