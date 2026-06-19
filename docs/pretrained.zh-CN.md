# 预训练权重（`cracks_yolo.weights`）

[English](pretrained.md) | [中文](pretrained.zh-CN.md)

## 设计

`load_pretrained(model, spec, weights_dir, strict=False) -> LoadReport` 获取一个 `.pt` 检查点，重映射 `state_dict` 的键，并使用 `strict=False` 加载。返回一个 `LoadReport`，描述哪些匹配成功、哪些缺失、以及哪些是意外的。

每个 zoo 类上的 `DetectorModel.from_pretrained` 类方法委托给 `load_pretrained`。如果下载失败，则回退到随机初始化（并发出警告）。

Torchvision 封装（RetinaNet、Faster/Mask R-CNN、FCOS、SSD300、SSDlite320）**不使用**此系统——它们通过 torchvision 内置的 `weights="DEFAULT"` API 加载预训练权重，由 torchvision 内部处理下载和缓存。

## `PretrainedSpec`

```python
@dataclass(frozen=True)
class PretrainedSpec:
    key: str                       # 缓存文件名：weights/{key}.pt
    url: str                       # 官方 .pt 发行版的 HTTPS URL
    state_dict_key_map: dict[str, str]  # 前缀重映射表
```

每个 zoo 类声明 `pretrained_spec: PretrainedSpec | None` 作为类属性。基线变体（无 SAC/TR）声明一个 spec；SAC/TR 变体声明 `None`（SAC/TR 层不存在 COCO 权重）。

## `LoadReport`

```python
@dataclass
class LoadReport:
    model: nn.Module
    matched: list[str]      # 成功加载的键
    missing: list[str]      # 模型中存在但检查点中缺失的键（例如 SAC/TR 层）
    unexpected: list[str]   # 检查点中存在但模型中缺失的键
    key: str
    url: str
    cached: bool            # 如果文件已存在磁盘上则为 True
```

## 缓存

- 检查点路径：`{weights_dir}/{key}.pt`（默认 `weights/{key}.pt`）。
- 如果文件已存在，则直接加载——无需网络访问。
- 如果文件不存在，`_download(url, dest)` 通过 `requests.get(stream=True)` 以 60 秒超时流式下载，以 64KiB 块写入。进度通过 loguru 记录。

## 检查点格式处理

`_load_state_dict(path)` 处理三种检查点布局：

1. **原始 state_dict**（`{key: Tensor}`）—— 按原样返回。
2. **`{"state_dict": {...}, ...}`** —— 提取 `"state_dict"` 的值（常见于 PyTorch Lightning / mmdet 检查点）。
3. **`{"model": nn.Module, ...}`** —— Ultralytics 风格。调用 `model.state_dict()` 来提取 state_dict。这使用了一种虚拟模块反序列化技术：YOLOv5/v8 检查点包含 `nn.Module` 对象，其 `__setattr__` 可能引用加载时尚不存在的模型内部结构。在反序列化之前，一组占位模块被注入 `sys.modules`，使检查点能够无错误地完成反序列化。

如果检查点不是一个 dict（或者是一个 dict 但不包含以上任何键），则抛出 `TypeError`。

## 键重映射

`_remap_keys(state_dict, key_map)` 根据 spec 重写 state_dict 的键前缀。用于将官方发行版的键与 `cracks_yolo` 的类布局对齐。空的 `key_map` 表示无需重映射。

示例：YOLOv5s 官方发行版使用 `model.0.weight`（Ultralytics 容器的索引）；`cracks_yolo` 将骨干网络存储为 `backbone.0.weight`。`state_dict_key_map={"model.": "backbone."}` 重写了前缀。

## SAC/TR 层：部分加载语义

SAC（`SAConv2d`、`BottleneckSAC`、`C3SAC`、`C2fSAC`）和 TR（`TransformerBlock`、`C3TR`）层 **不在** COCO 预训练权重中——它们是本项目添加的增强模块。使用 `strict=False`（默认值）：

- SAC/TR 的键出现在 `LoadReport.missing` 中（模型期望它们，检查点不包含它们 -> 随机初始化）。
- 骨干网络/颈部/检测头的其余部分从 COCO 正常加载。

使用 `strict=True` 时，缺失的键会抛出 `RuntimeError`。用于在测试中捕获与上游命名的不经意偏离。

## 每个 zoo 类上的 `from_pretrained`

```python
@classmethod
def from_pretrained(cls, num_classes, weights_dir=None, strict=False) -> _YOLOv8sBase:
    spec = cls.pretrained_spec
    if spec is None:
        return cls(num_classes=num_classes)  # SAC/TR 变体——无 COCO 权重
    report = load_pretrained(
        model=cls(num_classes=num_classes),
        spec=spec,
        weights_dir=weights_dir,
        strict=strict,
    )
    return report.model
```

## URL 注册表

`cracks_yolo/weights/registry.py`：

```python
PRETRAINED_URLS: dict[str, str] = {
    "yolov5s":  "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
    "yolov7w":  "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt",
    "yolov8s":  "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    "yolov10s": "https://huggingface.co/jameslahm/yolov10s/resolve/main/yolov10s.pt",
}
```

（实际 URL 可能会调整以匹配最新的稳定发行版。）

## 添加新的预训练来源

1. 将 URL 添加到 `cracks_yolo/weights/registry.py` 中的 `PRETRAINED_URLS`。
2. 在模型类上，设置 `pretrained_spec = PretrainedSpec(key="<short-name>", url=PRETRAINED_URLS["<short-name>"], state_dict_key_map={...})`。
3. 如果检查点的键布局与 `cracks_yolo` 的不同，请填充 `state_dict_key_map` 以对齐它们。
4. 在 `tests/weights/test_loader.py` 中添加一个测试（模拟 `requests.get`），将新 spec 加载到 `_DummyModel` 中并验证 `LoadReport`。

## 测试

`tests/weights/test_loader.py` 使用 `_FakeResponse` 上下文管理器 + `patch("cracks_yolo.weights.loader.requests.get", return_value=fake_resp)` 来测试下载路径而无需网络访问。一个带有两个线性层的 `_DummyModel` 用于测试部分加载报告（缓存命中、下载、strict=False 缺失键、strict=True 抛出异常、意外键）。
