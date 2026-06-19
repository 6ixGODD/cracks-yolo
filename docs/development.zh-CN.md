# 开发：添加新的模型变体

[English](development.md) | [中文](development.zh-CN.md)

## 命名约定

类名 = `{baseline}_{improvements}_{loss}_{optimizer}_{activation}`。

- **baseline（基线）：** `YOLOv5s` / `YOLOv7w` / `YOLOv8s` / `YOLOv10s`
- **improvements（改进）：** `SAC` / `TR` / `SACTR` / `Deformable` / 等。
- **loss（损失）：** `CIoU_BCEObj_BCECls`（v5/v7）/ `CIoU_DFL`（v8/v10）
- **optimizer（优化器）：** `AdamW` / `SGD`
- **activation（激活函数）：** `SILU` / `MISH`

示例：`YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU`。

短别名（例如 `YOLOv5sSACTR`）从 `cracks_yolo.zoo.__init__`
重新导出，并在 `ZOO` 中注册。

## 分步指南

### 1. 不从 `cracks_yolo.zoo.base` 继承任何内容

`base.py` 定义了一个 `Protocol`，没有需要继承的类。直接定义一个新的
`nn.Module` 类。

### 2. 文件位置

将你的模型放在 `cracks_yolo/zoo/<arch>.py` 中（如果基线
与现有系列匹配，则使用已有文件，例如新的 v5 变体放在 `yolov5.py`；
新系列则新建文件 `yolov6.py`）。

### 3. 包含所有必需组件

```python
class YOLOv5sNewVariant_CIoU_BCEObj_BCECls_AdamW_SILU(nn.Module):  # noqa: N801
    """YOLOv5s + NewVariant。<一段架构描述。>"""

    pretrained_spec: PretrainedSpec | None = None  # 如果存在 COCO 权重则设置

    def __init__(self, num_classes: int = 80, input_size: int = 640) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        # ... 构建 backbone、neck、head ...
        # ... 将损失模块附加为 self._loss_fn ...
        # ... 通过虚拟训练前向传播初始化 stride ...

    @property
    def stride(self) -> torch.Tensor:
        return self.head.stride

    def forward(self, x: torch.Tensor) -> ...:
        ...

    def compute_loss(self, preds, targets, imgs=None) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def decode(self, preds) -> torch.Tensor:
        ...

    def build_optimizer(self) -> torch.optim.Optimizer:
        return default_optimizer(self, lr=1e-3)

    @classmethod
    def from_pretrained(cls, num_classes, weights_dir=None, strict=False) -> ...:
        ...
```

### 4. 注册短别名

在 `cracks_yolo/zoo/__init__.py` 中：

```python
from cracks_yolo.zoo.yolov5 import YOLOv5sNewVariant_CIoU_BCEObj_BCECls_AdamW_SILU
YOLOv5sNewVariant = YOLOv5sNewVariant_CIoU_BCEObj_BCECls_AdamW_SILU

ZOO: dict[str, type[nn.Module]] = {
    ...,
    "yolov5s_newvariant": YOLOv5sNewVariant,
}
```

### 5. 声明预训练 spec（如适用）

如果基线存在 COCO 权重（无 SAC/TR）：

```python
from cracks_yolo.weights.registry import PRETRAINED_URLS

class YOLOv5sNewVariant_CIoU_BCEObj_BCECls_AdamW_SILU(nn.Module):
    pretrained_spec = PretrainedSpec(
        key="yolov5s",
        url=PRETRAINED_URLS["yolov5s"],
        state_dict_key_map={},
    )
```

对于 SAC/TR 变体：`pretrained_spec = None`。

### 6. 编写测试

在 `tests/zoo/test_<arch>.py` 中（扩展现有文件或新建文件）：

- **前向传播形状：** `model(x)` 产生预期的形状。
- **`compute_loss` 有限 + 非零梯度：** 损失是有限的，`backward()`
  为每个可训练参数填充梯度。
- **`build_optimizer` 返回 Optimizer**，包含模型的参数。
- **`from_pretrained` 部分加载行为**（如果基线有 COCO 权重）：
  `LoadReport.missing` 包含 SAC/TR 的键。
- **Protocol 结构性检查：** `isinstance(model, DetectorModel)`。

`tests/zoo/test_zoo.py` 中的参数化测试会自动检测新的
ZOO 条目 -- 只需在 `ZOO` 中添加一个键，即可在其上运行
前向传播/损失/stride/优化器/Protocol 测试。

### 7. 更新文档

- `docs/models.md` -- 添加一个章节，包含架构图、层
  表格、损失公式、SAC/TR 插入点、论文引用。
- `docs/ops.md` -- 如果引入了新的 op，在此处记录。
- `README.md` -- 如果适用，更新模型表格。

## 声称完成前的验证

```bash
uv run ruff check cracks_yolo tests
uv run mypy --strict cracks_yolo tests
uv run pytest -q
```

以上三项必须全部通过。此外：对于 `cracks_yolo.zoo.ZOO` 中的每个类，
实例化 -> `from_pretrained`（如果离线则随机初始化）-> 在 `(2, 3, 640, 640)`
上前向传播 -> `compute_loss` -> 反向传播 -> 所有可训练参数
都具有非 None 梯度。

## 约定检查清单

- 每个文件顶部有 `from __future__ import annotations`。
- 没有 `Any` -- 修复类型。
- Google 风格 docstring，双引号，4 空格缩进，行长度 100。
- 强制单行导入（由 `.ruff.toml` 强制执行）。
- Pre-commit 运行 ruff + ruff-format。
- 不要在运行时从 `deps/` 导入。
- 不要重新引入 `compat/` 层。
- 不要在运行时解析 `yolov5s-sac-tr.yml`。
- 不要编写需要网络访问的测试（模拟 `requests`）。
