# Development: adding a new model variant

## Naming convention

Class name = `{baseline}_{improvements}_{loss}_{optimizer}_{activation}`.

- **baseline:** `YOLOv5s` / `YOLOv7w` / `YOLOv8s` / `YOLOv10s`
- **improvements:** `SAC` / `TR` / `SACTR` / `Deformable` / etc.
- **loss:** `CIoU_BCEObj_BCECls` (v5/v7) / `CIoU_DFL` (v8/v10)
- **optimizer:** `AdamW` / `SGD`
- **activation:** `SILU` / `MISH`

Example: `YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU`.

Short alias (e.g. `YOLOv5sSACTR`) re-exported from
`cracks_yolo.zoo.__init__` and registered in `ZOO`.

## Step-by-step

### 1. Subclass nothing from `cracks_yolo.zoo.base`

`base.py` is a `Protocol` — there's nothing to subclass. Define a fresh
`nn.Module` class.

### 2. File location

Place your model in `cracks_yolo/zoo/<arch>.py` (existing file if the
baseline matches an existing family, e.g. `yolov5.py` for a new v5
variant; new file `yolov6.py` for a new family).

### 3. Bake in everything

```python
class YOLOv5sNewVariant_CIoU_BCEObj_BCECls_AdamW_SILU(nn.Module):  # noqa: N801
    """YOLOv5s + NewVariant. <one-paragraph architecture description>."""

    pretrained_spec: PretrainedSpec | None = None  # or set if COCO weights exist

    def __init__(self, num_classes: int = 80, input_size: int = 640) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        # ... build backbone, neck, head ...
        # ... attach loss module(s) as self._loss_fn ...
        # ... init stride via dummy training forward ...

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

### 4. Register the short alias

In `cracks_yolo/zoo/__init__.py`:

```python
from cracks_yolo.zoo.yolov5 import YOLOv5sNewVariant_CIoU_BCEObj_BCECls_AdamW_SILU
YOLOv5sNewVariant = YOLOv5sNewVariant_CIoU_BCEObj_BCECls_AdamW_SILU

ZOO: dict[str, type[nn.Module]] = {
    ...,
    "yolov5s_newvariant": YOLOv5sNewVariant,
}
```

### 5. Declare pretrained spec (if applicable)

If COCO weights exist for the baseline (no SAC/TR):

```python
from cracks_yolo.weights.registry import PRETRAINED_URLS

class YOLOv5sNewVariant_CIoU_BCEObj_BCECls_AdamW_SILU(nn.Module):
    pretrained_spec = PretrainedSpec(
        key="yolov5s",
        url=PRETRAINED_URLS["yolov5s"],
        state_dict_key_map={},
    )
```

For SAC/TR variants: `pretrained_spec = None`.

### 6. Write tests

In `tests/zoo/test_<arch>.py` (extend existing or new file):

- **Forward shape:** `model(x)` produces expected shapes.
- **`compute_loss` finite + non-zero grad:** loss is finite, `backward()`
  populates grad on every trainable param.
- **`build_optimizer` returns Optimizer** with the model's params.
- **`from_pretrained` partial-load behavior** (if baseline has COCO
  weights): `LoadReport.missing` includes the SAC/TR keys.
- **Protocol structural check:** `isinstance(model, DetectorModel)`.

The parametrized tests in `tests/zoo/test_zoo.py` automatically pick up
new ZOO entries — adding a key to `ZOO` is enough to get forward/loss/
stride/optimizer/protocol tests run on it.

### 7. Update docs

- `docs/models.md` — add a section with architecture diagram, layer
  table, loss formula, SAC/TR insertion points, paper citation.
- `docs/ops.md` — if you introduced a new op, document it here.
- `README.md` — update the model table if applicable.

## Verification before claiming done

```bash
uv run ruff check cracks_yolo tests
uv run mypy --strict cracks_yolo tests
uv run pytest -q
```

All three must be green. Plus: for every class in `cracks_yolo.zoo.ZOO`,
instantiate → `from_pretrained` (or random init if offline) → forward on
`(2, 3, 640, 640)` → `compute_loss` → backward → all trainable params
have non-None grad.

## Conventions checklist

- `from __future__ import annotations` at top of every file.
- No `Any` — fix the type.
- Google-style docstrings, double quotes, 4-space indent, line length 100.
- Force-single-line imports (enforced by `.ruff.toml`).
- Pre-commit runs ruff + ruff-format.
- Don't import from `deps/` at runtime.
- Don't reintroduce a `compat/` layer.
- Don't parse `yolov5s-sac-tr.yml` at runtime.
- Don't write tests that require network access (mock `requests`).
