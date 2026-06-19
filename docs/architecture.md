# Architecture

[English](architecture.md) | [中文](architecture.zh-CN.md)

## Design philosophy

`cracks_yolo` is a **self-contained PyTorch model zoo** for tongue surface
crack detection. Three principles govern every design decision:

1. **One model = one `nn.Module` class.** Each zoo model owns its layers,
   its loss module(s), its optimizer-builder, and its pretrained-weight
   loader. There is no abstract base class with abstract hooks, no plugin
   registry, and no component decoupling. The long class name
   (`YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU`) is the documentation.
2. **Pipeline contract is structural, not inherited.** `DetectorModel` is a
   `typing.Protocol` (see `cracks_yolo/zoo/base.py`). Pipelines depend only
   on the Protocol -- never on `isinstance(model, ...)`. Adding a new model
   variant never touches pipeline code.
3. **YAML is a memo, not a runtime artifact.** `yolov5s-sac-tr.yml` at the
   repo root records the target architecture for human readers. No code
   reads it.

The library avoids runtime YAML parsing and external framework coupling.
Vendored trees in `deps/{yolov5,yolov7,ultralytics,yolov9}` are **port-from
references only** -- never imported at runtime.

## Package layout

```
cracks_yolo/
  ops/         # Conv, CSP, transformer, detect heads, SAC/TR, YOLOv9 ops. Plain nn.Modules.
  losses/      # ComputeLoss (v5), ComputeLossOTA (v7), v8DetectionLoss, E2ELoss (v10).
  zoo/         # 26 model classes (YOLOv5/7/8/9/10 + SAC/TR variants + 6 torchvision detectors).
               # base.py = DetectorModel Protocol + PretrainedSpec.
  weights/     # load_pretrained: download, key-remap, strict=False load + LoadReport.
  logging/     # loguru JSONL sink + TypedDict log record schemas.
  metrics/     # COCOMetricsCalculator + PR/ROC/confusion + paired t-test/Wilcoxon/bootstrap CI.
  pipeline/    # TrainPipelineImpl / TestPipelineImpl / run_cross_validation / compare_models_cross_val.
  dataset/     # YOLOSource, COCOSource, DetectionDataset (torch.utils.data.Dataset), transforms, yolo<->coco convert.
  viz/         # loss/metric/PR/ROC curves, confusion matrix, Grad-CAM, dataset distribution plots.
  analysis/    # DatasetAnalysisReport (diversity metrics), ModelAnalysisReport (params/MACs/latency/VRAM).
scripts/       # CLI: train, test, convert_dataset, heatmap, analyze_dataset, analyze_model,
               # schedule_experiments (YAML-driven subprocess scheduler + retry mode), compare_models.
```

## Why no abstract base class?

An abstract base class couples every model to a shared inheritance chain.
Adding a model that needs a different forward signature (e.g. v10's dual
one2many/one2one head) requires either widening the ABC (breaking every
other model) or overriding every method (defeating the ABC's purpose).

A `typing.Protocol` decouples the contract from the implementation.
Pipelines depend on the Protocol; models satisfy it independently.
Structural subtyping means a model class does not even need to *declare*
that it satisfies the Protocol -- it just needs to have the right methods
and attributes. The `@runtime_checkable` decorator on `DetectorModel` lets
tests assert `isinstance(model, DetectorModel)` without forcing a
subclass relationship.

## The DetectorModel contract

Every zoo model satisfies:

```python
class DetectorModel(Protocol):
    input_size: int
    num_classes: int
    class_names: list[str]
    stride: torch.Tensor
    pretrained_spec: PretrainedSpec | None

    def forward(self, x): ...  # train: raw head outputs; eval: decoded
    def compute_loss(self, preds, targets, imgs=None) -> tuple[Tensor, Tensor]: ...
    def decode(self, preds) -> torch.Tensor: ...
    @classmethod
    def from_pretrained(cls, num_classes, weights_dir=None, strict=False) -> DetectorModel: ...
    def build_optimizer(self) -> torch.optim.Optimizer: ...
```

Pipelines call only these methods. Changing a model's internals never
touches pipeline code.

## The pipeline contract

```python
class TrainPipeline(Protocol):
    def run(self, model: DetectorModel, train_loader, val_loader, cfg: TrainConfig) -> TrainReport: ...

class TestPipeline(Protocol):
    def run(self, model: DetectorModel, test_loader, cfg: TestConfig) -> TestReport: ...
```

Concrete `TrainPipelineImpl` and `TestPipelineImpl` implement these
Protocols, backed by pydantic configs (`TrainConfig`, `TestConfig`).

## Logging

`loguru` configured by `cracks_yolo.logging.configure(output_dir)`. Writes
JSONL to `{output_dir}/run.log.jsonl` plus a human-readable stderr sink.
Log record schemas are `TypedDict`s in `cracks_yolo.logging.schema`:

- `TrainStepLog` -- one optimizer step
- `TrainEpochLog` -- end-of-epoch summary
- `ValLog` -- validation pass
- `TestLog` -- test-set evaluation
- `MetricLog` -- a single scalar emission
- `PretrainedLoadLog` -- pretrained weight load report

Each record carries a `record_type: Literal[...]` discriminator so
post-hoc queries can filter by record type. Usage:

```python
from cracks_yolo.logging.configure import configure_logger
from loguru import logger
configure_logger(output_dir=Path("output/run1"))
logger.bind(**{"record_type": "train_step", "step": 0, ...}).info("step done")
```

## Protocol self-description

Every zoo class declares two class attributes that the pipeline reads
instead of branching on class name:

- `loss_parts_schema: tuple[str, ...]` -- names of each entry in the
  `parts` tensor returned by `compute_loss`. v5/v7 =
  `("box","cls","obj")`; v8/v9/v10 = `("box","cls","dfl")`; torchvision
  wrappers = `("total","cls","box_reg","rpn_box_reg")`.
- `decode_format: str` -- `"anchor_free"` (v8/v9/v10:
  `(B, 4+nc, N)`) or `"anchor_based"` (v5/v7 + torchvision wrappers:
  `(B, N, nc+5)` or `(B, N_max, 6)`).

## Loss device-sync convention

Models are constructed on CPU, then moved to CUDA via `model.to(device)`
in the pipeline. Loss modules hold internal tensors (anchors, BCE
pos-weights, stride) that are not `nn.Parameter`s -- they do not move
with `.to()`. Each loss's `__call__` must sync `self.device`,
`self.anchors`, `self.stride`, and BCE submodules to `preds[0].device`
at entry. See `cracks_yolo/losses/yolov5.py` for the canonical pattern.
