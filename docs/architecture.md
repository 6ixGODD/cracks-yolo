# Architecture

## Design philosophy

`cracks_yolo` is a **self-contained PyTorch YOLO model zoo** for cracks
detection. Three principles govern every design decision:

1. **One model = one `nn.Module` class.** Each zoo model owns its layers,
   its loss module(s), its optimizer-builder, and its pretrained-weight
   loader. There is no abstract base class with abstract hooks; no plugin
   registry; no component decoupling. The long class name
   (`YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU`) is the documentation.
2. **Pipeline contract is structural, not inherited.** `DetectorModel` is a
   `typing.Protocol` (see `cracks_yolo/zoo/base.py`). Pipelines depend only
   on the Protocol — never on `isinstance(model, ...)`. Adding a new model
   variant never touches pipeline code.
3. **YAML is a memo, not a runtime artifact.** `yolov5s-sac-tr.yml` at the
   repo root records the target architecture for human readers. No code
   reads it.

This is the deliberate inverse of `ultralytics`' approach (YAML parsing +
monkey-patching `parse_model`). Vendored trees in `deps/{yolov5,yolov7,
ultralytics}` are **port-from references only** — never imported at runtime.

## Package layout

```
cracks_yolo/
  ops/         # Conv, CSP, transformer, detect heads, SAC/TR, YOLOv9 ops. Plain nn.Modules.
  losses/      # ComputeLoss (v5), ComputeLossOTA (v7), v8DetectionLoss, E2ELoss (v10).
  zoo/         # 22 model classes (YOLOv5/7/8/9/10 + SAC/TR variants + torchvision RetinaNet/Faster-RCNN).
               # base.py = DetectorModel Protocol + PretrainedSpec.
  weights/     # load_pretrained: download, key-remap, strict=False load + LoadReport.
  logging/     # loguru JSONL sink + TypedDict log record schemas.
  metrics/     # COCOMetricsCalculator + PR/ROC/confusion + paired t-test/Wilcoxon/bootstrap CI.
  pipeline/    # TrainPipelineImpl / TestPipelineImpl / run_cross_validation / compare_models_cross_val.
  dataset/     # YOLOSource, COCOSource, DetectionDataset (torch.utils.data.Dataset), transforms, yolo↔coco convert.
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
Structural subtyping means a model class doesn't even need to *declare*
that it satisfies the Protocol — it just needs to have the right methods
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

Pipelines call only these. Changing a model's internals never touches
pipeline code.

## The pipeline contract

```python
class TrainPipeline(Protocol):
    def run(self, model: DetectorModel, train_loader, val_loader, cfg: TrainConfig) -> TrainReport: ...

class TestPipeline(Protocol):
    def run(self, model: DetectorModel, test_loader, cfg: TestConfig) -> TestReport: ...
```

Concrete pipeline implementations land in a later pass. The Protocols
exist today so future code slots in without re-architecting.

## Logging

`loguru` configured by `cracks_yolo.logging.configure(output_dir)`. Writes
JSONL to `{output_dir}/run.log.jsonl` plus a human-readable stderr sink.
Log record schemas are `TypedDict`s in `cracks_yolo.logging.schema`:

- `TrainStepLog` — one optimizer step
- `TrainEpochLog` — end-of-epoch summary
- `ValLog` — validation pass
- `TestLog` — test-set evaluation
- `MetricLog` — a single scalar emission
- `PretrainedLoadLog` — pretrained weight load report

Each record carries a `record_type: Literal[...]` discriminator so
post-hoc queries can filter by record type. Usage:

```python
from cracks_yolo.logging.configure import configure_logger
from loguru import logger
configure_logger(output_dir=Path("output/run1"))
logger.bind(**{"record_type": "train_step", "step": 0, ...}).info("step done")
```

## What this pass does NOT include (explicitly deferred)

- Dataset adapter (`cracks_yolo.dataset`) — pending COCO dataset.
- Train/test pipeline **implementations** — only Protocols + pydantic
  configs.
- Metrics **implementation** — only schemas + Protocol.
- Visualization code (confusion matrix, PR curve drawing).
- Scripts/entries.
- torchvision transforms wiring.
- `torch.compile` or quantization.
