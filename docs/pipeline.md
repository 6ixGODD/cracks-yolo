# Pipeline

[English](pipeline.md) | [中文](pipeline.zh-CN.md)

The pipeline layer provides thin procedural wrappers over model methods. Pipelines own no model-specific logic; all branching is mediated by model-declared attributes (`decode_format`, `loss_parts_schema`) or by structural conformance to the `DetectorModel` Protocol.

## Architecture overview

```
CLI (Typer)
  ├─ train  → run_train()           → model.train_model(config)
  ├─ test   → run_test()            → model.load() → model.inference() → COCO eval
  ├─ run    → run_train() + run_test()  (train-then-test from YAML config)
  └─ compose → run_compose()        → subprocess per experiment
```

The pipeline layer sits at `cracks_yolo/pipeline/`. The modules are:

| Module | Responsibility |
| --- | --- |
| `train.py` | `run_train()` — model construction, pretrained loading, dispatch to `model.train_model()` |
| `test.py` | `run_test()` — checkpoint loading, batched inference, COCO metric computation |
| `compose.py` | `run_compose()` — YAML-driven experiment scheduler with `$include` recursion |
| `_helpers.py` | Shared utilities: NMS, target conversion, seed setting, anchor-free detection |

## Train flow

### Entry point

`run_train(model_name, dataset, output_dir, epochs, batch_size, lr, pretrained, device, seed, num_workers, **kwargs) -> TrainReport`

The function performs three steps:

1. **Model lookup.** The `model_name` is resolved against the `ZOO` dictionary (a `dict[str, type[BaseModel]]`). Unknown keys raise `ValueError`.

2. **Construction.** The model class is instantiated. If `pretrained=True` and the class exposes `from_pretrained`, the constructor is called first to obtain architecture metadata, then `from_pretrained` reloads the model with COCO-pretrained weights via key-matched `load_state_dict(strict=False)`. The ultralytics adapter models additionally accept a `model_name` argument to select the YAML configuration file.

3. **Dispatch.** A `TrainConfig` dataclass is populated from the function arguments and passed to `model.train_model(config)`. The pipeline does not inspect the model class beyond checking for `from_pretrained` and `model_name` in the constructor signature.

### TrainConfig

`TrainConfig` is a plain `@dataclass` (not a pydantic model) defined in `cracks_yolo/zoo/base.py`. Fields: `output_dir`, `dataset`, `data_yaml`, `epochs`, `batch_size`, `lr`, `weight_decay`, `warmup_epochs`, `warmup_lr`, `momentum`, `amp`, `clip_grad_norm`, `early_stopping_patience`, `cosine_lr`, `cosine_lrf`, `use_ema`, `ema_decay`, `optimizer`, `seed`, `device`, `num_workers`, `pretrained`, `extra_kwargs`.

It is a plain dataclass so that model subclasses may extend it without pydantic validation constraints.

### Two training implementations

The `train_model` method is model-defined and has two canonical forms:

**UltralyticsAdapter.train_model** (YOLOv3--v10, YOLO12). Constructs an ultralytics `DetectionTrainer` (or `RTDETRTrainer`) directly, bypassing `YOLO().train()` which would rebuild a vanilla `DetectionModel` and discard SAC/TR injections. The trainer's `model` attribute is replaced with the SAC-injected `nn.Module` before `trainer.train()` is called. After training completes, ultralytics output artifacts are synchronised into `config.output_dir`.

**TorchvisionBase._run_train_loop** (RetinaNet, Faster-RCNN, Mask-RCNN, FCOS, SSD300, SSDlite320). A hand-written epoch loop with: cosine LR schedule with linear warmup, optional AMP via `torch.amp.GradScaler`, optional gradient clipping, per-epoch validation via `pycocotools`, early stopping, and CSV metric logging. Each subclass calls `_run_train_loop` from its own `train_model` with an architecture-specific `score_thresh`.

### Post-train

After `train_model` returns, `run_train` saves the model checkpoint to `best.pt` and, if `thop` is installed, writes `analysis.json` with parameter count, GFLOPs, FPS, latency, and peak VRAM.

## Test flow

### Entry point

`run_test(model_name, weights, dataset, output_dir, batch_size, device, seed) -> dict[str, Any]`

The function:

1. Instantiates the model class from the ZOO dictionary.
2. Calls `model.load(weights)` to restore trained parameters. The load method handles both ultralytics checkpoint format (`ckpt["model"]` is a `DetectionModel` instance) and the generic format (`model_state_dict` key).
3. Moves the model to the requested device.
4. For each split in `("test", "valid")`:
   - Loads records from the YOLO-format dataset directory.
   - Builds a `DetectionDataset` with evaluation transforms (resize only, no augmentation).
   - Iterates batches through `model.inference(images)`, which returns `list[InferenceResult]`.
   - Converts predictions to COCO JSON format (boxes from xyxy to xywh).
   - Computes metrics via `pycocotools.cocoeval.COCOeval` at IoU thresholds 0.5 and 0.5:0.95.
   - Computes per-image PR/ROC curves, confusion matrix, sensitivity, specificity, PPV, NPV.
5. Writes `metrics.csv` and per-split prediction JSON files.
6. Optionally runs `model.analyze()` for efficiency metrics.

### Inference decoding

Each model class declares a `decode_format` class attribute: `"anchor_free"` (YOLOv8/v9/v10) or `"anchor_based"` (YOLOv5/v7, torchvision wrappers). The pipeline reads this attribute; it never branches on class name.

**UltralyticsAdapter.inference.** Calls `self._inner(images)` to obtain raw grid predictions, then applies `ultralytics.utils.nms.non_max_suppression` with `conf_thres=0.001`, `iou_thres=0.7`, `max_det=300`. Returns `InferenceResult` objects with boxes clipped to `[0, input_size]`.

**TorchvisionBase.inference.** Calls `self._inner(images)` to obtain raw torchvision output (box regression, classification logits), then applies the model's post-processing (score thresholding, NMS) to produce `InferenceResult` objects in original image coordinates.

### InferenceResult

A plain container with three `torch.Tensor` fields:
- `boxes`: `(N, 4)` xyxy pixel coordinates.
- `scores`: `(N,)` confidence in `[0, 1]`.
- `labels`: `(N,)` integer class indices (0-based).

## Run command (train + auto-test)

`run(config, output_dir, device, test_only, weights)` loads a single-experiment YAML config. If `type: train`, it calls `run_train()` and then automatically runs `run_test()` on the best checkpoint. If `type: test`, it calls `run_test()` directly. The `--test-only` flag skips training and requires `--weights`.

## Experiment compose

`run_compose(config, output_dir, max_parallel)` loads a compose YAML with recursive `$include` resolution. Each included file is either a single experiment config or a file with an `experiments` list. Experiments are run sequentially via `subprocess.run`, capturing stdout/stderr to per-experiment log files. Failures are recorded to `errors.jsonl`; successes to `results.jsonl`. Per-experiment `env` overrides (e.g., `CUDA_VISIBLE_DEVICES`) are supported for multi-GPU scheduling.

## Key design decisions

**Thin pipeline, thick model.** The pipeline layer is deliberately thin. `run_train` is 99 lines; `run_test` is 342 lines, most of which is COCO metric computation. All training loop logic lives in `model.train_model()`. Adding a new model family requires no pipeline changes as long as the model satisfies the implicit Protocol.

**No abstract hooks.** There is no `on_epoch_start` / `on_batch_end` callback system. Each model's `train_model` is a self-contained method that owns its loop, its optimizer construction, its LR schedule, and its validation. This trades some code duplication for total explicitness.

**Dataclass, not pydantic.** `TrainConfig` is a plain `@dataclass`. Pydantic validation would add a dependency and constrain model-specific extensions. Validation happens at the CLI boundary (Typer type coercion) and in the model's `train_model` method.

**Decode format self-description.** Models declare `decode_format: str` so the pipeline can apply the correct tensor rearrangement without `isinstance` checks. The same principle applies to `loss_parts_schema` for logging loss components.

**State machine.** `BaseModel` maintains a `ModelState` enum (`UNINITIALIZED`, `PRETRAINED`, `TRAINED`). `inference()` asserts `TRAINED`; `from_pretrained()` transitions to `PRETRAINED`; `train_model()` transitions to `TRAINED`. This prevents silent misuse (e.g., running inference on an untrained model).

**Two trainer strategies.** The ultralytics adapter reuses the upstream `DetectionTrainer` to preserve compatibility with ultralytics' augmentation pipeline, EMA, and metric logging. The torchvision adapter implements a hand-written loop because torchvision detection models use a different target format (`List[Dict[str, Tensor]]`) and do not share ultralytics' training infrastructure. Both strategies produce `TrainReport` with the same schema.

## Artifacts produced

| Artifact | Producer | Content |
| --- | --- | --- |
| `best.pt` | train | Model state dict checkpoint at best val mAP@50 |
| `metrics.csv` | train, test | Per-epoch losses (train); per-split metrics (test) |
| `analysis.json` | train | Parameter count, GFLOPs, FPS, latency, VRAM |
| `best_predictions_{split}.json` | test | COCO-format detection results per image |
| `run.log.jsonl` | train (ultralytics) | Structured JSONL training log |
| `results.csv` | train (ultralytics) | Ultralytics-format training metrics |
| `config.yaml` | train (ultralytics) | Frozen training configuration |
| `errors.jsonl` | compose | Per-experiment failure records |
| `results.jsonl` | compose | Per-experiment success records |

## CLI reference

```
cy train  --model yolov5s_sac --dataset data/ --output-dir runs/exp1
cy test   --model yolov5s_sac --weights runs/exp1/weights/best.pt --dataset data/ --output-dir runs/exp1/test
cy run    --config experiments/models/yolov5s.yaml --output-dir runs/exp1
cy compose --config experiments/all_models_direct.yaml --output-dir runs/batch1 --max-parallel 2
```

The `run` command supports `--test-only` for re-evaluating an existing checkpoint without retraining.
