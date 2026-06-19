# Pipeline

`cracks_yolo.pipeline` provides the train loop, test loop, 5-fold
cross-validation, and multi-model comparison. All four rely only on the
`DetectorModel` Protocol — no model-specific branching.

## TrainPipelineImpl

`TrainPipelineImpl.run(model, train_loader, val_loader, cfg) -> TrainReport`:

1. `configure_logger(cfg.output_dir)` — JSONL sink at `run.log.jsonl`.
2. `optimizer = model.build_optimizer()` (lr/weight_decay overridden from cfg).
3. Optional AMP `torch.amp.GradScaler` (enabled by `--amp`, default off).
4. For each epoch:
   - Train one pass: `preds = model(images)` → `loss, parts = model.compute_loss(preds, yolo_targets, imgs=images)`.
   - Backward + step (with optional grad clipping via `cfg.clip_grad_norm`).
   - Log `TrainStepLog` every N steps; `TrainEpochLog` per epoch.
   - Validate (if `val_loader` provided and `(epoch+1) % cfg.val_interval == 0`): forward → `decode` → NMS → COCO mAP.
   - Save `best.pt` if val mAP@50 improved.
5. Final: write `metrics.csv` (per-epoch losses + val metrics), `loss_curve.png`, `metric_curve.png`, `config.yaml`.
6. Returns `TrainReport(output_dir, best_weights_path, best_epoch, best_map50, history)`.

### Parts interpretation

Each model declares `loss_parts_schema: tuple[str, ...]`:
- v5/v7: `("box", "cls", "obj")`
- v8/v9/v10: `("box", "cls", "dfl")`
- torchvision RetinaNet/Faster-RCNN: `("total", "cls", "box_reg", "rpn_box_reg")`

The pipeline reads the schema from the model (no class-name branching) and
maps each entry to the appropriate `TrainStepLog` field. Unknown entries
(like `total`) are summed into the loss but not surfaced as a named part.

### TrainConfig (pydantic BaseModel)

```python
class TrainConfig(BaseModel):
    output_dir: Path
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 16
    input_size: int = 640
    amp: bool = False
    clip_grad_norm: float | None = None
    val_interval: int = 1
    log_every_n_steps: int = 10
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
```

## TestPipelineImpl

`TestPipelineImpl.run(model, test_loader, cfg) -> TestReport`:

1. `configure_logger(cfg.output_dir)`.
2. For each batch: `model.eval()` → `preds = model(images)` → `decoded = model.decode(preds)` → NMS (via `detections_to_per_image`).
3. `COCOMetricsCalculator.update(all_per_image)` → `report = calculator.run()`.
4. Compute PR/ROC curves, confusion matrix, AUCs (via `cracks_yolo.metrics.curves` + `confusion`).
5. Write `metrics.csv`, `per_image/<id>.json`, `predictions/<id>.jpg`, `curves/{pr,roc,confusion}.png`, `TestLog`.
6. Returns `TestReport(output_dir, metrics, elapsed_sec)`.

### Decode format

Each model declares `decode_format: str`:
- `"anchor_free"` (v8/v9/v10): output `(B, 4+nc, N)`. Pipeline permutes to `(B, N, 4+nc)` and applies NMS.
- `"anchor_based"` (v5/v7, torchvision wrappers): output `(B, N, nc+5)` already in xyxy-score-cls layout.

The pipeline reads `decode_format` from the model to decide the permute step
(no class-name branching).

## run_cross_validation

`run_cross_validation(model_cls, dataset, cfg, n_folds=5, seed=42) -> CrossValReport`:

1. Stratified 5-fold split via `sklearn.model_selection.StratifiedKFold(random_state=seed)`.
   - Stratification is by **image-level class composition** (an image is
     assigned the set of class ids it contains; stratify on the most-frequent
     class per image to keep per-fold class balance roughly stable).
2. For each fold: instantiate a fresh model, build train/val loaders on the
   fold split, run `TrainPipelineImpl.run`, save to `output/cv/fold_<i>/`.
3. Aggregate: mean ± std for every metric across folds; write `cv_summary.csv`
   + `cv_report.json`.
4. Returns `CrossValReport(folds, mean_metrics, std_metrics, output_dir)`.

## compare_models_cross_val

`compare_models_cross_val(model_keys, dataset, cfg, n_folds=5, seed=42, metric="map50") -> ComparisonReport`:

1. Run 5-fold CV for each model in `model_keys`.
2. Per-fold paired t-test on the chosen metric (default `map50`) across all
   model pairs: `scipy.stats.ttest_rel`. Also Wilcoxon signed-rank (for small
   N) and bootstrap CI on the per-fold difference.
3. Write `comparison.csv` (per-model mean ± std), `paired_t_test.csv`
   (pairwise p-values), `comparison_plot.png`.
4. Returns `ComparisonReport(per_model, pairwise_tests, output_dir)`.

See `docs/cross_validation.md` for the statistical methodology.

## CLI

- `scripts/train.py` — single train or `--cross-val` flag triggers `run_cross_validation`.
- `scripts/test.py` — single test run.
- `scripts/compare_models.py` — wraps `compare_models_cross_val`.
