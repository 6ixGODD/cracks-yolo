# Cross-validation

[English](cross_validation.md) | [中文](cross_validation.zh-CN.md)

`cracks_yolo.pipeline.crossval.run_cross_validation` and `cracks_yolo.pipeline.compare.compare_models_cross_val` implement N-fold cross-validation with statistical comparison.

## Data merging

**CV mode ignores the original train/valid/test split.** The caller is responsible for merging ALL dataset records into a single list before calling `run_cross_validation`. `scripts/train.py --cross-val` does this automatically:

```python
records = (
    src.load_split("train")
    + src.load_split("valid")
    + src.load_split("test")
)
```

The merged pool is what `StratifiedKFold` partitions. This guarantees the test set in each fold is drawn from the same distribution as the training data — important for the paired t-test's validity.

## N-fold mechanics

1. **Outer split**: `sklearn.model_selection.StratifiedKFold(n_splits=N, shuffle=True, random_state=seed)`.
   - Stratification key: per-image most-frequent class id. This keeps per-fold class balance roughly stable. For single-class datasets (like tongue surface cracks), this reduces to a plain shuffle — `StratifiedKFold` degenerates gracefully.
2. **Per fold**:
   - **Held-out fold (1/N)** → **TEST** records (no training sees them).
   - **Remaining records (N-1/N)** → training pool, further split into:
     - **train** (`1 - val_fraction`, default 90%) — used for backprop.
     - **val** (`val_fraction`, default 10%) — used for backprop-validation (best-checkpoint selection, learning-rate scheduling).
   - The inner split uses `sklearn.model_selection.train_test_split` with `random_state=seed + fold_idx` and stratification on the same per-image labels (falls back to plain random split if a class has too few samples for stratification).
   - Instantiate a fresh model from `model_factory`.
   - Run `TrainPipelineImpl.run(model, train_loader, val_loader, cfg)`.
   - Run `TestPipelineImpl.run(model, test_loader, test_cfg)` on the held-out fold.
   - Save to `output/cv/fold_<i>/` (full artifact set per fold) + `output/cv/fold_<i>/test/` (test artifacts).
3. **Aggregate**:
   - Mean ± std for every metric across the N folds (computed from per-fold TEST metrics).
   - Write `cv_summary.csv` (per-fold metrics) + `cv_report.json` (per-fold train + test summaries + aggregated mean/std).

The split is deterministic given `seed`. Same `seed` + same dataset = same N folds, so multi-model comparisons are paired (same fold indices for every model — required for the paired t-test).

## Why held-out fold = test (not val)

The original implementation used the held-out fold for BOTH validation during training AND test — this conflates the two roles and biases the test metric (the model selected its best checkpoint by the same data it's evaluated on). The current implementation separates them:

- **val** (from the training pool) → drives checkpoint selection during training. The model has indirectly seen this data via the scheduler.
- **test** (the held-out fold) → never influences training. The test metric is an honest generalization estimate.

`val_fraction=0.0` disables validation entirely (train on the full N-1/N pool, test on the held-out fold). Useful for very small datasets where the val carve-out would starve training.

## Multi-model comparison

`compare_models_cross_val(model_keys, dataset, cfg, n_folds=5, seed=42, metric="map50")`:

1. For each model in `model_keys`: run 5-fold CV → store per-fold `metric` values.
2. Per-fold paired t-test on `metric` across all model pairs: `scipy.stats.ttest_rel(per_fold_A, per_fold_B)`.
3. Wilcoxon signed-rank test (non-parametric alternative, useful for small N=5): `scipy.stats.wilcoxon(per_fold_A, per_fold_B)`.
4. Bootstrap CI on the per-fold difference `A - B`:
   - 1000 resamples with replacement, compute mean of each resample.
   - 95% CI = 2.5th and 97.5th percentiles of the bootstrap distribution.
5. Write:
   - `comparison.csv` — per-model mean ± std.
   - `paired_t_test.csv` — pairwise (model_A, model_B, t_stat, p_value, wilcoxon_p, bootstrap_ci_low, bootstrap_ci_high).
   - `comparison_plot.png` — box plot of per-fold `metric` per model.

## Statistical interpretation

A comparison between two models on a tongue surface crack detection task should be reported as:

> Over 5 stratified folds, Model A achieved mAP@50 = X.XX ± Y.YY, outperforming Model B (A.AA ± B.BB) by a mean per-fold difference of D.DD (paired t(4) = t.ttt, p = 0.0PPP; Wilcoxon p = 0.0WWW; 95% bootstrap CI [Lo.Lo, Hi.Hi]).

**Caveats**:
- N=5 is a small sample — the t-test assumes normality of per-fold differences. The Wilcoxon and bootstrap CI are non-parametric alternatives that don't assume normality; report all three.
- A p < 0.05 with N=5 should be reported with effect size (mean difference) + CI, not as a binary "significant / not" claim.
- If the bootstrap CI includes 0, the difference is not statistically significant at the 0.05 level even if the t-test p < 0.05.

## Metric choice

The default metric is `map50` (COCO AP at IoU 0.5). For tongue surface crack detection, `map50` is the primary metric — crack boxes are thin and IoU is sensitive to small localization errors, so `map5095` (averaged over IoU 0.5-0.95) penalizes thin-crack detectors more harshly than warranted.

To compare on a different metric:

```bash
python -m scripts.compare_models \
    --models yolov5s,yolov5s_sactr \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --n-folds 5 --epochs 100 \
    --metric map5095 \
    --output-dir output/comparison_map5095
```

Supported metric keys: `map50`, `map5095`, `map75`, `precision`, `recall`, `f1`, `auc_pr`, `auc_roc`.

## Per-fold artifacts

Each fold's `output/cv/fold_<i>/` directory contains the full artifact set of a single train run (see `docs/pipeline.md`), so individual folds can be inspected post-hoc — e.g. to confirm a particular fold's training curve didn't diverge.
