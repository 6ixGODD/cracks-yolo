# Cross-validation

`cracks_yolo.pipeline.crossval.run_cross_validation` and
`cracks_yolo.pipeline.compare.compare_models_cross_val` implement 5-fold
cross-validation with statistical comparison.

## 5-fold mechanics

1. **Split**: `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`.
   - Stratification key: per-image most-frequent class id. This keeps
     per-fold class balance roughly stable. For single-class datasets (like
     cracks), this reduces to a plain shuffle — `StratifiedKFold` degenerates
     gracefully.
2. **Per fold**:
   - Instantiate a fresh model from `model_cls`.
   - Build train/val `DetectionDataset` + `DataLoader` on the fold split.
   - Run `TrainPipelineImpl.run(model, train_loader, val_loader, cfg)`.
   - Save to `output/cv/fold_<i>/` (full artifact set per fold).
3. **Aggregate**:
   - Mean ± std for every metric across the 5 folds.
   - Write `cv_summary.csv` (per-fold metrics + mean/std rows) + `cv_report.json`.

The split is deterministic given `seed`. Same `seed` + same dataset = same
5 folds, so multi-model comparisons are paired (same fold indices for every
model — required for the paired t-test).

## Multi-model comparison

`compare_models_cross_val(model_keys, dataset, cfg, n_folds=5, seed=42, metric="map50")`:

1. For each model in `model_keys`: run 5-fold CV → store per-fold `metric` values.
2. Per-fold paired t-test on `metric` across all model pairs:
   `scipy.stats.ttest_rel(per_fold_A, per_fold_B)`.
3. Wilcoxon signed-rank test (non-parametric alternative, useful for small N=5):
   `scipy.stats.wilcoxon(per_fold_A, per_fold_B)`.
4. Bootstrap CI on the per-fold difference `A - B`:
   - 1000 resamples with replacement, compute mean of each resample.
   - 95% CI = 2.5th and 97.5th percentiles of the bootstrap distribution.
5. Write:
   - `comparison.csv` — per-model mean ± std.
   - `paired_t_test.csv` — pairwise (model_A, model_B, t_stat, p_value, wilcoxon_p, bootstrap_ci_low, bootstrap_ci_high).
   - `comparison_plot.png` — box plot of per-fold `metric` per model.

## Statistical interpretation

For the cracks-detection thesis (`YOLOv5sSACTR` is SOTA), the comparison
should be reported as:

> Over 5 stratified folds, YOLOv5sSACTR achieved mAP@50 = X.XX ± Y.YY,
> outperforming the YOLOv5s baseline (A.AA ± B.BB) by a mean per-fold
> difference of D.DD (paired t(4) = t.ttt, p = 0.0PPP; Wilcoxon p = 0.0WWW;
> 95% bootstrap CI [Lo.Lo, Hi.Hi]).

**Caveats**:
- N=5 is a small sample — the t-test assumes normality of per-fold
  differences. The Wilcoxon and bootstrap CI are non-parametric alternatives
  that don't assume normality; report all three.
- A p < 0.05 with N=5 should be reported with effect size (mean difference)
  + CI, not as a binary "significant / not" claim.
- If the bootstrap CI includes 0, the difference is not statistically
  significant at the 0.05 level even if the t-test p < 0.05.

## Metric choice

The default metric is `map50` (COCO AP at IoU 0.5). For cracks detection,
`map50` is the primary metric — crack boxes are thin and IoU is sensitive
to small localization errors, so `map5095` (averaged over IoU 0.5-0.95)
penalizes thin-crack detectors more harshly than warranted.

To compare on a different metric:

```bash
python -m scripts.compare_models \
    --models yolov5s,yolov5s_sactr \
    --dataset data/CrackDetection_Augmentation.v1.yolov5pytorch \
    --n-folds 5 --epochs 100 \
    --metric map5095 \
    --output-dir output/comparison_map5095
```

Supported metric keys: `map50`, `map5095`, `map75`, `precision`, `recall`,
`f1`, `auc_pr`, `auc_roc`.

## Per-fold artifacts

Each fold's `output/cv/fold_<i>/` directory contains the full artifact set
of a single train run (see `docs/pipeline.md`), so individual folds can be
inspected post-hoc — e.g. to confirm a particular fold's training curve
didn't diverge.
