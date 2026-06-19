"""5-fold (or N-fold) cross-validation for a single model class.

Splits the full list of :class:`RawDetection` records into N folds via
``sklearn.model_selection.StratifiedKFold`` (stratified by the most-common
class label per image, to keep class balance across folds). For each fold:

1. Build train/val :class:`DetectionDataset` from the split.
2. Instantiate a fresh model from ``model_cls``.
3. Run :class:`TrainPipelineImpl`.
4. Run :class:`TestPipelineImpl` on the held-out fold.
5. Save all artifacts under ``output_dir/fold_<i>/``.

After all folds: aggregate mean ± std for every metric in
:class:`MetricReport` and write ``cv_summary.csv`` + ``cv_report.json``.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import statistics
from typing import Any

from loguru import logger
import torch

from cracks_yolo.dataset.torchadapter import DetectionDataset
from cracks_yolo.dataset.torchadapter import build_dataloader
from cracks_yolo.dataset.transforms import build_transforms
from cracks_yolo.dataset.types import RawDetection
from cracks_yolo.logging.configure import configure_logger
from cracks_yolo.metrics.schemas import MetricReport
from cracks_yolo.pipeline._utils import set_seed
from cracks_yolo.pipeline.protocol import TestConfig
from cracks_yolo.pipeline.protocol import TrainConfig
from cracks_yolo.pipeline.protocol import TrainReport
from cracks_yolo.pipeline.test import TestPipelineImpl
from cracks_yolo.pipeline.train import TrainPipelineImpl
from cracks_yolo.zoo.base import DetectorModel


def _stratify_labels(records: list[RawDetection]) -> list[int]:
    """One label per image — the most common class in that image (or 0)."""
    out: list[int] = []
    for rec in records:
        if not rec.labels:
            out.append(0)
            continue
        counts: dict[int, int] = {}
        for lab in rec.labels:
            counts[lab] = counts.get(lab, 0) + 1
        out.append(max(counts.items(), key=lambda kv: kv[1])[0])
    return out


class CrossValReport:
    """Aggregate result of an N-fold CV run. Plain holder for JSON dump."""

    def __init__(
        self,
        model_name: str,
        n_folds: int,
        per_fold_train: list[dict[str, Any]],
        per_fold_test: list[dict[str, Any]],
        metric_fields: list[str],
    ) -> None:
        self.model_name = model_name
        self.n_folds = n_folds
        self.per_fold_train = per_fold_train
        self.per_fold_test = per_fold_test
        self.metric_fields = metric_fields

    def aggregated(self) -> dict[str, dict[str, float]]:
        """Compute mean/std for every metric field across folds."""
        agg: dict[str, dict[str, float]] = {}
        for field in self.metric_fields:
            values = [float(fold.get(field, 0.0)) for fold in self.per_fold_test]
            if not values:
                agg[field] = {"mean": 0.0, "std": 0.0}
                continue
            mean = statistics.fmean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            agg[field] = {"mean": mean, "std": std}
        return agg

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_folds": self.n_folds,
            "per_fold_train": self.per_fold_train,
            "per_fold_test": self.per_fold_test,
            "aggregated": self.aggregated(),
        }


def run_cross_validation(
    model_factory: Callable[[], DetectorModel],
    records: list[RawDetection],
    input_size: int,
    train_cfg: TrainConfig,
    n_folds: int = 5,
    seed: int = 42,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> CrossValReport:
    """Run N-fold CV for one model class.

    Args:
        model_factory: zero-arg callable that returns a fresh model each fold.
        records: full dataset records (will be split per fold).
        input_size: model input size (also transform resize target).
        train_cfg: base training config — its ``output_dir`` is used as the
            CV root; per-fold outputs land in ``output_dir/fold_<i>/``.
        n_folds: number of CV folds.
        seed: stratification + seeding seed.
        batch_size: optional override for the dataloader batch size; falls
            back to ``train_cfg.batch_size``.
        num_workers: dataloader workers.

    Returns:
        :class:`CrossValReport` with per-fold train/test summaries + mean/std.
    """
    from sklearn.model_selection import StratifiedKFold

    cv_root = train_cfg.output_dir
    cv_root.mkdir(parents=True, exist_ok=True)
    configure_logger(cv_root, level="INFO", stderr=True)
    set_seed(seed)

    bs = batch_size if batch_size is not None else train_cfg.batch_size
    labels = _stratify_labels(records)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_fold_train: list[dict[str, Any]] = []
    per_fold_test: list[dict[str, Any]] = []
    metric_fields = [
        "map50",
        "map5095",
        "ap50",
        "ap75",
        "precision",
        "recall",
        "f1",
        "ar1",
        "ar10",
        "ar100",
        "ar300",
        "ar1000",
        "ar_small",
        "ar_medium",
        "ar_large",
        "auc_pr",
        "auc_roc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
    ]

    train_pipeline = TrainPipelineImpl()
    test_pipeline = TestPipelineImpl()

    sample_model = model_factory()
    model_name = type(sample_model).__name__
    del sample_model

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(records)), labels)):
        fold_dir = cv_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CV fold {fold_idx}: train={len(train_idx)} val={len(val_idx)}")

        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

        train_ds = DetectionDataset(
            train_records,
            transform=build_transforms(input_size, train=True, augment=True),
        )
        val_ds = DetectionDataset(
            val_records,
            transform=build_transforms(input_size, train=False, augment=False),
        )
        train_loader = build_dataloader(
            train_ds,
            batch_size=bs,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = build_dataloader(
            val_ds,
            batch_size=bs,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        fold_train_cfg = TrainConfig(**{
            **train_cfg.model_dump(),
            "output_dir": fold_dir,
        })

        model = model_factory()
        train_report: TrainReport = train_pipeline.run(
            model, train_loader, val_loader, fold_train_cfg
        )
        per_fold_train.append({
            "fold": fold_idx,
            "best_epoch": train_report.best_epoch,
            "best_map50": train_report.best_map50,
            "final_train_loss": train_report.final_train_loss,
            "elapsed_sec": train_report.elapsed_sec,
        })

        test_cfg = TestConfig(
            output_dir=fold_dir / "test",
            batch_size=bs,
            device=train_cfg.device,
            num_workers=num_workers,
        )
        test_report = test_pipeline.run(model, val_loader, test_cfg)
        metrics: MetricReport = test_report.metrics
        per_fold_test.append({
            "fold": fold_idx,
            **{f: float(getattr(metrics, f)) for f in metric_fields},
        })

        # Free memory between folds.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = CrossValReport(
        model_name=model_name,
        n_folds=n_folds,
        per_fold_train=per_fold_train,
        per_fold_test=per_fold_test,
        metric_fields=metric_fields,
    )

    # Write CV summary artifacts.
    import csv

    summary_path = cv_root / "cv_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", *metric_fields])
        writer.writeheader()
        for row in per_fold_test:
            writer.writerow(row)
    agg_path = cv_root / "cv_report.json"
    agg_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


__all__ = ["CrossValReport", "run_cross_validation"]
