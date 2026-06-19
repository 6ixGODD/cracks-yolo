"""Multi-model comparison with paired statistical tests across folds.

Runs N-fold CV for each of N models on the same dataset splits, then
applies paired statistical tests (paired t-test, Wilcoxon, bootstrap CI)
on the per-fold metric of choice (default mAP@50). Writes:

- ``comparison.csv`` — per-model mean ± std for every metric.
- ``paired_t_test.csv`` — pairwise p-values between all model pairs.
- ``comparison_plot.png`` — bar plot of mean ± std (best-effort).
- ``comparison_report.json`` — full structured report.
"""

from __future__ import annotations

from collections.abc import Callable
import csv
import itertools
import json
from pathlib import Path
import statistics

from loguru import logger

from cracks_yolo.dataset.types import RawDetection
from cracks_yolo.logging.configure import configure_logger
from cracks_yolo.metrics.schemas import StatisticalTest
from cracks_yolo.metrics.statistical import bootstrap_ci
from cracks_yolo.metrics.statistical import paired_t_test
from cracks_yolo.metrics.statistical import wilcoxon
from cracks_yolo.pipeline.crossval import CrossValReport
from cracks_yolo.pipeline.crossval import run_cross_validation
from cracks_yolo.pipeline.protocol import TrainConfig
from cracks_yolo.zoo.base import DetectorModel


class ComparisonReport:
    """Aggregate result of comparing multiple models via N-fold CV."""

    def __init__(
        self,
        model_names: list[str],
        n_folds: int,
        per_model: dict[str, CrossValReport],
        metric: str,
    ) -> None:
        self.model_names = model_names
        self.n_folds = n_folds
        self.per_model = per_model
        self.metric = metric

    def per_fold_metric(self, model_name: str) -> list[float]:
        cv = self.per_model[model_name]
        return [float(fold.get(self.metric, 0.0)) for fold in cv.per_fold_test]

    def pairwise_tests(self) -> list[dict[str, object]]:
        tests: list[dict[str, object]] = []
        for a, b in itertools.combinations(self.model_names, 2):
            a_vals = self.per_fold_metric(a)
            b_vals = self.per_fold_metric(b)
            t: StatisticalTest = paired_t_test(a_vals, b_vals)
            w: StatisticalTest = wilcoxon(a_vals, b_vals)
            boot: StatisticalTest = bootstrap_ci(a_vals, b_vals, seed=0)
            tests.append({
                "model_a": a,
                "model_b": b,
                "metric": self.metric,
                "paired_t_stat": t["statistic"],
                "paired_t_p": t["p_value"],
                "wilcoxon_stat": w["statistic"],
                "wilcoxon_p": w["p_value"],
                "bootstrap_ci_low": boot["ci_low"],
                "bootstrap_ci_high": boot["ci_high"],
                "mean_diff": statistics.fmean([x - y for x, y in zip(a_vals, b_vals, strict=True)]),
            })
        return tests

    def to_dict(self) -> dict[str, object]:
        return {
            "model_names": self.model_names,
            "n_folds": self.n_folds,
            "metric": self.metric,
            "aggregated": {name: self.per_model[name].aggregated() for name in self.model_names},
            "pairwise_tests": self.pairwise_tests(),
        }


def compare_models_cross_val(
    model_factories: dict[str, Callable[[], DetectorModel]],
    records: list[RawDetection],
    input_size: int,
    train_cfg: TrainConfig,
    n_folds: int = 5,
    seed: int = 42,
    metric: str = "map50",
    batch_size: int | None = None,
    num_workers: int = 0,
) -> ComparisonReport:
    """Run N-fold CV for each model and compare via paired statistical tests.

    Args:
        model_factories: ``{short_name: zero_arg_callable}`` — each callable
            must return a fresh :class:`DetectorModel` instance.
        records: full dataset records.
        input_size: model input size.
        train_cfg: base training config — ``output_dir`` is the comparison
            root; per-model CV runs land in ``output_dir/<short_name>/``.
        n_folds: number of CV folds per model.
        seed: stratification + seeding seed (same splits across models).
        metric: metric name to compare (must be a field of
            :class:`MetricReport`).
        batch_size: optional override.
        num_workers: dataloader workers.

    Returns:
        :class:`ComparisonReport` with per-model aggregation + pairwise tests.
    """
    root = train_cfg.output_dir
    root.mkdir(parents=True, exist_ok=True)
    configure_logger(root, level="INFO", stderr=True)

    per_model: dict[str, CrossValReport] = {}
    model_names = list(model_factories.keys())

    for name in model_names:
        logger.info(f"compare: running CV for model '{name}'")
        model_dir = root / name
        per_model_cfg = TrainConfig(**{**train_cfg.model_dump(), "output_dir": model_dir})
        cv_report = run_cross_validation(
            model_factory=model_factories[name],
            records=records,
            input_size=input_size,
            train_cfg=per_model_cfg,
            n_folds=n_folds,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        per_model[name] = cv_report

    report = ComparisonReport(
        model_names=model_names,
        n_folds=n_folds,
        per_model=per_model,
        metric=metric,
    )

    _write_comparison_artifacts(report, root)
    return report


def _write_comparison_artifacts(report: ComparisonReport, root: Path) -> None:
    # comparison.csv — per-model mean ± std for every metric field.
    metric_fields = report.per_model[report.model_names[0]].metric_fields
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model"] + [f"{m}_mean" for m in metric_fields] + [f"{m}_std" for m in metric_fields]
        )
        for name in report.model_names:
            agg = report.per_model[name].aggregated()
            mean_row = (
                [name]
                + [agg[m]["mean"] for m in metric_fields]
                + [agg[m]["std"] for m in metric_fields]
            )
            writer.writerow(mean_row)

    # paired_t_test.csv — pairwise.
    tests = report.pairwise_tests()
    if tests:
        with (root / "paired_t_test.csv").open("w", newline="", encoding="utf-8") as f:
            dict_writer = csv.DictWriter(f, fieldnames=list(tests[0].keys()))
            dict_writer.writeheader()
            for test_row in tests:
                dict_writer.writerow(test_row)

    # Full JSON report.
    (root / "comparison_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    # Best-effort plot.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = report.model_names
        means = [report.per_model[n].aggregated()[report.metric]["mean"] for n in names]
        stds = [report.per_model[n].aggregated()[report.metric]["std"] for n in names]
        fig, ax = plt.subplots(figsize=(max(6, 2 * len(names)), 4))
        ax.bar(names, means, yerr=stds, capsize=5, color="steelblue")
        ax.set_ylabel(report.metric)
        ax.set_title(f"Per-model {report.metric} (mean ± std, n={report.n_folds} folds)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(root / "comparison_plot.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"comparison plot failed: {e}")


__all__ = ["ComparisonReport", "compare_models_cross_val"]
