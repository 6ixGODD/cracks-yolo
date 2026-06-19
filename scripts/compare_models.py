"""CLI entry: multi-model comparison via N-fold CV with paired statistical tests.

Example:
    python -m scripts.compare_models --models yolov5s,yolov5s_sactr,yolov8s,yolov10s \
        --dataset data/Crack --n-folds 5 --epochs 100 --output-dir output/comparison
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import cast

from cracks_yolo.dataset.yolo import YOLOSource
from cracks_yolo.pipeline import TrainConfig
from cracks_yolo.pipeline.compare import compare_models_cross_val
from cracks_yolo.zoo import ZOO
from cracks_yolo.zoo.base import DetectorModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple models via N-fold CV.")
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated ZOO keys (e.g. yolov5s,yolov5s_sactr)",
    )
    parser.add_argument("--dataset", required=True, help="YOLO dataset root (data.yaml)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--metric", default="map50", help="Metric for pairwise tests.")
    parser.add_argument("--train-split", default="train")
    args = parser.parse_args()

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for k in keys:
        if k not in ZOO:
            raise SystemExit(f"unknown model: {k!r}. Available: {sorted(ZOO.keys())}")

    src = YOLOSource(args.dataset)
    records = src.load_split(args.train_split)

    train_cfg = TrainConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=args.amp,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        val_interval=args.val_interval,
        log_every_n_steps=args.log_every_n_steps,
    )

    def _make_factory(key: str) -> Callable[[], DetectorModel]:
        cls = ZOO[key]

        def _factory() -> DetectorModel:
            return cast(DetectorModel, cls(num_classes=src.num_classes, input_size=args.input_size))

        return _factory

    model_factories: dict[str, Callable[[], DetectorModel]] = {k: _make_factory(k) for k in keys}

    report = compare_models_cross_val(
        model_factories=model_factories,
        records=records,
        input_size=args.input_size,
        train_cfg=train_cfg,
        n_folds=args.n_folds,
        seed=args.seed,
        metric=args.metric,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Comparison complete: {args.output_dir}")
    for name in report.model_names:
        agg = report.per_model[name].aggregated()
        m = agg.get(args.metric, {"mean": 0.0, "std": 0.0})
        print(f"  {name}: {args.metric}={m['mean']:.4f} ± {m['std']:.4f}")
    print(f"pairwise tests: {args.output_dir}/paired_t_test.csv")


if __name__ == "__main__":
    main()
