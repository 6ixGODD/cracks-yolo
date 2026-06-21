"""CLI entry: train a single model or run 5-fold cross-validation.

Examples
--------
Train YOLOv5s for 100 epochs:
    python -m scripts.train --model yolov5s --dataset data/Crack --epochs 100 --output-dir output/yolov5s

Run 2-fold CV on yolov5s_sactr with batch_size=2 (smoke test):
    python -m scripts.train --model yolov5s_sactr --dataset data/Crack --epochs 2 --batch-size 2 --cross-val --n-folds 2 --output-dir output/smoke_cv
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from typing import cast

from cracks_yolo.dataset.torchadapter import DetectionDataset
from cracks_yolo.dataset.torchadapter import build_dataloader
from cracks_yolo.dataset.transforms import build_transforms
from cracks_yolo.dataset.yolo import YOLOSource
from cracks_yolo.pipeline import TrainConfig
from cracks_yolo.pipeline import TrainPipelineImpl
from cracks_yolo.pipeline.crossval import run_cross_validation
from cracks_yolo.zoo import ZOO
from cracks_yolo.zoo.base import DetectorModel


class _PretrainedModel(Protocol):
    """Type narrowing for zoo classes that support ``from_pretrained``."""

    def __call__(self, num_classes: int, input_size: int = 640) -> DetectorModel: ...

    @classmethod
    def from_pretrained(
        cls, num_classes: int, weights_dir: Path | None = None, strict: bool = False
    ) -> DetectorModel: ...


def _instantiate(
    cls: type[object],
    num_classes: int,
    input_size: int,
    pretrained: bool,
    weights_dir: Path,
) -> DetectorModel:
    """Build a fresh model instance, optionally loading COCO pretrained weights."""
    if pretrained:
        loader = cast(Callable[..., DetectorModel], getattr(cls, "from_pretrained", None))
        if loader is None:
            raise AttributeError(
                f"{cls.__name__} does not implement from_pretrained; cannot use --pretrained."
            )
        return loader(num_classes=num_classes, weights_dir=weights_dir, strict=False)
    builder = cast(Callable[..., DetectorModel], cls)
    return builder(num_classes=num_classes, input_size=input_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a cracks_yolo model.")
    parser.add_argument("--model", required=True, help="ZOO key (e.g. yolov5s_sactr)")
    parser.add_argument("--dataset", required=True, help="YOLO dataset root (contains data.yaml)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=None,
        help="Max grad norm for gradient clipping (stabilizes AMP + higher lr).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop training after N epochs without val mAP@50 improvement.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument(
        "--cross-val",
        action="store_true",
        help="Run N-fold CV instead of a single train/val split.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="CV mode only: fraction of each fold's training pool carved out as "
        "validation for backprop. Held-out fold is always the test set.",
    )
    parser.add_argument(
        "--val-split", default="valid", help="Split name for single-run validation."
    )
    parser.add_argument("--train-split", default="train", help="Split name for training.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load official COCO pretrained weights via from_pretrained (strict=False).",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Directory for cached pretrained .pt files.",
    )
    args = parser.parse_args()

    if args.model not in ZOO:
        raise SystemExit(f"unknown model: {args.model!r}. Available: {sorted(ZOO.keys())}")
    cls = ZOO[args.model]

    src = YOLOSource(args.dataset)

    if args.cross_val:
        # CV mode: merge ALL dataset splits (train + valid + test) into a
        # single pool. The original split is ignored — StratifiedKFold
        # re-partitions the pool into N folds; held-out fold = test,
        # remaining records further split into train + val (val_fraction).
        records = src.load_split("train") + src.load_split("valid") + src.load_split("test")
        train_cfg = TrainConfig(
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            amp=args.amp,
            clip_grad_norm=args.clip_grad_norm,
            early_stopping_patience=args.early_stopping_patience,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            val_interval=args.val_interval,
            log_every_n_steps=args.log_every_n_steps,
        )

        def _factory() -> DetectorModel:
            return _instantiate(
                cls,
                num_classes=src.num_classes,
                input_size=args.input_size,
                pretrained=args.pretrained,
                weights_dir=args.weights_dir,
            )

        cv_report = run_cross_validation(
            model_factory=_factory,
            records=records,
            input_size=args.input_size,
            train_cfg=train_cfg,
            n_folds=args.n_folds,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_fraction=args.val_fraction,
        )
        print(f"CV complete: {args.output_dir}/cv_summary.csv")
        print(f"aggregated map50: {cv_report.aggregated()['map50']}")
        return

    # Single train/val run.
    train_records = src.load_split(args.train_split)
    val_records = src.load_split(args.val_split)
    train_ds = DetectionDataset(
        train_records,
        transform=build_transforms(args.input_size, train=True, augment=True),
    )
    val_ds = DetectionDataset(
        val_records,
        transform=build_transforms(args.input_size, train=False, augment=False),
    )
    train_loader = build_dataloader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.device.startswith("cuda"),
    )

    cfg = TrainConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=args.amp,
        clip_grad_norm=args.clip_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        val_interval=args.val_interval,
        log_every_n_steps=args.log_every_n_steps,
    )
    model = _instantiate(
        cls,
        num_classes=src.num_classes,
        input_size=args.input_size,
        pretrained=args.pretrained,
        weights_dir=args.weights_dir,
    )
    report = TrainPipelineImpl().run(model, train_loader, val_loader, cfg)
    print(f"Train complete: best_epoch={report.best_epoch} best_map50={report.best_map50:.4f}")
    print(f"  output: {args.output_dir}")


if __name__ == "__main__":
    main()
