"""CLI entry: run a trained model on a test split.

Examples
--------
    python -m scripts.test --model yolov5s --weights output/yolov5s/best.pt --dataset data/Crack --output-dir output/yolov5s_test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch

from cracks_yolo.dataset.torchadapter import DetectionDataset
from cracks_yolo.dataset.torchadapter import build_dataloader
from cracks_yolo.dataset.transforms import build_transforms
from cracks_yolo.dataset.yolo import YOLOSource
from cracks_yolo.pipeline import TestConfig
from cracks_yolo.pipeline import TestPipelineImpl
from cracks_yolo.zoo import ZOO
from cracks_yolo.zoo.base import DetectorModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained cracks_yolo model.")
    parser.add_argument("--model", required=True, help="ZOO key")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    parser.add_argument("--dataset", required=True, help="YOLO dataset root (contains data.yaml)")
    parser.add_argument("--split", default="test", help="Split name to evaluate (default: test)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf-thr", type=float, default=0.001)
    parser.add_argument("--iou-thr", type=float, default=0.6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.model not in ZOO:
        raise SystemExit(f"unknown model: {args.model!r}")
    cls = ZOO[args.model]

    src = YOLOSource(args.dataset)
    records = src.load_split(args.split)
    ds = DetectionDataset(
        records,
        transform=build_transforms(args.input_size, train=False, augment=False),
    )
    loader = build_dataloader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.device.startswith("cuda"),
    )

    model = cast(DetectorModel, cls(num_classes=src.num_classes, input_size=args.input_size))
    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)

    cfg = TestConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        conf_thr=args.conf_thr,
        iou_thr=args.iou_thr,
        num_workers=args.num_workers,
    )
    report = TestPipelineImpl().run(model, loader, cfg)
    print(f"Test complete: map50={report.metrics.map50:.4f} map5095={report.metrics.map5095:.4f}")
    print(f"  output: {args.output_dir}")


if __name__ == "__main__":
    main()
