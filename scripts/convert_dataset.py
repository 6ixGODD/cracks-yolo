"""CLI entry: convert YOLO <-> COCO dataset formats.

Examples
--------
    python -m scripts.convert_dataset --input data/Crack --from yolo --to coco --output data/Crack_coco
    python -m scripts.convert_dataset --input data/Crack_coco/instances_train.json --image-dir data/Crack_coco/train --from coco --to yolo --output data/Crack_yolo/labels/train
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cracks_yolo.dataset.convert import coco_to_yolo
from cracks_yolo.dataset.convert import yolo_to_coco


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert between YOLO and COCO formats.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--from", dest="src_fmt", choices=["yolo", "coco"], required=True)
    parser.add_argument("--to", dest="dst_fmt", choices=["yolo", "coco"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image-dir", type=Path, default=None, help="Image dir (coco -> yolo only)"
    )
    args = parser.parse_args()

    if args.src_fmt == "yolo" and args.dst_fmt == "coco":
        paths = yolo_to_coco(args.input, args.output)
        for split, p in paths.items():
            print(f"  {split}: {p}")
    elif args.src_fmt == "coco" and args.dst_fmt == "yolo":
        if args.image_dir is None:
            raise SystemExit("--image-dir required for coco -> yolo")
        n = coco_to_yolo(args.input, args.image_dir, args.output)
        print(f"  wrote {n} label files to {args.output}")
    else:
        raise SystemExit(f"unsupported conversion: {args.src_fmt} -> {args.dst_fmt}")


if __name__ == "__main__":
    main()
