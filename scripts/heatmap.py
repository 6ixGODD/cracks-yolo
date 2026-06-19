"""CLI entry: Grad-CAM heatmaps over backbone layers for a trained model.

Examples
--------
    python -m scripts.heatmap --model yolov5s --weights output/yolov5s/best.pt --input data/Crack/test --layers backbone.8,backbone.10 --output-dir output/heatmaps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from cracks_yolo.dataset.transforms import build_transforms
from cracks_yolo.viz.heatmap import GradCAMExtractor
from cracks_yolo.viz.heatmap import save_heatmap_overlay
from cracks_yolo.zoo import ZOO


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps.")
    parser.add_argument("--model", required=True, help="ZOO key")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    parser.add_argument("--input", type=Path, required=True, help="Image dir or single image")
    parser.add_argument(
        "--layers", required=True, help="Comma-separated module paths (e.g. backbone.8,backbone.10)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-images", type=int, default=0, help="Max images to process (0 = all)")
    args = parser.parse_args()

    if args.model not in ZOO:
        raise SystemExit(f"unknown model: {args.model!r}")
    cls = ZOO[args.model]
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = cls(num_classes=1, input_size=args.input_size)
    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    transform = build_transforms(args.input_size, train=False, augment=False)
    image_paths: list[Path]
    if args.input.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        image_paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in exts)
    else:
        image_paths = [args.input]
    if args.num_images > 0:
        image_paths = image_paths[: args.num_images]

    with GradCAMExtractor(model, layers) as ge:
        for i, img_path in enumerate(image_paths):
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                image_tensor, _, _ = transform(rgb, [], [])
            image_tensor = image_tensor.unsqueeze(0).to(device)
            try:
                heatmaps = ge.generate(image_tensor, target_class=0)
            except Exception as e:
                print(f"  [{i}] {img_path.name} failed: {e}")
                continue
            for layer, cam in heatmaps.items():
                layer_dir = args.output_dir / layer
                layer_dir.mkdir(parents=True, exist_ok=True)
                np.save(layer_dir / f"{img_path.stem}.npy", cam)
                save_heatmap_overlay(image_tensor, cam, layer_dir / f"{img_path.stem}.png")
            print(f"  [{i}] {img_path.name} -> {len(heatmaps)} layer(s)")

    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
