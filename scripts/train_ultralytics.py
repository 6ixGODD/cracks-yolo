"""Train any YOLO via ultralytics' official YOLO().train() API.

This uses ultralytics' own training loop (correct loss for each family:
v5 gets v5-style loss, v8 gets v8DetectionLoss, etc.), mosaic augmentation,
cosine lr, EMA, SGD/AdamW — everything the official repo does. SAC/TR
variants are produced by apply_sac_tr() after model construction.

Usage:
    python -m scripts.train_ultralytics --model yolov5s_sactr \\
        --dataset /root/autodl-tmp/CrackDetection_Augmentation.v1.yolov5pytorch \\
        --epochs 1200 --batch-size 64 --device 0
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _write_data_yaml(dataset: Path, out: Path, nc: int = 1) -> None:
    names = {0: "cracks"} if nc == 1 else {i: f"class_{i}" for i in range(nc)}
    out.write_text(
        f"path: {dataset}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnc: {nc}\nnames: {names}\n",
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="ZOO key: yolov5s, yolov5s_sactr, yolov8s_sac, ...")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-root", type=Path, default=Path("output"))
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--cos-lr", action="store_true", default=True)
    p.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")
    p.add_argument("--single-cls", action="store_true", default=True)
    args = p.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = args.output_root / ts / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = out_dir / "data.yaml"
    _write_data_yaml(Path(args.dataset), data_yaml, args.num_classes)

    # Determine the ultralytics model spec: baseline loads from .pt, SAC/TR
    # loads from .yaml + apply_sac_tr.
    from cracks_yolo.zoo import ZOO
    from cracks_yolo.zoo.ultralytics import _CFG_ASSET
    from cracks_yolo.zoo.ultralytics.sac import apply_sac_tr
    from ultralytics import YOLO

    cls = ZOO[args.model]
    cfg = cls.cfg  # e.g. "yolov5s.yaml"
    has_sac = bool(cls.sac_indices)
    has_tr = bool(cls.tr_indices)
    asset = _CFG_ASSET.get(cfg.replace(".yaml", ""), cfg.replace(".yaml", ""))
    is_anchor_based = cls.decode_format == "anchor_based"  # v10

    print(f"=== model={args.model} cfg={cfg} sac={has_sac} tr={has_tr} asset={asset} ===")

    if has_sac or has_tr:
        # Build from YAML, apply SAC/TR, load pretrained, then inject directly
        # into YOLO instance (NO serialization — avoids pickle failure on
        # SAC closure classes).
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import DEFAULT_CFG
        model = DetectionModel(cfg, ch=3, nc=args.num_classes, verbose=False)
        model.args = DEFAULT_CFG
        apply_sac_tr(model, sac_indices=cls.sac_indices, tr_indices=cls.tr_indices)
        # Load pretrained COCO weights (intersect).
        try:
            src = YOLO(f"{asset}.pt").model.state_dict()
            msd = model.state_dict()
            matched = 0
            for k in list(src.keys()):
                if k in msd and src[k].shape == msd[k].shape:
                    msd[k] = src[k]; matched += 1
            model.load_state_dict(msd, strict=False)
            print(f"pretrained: matched {matched}/{len(msd)}")
        except Exception as e:
            print(f"WARNING: pretrained load failed: {e}")
        # Direct injection: build YOLO from cfg, then replace its model.
        trainer = YOLO(cfg)
        trainer.model = model
        print("SAC/TR model injected directly (no serialization)")
    else:
        # Baseline: load pretrained directly.
        trainer = YOLO(f"{asset}.pt")

    # Train via ultralytics' official loop.
    trainer.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img,
        device=args.device,
        workers=args.workers,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        patience=args.patience,
        single_cls=args.single_cls,
        pretrained=True if not (has_sac or has_tr) else False,
        verbose=True,
    )
    print(f"=== training done. results in {out_dir} ===")


if __name__ == "__main__":
    main()
