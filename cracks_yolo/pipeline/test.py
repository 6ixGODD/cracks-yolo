"""Test pipeline: run inference on test + val splits, compute all metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch

from cracks_yolo.dataset.torchadapter import DetectionDataset
from cracks_yolo.dataset.torchadapter import detection_collate
from cracks_yolo.dataset.transforms import build_transforms
from cracks_yolo.dataset.yolo import YOLOSource
from cracks_yolo.pipeline._helpers import set_seed
from cracks_yolo.zoo import ZOO
from cracks_yolo.zoo.base import BaseModel


def run_test(
    model_name: str,
    weights: Path,
    dataset: str,
    output_dir: Path,
    batch_size: int = 32,
    device: str = "cuda",
    seed: int = 42,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Load a trained model, run inference on test + val, compute metrics.

    Returns a dict with test_metrics + val_metrics.
    """
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name not in ZOO:
        raise ValueError(f"unknown model: {model_name}")

    cls = ZOO[model_name]
    import inspect

    sig = inspect.signature(cls.__init__)
    if "model_name" in sig.parameters:
        model: BaseModel = cls(model_name=model_name, num_classes=1)
    else:
        model = cls(num_classes=1)

    model.load(weights)
    model.to(device)

    src = YOLOSource(dataset)
    isize = model.input_size

    results: dict[str, Any] = {"model": model_name}

    for split_name in ("test", "valid"):
        records = src.load_split(split_name)
        if not records:
            continue
        ds = DetectionDataset(
            records,
            transform=build_transforms(isize, train=False, augment=False),
        )
        from torch.utils.data import DataLoader

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=detection_collate,
        )

        # Run inference
        all_preds: list[dict] = []
        model._inner.eval() if hasattr(model, "_inner") else model.eval()
        with torch.no_grad():
            for imgs, targets in loader:
                imgs = imgs.to(device)
                results_list = model.inference(imgs)
                for b, res in enumerate(results_list):
                    # Dataset target image_id is 0-based; GT records are 1-based.
                    raw_id = (
                        targets[b].get("image_id", torch.tensor(b)).item()
                        if isinstance(targets, list)
                        else b
                    )
                    img_id = int(raw_id) + 1
                    for j in range(len(res.boxes)):
                        bx = res.boxes[j].tolist()
                        all_preds.append({
                            "image_id": img_id,
                            "category_id": int(res.labels[j]) + 1,
                            "bbox": [bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]],
                            "score": float(res.scores[j]),
                        })

        # Compute metrics via pycocotools
        metrics = _compute_metrics(records, all_preds, isize, dataset)
        results[f"{split_name}_metrics"] = metrics
        results[f"{split_name}_predictions"] = all_preds

        # Save predictions
        pred_file = output_dir / f"best_predictions_{split_name}.json"
        pred_file.write_text(json.dumps(all_preds))

    # Save metrics CSV
    _save_metrics_csv(results, output_dir / "metrics.csv")

    # Efficiency
    try:
        analysis = model.analyze(device=device)
        results["efficiency"] = {
            "n_parameters": analysis.n_parameters,
            "gflops": analysis.gflops,
            "fps_mean": analysis.fps_mean,
            "latency_mean_ms": analysis.latency_mean_ms,
            "peak_vram_bytes": analysis.peak_vram_bytes,
        }
    except Exception:
        pass

    print(f"Test complete: {model_name}")
    if "test_metrics" in results:
        m = results["test_metrics"]
        print(f"  test: mAP@0.5={m.get('map50', 0):.4f} mAP@0.5:0.95={m.get('map5095', 0):.4f}")
    if "valid_metrics" in results:
        m = results["valid_metrics"]
        print(f"  val:  mAP@0.5={m.get('map50', 0):.4f} mAP@0.5:0.95={m.get('map5095', 0):.4f}")
    print(f"  output: {output_dir}")
    return results


def _compute_metrics(
    records: list,
    preds: list[dict],
    _isize: int,
    _dataset: str,
) -> dict[str, float]:
    """Compute COCO metrics from predictions + GT."""
    from collections import defaultdict
    import tempfile

    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve

    # Build COCO GT
    images = []
    annotations = []
    ann_id = 1
    fname_to_id: dict[str, int] = {}
    for img_id, rec in enumerate(records, 1):
        fname = Path(rec.image_path).name
        stem = Path(rec.image_path).stem
        fname_to_id[fname] = img_id
        fname_to_id[stem] = img_id
        with Image.open(rec.image_path) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
        for _i, (box, _label) in enumerate(zip(rec.boxes_norm, rec.labels, strict=False)):
            x1, y1, x2, y2 = box
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [
                    round(x1 * w, 2),
                    round(y1 * h, 2),
                    round((x2 - x1) * w, 2),
                    round((y2 - y1) * h, 2),
                ],
                "area": round((x2 - x1) * w * (y2 - y1) * h, 2),
                "iscrowd": 0,
            })
            ann_id += 1
    coco_gt = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "cracks"}],
    }

    # Convert predictions
    coco_dt = []
    for p in preds:
        iid = p["image_id"]
        if isinstance(iid, str):
            iid = fname_to_id.get(iid, fname_to_id.get(iid + ".jpg", 0))
        coco_dt.append({"image_id": iid, "category_id": 1, "bbox": p["bbox"], "score": p["score"]})

    if not coco_dt:
        return {"map50": 0.0, "map5095": 0.0}

    # Run COCO eval
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as gtf:
        json.dump(coco_gt, gtf)
        gt_name = gtf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as dtf:
        json.dump(coco_dt, dtf)
        dt_name = dtf.name

    try:
        cgt = COCO(gt_name)
        cdt = cgt.loadRes(dt_name)
        ev50 = COCOeval(cgt, cdt, "bbox")
        ev50.params.iouThrs = np.array([0.5])
        ev50.evaluate()
        ev50.accumulate()
        ev50.summarize()
        ev = COCOeval(cgt, cdt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        m = {
            "map50": float(ev50.stats[1]),
            "map5095": float(ev.stats[0]),
            "ap75": float(ev.stats[2]),
            "ar1": float(ev.stats[6]),
            "ar10": float(ev.stats[7]),
            "ar100": float(ev.stats[8]),
        }
    except Exception:
        m = {"map50": 0.0, "map5095": 0.0}
    finally:
        Path(gt_name).unlink(missing_ok=True)
        Path(dt_name).unlink(missing_ok=True)

    # PR/ROC
    gt_by_img = defaultdict(list)
    for a in annotations:
        gt_by_img[a["image_id"]].append(a["bbox"])
    dt_by_img = defaultdict(list)
    for d in coco_dt:
        dt_by_img[d["image_id"]].append((d["bbox"], d["score"]))

    scores_list = []
    labels_list = []

    def iou_xywh(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0

    for img_id, gts in gt_by_img.items():
        dts = sorted(dt_by_img.get(img_id, []), key=lambda x: -x[1])
        matched = [False] * len(gts)
        for bbox, score in dts:
            best_iou = 0
            best_idx = -1
            for i, gb in enumerate(gts):
                if matched[i]:
                    continue
                iou = iou_xywh(bbox, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= 0.5 and best_idx >= 0:
                matched[best_idx] = True
                scores_list.append(score)
                labels_list.append(1)
            else:
                scores_list.append(score)
                labels_list.append(0)

    try:
        if scores_list:
            scores_arr = np.array(scores_list)
            labels_arr = np.array(labels_list)
            precision, recall, _ = precision_recall_curve(labels_arr, scores_arr)
            fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
            m["auc_pr"] = float(auc(recall, precision))
            m["auc_roc"] = float(auc(fpr, tpr))
            f1s = 2 * precision * recall / (precision + recall + 1e-12)
            bi = int(np.argmax(f1s))
            m["precision"] = float(precision[bi])
            m["recall"] = float(recall[bi])
            m["f1"] = float(f1s[bi])
            m["sensitivity"] = m["recall"]
            m["ppv"] = m["precision"]
            # Image-level binary classification: crack present vs absent
            gt_img_ids = {a["image_id"] for a in annotations}
            dt_img_ids = {d["image_id"] for d in coco_dt if d.get("score", 0) >= 0.25}
            n_pos_imgs = len(gt_img_ids)
            n_neg_imgs = len(records) - n_pos_imgs
            fp_img = len(dt_img_ids - gt_img_ids)
            fn_img = len(gt_img_ids - dt_img_ids)
            tn_img = n_neg_imgs - fp_img
            m["specificity"] = tn_img / (tn_img + fp_img) if (tn_img + fp_img) > 0 else 1.0
            m["npv"] = tn_img / (tn_img + fn_img) if (tn_img + fn_img) > 0 else 1.0
        else:
            m["auc_pr"] = 0.0
            m["auc_roc"] = 0.5
            m["precision"] = 0.0
            m["recall"] = 0.0
            m["f1"] = 0.0
    except Exception:
        m["auc_pr"] = 0.0
        m["auc_roc"] = 0.5
        m["precision"] = 0.0
        m["recall"] = 0.0
        m["f1"] = 0.0

    return m


def _save_metrics_csv(results: dict, path: Path) -> None:
    """Save metrics to CSV."""
    fields = [
        "model",
        "map50",
        "map5095",
        "ap75",
        "precision",
        "recall",
        "f1",
        "ar1",
        "ar10",
        "ar100",
        "auc_pr",
        "auc_roc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for split in ("test", "valid"):
            key = f"{split}_metrics" if split != "valid" else "valid_metrics"
            if key in results:
                row = {"model": f"{results['model']}_{split}"}
                row.update(results[key])
                w.writerow({k: row.get(k, "") for k in fields})
