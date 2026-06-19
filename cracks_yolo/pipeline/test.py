"""TestPipelineImpl — real evaluation loop.

Forwards each batch through ``model.forward`` → ``model.decode`` → NMS,
accumulates :class:`PerImageDetection` records, then runs the
:class:`COCOMetricsCalculator` to produce a full :class:`MetricReport`.

Artifacts written to ``cfg.output_dir``:
- ``metrics.csv`` — flat scalar metrics (one row).
- ``per_image/<image_id>.json`` — detections + ground truths per image.
- ``predictions/<image_id>.jpg`` — input image with predicted boxes drawn.
- ``curves/{pr,roc,confusion}.png`` — curve plots (best-effort, skipped if
  the viz module is unavailable).
- ``run.log.jsonl`` — TestLog record.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

from loguru import logger
import torch

from cracks_yolo.logging.configure import configure_logger
from cracks_yolo.logging.schema import TestLog
from cracks_yolo.metrics.calculator import COCOMetricsCalculator
from cracks_yolo.metrics.schemas import MetricReport
from cracks_yolo.metrics.schemas import PerImageDetection
from cracks_yolo.pipeline._utils import detections_to_per_image
from cracks_yolo.pipeline._utils import pick_device
from cracks_yolo.pipeline.protocol import TestConfig
from cracks_yolo.pipeline.protocol import TestReport
from cracks_yolo.zoo.base import DetectorModel


class TestPipelineImpl:
    """Real test pipeline. Satisfies the ``TestPipeline`` Protocol."""

    def run(
        self,
        model: DetectorModel,
        test_loader: Any,
        cfg: TestConfig,
    ) -> TestReport:
        """Evaluate ``model`` on ``test_loader`` and emit all artifacts."""
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        configure_logger(cfg.output_dir, level="INFO", stderr=True)

        device = pick_device(cfg.device)
        model = model.to(device)
        model.eval()

        per_image_dir = cfg.output_dir / "per_image"
        predictions_dir = cfg.output_dir / "predictions"
        curves_dir = cfg.output_dir / "curves"
        per_image_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        curves_dir.mkdir(parents=True, exist_ok=True)

        decode_format: str = getattr(model, "decode_format", "anchor_based")
        anchor_free = decode_format == "anchor_free"
        num_classes = int(getattr(model, "num_classes", 1))
        calc = COCOMetricsCalculator(
            num_classes=num_classes,
            iou_threshold=0.5,
            conf_threshold=cfg.conf_thr,
        )

        all_per_image: list[PerImageDetection] = []
        start_time = time.time()
        image_index = 0

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                preds = model(images)
                decoded = model.decode(preds)
                if not isinstance(decoded, torch.Tensor):
                    continue  # type: ignore[unreachable]
                per_image = detections_to_per_image(
                    decoded,
                    targets,
                    model.input_size,
                    conf_thr=cfg.conf_thr,
                    iou_thr=cfg.iou_thr,
                    is_anchor_free=anchor_free,
                )
                calc.update(per_image)
                for rec in per_image:
                    self._write_per_image(rec, per_image_dir, image_index)
                    try:
                        self._write_prediction_image(images, rec, predictions_dir, image_index)
                    except Exception as e:
                        logger.warning(f"prediction drawing failed for {image_index}: {e}")
                    image_index += 1
                all_per_image.extend(per_image)

        report = calc.run()
        self._write_metrics_csv(report, cfg.output_dir / "metrics.csv")
        self._write_curves(all_per_image, curves_dir, num_classes, cfg)

        elapsed = time.time() - start_time
        record: TestLog = {
            "record_type": "test",
            "map50": report.map50,
            "map5095": report.map5095,
            "per_class_ap": [v for _, v in sorted(report.per_class_ap.items())],
            "precision": report.precision,
            "recall": report.recall,
            "f1": report.f1,
            "elapsed_sec": elapsed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        logger.bind(**record).info("test")

        return TestReport(
            output_dir=cfg.output_dir,
            metrics=report,
            elapsed_sec=elapsed,
        )

    def _write_per_image(
        self,
        rec: PerImageDetection,
        out_dir: Path,
        idx: int,
    ) -> None:
        path = out_dir / f"{idx:06d}.json"
        path.write_text(json.dumps(rec, indent=2, default=_json_default), encoding="utf-8")

    def _write_prediction_image(
        self,
        images: torch.Tensor,
        rec: PerImageDetection,
        out_dir: Path,
        idx: int,
    ) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            return
        b = idx % images.shape[0]
        img = images[b].detach().cpu().float().numpy()
        img = (img * 255.0).clip(0, 255).astype("uint8")
        img = np.transpose(img, (1, 2, 0))
        img = np.ascontiguousarray(img[:, :, ::-1].copy())  # RGB -> BGR for cv2
        for d in rec["detections"]:
            x1, y1, x2, y2 = (int(v) for v in d["bbox_xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{d['score']:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        for g in rec["ground_truths"]:
            x1, y1, x2, y2 = (int(v) for v in g["bbox_xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), img)

    def _write_metrics_csv(self, report: MetricReport, path: Path) -> None:
        fields = [
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
            "iou_threshold",
            "conf_threshold",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow({k: getattr(report, k) for k in fields})

    def _write_curves(
        self,
        per_image: list[PerImageDetection],
        curves_dir: Path,
        num_classes: int,
        cfg: TestConfig,
    ) -> None:
        try:
            from cracks_yolo.metrics.curves import compute_pr_curve
            from cracks_yolo.metrics.curves import compute_roc_curve
        except ImportError:
            return
        det_arr = [
            (d["image_id"], d["class_id"], d["score"], d["bbox_xyxy"])
            for img in per_image
            for d in img["detections"]
        ]
        gt_arr = [
            (g["image_id"], g["class_id"], g["bbox_xyxy"])
            for img in per_image
            for g in img["ground_truths"]
        ]
        try:
            p, r, _ = compute_pr_curve(det_arr, gt_arr, iou_thr=0.5)
            fpr, tpr, _ = compute_roc_curve(det_arr, gt_arr, iou_thr=0.5)
        except Exception as e:
            logger.warning(f"curve computation failed: {e}")
            return
        try:
            from cracks_yolo.viz.curves import plot_pr_curve
            from cracks_yolo.viz.curves import plot_roc_curve

            plot_pr_curve(p, r, curves_dir / "pr.png")
            plot_roc_curve(fpr, tpr, curves_dir / "roc.png")
        except Exception as e:
            logger.warning(f"curve plotting failed: {e}")
        try:
            from cracks_yolo.metrics.confusion import compute_confusion_matrix
            from cracks_yolo.viz.confusion import plot_confusion_matrix

            cm = compute_confusion_matrix(
                detections=det_arr,
                ground_truths=gt_arr,
                iou_thr=cfg.iou_thr,
                num_classes=num_classes,
                conf_thr=cfg.conf_thr,
            )
            plot_confusion_matrix(
                cm, [f"cls_{i}" for i in range(num_classes)], curves_dir / "confusion.png"
            )
        except Exception as e:
            logger.warning(f"confusion matrix plotting failed: {e}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


__all__ = ["TestPipelineImpl"]
