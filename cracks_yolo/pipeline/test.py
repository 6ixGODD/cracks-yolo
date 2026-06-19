"""TestPipelineImpl — real evaluation loop.

Forwards each batch through ``model.forward`` → ``model.decode`` → NMS,
accumulates :class:`PerImageDetection` records, then runs the
:class:`COCOMetricsCalculator` to produce a full :class:`MetricReport`.

Artifacts written to ``cfg.output_dir``:
- ``metrics.csv`` — flat scalar metrics (one row): detection accuracy +
  efficiency (FPS, latency, params, GFLOPs, peak VRAM).
- ``model_analysis.json`` — full :func:`analyze_model` report (params,
  MACs, GFLOPs, single-image latency, peak VRAM).
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
import statistics
import time
from typing import Any
from typing import cast

from loguru import logger
import torch
import torch.nn as nn

from cracks_yolo.analysis.model import analyze_model
from cracks_yolo.analysis.model import save_model_analysis
from cracks_yolo.logging.configure import configure_logger
from cracks_yolo.logging.schema import TestLog
from cracks_yolo.metrics.calculator import COCOMetricsCalculator
from cracks_yolo.metrics.schemas import MetricReport
from cracks_yolo.metrics.schemas import PerImageDetection
from cracks_yolo.pipeline._utils import detections_to_per_image
from cracks_yolo.pipeline._utils import pick_device
from cracks_yolo.pipeline.protocol import EfficiencyReport
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

        # Efficiency timing: measure real end-to-end inference time
        # (forward + decode + NMS) per batch. File I/O and image drawing are
        # excluded so FPS reflects inference throughput, not artifact writing.
        is_cuda = device.type == "cuda"
        batch_times_ms: list[float] = []
        batch_image_counts: list[int] = []
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                if is_cuda:
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
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
                if is_cuda:
                    torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                batch_times_ms.append((t1 - t0) * 1000.0)
                batch_image_counts.append(int(images.shape[0]))
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
        efficiency = self._measure_efficiency(
            model, cfg, device, batch_times_ms, batch_image_counts
        )
        self._write_metrics_csv(report, cfg.output_dir / "metrics.csv", efficiency)
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
            "n_images": efficiency.n_images if efficiency else 0,
            "fps_mean": efficiency.fps_mean if efficiency else 0.0,
            "latency_mean_ms": efficiency.latency_mean_ms if efficiency else 0.0,
            "gflops": efficiency.gflops if efficiency else 0.0,
            "n_parameters": efficiency.n_parameters if efficiency else 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        logger.bind(**record).info("test")

        return TestReport(
            output_dir=cfg.output_dir,
            metrics=report,
            elapsed_sec=elapsed,
            efficiency=efficiency,
        )

    def _measure_efficiency(
        self,
        model: DetectorModel,
        cfg: TestConfig,
        device: torch.device,
        batch_times_ms: list[float],
        batch_image_counts: list[int],
    ) -> EfficiencyReport | None:
        """Build the EfficiencyReport from real loop timings + analyze_model.

        Returns None when ``cfg.measure_efficiency`` is False.
        """
        if not cfg.measure_efficiency:
            return None

        input_size = int(getattr(model, "input_size", 640))
        total_images = sum(batch_image_counts)
        per_image_latencies = [
            bt / max(n, 1) for bt, n in zip(batch_times_ms, batch_image_counts, strict=True)
        ]
        total_inference_sec = sum(batch_times_ms) / 1000.0
        fps_mean = total_images / total_inference_sec if total_inference_sec > 0 else 0.0
        latency_mean_ms = statistics.fmean(per_image_latencies) if per_image_latencies else 0.0
        latency_p50_ms = _percentile(per_image_latencies, 50.0)
        latency_p95_ms = _percentile(per_image_latencies, 95.0)
        fps_p50 = 1000.0 / latency_p50_ms if latency_p50_ms > 0 else 0.0
        fps_p95 = 1000.0 / latency_p95_ms if latency_p95_ms > 0 else 0.0

        # Real peak VRAM over the inference loop (test batch size); falls back
        # to the synthetic single-image value from analyze_model on CPU.
        real_peak_vram = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )

        # Structural metrics (params, MACs, GFLOPs) via a single-image dummy
        # forward. Modest n_runs keeps the extra cost low; MACs are
        # input-size dependent, not batch-size, so batch=1 is correct here.
        try:
            analysis = analyze_model(
                cast(nn.Module, model),
                input_size=input_size,
                device=str(device),
                n_warmup=1,
                n_runs=5,
            )
            save_model_analysis(analysis, cfg.output_dir)
            n_params = analysis.n_parameters
            n_trainable = analysis.n_trainable_parameters
            macs = analysis.macs
            gflops = analysis.flops
            synthetic_vram = analysis.peak_vram_bytes
        except Exception as e:
            logger.warning(f"analyze_model failed; efficiency structural metrics set to 0: {e}")
            n_params = 0
            n_trainable = 0
            macs = 0.0
            gflops = 0.0
            synthetic_vram = 0

        return EfficiencyReport(
            n_images=total_images,
            fps_mean=fps_mean,
            fps_p50=fps_p50,
            fps_p95=fps_p95,
            latency_mean_ms=latency_mean_ms,
            latency_p50_ms=latency_p50_ms,
            latency_p95_ms=latency_p95_ms,
            n_parameters=n_params,
            n_trainable_parameters=n_trainable,
            macs=macs,
            gflops=gflops,
            peak_vram_bytes=real_peak_vram if real_peak_vram > 0 else synthetic_vram,
            input_size=input_size,
            device=str(device),
            batch_size=cfg.batch_size,
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

    def _write_metrics_csv(
        self,
        report: MetricReport,
        path: Path,
        efficiency: EfficiencyReport | None,
    ) -> None:
        accuracy_fields = [
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
        efficiency_fields = [
            "n_images",
            "fps_mean",
            "fps_p50",
            "fps_p95",
            "latency_mean_ms",
            "latency_p50_ms",
            "latency_p95_ms",
            "n_parameters",
            "n_trainable_parameters",
            "macs",
            "gflops",
            "peak_vram_bytes",
            "input_size",
            "device",
            "batch_size",
        ]
        fields = accuracy_fields + efficiency_fields
        row: dict[str, object] = {k: getattr(report, k) for k in accuracy_fields}
        if efficiency is not None:
            row.update({k: getattr(efficiency, k) for k in efficiency_fields})
        else:
            row.update(dict.fromkeys(efficiency_fields, 0))
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

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


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile of ``values`` (no sort mutation)."""
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


__all__ = ["TestPipelineImpl"]
