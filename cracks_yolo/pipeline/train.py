"""TrainPipelineImpl — real training loop.

Owns the train/val loop, optimizer step, checkpointing, and per-epoch
artifact emission (loss curves, metric curves, config.yaml). Calls only
the ``DetectorModel`` Protocol methods — no model-specific branching.

Returns a :class:`TrainReport` with the path to ``best.pt`` plus summary
metrics.
"""

from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any

from loguru import logger
import torch

from cracks_yolo.logging.configure import configure_logger
from cracks_yolo.logging.schema import TrainEpochLog
from cracks_yolo.logging.schema import TrainStepLog
from cracks_yolo.logging.schema import ValLog
from cracks_yolo.metrics.calculator import COCOMetricsCalculator
from cracks_yolo.pipeline._utils import detections_to_per_image
from cracks_yolo.pipeline._utils import pick_device
from cracks_yolo.pipeline._utils import set_seed
from cracks_yolo.pipeline._utils import targets_to_yolo
from cracks_yolo.pipeline.protocol import TrainConfig
from cracks_yolo.pipeline.protocol import TrainReport
from cracks_yolo.zoo.base import DetectorModel


class TrainPipelineImpl:
    """Real training pipeline. Satisfies the ``TrainPipeline`` Protocol."""

    def run(
        self,
        model: DetectorModel,
        train_loader: Any,
        val_loader: Any | None,
        cfg: TrainConfig,
    ) -> TrainReport:
        """Train ``model`` on ``train_loader``, validate on ``val_loader``.

        See :class:`cracks_yolo.pipeline.protocol.TrainConfig` for options.
        """
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        configure_logger(cfg.output_dir, level="INFO", stderr=True)
        set_seed(cfg.seed)

        device = pick_device(cfg.device)
        model = model.to(device)
        optimizer = model.build_optimizer()
        # Override lr/weight_decay from cfg.
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr
            g["weight_decay"] = cfg.weight_decay

        scaler = (
            torch.amp.GradScaler("cuda")  # type: ignore[attr-defined]
            if cfg.amp and device.type == "cuda"
            else None
        )

        # Save the resolved config for reproducibility.
        (cfg.output_dir / "config.yaml").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

        best_map50 = -1.0
        best_epoch = -1
        checkpoint_paths: list[Path] = []
        history: list[dict[str, float]] = []
        total_steps = 0
        start_time = time.time()

        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss_sum = 0.0
            epoch_box_sum = 0.0
            epoch_cls_sum = 0.0
            epoch_obj_sum = 0.0
            epoch_dfl_sum = 0.0
            epoch_steps = 0

            for step, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                # Re-stack targets on device.
                targets_dev = [
                    {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
                    for t in targets
                ]
                yolo_targets = targets_to_yolo(targets_dev, model.input_size).to(device)

                with torch.amp.autocast("cuda", enabled=scaler is not None):  # type: ignore[attr-defined]
                    preds = model(images)
                    loss, parts = model.compute_loss(preds, yolo_targets, imgs=images)

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                    if cfg.clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()  # type: ignore[no-untyped-call]
                    if cfg.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                    optimizer.step()

                # Interpret parts via the model's declared schema (no class-name branching).
                schema: tuple[str, ...] = getattr(model, "loss_parts_schema", ("box", "cls", "obj"))
                parts_list = parts.detach().tolist()
                parts_map: dict[str, float] = dict(zip(schema, parts_list, strict=False))
                box_l = parts_map.get("box", 0.0)
                cls_l = parts_map.get("cls", 0.0)
                obj_l = parts_map.get("obj")
                dfl_l = parts_map.get("dfl")

                epoch_loss_sum += float(loss.detach().item())
                epoch_box_sum += box_l
                epoch_cls_sum += cls_l
                if obj_l is not None:
                    epoch_obj_sum += obj_l
                if dfl_l is not None:
                    epoch_dfl_sum += dfl_l
                epoch_steps += 1
                total_steps += 1

                if step % cfg.log_every_n_steps == 0:
                    record: TrainStepLog = {
                        "record_type": "train_step",
                        "step": total_steps,
                        "epoch": epoch,
                        "total_loss": float(loss.detach().item()),
                        "box_loss": box_l,
                        "cls_loss": cls_l,
                        "obj_loss": obj_l,
                        "dfl_loss": dfl_l,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    logger.bind(**record).info("train step")

            mean_loss = epoch_loss_sum / max(epoch_steps, 1)
            mean_box = epoch_box_sum / max(epoch_steps, 1)
            mean_cls = epoch_cls_sum / max(epoch_steps, 1)
            mean_obj = epoch_obj_sum / max(epoch_steps, 1) if epoch_obj_sum > 0 else None
            mean_dfl = epoch_dfl_sum / max(epoch_steps, 1) if epoch_dfl_sum > 0 else None

            # Validate.
            val_map50 = 0.0
            val_map5095 = 0.0
            if val_loader is not None and (epoch + 1) % cfg.val_interval == 0:
                val_map50, val_map5095 = self._validate(model, val_loader, device)

            elapsed = time.time() - start_time
            epoch_record: TrainEpochLog = {
                "record_type": "train_epoch",
                "epoch": epoch,
                "mean_total_loss": mean_loss,
                "mean_box_loss": mean_box,
                "mean_cls_loss": mean_cls,
                "mean_obj_loss": mean_obj,
                "mean_dfl_loss": mean_dfl,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_sec": elapsed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            logger.bind(**epoch_record).info("train epoch")
            if val_loader is not None:
                val_record: ValLog = {
                    "record_type": "val",
                    "epoch": epoch,
                    "map50": val_map50,
                    "map5095": val_map5095,
                    "per_class_ap": [],
                    "elapsed_sec": elapsed,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                logger.bind(**val_record).info("val")

            history.append({
                "epoch": epoch,
                "train_loss": mean_loss,
                "box_loss": mean_box,
                "cls_loss": mean_cls,
                "val_map50": val_map50,
                "val_map5095": val_map5095,
            })

            # Save best checkpoint.
            if val_map50 > best_map50:
                best_map50 = val_map50
                best_epoch = epoch
                best_path = cfg.output_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_map50": best_map50,
                    },
                    best_path,
                )
                if best_path not in checkpoint_paths:
                    checkpoint_paths.append(best_path)

        # Write metrics.csv.
        csv_path = cfg.output_dir / "metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()) if history else ["epoch"])
            writer.writeheader()
            for row in history:
                writer.writerow(row)

        # Plot curves.
        try:
            from cracks_yolo.viz.curves import plot_loss_curve
            from cracks_yolo.viz.curves import plot_metric_curve

            plot_loss_curve(csv_path, cfg.output_dir / "loss_curve.png")
            plot_metric_curve(csv_path, cfg.output_dir / "metric_curve.png")
        except Exception as e:
            logger.warning(f"curve plotting failed: {e}")

        elapsed_total = time.time() - start_time
        return TrainReport(
            output_dir=cfg.output_dir,
            best_epoch=best_epoch,
            best_map50=best_map50,
            final_train_loss=history[-1]["train_loss"] if history else 0.0,
            final_val_map50=history[-1]["val_map50"] if history else 0.0,
            final_val_map5095=history[-1]["val_map5095"] if history else 0.0,
            total_steps=total_steps,
            total_epochs=cfg.epochs,
            elapsed_sec=elapsed_total,
            checkpoint_paths=checkpoint_paths,
        )

    def _validate(
        self,
        model: DetectorModel,
        val_loader: Any,
        device: torch.device,
    ) -> tuple[float, float]:
        """Run validation: forward, decode, NMS, compute mAP."""
        model.eval()
        calc = COCOMetricsCalculator(
            num_classes=getattr(model, "num_classes", 1), iou_threshold=0.5
        )
        decode_format: str = getattr(model, "decode_format", "anchor_based")
        anchor_free = decode_format == "anchor_free"
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                preds = model(images)
                decoded = model.decode(preds)
                if not isinstance(decoded, torch.Tensor):
                    continue  # type: ignore[unreachable]
                per_image = detections_to_per_image(
                    decoded,
                    targets,
                    model.input_size,
                    conf_thr=0.25,
                    iou_thr=0.6,
                    is_anchor_free=anchor_free,
                )
                calc.update(per_image)
        report = calc.run()
        return report.map50, report.map5095


__all__ = ["TrainPipelineImpl"]
