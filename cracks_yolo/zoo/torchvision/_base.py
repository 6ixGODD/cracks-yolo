"""Shared base for torchvision detectors.

Each model file (retinanet.py, faster_rcnn.py, etc.) inherits from
TorchvisionBase and implements its own train_model + inference with
model-specific score thresholds, coordinate scaling, and loss handling.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn

from cracks_yolo.zoo.base import BaseModel
from cracks_yolo.zoo.base import InferenceResult
from cracks_yolo.zoo.base import ModelState
from cracks_yolo.zoo.base import TrainConfig
from cracks_yolo.zoo.base import TrainReport


class TorchvisionBase(BaseModel):
    """Shared base for torchvision detection models.

    Subclasses set ``self._inner`` to the torchvision model and implement:
        - ``train_model(config)``: own training loop
        - ``inference(images)``: own decode logic
    """

    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, logger=logger)
        self._inner: nn.Module | None = None

    @property
    def stride(self) -> torch.Tensor:
        return torch.tensor([8.0, 16.0, 32.0], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> Any:
        if self._inner is None:
            raise RuntimeError("inner model not built")
        return self._inner(x)

    def _yolo_targets_to_tv(
        self,
        targets: torch.Tensor,
        images: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Convert (N, 6) YOLO targets → torchvision list[dict] format.

        YOLO: (img_idx, cls, xc, yc, w, h) normalized
        torchvision: [{boxes: (M, 4) xyxy abs, labels: (M,)}]
        """
        size = images.shape[-1]
        num_images = images.shape[0]
        per_image: list[list[torch.Tensor]] = [[] for _ in range(num_images)]
        per_labels: list[list[int]] = [[] for _ in range(num_images)]
        if targets.numel() > 0:
            for row in targets:
                img_idx = int(row[0].item())
                cls = int(row[1].item())
                xc, yc, bw, bh = (float(v) for v in row[2:6].tolist())
                x1 = (xc - bw / 2) * size
                y1 = (yc - bh / 2) * size
                x2 = (xc + bw / 2) * size
                y2 = (yc + bh / 2) * size
                if 0 <= img_idx < num_images:
                    per_image[img_idx].append(
                        torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=images.device)
                    )
                    per_labels[img_idx].append(cls + 1)
        result: list[dict[str, torch.Tensor]] = []
        for img_idx in range(num_images):
            if per_image[img_idx]:
                boxes = torch.stack(per_image[img_idx], dim=0)
                labels = torch.tensor(per_labels[img_idx], dtype=torch.int64, device=images.device)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=images.device)
                labels = torch.zeros(0, dtype=torch.int64, device=images.device)
            target: dict[str, torch.Tensor] = {"boxes": boxes, "labels": labels}
            if getattr(self, "_needs_masks", False):
                target["masks"] = torch.ones(
                    (len(boxes), size, size), dtype=torch.uint8, device=images.device
                )
            result.append(target)
        return result

    def _tv_output_to_inference(
        self,
        out: list[dict],
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> list[InferenceResult]:
        """Convert torchvision list[dict] output → list[InferenceResult].

        Scales boxes from model input space to original image size if needed.
        """
        results: list[InferenceResult] = []
        for det in out:
            if not isinstance(det, dict):
                results.append(
                    InferenceResult(
                        boxes=torch.zeros((0, 4)),
                        scores=torch.zeros(0),
                        labels=torch.zeros(0, dtype=torch.long),
                    )
                )
                continue
            boxes = det.get("boxes", torch.tensor([])).cpu()
            scores = det.get("scores", torch.tensor([])).cpu()
            labels = det.get("labels", torch.tensor([])).cpu()
            if len(boxes) > 0 and (scale_x != 1.0 or scale_y != 1.0):
                boxes = boxes.clone()
                boxes[:, 0] *= scale_x
                boxes[:, 1] *= scale_y
                boxes[:, 2] *= scale_x
                boxes[:, 3] *= scale_y
            # labels from torchvision are 1-indexed (bg=0); convert to 0-indexed
            if len(labels) > 0:
                labels = (labels - 1).clamp(min=0)
            results.append(InferenceResult(boxes=boxes, scores=scores, labels=labels.long()))
        return results

    def _build_optimizer(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> torch.optim.Optimizer:
        """Default optimizer for torchvision detectors (SGD)."""
        return torch.optim.SGD(
            self._inner.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    def _run_train_loop(
        self,
        config: TrainConfig,
        train_loader: Any,
        val_loader: Any | None,
        score_thresh: float = 0.05,
    ) -> TrainReport:
        """Generic training loop for torchvision models.

        Subclasses can override train_model entirely, or call this with
        model-specific score_thresh.
        """
        from cracks_yolo.pipeline._helpers import set_seed

        set_seed(config.seed)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._inner = self._inner.to(device)

        optimizer = self._build_optimizer(lr=config.lr, weight_decay=config.weight_decay)
        scaler = torch.amp.GradScaler("cuda") if config.amp and device.type == "cuda" else None

        best_map50 = -1.0
        best_epoch = 0
        history: list[dict[str, float]] = []
        epochs_since_best = 0
        start_time = time.time()

        for epoch in range(config.epochs):
            # Cosine LR
            if config.cosine_lr and epoch >= config.warmup_epochs:
                progress = (epoch - config.warmup_epochs) / max(
                    1, config.epochs - config.warmup_epochs
                )
                cos_lr = config.lr * (
                    config.cosine_lrf
                    + (1 - config.cosine_lrf) * (1 + math.cos(math.pi * progress)) / 2
                )
                for g in optimizer.param_groups:
                    g["lr"] = cos_lr
            # Warmup
            elif config.warmup_epochs > 0 and epoch < config.warmup_epochs:
                warmup_lr = config.warmup_lr + (config.lr - config.warmup_lr) * (
                    epoch / config.warmup_epochs
                )
                for g in optimizer.param_groups:
                    g["lr"] = warmup_lr

            self._inner.train()
            epoch_loss = 0.0
            n_steps = 0

            for images, targets in train_loader:
                images = images.to(device)
                tv_targets = self._yolo_targets_to_tv(
                    targets
                    if isinstance(targets, torch.Tensor)
                    else self._targets_to_tensor(targets),
                    images,
                )

                optimizer.zero_grad()
                if scaler:
                    with torch.amp.autocast("cuda"):
                        loss_dict = self._inner(images, tv_targets)
                    loss = sum(loss_dict.values())
                    scaler.scale(loss).backward()
                    if config.clip_grad_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self._inner.parameters(),
                            config.clip_grad_norm,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict = self._inner(images, tv_targets)
                    loss = sum(loss_dict.values())
                    loss.backward()
                    if config.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self._inner.parameters(),
                            config.clip_grad_norm,
                        )
                    optimizer.step()

                epoch_loss += float(loss.item())
                n_steps += 1

            mean_loss = epoch_loss / max(n_steps, 1)

            # Validation
            val_map50 = 0.0
            if (
                val_loader and (epoch + 1) % max(1, config.early_stopping_patience or 1) == 0
            ) or val_loader:
                val_map50 = self._validate_torchvision(val_loader, device, score_thresh)

            history.append({"epoch": epoch, "train_loss": mean_loss, "val_map50": val_map50})

            if val_map50 > best_map50:
                best_map50 = val_map50
                best_epoch = epoch
                epochs_since_best = 0
                # Save best
                best_path = config.output_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self._inner.state_dict(),
                        "best_map50": best_map50,
                    },
                    best_path,
                )
            else:
                epochs_since_best += 1

            if self._logger:
                self._logger.info(
                    f"epoch {epoch}: loss={mean_loss:.4f} val_map50={val_map50:.4f} "
                    f"best={best_map50:.4f}@{best_epoch}"
                )

            if (
                config.early_stopping_patience
                and epochs_since_best >= config.early_stopping_patience
            ):
                if self._logger:
                    self._logger.info(f"early stopping at epoch {epoch}")
                break

        # Write metrics.csv
        csv_path = config.output_dir / "metrics.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_map50"])
            w.writeheader()
            for row in history:
                w.writerow(row)

        self._set_state(ModelState.TRAINED)
        elapsed = time.time() - start_time
        final = history[-1] if history else {}
        return TrainReport(
            output_dir=config.output_dir,
            best_epoch=best_epoch,
            best_map50=best_map50,
            final_train_loss=final.get("train_loss", 0.0),
            final_val_map50=final.get("val_map50", 0.0),
            final_val_map5095=0.0,
            total_epochs=len(history),
            elapsed_sec=elapsed,
            checkpoint_path=config.output_dir / "best.pt",
        )

    def _validate_torchvision(
        self,
        val_loader: Any,
        device: torch.device,
        score_thresh: float = 0.05,
    ) -> float:
        """Run validation, return mAP@0.5."""
        if self._inner is None:
            raise RuntimeError("inner model not built")

        from cracks_yolo.metrics.calculator import COCOMetricsCalculator

        # Temporarily lower score threshold for validation
        old_thresh = getattr(self._inner, "score_thresh", None)
        if old_thresh is not None:
            self._inner.score_thresh = min(score_thresh, 0.01)

        self._inner.eval()
        calc = COCOMetricsCalculator(num_classes=self.num_classes, iou_threshold=0.5)
        with torch.no_grad():
            for images, _targets in val_loader:
                images = images.to(device)
                outs = self._inner(images)
                # Convert to per-image for metrics
                for b, out in enumerate(outs if isinstance(outs, list) else [outs]):
                    if not isinstance(out, dict):
                        continue
                    boxes = out.get("boxes", torch.tensor([])).cpu()
                    scores = out.get("scores", torch.tensor([])).cpu()
                    labels = out.get("labels", torch.tensor([])).cpu()
                    for j in range(len(boxes)):
                        if scores[j] > 0.001:
                            calc.update([
                                {
                                    "image_id": b,
                                    "detections": [
                                        {
                                            "image_id": b,
                                            "class_id": int(labels[j]) - 1 if labels[j] > 0 else 0,
                                            "score": float(scores[j]),
                                            "bbox_xyxy": (
                                                float(boxes[j][0]),
                                                float(boxes[j][1]),
                                                float(boxes[j][2]),
                                                float(boxes[j][3]),
                                            ),
                                        }
                                    ],
                                    "ground_truths": [],
                                }
                            ])

        if old_thresh is not None:
            self._inner.score_thresh = old_thresh

        report = calc.run()
        return report.map50

    def _targets_to_tensor(self, targets: list[dict]) -> torch.Tensor:
        """Convert list[dict] targets → (N, 6) tensor."""
        rows: list[list[float]] = []
        size = self.input_size
        for img_idx, t in enumerate(targets):
            if t["boxes"].numel() == 0:
                continue
            boxes = t["boxes"].float() / size
            labels = t["labels"].long()
            for j in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[j].tolist()
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                rows.append([float(img_idx), float(labels[j].item()), xc, yc, w, h])
        if not rows:
            return torch.zeros((0, 6), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def save(self, path: Path, torchscript: bool = False, onnx: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self._inner.state_dict()}, path)
        if torchscript:
            scripted = torch.jit.script(self._inner)
            torch.jit.save(scripted, path.with_suffix(".torchscript"))
        if onnx:
            dummy = torch.zeros(1, 3, self.input_size, self.input_size)
            torch.onnx.export(self._inner.eval(), dummy, path.with_suffix(".onnx"))

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        self._inner.load_state_dict(sd, strict=False)
        self._set_state(ModelState.TRAINED)
