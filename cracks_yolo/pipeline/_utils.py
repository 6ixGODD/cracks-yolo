"""Shared helpers for the pipeline implementation modules.

Kept separate so :mod:`cracks_yolo.pipeline.train`, :mod:`...test`,
:mod:`...crossval`, and :mod:`...compare` can all import the same
target-conversion / NMS / seeding helpers without circular imports.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from cracks_yolo.metrics.schemas import DetectionMetric
from cracks_yolo.metrics.schemas import PerImageDetection


def is_anchor_free_model(model: Any) -> bool:
    """True if the model's decode output is anchor-free (v8/v10 style).

    Prefers the model's declared ``decode_format`` attribute; falls back to
    class-name detection for backward compatibility with models that haven't
    been migrated to the self-description Protocol.
    """
    fmt = getattr(model, "decode_format", None)
    if fmt is not None:
        return bool(fmt == "anchor_free")
    return type(model).__name__.startswith(("YOLOv8", "YOLOv10"))


def is_v7_model(model: Any) -> bool:
    """True if the model's loss historically expected the ``imgs`` kwarg.

    Deprecated: the Protocol now requires every ``compute_loss`` to accept
    ``imgs`` (ignored if unused). Pipelines always pass ``imgs=images``.
    Kept for backward compat with external callers.
    """
    return type(model).__name__.startswith("YOLOv7")


def targets_to_yolo(
    targets: list[dict[str, torch.Tensor]],
    image_size: int,
) -> torch.Tensor:
    """Convert list-of-targets-dicts to the (N, 6) YOLO target tensor.

    YOLO target format: (img_idx, cls, x_center, y_center, w, h) normalized.
    Input boxes are xyxy absolute pixels.
    """
    rows: list[list[float]] = []
    for img_idx, t in enumerate(targets):
        if t["boxes"].numel() == 0:
            continue
        boxes = t["boxes"].float() / image_size
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


def detections_to_per_image(
    decoded: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    image_size: int,  # noqa: ARG001 — kept for API symmetry with targets_to_yolo
    conf_thr: float = 0.25,
    iou_thr: float = 0.6,
    is_anchor_free: bool = True,
) -> list[PerImageDetection]:
    """Run NMS + format decoded predictions into PerImageDetection records.

    Handles both anchor-based (B, N, nc+5) and anchor-free (B, 4+nc, N)
    output shapes.
    """
    from torchvision.ops import nms

    if decoded.dim() != 3:
        return []
    n_images = decoded.shape[0]
    out: list[PerImageDetection] = []
    for b in range(n_images):
        if is_anchor_free:
            preds = decoded[b].permute(1, 0).cpu().float()
        else:
            preds = decoded[b].cpu().float()
        # Anchor-based (v5/v7) decode emits (cx, cy, w, h, obj, cls...) —
        # convert to xyxy. Anchor-free (v8/v10) decode already emits xyxy.
        if not is_anchor_free and preds.shape[1] >= 4:
            cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            preds = preds.clone()
            preds[:, 0] = cx - w / 2
            preds[:, 1] = cy - h / 2
            preds[:, 2] = cx + w / 2
            preds[:, 3] = cy + h / 2
        boxes = preds[:, :4]
        if preds.shape[1] > 5:
            scores = preds[:, 4] * preds[:, 5:].max(dim=1).values
            class_ids = preds[:, 5:].argmax(dim=1)
        else:
            scores = preds[:, 4]
            class_ids = torch.zeros(preds.shape[0], dtype=torch.long)

        keep = scores >= conf_thr
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        # Drop degenerate boxes (x2<=x1 or y2<=y1) — low-quality anchor
        # predictions that survive NMS because their IoU with everything is 0.
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        if boxes.numel() == 0:
            det_list: list[DetectionMetric] = []
        else:
            keep_idx = nms(boxes, scores, iou_thr)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            class_ids = class_ids[keep_idx]
            det_list = [
                {
                    "image_id": int(_extract_image_id(targets[b])),
                    "class_id": int(class_ids[k].item()),
                    "score": float(scores[k].item()),
                    "bbox_xyxy": (
                        float(boxes[k][0]),
                        float(boxes[k][1]),
                        float(boxes[k][2]),
                        float(boxes[k][3]),
                    ),
                }
                for k in range(boxes.shape[0])
            ]
        gt_list: list[DetectionMetric] = [
            {
                "image_id": int(_extract_image_id(targets[b])),
                "class_id": int(targets[b]["labels"][k].item()),
                "score": 1.0,
                "bbox_xyxy": (
                    float(targets[b]["boxes"][k][0]),
                    float(targets[b]["boxes"][k][1]),
                    float(targets[b]["boxes"][k][2]),
                    float(targets[b]["boxes"][k][3]),
                ),
            }
            for k in range(targets[b]["boxes"].shape[0])
        ]
        out.append({
            "image_id": _extract_image_id(targets[b]),
            "detections": det_list,
            "ground_truths": gt_list,
        })
    return out


def _extract_image_id(target: dict[str, Any]) -> int:
    v = target["image_id"]
    if isinstance(v, torch.Tensor):
        return int(v.item())
    return int(v)


def set_seed(seed: int) -> None:
    """Seed Python / numpy / torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(preferred: str = "cuda") -> torch.device:
    """Return ``cuda`` if available and requested, else ``cpu``."""
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    return torch.device("cpu")


__all__ = [
    "detections_to_per_image",
    "is_anchor_free_model",
    "is_v7_model",
    "pick_device",
    "set_seed",
    "targets_to_yolo",
]
