"""Static-graph YOLOv5 inference via TorchScript — drop-in for ``run_test``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cracks_yolo.zoo.base import BaseModel
from cracks_yolo.zoo.base import InferenceResult
from cracks_yolo.zoo.base import ModelState
from cracks_yolo.zoo.base import TrainConfig
from cracks_yolo.zoo.base import TrainReport


class StaticYOLOv5(BaseModel):
    """TorchScript YOLOv5 inference wrapper.  Satisfies the ``BaseModel``
    interface so ``run_test`` works without modification.

    Usage::

        model = StaticYOLOv5(weights="yolov5s-baseline.torchscript")
        model.load(
            Path("yolov5s-baseline.torchscript")
        )  # set state
        predictions = model.inference(batch_tensor)
    """

    def __init__(
        self,
        num_classes: int = 1,
        input_size: int = 640,
        logger: Any = None,
        weights: str | Path = "",
    ) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, logger=logger)
        self._weights = Path(weights)
        self._model: torch.jit.ScriptModule | None = None

    # -- BaseModel abstract methods ----------------------------------------

    def train_model(self, config: TrainConfig) -> TrainReport:
        raise NotImplementedError("static model cannot be trained")

    def inference(self, images: torch.Tensor) -> list[InferenceResult]:
        """images: (B, 3, H, W) float in [0, 1] — same contract as BaseModel."""
        if self._model is None:
            raise RuntimeError("call .load(weights_path) before inference")

        # TS model has mixed devices from old export → run on CPU
        images_cpu = images.cpu()
        results: list[InferenceResult] = []
        with torch.no_grad():
            raw = self._model(images_cpu)
            if isinstance(raw, (list, tuple)):
                raw = raw[0]  # old YOLOv5 outputs (pred, train_out)

        # Old YOLOv5 non_max_suppression — apply per image
        for b in range(images.shape[0]):
            det = _old_yolov5_nms(raw[b : b + 1], conf_thres=0.001, iou_thres=0.7)
            if det is None or len(det) == 0:
                results.append(
                    InferenceResult(
                        boxes=torch.zeros((0, 4)),
                        scores=torch.zeros(0),
                        labels=torch.zeros(0, dtype=torch.long),
                    )
                )
            else:
                d = det[0]
                boxes = d[:, :4].clamp(0, self.input_size)
                results.append(
                    InferenceResult(
                        boxes=boxes.cpu(),
                        scores=d[:, 4].cpu(),
                        labels=d[:, 5].long().cpu(),
                    )
                )
        return results

    def save(self, path: Path, torchscript: bool = False, onnx: bool = False) -> None:
        pass  # already persisted

    def load(self, path: Path) -> None:
        self._model = torch.jit.load(str(path))
        self._set_state(ModelState.TRAINED)

    @classmethod
    def from_pretrained(cls, num_classes: int = 1, **kwargs: Any) -> StaticYOLOv5:
        return cls(num_classes=num_classes, **kwargs)

    @property
    def stride(self) -> torch.Tensor:
        return torch.tensor([8.0, 16.0, 32.0])


# ---------------------------------------------------------------------------
# Old YOLOv5 NMS (self-contained — no dependency on old code at runtime)
# ---------------------------------------------------------------------------


def _old_yolov5_nms(
    pred: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> list[torch.Tensor | None]:
    """Re-implementation of old YOLOv5 non_max_suppression.

    Args:
        pred: (B, N, 6) raw output = (cx, cy, w, h, obj_conf, cls_conf)

    Returns:
        list of (M, 6) tensors in xyxy pixel coordinates, or None.
    """

    bs = pred.shape[0]
    # single-class crack detection
    output: list[torch.Tensor | None] = [torch.zeros((0, 6), device=pred.device)] * bs

    for bi in range(bs):
        x = pred[bi]
        # Confidence = obj * cls (single class)
        x[:, 5:6] = x[:, 4:5]  # cls_conf = obj_conf for single class
        # Filter by confidence
        mask = x[:, 4] > conf_thres
        x = x[mask]
        if not len(x):
            continue

        # Convert cxcywh → xyxy
        box = torch.zeros_like(x[:, :4])
        box[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        box[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        box[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        box[:, 3] = x[:, 1] + x[:, 3] / 2  # y2

        # NMS per class (only 1 class)
        conf = x[:, 4]
        order = conf.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            if len(keep) >= max_det:
                break
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            iou = _box_iou(box[i : i + 1], box[order[1:]])
            mask = iou <= iou_thres
            order = order[1:][mask]

        if keep:
            kept = torch.stack(keep)
            out = torch.cat([box[kept], conf[kept].unsqueeze(1), x[kept, 5:6]], dim=1)
            output[bi] = out

    return output


def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU of xyxy boxes. box1: (1, 4), box2: (N, 4)."""
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1 + area2 - inter + 1e-16)
