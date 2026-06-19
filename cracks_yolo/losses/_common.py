"""Shared loss utilities for the cracks_yolo loss functions.

Provides:
- :func:`bbox_iou` — IoU/GIoU/DIoU/CIoU between two boxes.
- :func:`xywh2xyxy` / :func:`xyxy2xywh` — box format conversions.
- :func:`smooth_BCE` — label-smoothed BCE target pair.
- :class:`BCEBlurWithLogitsLoss`, :class:`FocalLoss`, :class:`QFocalLoss` —
  v5/v7 BCE variants.
- :class:`DFLoss` — Distribution Focal Loss (v8/v10).

Ported verbatim (with type annotations) from ``deps/yolov5/utils/loss.py`` and
``deps/ultralytics/ultralytics/utils/loss.py``. No runtime deps on either.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,  # noqa: N803
    DIoU: bool = False,  # noqa: N803
    CIoU: bool = False,  # noqa: N803
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calculate IoU/GIoU/DIoU/CIoU between ``box1`` and ``box2``.

    Args:
        box1: ``(N, 4)`` boxes in xywh or xyxy depending on ``xywh``.
        box2: ``(M, 4)`` or broadcastable to ``box1``.
        xywh: If True, interpret inputs as ``(cx, cy, w, h)``; else xyxy.
        GIoU/DIoU/CIoU: Which variant to return.
        eps: Numerical stability.

    Returns:
        Per-pair IoU (or variant) tensor.
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou: torch.Tensor = inter / union

    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw**2 + ch**2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return torch.as_tensor(iou - (rho2 / c2 + v * alpha))
            return torch.as_tensor(iou - rho2 / c2)
        c_area = cw * ch + eps
        return torch.as_tensor(iou - (c_area - union) / c_area)
    return iou


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert ``(cx, cy, w, h)`` to ``(x1, y1, x2, y2)``."""
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """Convert ``(x1, y1, x2, y2)`` to ``(cx, cy, w, h)``."""
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def bbox_iou_v7(
    box1: torch.Tensor,
    box2: torch.Tensor,
    x1y1x2y2: bool = True,
    GIoU: bool = False,  # noqa: N803
    DIoU: bool = False,  # noqa: N803
    CIoU: bool = False,  # noqa: N803
    eps: float = 1e-7,
) -> torch.Tensor:
    """YOLOv7-style ``bbox_iou`` — ``box1`` is ``(4, N)``, ``box2`` is ``(N, 4)``.

    The two arguments use different conventions on purpose (matches the
    upstream v7 loss call site: ``bbox_iou(pbox.T, selected_tbox, ...)``).
    Internally ``box2`` is transposed to ``(4, N)`` and elementwise ops
    produce an ``(N,)`` IoU vector.
    """
    box2 = box2.T

    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou: torch.Tensor = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = cw**2 + ch**2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                return torch.as_tensor(iou - rho2 / c2)
            v = (4 / math.pi**2) * torch.pow(
                torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
            )
            with torch.no_grad():
                alpha = v / (v - iou + (1 + eps))
            return torch.as_tensor(iou - (rho2 / c2 + v * alpha))
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou


def smooth_BCE(eps: float = 0.1) -> tuple[float, float]:  # noqa: N802
    """Return label-smoothed BCE targets ``(pos, neg)``.

    pos = ``1 - 0.5 * eps``, neg = ``0.5 * eps``. See
    https://arxiv.org/pdf/1902.04103.pdf.
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified ``BCEWithLogitsLoss`` that reduces missing-label effects."""

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss = loss * alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss wrapping a ``BCEWithLogitsLoss``."""

    def __init__(
        self,
        loss_fcn: nn.BCEWithLogitsLoss,
        gamma: float = 1.5,
        alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = loss * alpha_factor * modulating_factor
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class QFocalLoss(nn.Module):
    """Quality focal loss wrapping a ``BCEWithLogitsLoss``."""

    def __init__(
        self,
        loss_fcn: nn.BCEWithLogitsLoss,
        gamma: float = 1.5,
        alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss = loss * alpha_factor * modulating_factor
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DFLoss(nn.Module):
    """Distribution Focal Loss criterion (v8/v10)."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        loss: torch.Tensor = (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)
        return loss
