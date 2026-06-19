"""torchvision detection model wrappers.

torchvision's detection models have a fundamentally different API from the
YOLO family:
- In train mode, ``model(images, targets)`` returns a ``dict[str, Tensor]``
  of named losses (``loss_classifier``, ``loss_box_reg``, ...).
- In eval mode, ``model(images)`` returns a ``list[dict]`` with ``boxes``,
  ``labels``, ``scores`` per image.

This module adapts that API to the :class:`cracks_yolo.zoo.base.DetectorModel`
Protocol so the existing train/test pipelines can consume RetinaNet,
Faster R-CNN, Mask R-CNN, FCOS, SSD300, and SSDlite320 as cross-paradigm
comparison baselines without any model-specific branching.

Adaptation strategy:
- ``forward(x)`` in train mode stashes ``x`` and returns a dict ``{"_tv_images": x}``.
  In eval mode it calls the inner model and returns the ``list[dict]``.
- ``compute_loss(preds, targets, imgs=None)`` extracts images from ``preds``
  (or ``imgs``), converts ``(N, 6)`` YOLO targets to ``list[dict]`` with
  ``boxes``/``labels``, calls the inner model in train mode, sums the loss
  dict (flattening list-valued losses from SSD), returns
  ``(total_loss, parts_tensor)`` where ``parts_tensor`` is a
  ``(4,)`` tensor of ``(total, loss_classifier, loss_box_reg, loss_rpn_box_reg)``
  (RPN-specific losses are zero for RetinaNet/FCOS/SSD; mask loss is folded
  into ``total`` but not surfaced as a parts slot).
- ``decode(preds)`` converts the eval-mode ``list[dict]`` to a single
  ``(B, N_max, 6)`` tensor of ``(x1, y1, x2, y2, score, class_id)`` — the
  anchor-based format the existing pipeline expects. Shorter detections
  are zero-padded to ``N_max``; the pipeline's NMS step ignores zero-score
  rows via the ``conf_thr`` filter.

The COCO pretrained weights load via torchvision's own weight machinery
(``weights="DEFAULT"`` or a ``Weights`` enum), not via the cracks_yolo
``weights/`` registry — torchvision handles download + cache internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.retinanet import RetinaNetRegressionHead

from cracks_yolo.zoo.base import PretrainedSpec
from cracks_yolo.zoo.base import default_optimizer


class _TorchvisionDetectorBase(nn.Module):
    """Adapter wrapping a torchvision detection model.

    Subclasses set ``_builder`` (a callable returning a fresh torchvision
    model) and ``pretrained_spec``.
    """

    pretrained_spec: PretrainedSpec | None = None
    # Single-element schema — the parts tensor is (total, cls, box_reg, rpn_box_reg).
    # The pipeline's parts interpretation uses schema names, so the first
    # entry "total" is the only one matched against `box`/`cls`/`obj`/`dfl`.
    loss_parts_schema: tuple[str, ...] = ("total", "cls", "box_reg", "rpn_box_reg")
    decode_format: str = "anchor_based"

    # Subclass attributes.
    _builder: Any = None  # callable: (num_classes, pretrained) -> nn.Module
    # Set True on Mask R-CNN subclasses — adds a synthetic per-image mask
    # (bbox region filled with 1s) so the mask head has a target.
    _needs_masks: bool = False

    def __init__(
        self,
        num_classes: int = 80,
        input_size: int = 640,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        # Build the inner torchvision model with num_classes+1 (background=0).
        assert self._builder is not None
        self._inner: nn.Module = self._builder(num_classes=num_classes, pretrained=pretrained)
        # Stride is not a meaningful concept for torchvision detectors;
        # expose a placeholder [8, 16, 32] for Protocol compliance.
        self._stride = torch.tensor([8.0, 16.0, 32.0], dtype=torch.float32)
        # Stash for target conversion (set in forward()).
        self._last_input_size: int = input_size

    @property
    def stride(self) -> torch.Tensor:
        return self._stride

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Train: stash images, return dict. Eval: call inner, return list[dict]."""
        self._last_input_size = int(x.shape[-1])
        if self.training:
            return {"_tv_images": x}
        # Eval mode — torchvision returns list[dict] directly.
        # It expects a list of tensors, one per image.
        images = [x[i] for i in range(x.shape[0])]
        out: list[dict[str, torch.Tensor]] = self._inner(images)
        return out

    def compute_loss(
        self,
        preds: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | torch.Tensor,
        targets: torch.Tensor,
        imgs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss by calling inner model with images + tv-format targets."""
        # Extract the stashed image batch.
        if isinstance(preds, dict) and "_tv_images" in preds:
            images = preds["_tv_images"]
        elif imgs is not None:
            images = imgs
        else:
            raise TypeError(f"Cannot extract image batch from preds of type {type(preds)}")

        # Convert (N, 6) YOLO targets → list[dict] with xyxy abs boxes.
        size = int(images.shape[-1])
        tv_targets: list[dict[str, torch.Tensor]] = []
        if targets.numel() > 0:
            num_images = int(images.shape[0])
            per_image: list[list[torch.Tensor]] = [[] for _ in range(num_images)]
            per_image_labels: list[list[int]] = [[] for _ in range(num_images)]
            for row in targets:
                img_idx = int(row[0].item())
                cls = int(row[1].item())
                xc, yc, w, h = (float(v) for v in row[2:6].tolist())
                x1 = (xc - w / 2) * size
                y1 = (yc - h / 2) * size
                x2 = (xc + w / 2) * size
                y2 = (yc + h / 2) * size
                if 0 <= img_idx < num_images:
                    per_image[img_idx].append(
                        torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=images.device)
                    )
                    per_image_labels[img_idx].append(cls + 1)  # +1: bg=0
            for img_idx in range(num_images):
                if per_image[img_idx]:
                    boxes = torch.stack(per_image[img_idx], dim=0)
                    labels = torch.tensor(
                        per_image_labels[img_idx], dtype=torch.int64, device=images.device
                    )
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32, device=images.device)
                    labels = torch.zeros((0,), dtype=torch.int64, device=images.device)
                target: dict[str, torch.Tensor] = {"boxes": boxes, "labels": labels}
                if self._needs_masks:
                    # Mask R-CNN requires masks. Generate a (N, H, W) binary
                    # mask per image by filling each box region with 1. The
                    # mask head learns box-fill — not real segmentation, but
                    # sufficient for a comparison baseline when the dataset
                    # has no mask annotations.
                    h, w = int(images.shape[2]), int(images.shape[3])
                    if boxes.shape[0] > 0:
                        masks = torch.zeros(
                            (boxes.shape[0], h, w),
                            dtype=torch.uint8,
                            device=images.device,
                        )
                        for k in range(boxes.shape[0]):
                            x1 = int(max(0, min(w - 1, boxes[k, 0].item())))
                            y1 = int(max(0, min(h - 1, boxes[k, 1].item())))
                            x2 = int(max(1, min(w, boxes[k, 2].item())))
                            y2 = int(max(1, min(h, boxes[k, 3].item())))
                            masks[k, y1:y2, x1:x2] = 1
                    else:
                        masks = torch.zeros((0, h, w), dtype=torch.uint8, device=images.device)
                    target["masks"] = masks
                tv_targets.append(target)
        else:
            for _ in range(int(images.shape[0])):
                t: dict[str, torch.Tensor] = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=images.device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=images.device),
                }
                if self._needs_masks:
                    h, w = int(images.shape[2]), int(images.shape[3])
                    t["masks"] = torch.zeros((0, h, w), dtype=torch.uint8, device=images.device)
                tv_targets.append(t)

        # Call inner model in train mode — torchvision returns a loss dict.
        # Ensure inner is in train mode (the pipeline sets the wrapper's mode,
        # which propagates to children — but be defensive).
        was_training = self._inner.training
        self._inner.train()
        images_list = [images[i] for i in range(images.shape[0])]
        loss_dict: dict[str, torch.Tensor] = self._inner(images_list, tv_targets)
        if not was_training:
            self._inner.eval()

        total_loss = self._sum_loss_dict(loss_dict)
        # parts tensor: (total, cls, box_reg, rpn_box_reg) — zero-fill missing.
        # SSD returns classification_losses / regression_losses lists instead
        # of single tensors; sum them so the parts slots are populated.
        cls_loss = self._lookup_loss(
            loss_dict, ("loss_classifier", "classification_losses"), total_loss.device
        )
        box_reg = self._lookup_loss(
            loss_dict, ("loss_box_reg", "regression_losses"), total_loss.device
        )
        rpn_box = self._lookup_loss(
            loss_dict,
            ("loss_rpn_box_reg", "loss_objectness", "loss_centerness"),
            total_loss.device,
        )
        parts = torch.stack([total_loss, cls_loss, box_reg, rpn_box]).detach()
        return total_loss, parts

    @staticmethod
    def _sum_loss_dict(loss_dict: dict[str, Any]) -> torch.Tensor:
        """Sum every value in a torchvision loss dict.

        SSD returns lists of per-location tensors under
        ``classification_losses`` / ``regression_losses``; RetinaNet/Faster/
        Mask/FCOS return scalar tensors. This helper sums both shapes.
        """
        total: torch.Tensor | None = None
        for v in loss_dict.values():
            if isinstance(v, list):
                if not v:
                    continue
                t = torch.stack([x for x in v if isinstance(x, torch.Tensor)]).sum()
            elif isinstance(v, torch.Tensor):
                t = v
            else:
                continue
            total = t if total is None else total + t
        if total is None:
            raise TypeError("torchvision loss dict had no Tensor values to sum")
        return total

    @staticmethod
    def _lookup_loss(
        loss_dict: dict[str, Any],
        keys: tuple[str, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """Return the first matching loss key as a scalar tensor.

        Handles both scalar-tensor values (``loss_classifier``) and list-of-
        tensor values (``classification_losses`` from SSD).
        """
        for k in keys:
            if k not in loss_dict:
                continue
            v = loss_dict[k]
            if isinstance(v, list):
                if not v:
                    continue
                return torch.stack([x for x in v if isinstance(x, torch.Tensor)]).sum().detach()
            if isinstance(v, torch.Tensor):
                return v.detach()
        return torch.zeros((), device=device)

    def decode(self, preds: object) -> torch.Tensor:
        """Convert eval-mode list[dict] → (B, N_max, 6) anchor-based tensor.

        Each row: (x1, y1, x2, y2, score, class_id). Per-image detections
        are zero-padded to ``N_max`` (the max count across the batch) so the
        result is a dense tensor. The pipeline's NMS step filters zero-score
        rows via ``conf_thr``.
        """
        if not isinstance(preds, list):
            raise TypeError(f"Expected list[dict] from torchvision eval, got {type(preds)}")
        per_image_boxes: list[torch.Tensor] = []
        for det in preds:
            boxes = det.get("boxes")
            scores = det.get("scores")
            labels = det.get("labels")
            if boxes is None or boxes.numel() == 0:
                per_image_boxes.append(torch.zeros((0, 6), dtype=torch.float32))
                continue
            assert scores is not None and labels is not None
            # class_id 0-indexed (torchvision uses 1-indexed with bg=0).
            cls_idx = (labels.float() - 1).clamp(min=0).unsqueeze(1)
            row = torch.cat([boxes.float(), scores.float().unsqueeze(1), cls_idx], dim=1)
            per_image_boxes.append(row)
        n_max = max((t.shape[0] for t in per_image_boxes), default=0)
        if n_max == 0:
            # No detections at all — return an empty (B, 0, 6) tensor.
            return torch.zeros((len(per_image_boxes), 0, 6), dtype=torch.float32)
        out = torch.zeros((len(per_image_boxes), n_max, 6), dtype=torch.float32)
        for i, t in enumerate(per_image_boxes):
            if t.shape[0] > 0:
                out[i, : t.shape[0], :] = t
        return out

    def build_optimizer(self) -> torch.optim.Optimizer:
        return default_optimizer(self, lr=1e-4)  # torchvision uses lower lr

    @classmethod
    def from_pretrained(
        cls,
        num_classes: int,
        weights_dir: Path | None = None,  # noqa: ARG003 — torchvision handles caching internally
        strict: bool = False,  # noqa: ARG003 — torchvision weights are loaded at construction
    ) -> _TorchvisionDetectorBase:
        # torchvision loads COCO weights via its own machinery (pretrained=True).
        # We instantiate with num_classes + 1 (background) only when num_classes
        # matches COCO (80); otherwise we load COCO backbone + replace the head.
        return cls(num_classes=num_classes, pretrained=True)


def _build_retinanet(num_classes: int, pretrained: bool) -> nn.Module:
    """Build a torchvision RetinaNet with ``num_classes+1`` outputs (bg=0)."""
    if pretrained and num_classes == 80:
        model: nn.Module = retinanet_resnet50_fpn(weights="DEFAULT", num_classes=91)
        return model
    # Build with COCO backbone + replace head for the new class count.
    model = retinanet_resnet50_fpn(weights="DEFAULT" if pretrained else None, num_classes=91)
    if num_classes != 80:
        # Replace the head's classification + regression predictors.
        head = model.head
        num_anchors = head.classification_head.num_anchors
        head.classification_head = RetinaNetClassificationHead(
            in_channels=256, num_anchors=num_anchors, num_classes=num_classes + 1
        )
        head.regression_head = RetinaNetRegressionHead(in_channels=256, num_anchors=num_anchors)
    return model


def _build_fasterrcnn(num_classes: int, pretrained: bool) -> nn.Module:
    """Build a torchvision Faster R-CNN with ``num_classes+1`` outputs (bg=0)."""
    if pretrained and num_classes == 80:
        model: nn.Module = fasterrcnn_resnet50_fpn(weights="DEFAULT", num_classes=91)
        return model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None, num_classes=91)
    if num_classes != 80:
        # Replace the box predictor for the new class count.
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def _build_maskrcnn(num_classes: int, pretrained: bool) -> nn.Module:
    """Build a torchvision Mask R-CNN with ``num_classes+1`` outputs (bg=0)."""
    if pretrained and num_classes == 80:
        model: nn.Module = maskrcnn_resnet50_fpn(weights="DEFAULT", num_classes=91)
        return model
    model = maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None, num_classes=91)
    if num_classes != 80:
        # Replace both box predictor and mask predictor.
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        in_features_mask = model.roi_heads.mask_predictor.mask_fcn_logits.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask, dim_reduced=256, num_classes=num_classes + 1
        )
    return model


def _build_fcos(num_classes: int, pretrained: bool) -> nn.Module:
    """Build a torchvision FCOS with ``num_classes+1`` outputs (bg=0)."""
    if pretrained and num_classes == 80:
        model: nn.Module = fcos_resnet50_fpn(weights="DEFAULT", num_classes=91)
        return model
    # FCOS constructor accepts num_classes directly — head is built for nc+1.
    model2: nn.Module = fcos_resnet50_fpn(
        weights="DEFAULT" if pretrained else None, num_classes=num_classes + 1
    )
    return model2


def _build_ssd300(num_classes: int, pretrained: bool) -> nn.Module:
    """Build a torchvision SSD300 (VGG16 backbone) with ``num_classes+1``."""
    if pretrained and num_classes == 80:
        m: nn.Module = ssd300_vgg16(weights="DEFAULT", num_classes=91)
        return m
    m2: nn.Module = ssd300_vgg16(
        weights="DEFAULT" if pretrained else None, num_classes=num_classes + 1
    )
    return m2


def _build_ssdlite320(num_classes: int, pretrained: bool) -> nn.Module:
    """Build a torchvision SSDlite320 (MobileNetV3-Large) with ``num_classes+1``."""
    if pretrained and num_classes == 80:
        m: nn.Module = ssdlite320_mobilenet_v3_large(weights="DEFAULT", num_classes=91)
        return m
    m2: nn.Module = ssdlite320_mobilenet_v3_large(
        weights="DEFAULT" if pretrained else None, num_classes=num_classes + 1
    )
    return m2


# Long-form class names (documentation-is-the-name).
def _retinanet_init(
    self: _TorchvisionDetectorBase,
    num_classes: int = 80,
    input_size: int = 640,
    pretrained: bool = False,
) -> None:
    _TorchvisionDetectorBase.__init__(
        self, num_classes=num_classes, input_size=input_size, pretrained=pretrained
    )


def _fasterrcnn_init(
    self: _TorchvisionDetectorBase,
    num_classes: int = 80,
    input_size: int = 640,
    pretrained: bool = False,
) -> None:
    _TorchvisionDetectorBase.__init__(
        self, num_classes=num_classes, input_size=input_size, pretrained=pretrained
    )


_maskrcnn_init = _fasterrcnn_init
_fcos_init = _fasterrcnn_init
_ssd300_init = _fasterrcnn_init
_ssdlite320_init = _fasterrcnn_init


RetinaNet_R50_FocalLoss_SGD_SILU = type(
    "RetinaNet_R50_FocalLoss_SGD_SILU",
    (_TorchvisionDetectorBase,),
    {
        "__init__": _retinanet_init,
        "_builder": staticmethod(_build_retinanet),
        "pretrained_spec": None,  # torchvision handles weights internally
    },
)

FasterRCNN_R50_CEA_SmoothL1_SGD_SILU = type(
    "FasterRCNN_R50_CEA_SmoothL1_SGD_SILU",
    (_TorchvisionDetectorBase,),
    {
        "__init__": _fasterrcnn_init,
        "_builder": staticmethod(_build_fasterrcnn),
        "pretrained_spec": None,
    },
)

MaskRCNN_R50_CEA_SmoothL1_BCE_SGD_SILU = type(
    "MaskRCNN_R50_CEA_SmoothL1_BCE_SGD_SILU",
    (_TorchvisionDetectorBase,),
    {
        "__init__": _maskrcnn_init,
        "_builder": staticmethod(_build_maskrcnn),
        "pretrained_spec": None,
        "_needs_masks": True,
    },
)

FCOS_R50_FocalLoss_Centerness_SGD_SILU = type(
    "FCOS_R50_FocalLoss_Centerness_SGD_SILU",
    (_TorchvisionDetectorBase,),
    {
        "__init__": _fcos_init,
        "_builder": staticmethod(_build_fcos),
        "pretrained_spec": None,
    },
)

SSD300_VGG16_FocalLoss_SmoothL1_SGD_ReLU = type(
    "SSD300_VGG16_FocalLoss_SmoothL1_SGD_ReLU",
    (_TorchvisionDetectorBase,),
    {
        "__init__": _ssd300_init,
        "_builder": staticmethod(_build_ssd300),
        "pretrained_spec": None,
    },
)

SSDlite320_MobileNetV3_FocalLoss_SmoothL1_SGD_Hardswish = type(
    "SSDlite320_MobileNetV3_FocalLoss_SmoothL1_SGD_Hardswish",
    (_TorchvisionDetectorBase,),
    {
        "__init__": _ssdlite320_init,
        "_builder": staticmethod(_build_ssdlite320),
        "pretrained_spec": None,
    },
)

# Short aliases.
RetinaNetR50 = RetinaNet_R50_FocalLoss_SGD_SILU
FasterRCNNR50 = FasterRCNN_R50_CEA_SmoothL1_SGD_SILU
MaskRCNNR50 = MaskRCNN_R50_CEA_SmoothL1_BCE_SGD_SILU
FCOSR50 = FCOS_R50_FocalLoss_Centerness_SGD_SILU
SSD300VGG16 = SSD300_VGG16_FocalLoss_SmoothL1_SGD_ReLU
SSDlite320MobileNetV3 = SSDlite320_MobileNetV3_FocalLoss_SmoothL1_SGD_Hardswish
