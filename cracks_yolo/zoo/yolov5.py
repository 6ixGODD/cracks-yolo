"""YOLOv5 model zoo: YOLOv5s, YOLOv5sSAC, YOLOv5sTR, YOLOv5sSACTR.

Each class is a self-contained ``nn.Module`` owning its layers, its loss
module, its optimizer-builder, and its pretrained-spec. Long class names
encode the loss/optimizer/activation so the name *is* the documentation:

    YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU

Short aliases (``YOLOv5sSACTR``) re-exported from
:mod:`cracks_yolo.zoo.__init__`.

Architecture (from ``yolov5s-sac-tr.yml``): v5 v6.0 backbone with optional
SAC in the C3 stages (P2/P3/P4/P5), TR (transformer) in the P5 stage; FPN
+ PAN neck; anchor-based ``Detect`` head with hardcoded COCO anchors
(3 scales x 3 anchors).

Insertion matrix:

================  =====  =====  =====
Variant           SAC    TR     Notes
================  =====  =====  =====
YOLOv5s           -      -      baseline
YOLOv5sSAC        yes    -      SAC in C3 backbone stages
YOLOv5sTR         -      yes    TR in P5 backbone stage
YOLOv5sSACTR      yes    yes    both
================  =====  =====  =====
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from cracks_yolo.losses.yolov5 import ComputeLoss
from cracks_yolo.ops.conv import Conv
from cracks_yolo.ops.csp import C3
from cracks_yolo.ops.csp import C3SAC
from cracks_yolo.ops.csp import C3TR
from cracks_yolo.ops.csp import SPPF
from cracks_yolo.ops.csp import Concat
from cracks_yolo.ops.detect_heads import DetectAnchorBased
from cracks_yolo.weights.loader import load_pretrained
from cracks_yolo.weights.registry import PRETRAINED_URLS
from cracks_yolo.weights.remappers import yolo_remapper
from cracks_yolo.zoo.base import PretrainedSpec
from cracks_yolo.zoo.base import default_optimizer

# COCO v5 anchors: 3 scales x 3 anchors x 2 dims (w, h).
V5_ANCHORS: tuple[tuple[int, ...], ...] = (
    (10, 13, 16, 30, 33, 23),
    (30, 61, 62, 45, 59, 119),
    (116, 90, 156, 198, 373, 326),
)


# v5s depth_multiple=0.33, width_multiple=0.50. Channel scaling: c * 0.5,
# rounded to nearest multiple of 8.
def _c(c: int) -> int:
    return max(round(c * 0.5 / 8) * 8, 8)


def _n(n: int) -> int:
    return max(round(n * 0.33), 1)


V5S_HYP: dict[str, float] = {
    "box": 0.05,
    "obj": 0.7,
    "cls": 0.3,
    "cls_pw": 1.0,
    "obj_pw": 1.0,
    "anchor_t": 4.0,
    "fl_gamma": 0.0,
    "label_smoothing": 0.0,
}


def _build_v5_backbone(sac: bool, tr: bool) -> tuple[nn.Sequential, list[int], list[int]]:
    """Build the v5s backbone. Returns (seq, p3_p4_p5_channels, layer_indices).

    The returned layer_indices are the indices of P3, P4, P5 outputs inside
    the returned Sequential (used for FPN skip connections).
    """
    c3_block = C3SAC if sac else C3
    p5_block = C3TR if tr else C3
    layers: list[nn.Module] = [
        Conv(3, _c(64), 6, 2, 2),  # 0  P1/2
        Conv(_c(64), _c(128), 3, 2),  # 1  P2/4
        c3_block(_c(128), _c(128), n=_n(3)),  # 2
        Conv(_c(128), _c(256), 3, 2),  # 3  P3/8
        c3_block(_c(256), _c(256), n=_n(6)),  # 4
        Conv(_c(256), _c(512), 3, 2),  # 5  P4/16
        c3_block(_c(512), _c(512), n=_n(9)),  # 6
        Conv(_c(512), _c(1024), 3, 2),  # 7  P5/32
        p5_block(_c(1024), _c(1024), n=_n(3)),  # 8
        SPPF(_c(1024), _c(1024), k=5),  # 9
    ]
    # P3 is layer 4, P4 is layer 6, P5 is layer 9.
    return nn.Sequential(*layers), [_c(256), _c(512), _c(1024)], [4, 6, 9]


def _build_v5_neck(
    backbone_channels: list[int],
) -> tuple[nn.Sequential, list[int], list[int]]:
    """Build the FPN+PAN neck. Returns (seq, head_out_channels, head_input_indices).

    head_input_indices point into the concatenated [backbone_out, neck_layers]
    sequence: 0 = SPPF out, P4 (index 6), P3 (index 4) come from backbone.
    """
    c_p3, c_p4, c_p5 = backbone_channels
    layers: list[nn.Module] = [
        Conv(c_p5, _c(512), 1, 1),  # 10
        nn.Upsample(scale_factor=2, mode="nearest"),  # 11
        Concat(dimension=1),  # 12  cat backbone P4
        C3(_c(512) + c_p4, _c(512), n=_n(3), shortcut=False),  # 13
        Conv(_c(512), _c(256), 1, 1),  # 14
        nn.Upsample(scale_factor=2, mode="nearest"),  # 15
        Concat(dimension=1),  # 16  cat backbone P3
        C3(_c(256) + c_p3, _c(256), n=_n(3), shortcut=False),  # 17  P3/8-small
        Conv(_c(256), _c(256), 3, 2),  # 18
        Concat(dimension=1),  # 19  cat head P4
        C3(_c(256) + _c(256), _c(512), n=_n(3), shortcut=False),  # 20  P4/16-medium
        Conv(_c(512), _c(512), 3, 2),  # 21
        Concat(dimension=1),  # 22  cat head P5
        C3TR(_c(512) + _c(512), _c(1024), n=_n(3), shortcut=False),  # 23  P5/32-large
    ]
    return (
        nn.Sequential(*layers),
        [_c(256), _c(512), _c(1024)],
        [17, 20, 23],
    )


class _YOLOv5sBase(nn.Module):
    """Concrete base shared by the four v5s variants.

    Not part of the public Protocol — a private implementation detail for
    code reuse. Each public subclass passes its ``sac`` / ``tr`` flags and
    declares its own ``pretrained_spec``.
    """

    pretrained_spec: PretrainedSpec | None = None
    # Self-description consumed by the pipeline (no class-name branching).
    loss_parts_schema: tuple[str, ...] = ("box", "cls", "obj")
    decode_format: str = "anchor_based"

    def __init__(
        self,
        num_classes: int = 80,
        input_size: int = 640,
        sac: bool = False,
        tr: bool = False,
        hyp: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self._sac = sac
        self._tr = tr
        self._hyp = hyp if hyp is not None else V5S_HYP

        backbone, backbone_channels, _ = _build_v5_backbone(sac=sac, tr=tr)
        neck, head_channels, head_input_indices = _build_v5_neck(backbone_channels)
        self.backbone = backbone
        self.neck = neck
        # Concat (idx 12, 16, 19, 22) consumes two inputs: previous layer
        # output and a skip from backbone/neck. We handle this in forward.
        self._head_input_indices = head_input_indices
        self.head = DetectAnchorBased(
            nc=num_classes,
            anchors=V5_ANCHORS,
            ch=tuple(head_channels),
        )
        # Initialize stride via a dummy training forward (training mode
        # returns the per-level feature maps without needing stride).
        s = 256  # min input that yields 3 scales at stride 8/16/32
        with torch.no_grad():
            feats = self._forward_impl(torch.zeros(1, 3, s, s))
        assert isinstance(feats, list)
        self.head.stride = torch.tensor([s / f.shape[-2] for f in feats], dtype=torch.float32)
        # Scale anchors from image pixels to stride-relative units.
        self.head.anchors /= self.head.stride.view(-1, 1, 1)
        # Build loss.
        self._loss_fn = ComputeLoss(
            nc=num_classes,
            anchors=self.head.anchors,
            stride=self.stride,
            hyp=self._hyp,
            device=next(self.parameters()).device,
        )

    @property
    def stride(self) -> torch.Tensor:
        assert self.head.stride is not None
        return self.head.stride

    def _forward_impl(
        self, x: torch.Tensor
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        # Backbone: track outputs by index for skip connections.
        bb_out: list[torch.Tensor] = []
        for layer in self.backbone:
            x = layer(x)
            bb_out.append(x)
        # Neck: each layer's input is either the previous output, or (for
        # Concat layers) a list of [prev_output, skip_connection].
        # Concat layer indices (in neck): 2, 6, 9, 12.
        # Skip targets: neck[2] concat bb_out[6] (P4); neck[6] concat
        # bb_out[4] (P3); neck[9] concat nk_out[4] (head P4); neck[12]
        # concat nk_out[0] (head P5).
        # Encoded as: backbone index = itself, neck index = 100 + i.
        nk_out: list[torch.Tensor] = []
        concat_skips: dict[int, int] = {2: 6, 6: 4, 9: 104, 12: 100}
        prev = bb_out[-1]  # SPPF output
        for i, layer in enumerate(self.neck):
            if i in concat_skips:
                skip_idx = concat_skips[i]
                skip = bb_out[skip_idx] if skip_idx < 100 else nk_out[skip_idx - 100]
                out = layer([prev, skip])
            else:
                out = layer(prev)
            nk_out.append(out)
            prev = out
        head_inputs = [nk_out[7], nk_out[10], nk_out[13]]
        head_out: list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]] = self.head(
            head_inputs
        )
        return head_out

    def forward(
        self, x: torch.Tensor
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        return self._forward_impl(x)

    def compute_loss(
        self,
        preds: list[torch.Tensor],
        targets: torch.Tensor,
        imgs: torch.Tensor | None = None,  # noqa: ARG002 — v5 loss doesn't need images
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute v5 loss. ``targets`` is ``(N, 6)`` — (img_idx, cls, x, y, w, h) normalized."""
        loss_out: tuple[torch.Tensor, torch.Tensor] = self._loss_fn(preds, targets)
        return loss_out

    def decode(self, preds: object) -> torch.Tensor:
        """Decode eval-mode predictions (already decoded by the head)."""
        if isinstance(preds, tuple):
            out0: torch.Tensor = preds[0]
            return out0
        if isinstance(preds, torch.Tensor):
            return preds
        raise TypeError(f"Unsupported preds type: {type(preds)}")

    def build_optimizer(self) -> torch.optim.Optimizer:
        return default_optimizer(self, lr=1e-3)

    @classmethod
    def from_pretrained(
        cls,
        num_classes: int,
        weights_dir: Path | None = None,
        strict: bool = False,
    ) -> _YOLOv5sBase:
        spec = cls.pretrained_spec
        if spec is None:
            return cls(num_classes=num_classes)
        report = load_pretrained(
            model=cls(num_classes=num_classes),
            spec=spec,
            weights_dir=weights_dir,
            strict=strict,
        )
        # Return the model instance that load_pretrained populated.
        model: _YOLOv5sBase = report.model  # type: ignore[assignment]
        return model


class YOLOv5s_CIoU_BCEObj_BCECls_AdamW_SILU(_YOLOv5sBase):  # noqa: N801
    """YOLOv5s baseline (no SAC, no TR)."""

    pretrained_spec = PretrainedSpec(
        key="yolov5s",
        url=PRETRAINED_URLS["yolov5s"],
        state_dict_key_map={},
        remapper=yolo_remapper,
    )

    def __init__(self, num_classes: int = 80, input_size: int = 640) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, sac=False, tr=False)


class YOLOv5sSAC_CIoU_BCEObj_BCECls_AdamW_SILU(_YOLOv5sBase):  # noqa: N801
    """YOLOv5s with SAC in the backbone C3 stages (P2/P3/P4/P5).

    Loads the same COCO weights as the baseline; SAC-specific layers (the
    SAConv2d switches / ConvAWS2d) have no COCO counterpart and stay randomly
    initialized via ``strict=False``.
    """

    pretrained_spec = PretrainedSpec(
        key="yolov5s",
        url=PRETRAINED_URLS["yolov5s"],
        state_dict_key_map={},
        remapper=yolo_remapper,
    )

    def __init__(self, num_classes: int = 80, input_size: int = 640) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, sac=True, tr=False)


class YOLOv5sTR_CIoU_BCEObj_BCECls_AdamW_SILU(_YOLOv5sBase):  # noqa: N801
    """YOLOv5s with TR (TransformerBlock) in the P5 backbone stage.

    Loads COCO weights; the TransformerBlock layers stay random init.
    """

    pretrained_spec = PretrainedSpec(
        key="yolov5s",
        url=PRETRAINED_URLS["yolov5s"],
        state_dict_key_map={},
        remapper=yolo_remapper,
    )

    def __init__(self, num_classes: int = 80, input_size: int = 640) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, sac=False, tr=True)


class YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU(_YOLOv5sBase):  # noqa: N801
    """YOLOv5s with both SAC (backbone C3 stages) and TR (P5 stage).

    Loads COCO weights; SAC and TR layers stay random init.
    """

    pretrained_spec = PretrainedSpec(
        key="yolov5s",
        url=PRETRAINED_URLS["yolov5s"],
        state_dict_key_map={},
        remapper=yolo_remapper,
    )

    def __init__(self, num_classes: int = 80, input_size: int = 640) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, sac=True, tr=True)


# Short aliases for ergonomic use.
YOLOv5s = YOLOv5s_CIoU_BCEObj_BCECls_AdamW_SILU
YOLOv5sSAC = YOLOv5sSAC_CIoU_BCEObj_BCECls_AdamW_SILU
YOLOv5sTR = YOLOv5sTR_CIoU_BCEObj_BCECls_AdamW_SILU
YOLOv5sSACTR = YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU
