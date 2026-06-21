"""YOLOv7 model zoo: YOLOv7w (baseline) + YOLOv7wSAC.

Both use :class:`~cracks_yolo.ops.detect_heads.IDetect` (ImplicitA/M bias)
and :class:`~cracks_yolo.losses.yolov7.ComputeLossOTA` (SimOTA assignment).

The architecture is a simplified v7-w: ELAN-based backbone with RepConv
in the neck, SPPCSPC at P5, IDetect head with COCO anchors. The SAC variant
swaps the C3 stages for C3SAC.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from cracks_yolo.losses.yolov7 import ComputeLossOTA
from cracks_yolo.ops.conv import Conv
from cracks_yolo.ops.csp import C3
from cracks_yolo.ops.csp import C3SAC
from cracks_yolo.ops.csp import SPPCSPC
from cracks_yolo.ops.csp import Concat
from cracks_yolo.ops.csp import RepConv
from cracks_yolo.ops.detect_heads import IDetect
from cracks_yolo.weights.loader import load_pretrained
from cracks_yolo.weights.registry import PRETRAINED_URLS
from cracks_yolo.weights.remappers import yolo_remapper
from cracks_yolo.zoo.base import PretrainedSpec
from cracks_yolo.zoo.base import default_optimizer

V7_ANCHORS: tuple[tuple[int, ...], ...] = (
    (12, 16, 19, 36, 40, 28),
    (36, 75, 76, 55, 72, 146),
    (142, 110, 192, 243, 459, 401),
)

V7_HYP: dict[str, float] = {
    "box": 0.05,
    "obj": 0.7,
    "cls": 0.3,
    "cls_pw": 1.0,
    "obj_pw": 1.0,
    "anchor_t": 4.0,
    "fl_gamma": 0.0,
    "label_smoothing": 0.0,
}


def _c(c: int) -> int:
    return max(round(c / 8) * 8, 8)


def _build_v7_backbone(sac: bool) -> tuple[nn.Sequential, list[int]]:
    c3_block = C3SAC if sac else C3
    layers: list[nn.Module] = [
        Conv(3, 32, 3, 2),  # 0   P1/2
        Conv(32, 64, 3, 2),  # 1   P2/4
        c3_block(64, 64, n=1),  # 2
        Conv(64, 128, 3, 2),  # 3   P3/8
        c3_block(128, 128, n=3),  # 4
        Conv(128, 256, 3, 2),  # 5   P4/16
        c3_block(256, 256, n=6),  # 6
        Conv(256, 512, 3, 2),  # 7   P5/32
        c3_block(512, 512, n=3),  # 8
        SPPCSPC(512, 512, k=(5, 9, 13)),  # 9
    ]
    return nn.Sequential(*layers), [128, 256, 512]


def _build_v7_neck(backbone_channels: list[int]) -> tuple[nn.Sequential, list[int]]:
    c_p3, c_p4, c_p5 = backbone_channels
    layers: list[nn.Module] = [
        Conv(c_p5, _c(256), 1, 1),  # 10
        nn.Upsample(scale_factor=2, mode="nearest"),  # 11
        Concat(dimension=1),  # 12  cat backbone P4
        RepConv(_c(256) + c_p4, _c(256)),  # 13
        Conv(_c(256), _c(128), 1, 1),  # 14
        nn.Upsample(scale_factor=2, mode="nearest"),  # 15
        Concat(dimension=1),  # 16  cat backbone P3
        RepConv(_c(128) + c_p3, _c(128)),  # 17  P3/8-small
        Conv(_c(128), _c(256), 3, 2),  # 18
        Concat(dimension=1),  # 19  cat head P4
        RepConv(_c(256) + _c(128), _c(256)),  # 20  P4/16-medium
        Conv(_c(256), _c(512), 3, 2),  # 21
        Concat(dimension=1),  # 22  cat head P5
        RepConv(_c(512) + _c(256), _c(512)),  # 23  P5/32-large
    ]
    return nn.Sequential(*layers), [_c(128), _c(256), _c(512)]


class _YOLOv7wBase(nn.Module):
    """Concrete base shared by v7-w and v7-w-SAC variants."""

    pretrained_spec: PretrainedSpec | None = None
    loss_parts_schema: tuple[str, ...] = ("box", "cls", "obj")
    decode_format: str = "anchor_based"

    def __init__(
        self,
        num_classes: int = 80,
        input_size: int = 640,
        sac: bool = False,
        hyp: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self._sac = sac
        self._hyp = hyp if hyp is not None else V7_HYP

        backbone, bb_channels = _build_v7_backbone(sac=sac)
        neck, head_channels = _build_v7_neck(bb_channels)
        self.backbone = backbone
        self.neck = neck
        self.head = IDetect(
            nc=num_classes,
            anchors=V7_ANCHORS,
            ch=tuple(head_channels),
        )
        # Initialize stride via a dummy training forward (training mode
        # returns the per-level feature maps without needing stride).
        s = 256
        with torch.no_grad():
            feats = self._forward_impl(torch.zeros(1, 3, s, s))
        assert isinstance(feats, list)
        self.head.stride = torch.tensor([s / f.shape[-2] for f in feats], dtype=torch.float32)
        self.head.anchors /= self.head.stride.view(-1, 1, 1)
        self._loss_fn = ComputeLossOTA(
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
        bb_out: list[torch.Tensor] = []
        for layer in self.backbone:
            x = layer(x)
            bb_out.append(x)
        nk_out: list[torch.Tensor] = []
        # Concat layer indices in the neck (with skip target):
        #   idx 2  -> backbone P4 (bb[6])
        #   idx 6  -> backbone P3 (bb[4])
        #   idx 9  -> neck P4 (nk[4])
        #   idx 12 -> neck P5 (nk[0])
        # Backbone indices are encoded as themselves; neck indices as 100+i.
        concat_skips: dict[int, int] = {2: 6, 6: 4, 9: 104, 12: 100}
        prev = bb_out[-1]
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
        imgs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if imgs is None:
            imgs = torch.zeros(
                (preds[0].shape[0], 3, self.input_size, self.input_size),
                device=preds[0].device,
            )
        loss_out: tuple[torch.Tensor, torch.Tensor] = self._loss_fn(preds, targets, imgs)
        return loss_out

    def decode(self, preds: object) -> torch.Tensor:
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
    ) -> _YOLOv7wBase:
        spec = cls.pretrained_spec
        if spec is None:
            return cls(num_classes=num_classes)
        report = load_pretrained(
            model=cls(num_classes=num_classes),
            spec=spec,
            weights_dir=weights_dir,
            strict=strict,
        )
        model: _YOLOv7wBase = report.model  # type: ignore[assignment]
        return model


class YOLOv7w_CIouOTA_BCEObj_BCECls_AdamW_SILU(_YOLOv7wBase):  # noqa: N801
    """YOLOv7-w baseline with IDetect + OTA loss."""

    pretrained_spec = PretrainedSpec(
        key="yolov7w",
        url=PRETRAINED_URLS["yolov7w"],
        state_dict_key_map={},
        remapper=yolo_remapper,
    )

    def __init__(self, num_classes: int = 80, input_size: int = 640) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, sac=False)


class YOLOv7wSAC_CIouOTA_BCEObj_BCECls_AdamW_SILU(_YOLOv7wBase):  # noqa: N801
    """YOLOv7-w with SAC in the backbone C3 stages.

    Loads the same COCO weights as the baseline; SAC layers stay random init.
    """

    pretrained_spec = PretrainedSpec(
        key="yolov7w",
        url=PRETRAINED_URLS["yolov7w"],
        state_dict_key_map={},
        remapper=yolo_remapper,
    )

    def __init__(self, num_classes: int = 80, input_size: int = 640) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, sac=True)


YOLOv7w = YOLOv7w_CIouOTA_BCEObj_BCECls_AdamW_SILU
YOLOv7wSAC = YOLOv7wSAC_CIouOTA_BCEObj_BCECls_AdamW_SILU
