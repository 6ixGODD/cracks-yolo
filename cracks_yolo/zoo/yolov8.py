"""YOLOv8 model zoo: YOLOv8s + YOLOv8sSAC.

Each class is a self-contained ``nn.Module`` owning its layers, its loss
module, its optimizer-builder, and its pretrained-spec. Architecture: v8
backbone (C2f stages with optional SAC), SPPF at P5, FPN+PAN neck,
anchor-free ``DetectAnchorFree`` head with separate box (DFL) and cls
branches. Loss: ``v8DetectionLoss`` (TaskAlignedAssigner topk=10, BboxLoss
with CIoU+DFL, no obj loss).

The SAC variant swaps the backbone C2f stages for ``C2fSAC`` (a C2f whose
inner Bottleneck uses SAC). COCO weights load with ``strict=False``; SAC
layers are randomly initialized.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from cracks_yolo.losses.yolov8 import v8DetectionLoss
from cracks_yolo.ops.conv import Conv
from cracks_yolo.ops.csp import SPPF
from cracks_yolo.ops.csp import C2f
from cracks_yolo.ops.csp import C2fSAC
from cracks_yolo.ops.csp import Concat
from cracks_yolo.ops.detect_heads import DetectAnchorFree
from cracks_yolo.weights.loader import load_pretrained
from cracks_yolo.weights.registry import PRETRAINED_URLS
from cracks_yolo.weights.remappers import yolo_remapper
from cracks_yolo.zoo.base import PretrainedSpec
from cracks_yolo.zoo.base import default_optimizer

V8S_HYP: dict[str, float] = {
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
}


# Channel scaling: c * width_mult, rounded to nearest multiple of 8.
def _c(c: int, width_mult: float = 0.5) -> int:
    return max(round(c * width_mult / 8) * 8, 8)


def _n(n: int, depth_mult: float = 0.33) -> int:
    return max(round(n * depth_mult), 1)


def _build_v8_backbone(
    sac: bool, width_mult: float = 0.5, depth_mult: float = 0.33
) -> tuple[nn.Sequential, list[int]]:
    """Build the v8 backbone. Returns (seq, [P3, P4, P5] channels).

    Layer indices: P3 = layer 4, P4 = layer 6, P5 = layer 9 (SPPF out).
    """
    c2f_block = C2fSAC if sac else C2f
    layers: list[nn.Module] = [
        Conv(3, _c(64, width_mult), 3, 2),  # 0   P1/2
        Conv(_c(64, width_mult), _c(128, width_mult), 3, 2),  # 1   P2/4
        c2f_block(
            _c(128, width_mult), _c(128, width_mult), n=_n(3, depth_mult), shortcut=True
        ),  # 2
        Conv(_c(128, width_mult), _c(256, width_mult), 3, 2),  # 3   P3/8
        c2f_block(
            _c(256, width_mult), _c(256, width_mult), n=_n(6, depth_mult), shortcut=True
        ),  # 4
        Conv(_c(256, width_mult), _c(512, width_mult), 3, 2),  # 5   P4/16
        c2f_block(
            _c(512, width_mult), _c(512, width_mult), n=_n(6, depth_mult), shortcut=True
        ),  # 6
        Conv(_c(512, width_mult), _c(1024, width_mult), 3, 2),  # 7   P5/32
        c2f_block(
            _c(1024, width_mult), _c(1024, width_mult), n=_n(3, depth_mult), shortcut=True
        ),  # 8
        SPPF(_c(1024, width_mult), _c(1024, width_mult), k=5),  # 9
    ]
    return nn.Sequential(*layers), [
        _c(256, width_mult),
        _c(512, width_mult),
        _c(1024, width_mult),
    ]


def _build_v8_neck(
    backbone_channels: list[int],
    width_mult: float = 0.5,
    depth_mult: float = 0.33,
) -> tuple[nn.Sequential, list[int]]:
    """Build the v8 C2f FPN+PAN neck. Returns (seq, head_input_channels).

    Concat layer indices in the neck (with skip target encoded):
      idx 2  -> backbone P4 (bb[6])
      idx 5  -> backbone P3 (bb[4])
      idx 8  -> neck P4 (nk[4])
      idx 11 -> neck P5 (nk[0])
    Backbone indices encoded as themselves; neck indices as 100 + i.
    """
    c_p3, c_p4, c_p5 = backbone_channels
    layers: list[nn.Module] = [
        Conv(c_p5, _c(512, width_mult), 1, 1),  # 0
        nn.Upsample(scale_factor=2, mode="nearest"),  # 1
        Concat(dimension=1),  # 2  cat backbone P4
        C2f(
            _c(512, width_mult) + c_p4, _c(512, width_mult), n=_n(3, depth_mult), shortcut=False
        ),  # 3
        Conv(_c(512, width_mult), _c(256, width_mult), 1, 1),  # 4
        nn.Upsample(scale_factor=2, mode="nearest"),  # 5
        Concat(dimension=1),  # 6  cat backbone P3
        C2f(
            _c(256, width_mult) + c_p3, _c(256, width_mult), n=_n(3, depth_mult), shortcut=False
        ),  # 7   P3/8-small
        Conv(_c(256, width_mult), _c(256, width_mult), 3, 2),  # 8
        Concat(dimension=1),  # 9   cat head P4
        C2f(
            _c(256, width_mult) + _c(256, width_mult),
            _c(512, width_mult),
            n=_n(3, depth_mult),
            shortcut=False,
        ),  # 10  P4/16-medium
        Conv(_c(512, width_mult), _c(512, width_mult), 3, 2),  # 11
        Concat(dimension=1),  # 12  cat head P5
        C2f(
            _c(512, width_mult) + _c(512, width_mult),
            _c(1024, width_mult),
            n=_n(3, depth_mult),
            shortcut=False,
        ),  # 13  P5/32-large
    ]
    return nn.Sequential(*layers), [
        _c(256, width_mult),
        _c(512, width_mult),
        _c(1024, width_mult),
    ]


class _YOLOv8Base(nn.Module):
    """Concrete base shared by all v8 size variants (n/s/m/l/x) and their SAC versions."""

    pretrained_spec: PretrainedSpec | None = None
    loss_parts_schema: tuple[str, ...] = ("box", "cls", "dfl")
    decode_format: str = "anchor_free"

    def __init__(
        self,
        num_classes: int = 80,
        input_size: int = 640,
        sac: bool = False,
        hyp: dict[str, float] | None = None,
        width_mult: float = 0.5,
        depth_mult: float = 0.33,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self._sac = sac
        self._hyp = hyp if hyp is not None else V8S_HYP

        backbone, backbone_channels = _build_v8_backbone(
            sac=sac, width_mult=width_mult, depth_mult=depth_mult
        )
        neck, head_channels = _build_v8_neck(
            backbone_channels, width_mult=width_mult, depth_mult=depth_mult
        )
        self.backbone = backbone
        self.neck = neck
        self.head = DetectAnchorFree(
            nc=num_classes,
            reg_max=16,
            ch=tuple(head_channels),
        )
        # Initialize stride via a dummy training forward (the head's training
        # forward returns a dict {boxes, scores, feats} without needing stride).
        s = 256
        with torch.no_grad():
            feats_dict = self._forward_impl(torch.zeros(1, 3, s, s))
        assert isinstance(feats_dict, dict)
        feats = feats_dict["feats"]
        assert isinstance(feats, list)
        self.head.stride = torch.tensor([s / f.shape[-2] for f in feats], dtype=torch.float32)
        self.head.bias_init()
        self._loss_fn = v8DetectionLoss(
            nc=num_classes,
            reg_max=16,
            stride=self.stride,
            hyp=self._hyp,
            device=next(self.parameters()).device,
            tal_topk=10,
        )

    @property
    def stride(self) -> torch.Tensor:
        return self.head.stride

    def _forward_impl(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor | list[torch.Tensor]] | torch.Tensor:
        bb_out: list[torch.Tensor] = []
        for layer in self.backbone:
            x = layer(x)
            bb_out.append(x)
        nk_out: list[torch.Tensor] = []
        # Concat skip targets: neck[2]->bb[6] (P4), neck[6]->bb[4] (P3),
        # neck[9]->nk[4] (head P4), neck[12]->nk[0] (head P5). Backbone
        # indices encoded as themselves; neck indices as 100 + i.
        concat_skips: dict[int, int] = {2: 6, 6: 4, 9: 104, 12: 100}
        prev = bb_out[-1]
        for i, layer in enumerate(self.neck):
            if isinstance(layer, Concat) and i in concat_skips:
                skip_idx = concat_skips[i]
                skip = bb_out[skip_idx] if skip_idx < 100 else nk_out[skip_idx - 100]
                out = layer([prev, skip])
            else:
                out = layer(prev)
            nk_out.append(out)
            prev = out
        head_inputs = [nk_out[7], nk_out[10], nk_out[13]]
        head_out: dict[str, torch.Tensor | list[torch.Tensor]] | torch.Tensor = self.head(
            head_inputs
        )
        return head_out

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor | list[torch.Tensor]] | torch.Tensor:
        return self._forward_impl(x)

    def compute_loss(
        self,
        preds: dict[str, torch.Tensor | list[torch.Tensor]],
        targets: torch.Tensor,
        imgs: torch.Tensor | None = None,  # noqa: ARG002 — v8 loss doesn't need images
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute v8 loss.

        ``targets`` is ``(N, 6)`` — (img_idx, cls, x, y, w, h) normalized.
        Internally converted to the dict batch format the loss expects.
        """
        batch = {
            "batch_idx": targets[:, 0].long(),
            "cls": targets[:, 1].long(),
            "bboxes": targets[:, 2:6],
        }
        loss_out: tuple[torch.Tensor, torch.Tensor] = self._loss_fn(preds, batch)
        # v8 loss returns a (3,) tensor of [box, cls, dfl] — sum to a scalar.
        total_loss: torch.Tensor = loss_out[0].sum()
        return total_loss, loss_out[1]

    def decode(self, preds: object) -> torch.Tensor:
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
    ) -> _YOLOv8Base:
        spec = cls.pretrained_spec
        if spec is None:
            return cls(num_classes=num_classes)
        report = load_pretrained(
            model=cls(num_classes=num_classes),
            spec=spec,
            weights_dir=weights_dir,
            strict=strict,
        )
        model: _YOLOv8Base = report.model  # type: ignore[assignment]
        return model


# YOLOv8 size multipliers (from ultralytics yolov8.yaml family).
# (depth_multiple, width_multiple)
_V8_SIZE_MULTS: dict[str, tuple[float, float]] = {
    "n": (0.33, 0.25),
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.00, 1.00),
    "x": (1.00, 1.25),
}


def _make_v8_class(size: str, sac: bool, pretrained_key: str | None = None) -> type[_YOLOv8Base]:
    """Factory for v8 size variants. Used at module import to populate the namespace."""
    depth_mult, width_mult = _V8_SIZE_MULTS[size]
    suffix = ("SAC_" if sac else "") + "CIoU_DFL_AdamW_SILU"
    cls_name = f"YOLOv8{size}{'SAC' if sac else ''}_{suffix}"

    def _init(
        self: _YOLOv8Base,
        num_classes: int = 80,
        input_size: int = 640,
        _depth_mult: float = depth_mult,
        _width_mult: float = width_mult,
        _sac: bool = sac,
    ) -> None:
        _YOLOv8Base.__init__(
            self,
            num_classes=num_classes,
            input_size=input_size,
            sac=_sac,
            width_mult=_width_mult,
            depth_mult=_depth_mult,
        )

    attrs: dict[str, object] = {"__init__": _init}
    if pretrained_key is not None:
        attrs["pretrained_spec"] = PretrainedSpec(
            key=pretrained_key,
            url=PRETRAINED_URLS[pretrained_key],
            state_dict_key_map={},
            remapper=yolo_remapper,
        )
    else:
        attrs["pretrained_spec"] = None
    return type(cls_name, (_YOLOv8Base,), attrs)


# Generate all 10 v8 classes (5 sizes x {baseline, SAC}).
YOLOv8n_CIoU_DFL_AdamW_SILU = _make_v8_class("n", sac=False, pretrained_key="yolov8n")
YOLOv8s_CIoU_DFL_AdamW_SILU = _make_v8_class("s", sac=False, pretrained_key="yolov8s")
YOLOv8m_CIoU_DFL_AdamW_SILU = _make_v8_class("m", sac=False, pretrained_key="yolov8m")
YOLOv8l_CIoU_DFL_AdamW_SILU = _make_v8_class("l", sac=False, pretrained_key="yolov8l")
YOLOv8x_CIoU_DFL_AdamW_SILU = _make_v8_class("x", sac=False, pretrained_key="yolov8x")
YOLOv8nSAC_CIoU_DFL_AdamW_SILU = _make_v8_class("n", sac=True, pretrained_key="yolov8n")
YOLOv8sSAC_CIoU_DFL_AdamW_SILU = _make_v8_class("s", sac=True, pretrained_key="yolov8s")
YOLOv8mSAC_CIoU_DFL_AdamW_SILU = _make_v8_class("m", sac=True, pretrained_key="yolov8m")
YOLOv8lSAC_CIoU_DFL_AdamW_SILU = _make_v8_class("l", sac=True, pretrained_key="yolov8l")
YOLOv8xSAC_CIoU_DFL_AdamW_SILU = _make_v8_class("x", sac=True, pretrained_key="yolov8x")


# Short aliases for ergonomic use.
YOLOv8n = YOLOv8n_CIoU_DFL_AdamW_SILU
YOLOv8s = YOLOv8s_CIoU_DFL_AdamW_SILU
YOLOv8m = YOLOv8m_CIoU_DFL_AdamW_SILU
YOLOv8l = YOLOv8l_CIoU_DFL_AdamW_SILU
YOLOv8x = YOLOv8x_CIoU_DFL_AdamW_SILU
YOLOv8nSAC = YOLOv8nSAC_CIoU_DFL_AdamW_SILU
YOLOv8sSAC = YOLOv8sSAC_CIoU_DFL_AdamW_SILU
YOLOv8mSAC = YOLOv8mSAC_CIoU_DFL_AdamW_SILU
YOLOv8lSAC = YOLOv8lSAC_CIoU_DFL_AdamW_SILU
YOLOv8xSAC = YOLOv8xSAC_CIoU_DFL_AdamW_SILU
