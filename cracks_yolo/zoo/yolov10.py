"""YOLOv10 model zoo: YOLOv10s + YOLOv10sSAC.

Each class is a self-contained ``nn.Module`` owning its layers, its loss
module, its optimizer-builder, and its pretrained-spec. Architecture: v10
backbone (C2fCIB stages with optional SAC at the same positions, SCDown at
P4/P5, PSA at P5), FPN+PAN neck, dual-head ``v10Detect`` (one2many for
training supervision + one2one for NMS-free inference). Loss: ``E2ELoss``
wrapping two ``v8DetectionLoss`` instances with an o2m/o2o decay schedule.

The SAC variant swaps the backbone C2fCIB stages for SAC-enabled variants.
COCO weights load with ``strict=False``; SAC layers are randomly initialized.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from cracks_yolo.losses.yolov10 import E2ELoss
from cracks_yolo.ops.conv import Conv
from cracks_yolo.ops.csp import PSA
from cracks_yolo.ops.csp import SPPF
from cracks_yolo.ops.csp import C2f
from cracks_yolo.ops.csp import C2fCIB
from cracks_yolo.ops.csp import C2fSAC
from cracks_yolo.ops.csp import Concat
from cracks_yolo.ops.csp import SCDown
from cracks_yolo.ops.detect_heads import v10Detect
from cracks_yolo.weights.loader import load_pretrained
from cracks_yolo.weights.registry import PRETRAINED_URLS
from cracks_yolo.weights.remappers import yolo_remapper
from cracks_yolo.zoo.base import PretrainedSpec
from cracks_yolo.zoo.base import default_optimizer

V10S_HYP: dict[str, float] = {
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
}


# v10s: depth_multiple=0.50, width_multiple=0.50.
def _c(c: int) -> int:
    return max(round(c * 0.5 / 8) * 8, 8)


def _n(n: int) -> int:
    return max(round(n * 0.50), 1)


def _build_v10_backbone(sac: bool) -> tuple[nn.Sequential, list[int]]:
    """Build the v10s backbone. Returns (seq, [P3, P4, P5] channels).

    Layer indices: P3 = layer 4, P4 = layer 6, P5 = layer 9 (PSA out).
    """
    c2f_block = C2fSAC if sac else C2f
    layers: list[nn.Module] = [
        Conv(3, _c(64), 3, 2),  # 0   P1/2
        Conv(_c(64), _c(128), 3, 2),  # 1   P2/4
        c2f_block(_c(128), _c(128), n=_n(4), shortcut=True),  # 2
        Conv(_c(128), _c(256), 3, 2),  # 3   P3/8
        c2f_block(_c(256), _c(256), n=_n(6), shortcut=True),  # 4
        Conv(_c(256), _c(512), 3, 2),  # 5   P4/16
        C2fCIB(_c(512), _c(512), n=_n(6), shortcut=True),  # 6
        SCDown(_c(512), _c(1024), 3, 2),  # 7   P5/32
        C2fCIB(_c(1024), _c(1024), n=_n(3), shortcut=True),  # 8
        SPPF(_c(1024), _c(1024), k=5),  # 9
        PSA(_c(1024), _c(1024), e=0.5),  # 10
    ]
    return nn.Sequential(*layers), [_c(256), _c(512), _c(1024)]


def _build_v10_neck(
    backbone_channels: list[int],
) -> tuple[nn.Sequential, list[int]]:
    """Build the v10 neck. Returns (seq, head_input_channels).

    Concat layer indices in the neck (with skip target encoded):
      idx 2  -> backbone P4 (bb[6])
      idx 5  -> backbone P3 (bb[4])
      idx 8  -> neck P4 (nk[4])
      idx 11 -> neck P5 (nk[0])
    Backbone indices encoded as themselves; neck indices as 100 + i.
    """
    c_p3, c_p4, c_p5 = backbone_channels
    layers: list[nn.Module] = [
        Conv(c_p5, _c(512), 1, 1),  # 0
        nn.Upsample(scale_factor=2, mode="nearest"),  # 1
        Concat(dimension=1),  # 2  cat backbone P4
        C2f(_c(512) + c_p4, _c(512), n=_n(3), shortcut=False),  # 3
        Conv(_c(512), _c(256), 1, 1),  # 4
        nn.Upsample(scale_factor=2, mode="nearest"),  # 5
        Concat(dimension=1),  # 6  cat backbone P3
        C2f(_c(256) + c_p3, _c(256), n=_n(3), shortcut=False),  # 7   P3/8-small
        Conv(_c(256), _c(256), 3, 2),  # 8
        Concat(dimension=1),  # 9   cat head P4
        C2f(_c(256) + _c(256), _c(512), n=_n(3), shortcut=False),  # 10  P4/16-medium
        Conv(_c(512), _c(512), 3, 2),  # 11
        Concat(dimension=1),  # 12  cat head P5
        C2fCIB(_c(512) + _c(512), _c(1024), n=_n(3), shortcut=False),  # 13  P5/32-large
    ]
    return nn.Sequential(*layers), [_c(256), _c(512), _c(1024)]


class _YOLOv10sBase(nn.Module):
    """Concrete base shared by v10s and v10s-SAC variants."""

    pretrained_spec: PretrainedSpec | None = None
    loss_parts_schema: tuple[str, ...] = ("box", "cls", "dfl")
    decode_format: str = "anchor_free"

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
        self._hyp = hyp if hyp is not None else V10S_HYP

        backbone, backbone_channels = _build_v10_backbone(sac=sac)
        neck, head_channels = _build_v10_neck(backbone_channels)
        self.backbone = backbone
        self.neck = neck
        self.head = v10Detect(
            nc=num_classes,
            ch=tuple(head_channels),
        )
        # Initialize stride via a dummy training forward (v10Detect.training
        # returns {one2many: {...}, one2one: {...}} without needing stride).
        s = 256
        with torch.no_grad():
            feats_dict = self._forward_impl(torch.zeros(1, 3, s, s))
        assert isinstance(feats_dict, dict)
        one2many = feats_dict["one2many"]
        assert isinstance(one2many, dict)
        feats = one2many["feats"]
        assert isinstance(feats, list)
        self.head.stride = torch.tensor([s / f.shape[-2] for f in feats], dtype=torch.float32)
        self.head.bias_init()
        self._loss_fn = E2ELoss(
            nc=num_classes,
            reg_max=16,
            stride=self.stride,
            hyp=self._hyp,
            device=next(self.parameters()).device,
            initial_o2m=0.8,
            final_o2m=0.1,
        )

    @property
    def stride(self) -> torch.Tensor:
        return self.head.stride

    def _forward_impl(
        self, x: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor | list[torch.Tensor]]] | torch.Tensor:
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
        head_out: dict[str, dict[str, torch.Tensor | list[torch.Tensor]]] | torch.Tensor = (
            self.head(head_inputs)
        )
        return head_out

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor | list[torch.Tensor]]] | torch.Tensor:
        return self._forward_impl(x)

    def compute_loss(
        self,
        preds: dict[str, dict[str, torch.Tensor | list[torch.Tensor]]],
        targets: torch.Tensor,
        imgs: torch.Tensor | None = None,  # noqa: ARG002 — v10 E2E loss doesn't need images
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute v10 E2E loss.

        ``targets`` is ``(N, 6)`` — (img_idx, cls, x, y, w, h) normalized.
        """
        batch = {
            "batch_idx": targets[:, 0].long(),
            "cls": targets[:, 1].long(),
            "bboxes": targets[:, 2:6],
        }
        loss_out: tuple[torch.Tensor, torch.Tensor] = self._loss_fn(preds, batch)
        # E2E loss returns a (3,) tensor of [box, cls, dfl] — sum to scalar.
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
    ) -> _YOLOv10sBase:
        spec = cls.pretrained_spec
        if spec is None:
            return cls(num_classes=num_classes)
        report = load_pretrained(
            model=cls(num_classes=num_classes),
            spec=spec,
            weights_dir=weights_dir,
            strict=strict,
        )
        model: _YOLOv10sBase = report.model  # type: ignore[assignment]
        return model


class YOLOv10s_CIoU_DFL_E2E_AdamW_SILU(_YOLOv10sBase):  # noqa: N801
    """YOLOv10s baseline (anchor-free, dual-head, NMS-free)."""

    pretrained_spec = PretrainedSpec(
        key="yolov10s",
        url=PRETRAINED_URLS["yolov10s"],
        state_dict_key_map={},
        remapper=yolo_remapper,
    )

    def __init__(self, num_classes: int = 80, input_size: int = 640) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, sac=False)


class YOLOv10sSAC_CIoU_DFL_E2E_AdamW_SILU(_YOLOv10sBase):  # noqa: N801
    """YOLOv10s with SAC in the backbone C2f stages (P2/P3/P4/P5).

    Loads the same COCO weights as the baseline; SAC layers stay random init.
    """

    pretrained_spec = PretrainedSpec(
        key="yolov10s",
        url=PRETRAINED_URLS["yolov10s"],
        state_dict_key_map={},
        remapper=yolo_remapper,
    )

    def __init__(self, num_classes: int = 80, input_size: int = 640) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, sac=True)


# Short aliases for ergonomic use.
YOLOv10s = YOLOv10s_CIoU_DFL_E2E_AdamW_SILU
YOLOv10sSAC = YOLOv10sSAC_CIoU_DFL_E2E_AdamW_SILU
