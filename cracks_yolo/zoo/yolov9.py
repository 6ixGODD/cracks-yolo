"""YOLOv9 model zoo: YOLOv9c + YOLOv9cSAC (simplified, no PGI).

Each class is a self-contained ``nn.Module`` owning its layers, its loss
module, its optimizer-builder, and its pretrained-spec.

Architecture (simplified from upstream YOLOv9-c):
- Backbone: GELAN-style stages (``RepNCSPELAN4``) with ``ADown`` downsampling.
- Neck: ``SPPELAN`` bottom + FPN/PAN with ``RepNCSPELAN4`` stages.
- Head: v8 ``DetectAnchorFree`` (separate cv2/cv3 branches + DFL).
- Loss: ``v8DetectionLoss`` (TaskAlignedAssigner topk=10, BboxLoss with
  CIoU+DFL, no obj loss).

Upstream YOLOv9 ships a Programmable Gradient Information (PGI) auxiliary
supervision branch (``DualDDetect`` head + ``CBFuse`` fusion). That branch
is omitted here — the cracks_yolo YOLOv9 is a fair same-Protocol comparison
baseline, not a perfect reproduction. Inference uses the main head only.

The SAC variant swaps the backbone ``RepNCSPELAN4`` blocks' inner
``RepNBottleneck`` for an SAC-enabled form. Since ``RepConvN`` is structurally
incompatible with SAC (re-parameterization), the SAC variant falls back to
``C2fSAC`` stages at the same backbone positions (P2/P3/P4/P5). COCO weights
load with ``strict=False``; SAC layers are randomly initialized.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from cracks_yolo.losses.yolov8 import v8DetectionLoss
from cracks_yolo.ops.conv import Conv
from cracks_yolo.ops.csp import C2fSAC
from cracks_yolo.ops.csp import Concat
from cracks_yolo.ops.detect_heads import DetectAnchorFree
from cracks_yolo.ops.yolov9 import SPPELAN
from cracks_yolo.ops.yolov9 import ADown
from cracks_yolo.ops.yolov9 import RepNCSPELAN4
from cracks_yolo.weights.loader import load_pretrained
from cracks_yolo.weights.registry import PRETRAINED_URLS
from cracks_yolo.weights.remappers import yolo_remapper
from cracks_yolo.zoo.base import PretrainedSpec
from cracks_yolo.zoo.base import default_optimizer

V9C_HYP: dict[str, float] = {
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
}


def _build_v9_backbone(sac: bool) -> tuple[nn.Sequential, list[int]]:
    """Build the v9-c backbone. Returns (seq, [P3, P4, P5] channels).

    Layer indices: P3 = layer 5, P4 = layer 7, P5 = layer 9.

    When ``sac=True``, the four ``RepNCSPELAN4`` stages are replaced with
    ``C2fSAC`` blocks of matching output channels. This is an architectural
    compromise — ``RepConvN`` cannot host SAC — but preserves the per-stage
    channel plan and skip indices.
    """
    if sac:
        # SAC variant: replace RepNCSPELAN4 with C2fSAC at the same positions.
        layers: list[nn.Module] = [
            nn.Identity(),  # 0  (Silence equivalent)
            Conv(3, 64, 3, 2),  # 1   P1/2
            Conv(64, 128, 3, 2),  # 2   P2/4
            C2fSAC(128, 256, n=1, shortcut=True),  # 3
            ADown(256, 256),  # 4
            C2fSAC(256, 512, n=1, shortcut=True),  # 5   P3
            ADown(512, 512),  # 6
            C2fSAC(512, 512, n=1, shortcut=True),  # 7   P4
            ADown(512, 512),  # 8
            C2fSAC(512, 512, n=1, shortcut=True),  # 9   P5
        ]
    else:
        layers = [
            nn.Identity(),  # 0  (Silence equivalent)
            Conv(3, 64, 3, 2),  # 1   P1/2
            Conv(64, 128, 3, 2),  # 2   P2/4
            RepNCSPELAN4(128, 256, 128, 64, c5=1),  # 3
            ADown(256, 256),  # 4
            RepNCSPELAN4(256, 512, 256, 128, c5=1),  # 5   P3
            ADown(512, 512),  # 6
            RepNCSPELAN4(512, 512, 512, 256, c5=1),  # 7   P4
            ADown(512, 512),  # 8
            RepNCSPELAN4(512, 512, 512, 256, c5=1),  # 9   P5
        ]
    return nn.Sequential(*layers), [512, 512, 512]


def _build_v9_neck(backbone_channels: list[int]) -> tuple[nn.Sequential, list[int]]:
    """Build the v9-c neck. Returns (seq, head_input_channels).

    Concat layer indices in the neck (with skip target encoded):
      idx 2  -> backbone P4 (bb[7])
      idx 5  -> backbone P3 (bb[5])
      idx 8  -> neck P4 (nk[3])
      idx 11 -> neck P5 (nk[0])
    Backbone indices encoded as themselves; neck indices as 100 + i.
    """
    _c_p3, _c_p4, c_p5 = backbone_channels
    layers: list[nn.Module] = [
        SPPELAN(c_p5, 512, 256),  # 0
        nn.Upsample(scale_factor=2, mode="nearest"),  # 1
        Concat(dimension=1),  # 2  cat backbone P4 (512)
        RepNCSPELAN4(512 + 512, 512, 512, 256, c5=1),  # 3
        nn.Upsample(scale_factor=2, mode="nearest"),  # 4
        Concat(dimension=1),  # 5  cat backbone P3 (512)
        RepNCSPELAN4(512 + 512, 256, 256, 128, c5=1),  # 6  P3/8-small
        ADown(256, 256),  # 7
        Concat(dimension=1),  # 8  cat head P4 (nk[3]=512)
        RepNCSPELAN4(256 + 512, 512, 512, 256, c5=1),  # 9  P4/16-medium
        ADown(512, 512),  # 10
        Concat(dimension=1),  # 11 cat head P5 (nk[0]=512)
        RepNCSPELAN4(512 + 512, 512, 512, 256, c5=1),  # 12 P5/32-large
    ]
    return nn.Sequential(*layers), [256, 512, 512]


class _YOLOv9cBase(nn.Module):
    """Concrete base shared by YOLOv9c and YOLOv9cSAC."""

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
        self._hyp = hyp if hyp is not None else V9C_HYP

        backbone, backbone_channels = _build_v9_backbone(sac=sac)
        neck, head_channels = _build_v9_neck(backbone_channels)
        self.backbone = backbone
        self.neck = neck
        self.head = DetectAnchorFree(
            nc=num_classes,
            reg_max=16,
            ch=tuple(head_channels),
        )
        # Initialize stride via a dummy training forward.
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
        # Concat skip targets: neck[2]->bb[7] (P4), neck[5]->bb[5] (P3),
        # neck[8]->nk[3] (head P4), neck[11]->nk[0] (head P5). Backbone
        # indices encoded as themselves; neck indices as 100 + i.
        concat_skips: dict[int, int] = {2: 7, 5: 5, 8: 103, 11: 100}
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
        head_inputs = [nk_out[6], nk_out[9], nk_out[12]]
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
        imgs: torch.Tensor | None = None,  # noqa: ARG002 — v9 loss doesn't need images
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute v9 loss (delegates to v8DetectionLoss).

        ``targets`` is ``(N, 6)`` — (img_idx, cls, x, y, w, h) normalized.
        """
        batch = {
            "batch_idx": targets[:, 0].long(),
            "cls": targets[:, 1].long(),
            "bboxes": targets[:, 2:6],
        }
        loss_out: tuple[torch.Tensor, torch.Tensor] = self._loss_fn(preds, batch)
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
    ) -> _YOLOv9cBase:
        spec = cls.pretrained_spec
        if spec is None:
            return cls(num_classes=num_classes)
        report = load_pretrained(
            model=cls(num_classes=num_classes),
            spec=spec,
            weights_dir=weights_dir,
            strict=strict,
        )
        model: _YOLOv9cBase = report.model  # type: ignore[assignment]
        return model


# Long-form class names (documentation-is-the-name).
def _v9c_init(
    self: _YOLOv9cBase,
    num_classes: int = 80,
    input_size: int = 640,
    _sac: bool = False,
) -> None:
    _YOLOv9cBase.__init__(self, num_classes=num_classes, input_size=input_size, sac=_sac)


def _v9c_sac_init(
    self: _YOLOv9cBase,
    num_classes: int = 80,
    input_size: int = 640,
    _sac: bool = True,
) -> None:
    _YOLOv9cBase.__init__(self, num_classes=num_classes, input_size=input_size, sac=_sac)


YOLOv9c_CIoU_DFL_AdamW_SILU = type(
    "YOLOv9c_CIoU_DFL_AdamW_SILU",
    (_YOLOv9cBase,),
    {
        "__init__": _v9c_init,
        "pretrained_spec": PretrainedSpec(
            key="yolov9c",
            url=PRETRAINED_URLS["yolov9c"],
            state_dict_key_map={},
            remapper=yolo_remapper,
        ),
    },
)

YOLOv9cSAC_CIoU_DFL_AdamW_SILU = type(
    "YOLOv9cSAC_CIoU_DFL_AdamW_SILU",
    (_YOLOv9cBase,),
    {
        "__init__": _v9c_sac_init,
        "pretrained_spec": PretrainedSpec(
            key="yolov9c",
            url=PRETRAINED_URLS["yolov9c"],
            state_dict_key_map={},
            remapper=yolo_remapper,
        ),
    },
)

# Short aliases.
YOLOv9c = YOLOv9c_CIoU_DFL_AdamW_SILU
YOLOv9cSAC = YOLOv9cSAC_CIoU_DFL_AdamW_SILU
