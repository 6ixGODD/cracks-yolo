"""DetectorModel adapter for the bundled official DETR implementation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from cracks_yolo.weights.loader import load_pretrained
from cracks_yolo.zoo.base import PretrainedSpec
from cracks_yolo.zoo.detr.models.backbone import build_backbone
from cracks_yolo.zoo.detr.models.detr import DETR
from cracks_yolo.zoo.detr.models.detr import SetCriterion
from cracks_yolo.zoo.detr.models.matcher import build_matcher
from cracks_yolo.zoo.detr.models.transformer import build_transformer

_COCO_CATEGORY_IDS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)


def _detr_remapper(
    state_dict: dict[str, torch.Tensor], model: nn.Module
) -> dict[str, torch.Tensor]:
    mapped = {f"_inner.{key}": value for key, value in state_dict.items()}
    if getattr(model, "num_classes", None) == 80:
        rows = torch.tensor((*_COCO_CATEGORY_IDS, 91), dtype=torch.long)
        for suffix in ("weight", "bias"):
            key = f"_inner.class_embed.{suffix}"
            mapped[key] = mapped[key].index_select(0, rows)
    return mapped


class DETR_R50_CE_L1_GIoU_AdamW(nn.Module):  # noqa: N801
    pretrained_spec = PretrainedSpec(
        key="detr_r50",
        url="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
        state_dict_key_map={},
        remapper=_detr_remapper,
    )
    loss_parts_schema = ("cls", "box", "giou")
    decode_format = "anchor_based"

    def __init__(self, num_classes: int = 80, input_size: int = 640) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self.register_buffer("_stride", torch.tensor([32.0]), persistent=False)
        args = SimpleNamespace(
            lr_backbone=1e-5,
            masks=False,
            backbone="resnet50",
            dilation=False,
            position_embedding="sine",
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            dim_feedforward=2048,
            enc_layers=6,
            dec_layers=6,
            pre_norm=False,
            num_queries=100,
            aux_loss=True,
            set_cost_class=1,
            set_cost_bbox=5,
            set_cost_giou=2,
        )
        self._inner = DETR(
            build_backbone(args),
            build_transformer(args),
            num_classes,
            args.num_queries,
            args.aux_loss,
        )
        weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        auxiliary = {
            f"{key}_{index}": value for index in range(5) for key, value in weight_dict.items()
        }
        weight_dict.update(auxiliary)
        self._criterion = SetCriterion(
            num_classes,
            build_matcher(args),
            weight_dict,
            0.1,
            ["labels", "boxes", "cardinality"],
        )

    @property
    def stride(self) -> torch.Tensor:
        return self._stride

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return self._inner((x - mean) / std)

    def compute_loss(
        self,
        preds: dict[str, Any],
        targets: torch.Tensor,
        imgs: torch.Tensor | None = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        official_targets = []
        for index in range(int(preds["pred_logits"].shape[0])):
            rows = targets[targets[:, 0].long() == index]
            official_targets.append({"labels": rows[:, 1].long(), "boxes": rows[:, 2:6]})
        losses = self._criterion(preds, official_targets)
        total = sum(
            value * self._criterion.weight_dict[key]
            for key, value in losses.items()
            if key in self._criterion.weight_dict
        )
        parts = torch.stack([losses["loss_ce"], losses["loss_bbox"], losses["loss_giou"]]).detach()
        return total, parts

    def decode(self, preds: object) -> torch.Tensor:
        if not isinstance(preds, dict):
            raise TypeError(f"Expected DETR output mapping, got {type(preds)}")
        boxes = preds["pred_boxes"].clone()
        boxes[..., 0::2] *= self.input_size
        boxes[..., 1::2] *= self.input_size
        probabilities = preds["pred_logits"].softmax(-1)[..., :-1]
        objectness = torch.ones((*boxes.shape[:2], 1), device=boxes.device, dtype=boxes.dtype)
        return torch.cat((boxes, objectness, probabilities), dim=-1)

    def build_optimizer(self) -> torch.optim.Optimizer:
        backbone = list(self._inner.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone}
        rest = [parameter for parameter in self.parameters() if id(parameter) not in backbone_ids]
        return torch.optim.AdamW(
            [{"params": rest}, {"params": backbone, "lr": 1e-5}], lr=1e-4, weight_decay=1e-4
        )

    @classmethod
    def from_pretrained(
        cls, num_classes: int, weights_dir: Path | None = None, strict: bool = False
    ) -> DETR_R50_CE_L1_GIoU_AdamW:
        return load_pretrained(
            cls(num_classes=num_classes), cls.pretrained_spec, weights_dir, strict
        ).model  # type: ignore[return-value]


DETRR50 = DETR_R50_CE_L1_GIoU_AdamW
