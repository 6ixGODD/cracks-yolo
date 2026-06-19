"""DetectorModel Protocol + PretrainedSpec dataclass.

Defines the *structural* contract that every zoo model satisfies. No
inheritance — models satisfy the Protocol independently.

Each model class declares a ``pretrained_spec`` (or ``None``) so the loader
knows which COCO weights to fetch and how to remap state_dict keys.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import Self
from typing import runtime_checkable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PretrainedSpec:
    """Where to fetch COCO pretrained weights for a model class.

    Attributes:
        key: Short identifier used as ``weights/{key}.pt`` cache filename.
        url: HTTPS URL to the official ``.pt`` release.
        state_dict_key_map: Mapping from official state_dict key prefixes
            to the cracks_yolo layout. Empty means no remap needed.
    """

    key: str
    url: str
    state_dict_key_map: dict[str, str]


@runtime_checkable
class DetectorModel(Protocol):
    """Structural contract for cracks_yolo detector models.

    Pipelines rely only on this Protocol — never on ``isinstance(model, ...)``.
    The standard ``nn.Module`` methods (``to``, ``train``, ``eval``,
    ``parameters``, ``state_dict``, ``__call__``) are part of the contract
    so pipelines can move models to device, switch train/eval mode, iterate
    parameters for grad clipping, and save checkpoints.
    """

    # Class-level configuration exposed as properties.
    input_size: int
    num_classes: int
    class_names: list[str]
    stride: torch.Tensor
    pretrained_spec: PretrainedSpec | None
    # Self-description consumed by the pipeline (avoids class-name branching).
    # Names of each entry in the ``parts`` tensor returned by ``compute_loss``.
    loss_parts_schema: tuple[str, ...]
    # Decode output layout: "anchor_free" (B, 4+nc, N), "anchor_based"
    # (B, N, nc+5), or "list_dict" (B list of {boxes, labels, scores}).
    decode_format: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def forward(
        self, x: torch.Tensor | list[torch.Tensor]
    ) -> (
        list[torch.Tensor]
        | tuple[torch.Tensor, list[torch.Tensor]]
        | dict[str, torch.Tensor | list[torch.Tensor]]
        | dict[str, dict[str, torch.Tensor | list[torch.Tensor]]]
        | torch.Tensor
    ):
        """Training forward returns raw head outputs; eval forward decodes."""
        ...

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move model to device/dtype (mirrors ``nn.Module.to``)."""
        ...

    def train(self, mode: bool = True) -> Self:
        """Set training mode (mirrors ``nn.Module.train``)."""
        ...

    def eval(self) -> Self:
        """Set eval mode (mirrors ``nn.Module.eval``)."""
        ...

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Iterate model parameters (mirrors ``nn.Module.parameters``)."""
        ...

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Return state dict (mirrors ``nn.Module.state_dict``)."""
        ...

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> Any:
        """Load state dict (mirrors ``nn.Module.load_state_dict``)."""
        ...

    def compute_loss(
        self,
        preds: object,
        targets: torch.Tensor | dict[str, torch.Tensor],
        imgs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (total_loss, parts_detach).

        ``imgs`` is optional (used by OTA-based losses like v7 that need
        image dimensions for assignment).
        """
        ...

    def decode(self, preds: object) -> torch.Tensor:
        """Decode model predictions into final detections."""
        ...

    def build_optimizer(self) -> torch.optim.Optimizer:
        """Construct the default optimizer for this model's parameters."""
        ...


def default_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    """Default AdamW optimizer for cracks_yolo models."""
    return torch.optim.AdamW(model.parameters(), lr=lr)
