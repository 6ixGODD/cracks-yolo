"""Shared types for dataset adapters.

Both YOLO and COCO sources return :class:`RawDetection` records — a
format-agnostic intermediate representation. The :class:`DetectionDataset`
torch wrapper consumes these and applies transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from typing_extensions import TypedDict


class DetectionTarget(TypedDict):
    """Target dict returned by DetectionDataset.__getitem__.

    All tensors are torch.Tensor; ``boxes`` are xyxy absolute-pixel coords.
    """

    boxes: object  # torch.Tensor of shape (N, 4) — avoid hard torch import here
    labels: object  # torch.Tensor of shape (N,) — int64
    image_id: int


@dataclass(frozen=True)
class RawDetection:
    """One image + its annotations, format-agnostic.

    ``boxes`` are xyxy normalized to [0, 1] (independent of image size —
    multiply by (w, h, w, h) at load time to get absolute pixels). This
    decouples parsing from image loading (we don't read the JPEG just to
    learn image dimensions).
    """

    image_path: Path
    image_id: int
    width: int  # original pixel width
    height: int  # original pixel height
    boxes_norm: list[tuple[float, float, float, float]]  # xyxy normalized
    labels: list[int]


SplitName = Literal["train", "valid", "test", "val"]
"""Split name. ``val`` is accepted as alias for ``valid``."""
