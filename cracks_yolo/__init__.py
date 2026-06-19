"""Cracks-YOLO: self-contained PyTorch YOLO model zoo for cracks detection."""

from __future__ import annotations

from cracks_yolo.zoo import ZOO
from cracks_yolo.zoo.base import DetectorModel
from cracks_yolo.zoo.base import PretrainedSpec

__version__ = "0.1.0"

__all__ = [
    "ZOO",
    "DetectorModel",
    "PretrainedSpec",
    "__version__",
]
