"""Implicit additive / multiplicative biases (YOLOR-style, used by YOLOv7).

Ported from ``deps/yolov7/models/common.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ImplicitA(nn.Module):
    """Implicit additive bias.

    Adds a learnable per-channel bias ``implicit`` of shape ``(1, C, 1, 1)``
    to the input. Used in YOLOv7's IDetect head before the output conv.

    Args:
        channel: Number of channels.
        mean: Init mean. Default 0.0.
        std: Init std. Default 0.02.

    Reference: Wang et al., "You Only Learn One Representation" (YOLOR).
    """

    def __init__(self, channel: int, mean: float = 0.0, std: float = 0.02) -> None:
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.implicit + x


class ImplicitM(nn.Module):
    """Implicit multiplicative gain.

    Multiplies the input by a learnable per-channel gain ``implicit`` of shape
    ``(1, C, 1, 1)``. Used in YOLOv7's IDetect head after the output conv.

    Args:
        channel: Number of channels.
        mean: Init mean. Default 1.0.
        std: Init std. Default 0.02.
    """

    def __init__(self, channel: int, mean: float = 1.0, std: float = 0.02) -> None:
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.implicit * x
