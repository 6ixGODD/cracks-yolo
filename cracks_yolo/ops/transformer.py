"""Transformer blocks for YOLO (C3TR / ViT-style).

Ported from ``deps/ultralytics/ultralytics/nn/modules/transformer.py`` (which
itself traces back to ``deps/yolov5/models/common.py``). LayerNorm is omitted
by design — matches upstream behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cracks_yolo.ops.conv import Conv


class TransformerLayer(nn.Module):
    """Single transformer encoder layer (no LayerNorm).

    Multi-head self-attention with explicit Q/K/V projections, then a 2-layer
    MLP. Both sub-layers use residual connections.

    Args:
        c: Channel dimension (input and output).
        num_heads: Number of attention heads.

    Reference: Carion et al., "End-to-End Object Detection with Transformers"
    (https://arxiv.org/abs/2010.11929).
    """

    def __init__(self, c: int, num_heads: int) -> None:
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer block.

    Optional 1x1 conv for channel adjustment, learnable position embedding,
    then ``num_layers`` stacked :class:`TransformerLayer` applied in the
    sequence dimension.

    Args:
        c1: Input channels.
        c2: Output channels.
        num_heads: Number of attention heads per layer.
        num_layers: Number of stacked :class:`TransformerLayer` blocks.

    Shape:
        - Input: ``(B, c1, H, W)``
        - Output: ``(B, c2, H, W)``
    """

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.conv: nn.Module | None = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            x = self.conv(x)
        b, _, h, w = x.shape
        # Flatten spatial dims to sequence: (H*W, B, C) for nn.MultiheadAttention.
        p = x.flatten(2).permute(2, 0, 1)
        out = self.tr(p + self.linear(p))
        return out.permute(1, 2, 0).reshape(b, self.c2, h, w)
