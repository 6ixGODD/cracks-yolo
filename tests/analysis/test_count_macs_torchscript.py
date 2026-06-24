"""Regression tests for cracks_yolo.analysis.model._count_macs.

Guards the torchscript / frozen-ScriptModule MACs path: thop raises on
ScriptModules and the result was silently 0 before the
FlopCounterMode fallback was added.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cracks_yolo.analysis.model import _count_macs


class _TinyConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_count_macs_eager() -> None:
    """Eager modules: thop path returns a positive MAC count."""
    model = _TinyConv().eval()
    macs = _count_macs(model, input_size=64)
    assert macs > 0.0


def test_count_macs_torchscript_nonzero() -> None:
    """Traced (frozen) ScriptModule: must NOT silently return 0.

    thop raises ``AttributeError`` on ScriptModules; the FlopCounterMode
    fallback must recover a non-zero MAC count.
    """
    model = _TinyConv().eval()
    x = torch.zeros(1, 3, 64, 64)
    ts = torch.jit.trace(model, x).eval()
    ts.requires_grad_(False)  # freeze like a deployed torchscript
    macs = _count_macs(ts, input_size=64)
    assert macs > 0.0, f"expected non-zero MACs for torchscript, got {macs}"


def test_count_macs_torchscript_matches_eager() -> None:
    """Torchscript MACs (via FlopCounterMode) match eager MACs (via thop).

    Both count the same conv; FlopCounterMode returns FLOPs and we halve it,
    so the torchscript value should equal the eager thop value.
    """
    model = _TinyConv().eval()
    eager_macs = _count_macs(model, input_size=64)
    x = torch.zeros(1, 3, 64, 64)
    ts = torch.jit.trace(model, x).eval()
    ts_macs = _count_macs(ts, input_size=64)
    assert abs(eager_macs - ts_macs) / max(eager_macs, 1.0) < 0.05
