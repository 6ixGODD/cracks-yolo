"""YOLOv9-specific operators: ADown, RepConvN, RepNCSPELAN4, SPPELAN, SP, Silence.

Ported from ``deps/yolov9/models/common.py``. These are the minimum ops
needed to build a YOLOv9-c-style backbone + neck. The v9 detection head
and PGI auxiliary branch are NOT ported — the cracks_yolo YOLOv9 wrapper
uses the v8 ``DetectAnchorFree`` head + ``v8DetectionLoss`` for a fair
same-Protocol comparison baseline (see ``cracks_yolo/zoo/yolov9.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from cracks_yolo.ops.conv import Conv


class Silence(nn.Module):
    """No-op layer (v9 backbone index 0 placeholder)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SP(nn.Module):
    """Spatial pooling — max-pool with kernel ``k`` and stride ``s``."""

    def __init__(self, k: int = 3, s: int = 1) -> None:
        super().__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


class ADown(nn.Module):
    """Avg-pool + chunk + (conv3x3/s2, maxpool+conv1x1) downsample (v9).

    Halves spatial resolution; doubles-ish channels.
    """

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = F.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class RepConvN(nn.Module):
    """RepVGG-style re-parameterizable conv (v9 variant).

    Training-time structure: 3x3 ``Conv`` (with BN+act=False) + 1x1 ``Conv``
    (with BN+act=False). At deployment, call :meth:`fuse_convs` to merge
    branches into a single ``conv`` ``Conv2d``.

    Unlike :class:`cracks_yolo.ops.csp.RepConv` (v7), this variant uses
    ``Conv`` wrappers (BN inside) and never includes an identity BN branch.

    Args:
        c1: Input channels.
        c2: Output channels.
        k: Kernel size (must be 3).
        s: Stride.
        p: Padding (must be 1).
        g: Groups.
        act: Activation spec (True = SiLU).
    """

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        g: int = 1,
        d: int = 1,  # noqa: ARG002 — kept for API symmetry; v9 never uses dilation
        act: bool | nn.Module = True,
        bn: bool = False,  # noqa: ARG002 — kept for API symmetry; v9 never uses identity BN
        deploy: bool = False,
    ) -> None:
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )
        self.deploy = deploy
        if deploy:
            self.conv = nn.Conv2d(c1, c2, k, s, padding=p, groups=g, bias=True)
        else:
            self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
            self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.conv(x))
        return self.act(self.conv1(x) + self.conv2(x))


class RepNBottleneck(nn.Module):
    """Bottleneck with :class:`RepConvN` as the first conv (v9)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    """CSP bottleneck with :class:`RepNBottleneck` inner blocks (v9)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN block — the main v9 backbone stage block.

    Two parallel RepNCSP branches concatenated with the chunked input
    projection. Output channels = ``c2``; inner branch channels = ``c4``.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
        c5: int = 1,
    ) -> None:
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class SPPELAN(nn.Module):
    """SPP-ELAN — v9 neck bottom block (replaces v5/v8 SPPF).

    Three parallel max-pool branches (k=5) concatenated with a 1x1 proj.
    """

    def __init__(self, c1: int, c2: int, c3: int) -> None:
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


__all__ = [
    "SP",
    "SPPELAN",
    "ADown",
    "RepConvN",
    "RepNBottleneck",
    "RepNCSP",
    "RepNCSPELAN4",
    "Silence",
]
