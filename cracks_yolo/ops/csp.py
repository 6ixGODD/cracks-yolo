"""CSP-style blocks, SPP variants, RepConv, and v10-specific blocks.

Ported from:
- ``deps/ultralytics/ultralytics/nn/modules/block.py`` for Bottleneck, C3, C3TR,
  C2f, SPPF, DFL (DFL is in detect_heads.py), CIB, C2fCIB, PSA, SCDown,
  RepVGGDW, Attention, Concat.
- ``deps/yolov7/models/common.py`` for SPPCSPC, RepConv.
- ``cracks_yolo/compat/ultralytics/conv.py`` (deleted) for BottleneckSAC, C3SAC.

Attribute names (``cv1``, ``cv2``, ``m``, etc.) match upstream so that COCO
pretrained state_dicts load with minimal remapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from cracks_yolo.ops.conv import Conv
from cracks_yolo.ops.transformer import TransformerBlock


class Concat(nn.Module):
    """Concatenate a list of tensors along a dimension."""

    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)


class Bottleneck(nn.Module):
    """Standard residual bottleneck: 1x1 -> k x k conv, optional residual.

    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: If True and ``c1 == c2``, add residual.
        g: Groups for the second conv.
        k: 2-tuple of kernel sizes for cv1 and cv2.
        e: Expansion ratio (hidden channels = ``c2 * e``).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int | tuple[int, int], int | tuple[int, int]] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)  # type: ignore[arg-type]
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)  # type: ignore[arg-type]
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions.

    Two parallel branches: ``cv1 -> m (n bottlenecks)`` and ``cv2`` (identity).
    Outputs are concatenated and fused by ``cv3``.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of Bottleneck blocks.
        shortcut: Pass-through to Bottleneck.
        g: Groups for Bottleneck's second conv.
        e: Expansion ratio.
    """

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
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3TR(C3):
    """C3 module with a :class:`TransformerBlock` instead of bottlenecks.

    The ``n`` parameter controls the number of transformer layers (not blocks).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)  # type: ignore[assignment]


class BottleneckSAC(nn.Module):
    """Bottleneck whose second conv is a :class:`SAConv2d`.

    Drop-in replacement for ultralytics' Bottleneck at SAC positions. Ported
    from the original ``cracks_yolo/compat/ultralytics/conv.py``.

    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: If True and ``c1 == c2``, add residual.
        g: Groups for the SAConv2d.
        k: 2-tuple of kernel sizes for cv1 and cv2.
        e: Expansion ratio (hidden channels = ``c2 * e``).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (1, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        # Late import to avoid circular dependency (conv.py imports activation,
        # not csp.py — but SAConv2d lives in conv.py and so does Conv).
        from cracks_yolo.ops.conv import SAConv2d

        self.cv2 = SAConv2d(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3SAC(C3):
    """C3 module with :class:`BottleneckSAC` blocks.

    Ported from the original ``cracks_yolo/compat/ultralytics/conv.py``. Inner
    bottlenecks use ``e=1.0`` to preserve channel width.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(BottleneckSAC(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class C2f(nn.Module):
    """Faster CSP Bottleneck with 2 convolutions (YOLOv8+).

    Splits ``cv1`` output into 2 chunks; the first chunk passes through ``n``
    bottlenecks, all intermediate outputs are concatenated and fused by ``cv2``.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of Bottleneck blocks.
        shortcut: Pass-through to Bottleneck.
        g: Groups for Bottleneck's second conv.
        e: Expansion ratio.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fSAC(C2f):
    """C2f with :class:`BottleneckSAC` inner blocks.

    Drop-in SAC variant of :class:`C2f`. The inner bottlenecks use SAC for
    the second conv, enabling adaptive receptive field selection inside the
    v8 C2f stages.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            BottleneckSAC(self.c, self.c, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)
        )


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (YOLOv5).

    Three cascaded ``MaxPool2d(k)`` operations on ``cv1`` output, all four
    tensors concatenated, fused by ``cv2``.

    Args:
        c1: Input channels.
        c2: Output channels.
        k: MaxPool kernel size.
        n: Number of pooling iterations. Default 3.
        shortcut: If True and ``c1 == c2``, add residual.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 5,
        n: int = 3,
        shortcut: bool = False,
    ) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class SPPCSPC(nn.Module):
    """CSP Spatial Pyramid Pooling (YOLOv7).

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of bottlenecks (unused in original; kept for API compat).
        shortcut: Unused in original; kept for API compat.
        g: Groups (unused in original; kept for API compat).
        e: Expansion ratio.
        k: Tuple of MaxPool kernel sizes.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,  # noqa: ARG002
        shortcut: bool = False,  # noqa: ARG002
        g: int = 1,  # noqa: ARG002
        e: float = 0.5,
        k: tuple[int, ...] = (5, 9, 13),
    ) -> None:
        super().__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class RepConv(nn.Module):
    """RepVGG-style re-parameterizable conv (YOLOv7).

    Training-time structure: 3x3 Conv+BN + 1x1 Conv+BN + (identity BN if
    ``c1 == c2 and s == 1``). At deployment, call :meth:`fuse` to merge all
    branches into a single ``rbr_reparam`` Conv2d.

    Args:
        c1: Input channels.
        c2: Output channels.
        k: Kernel size (must be 3).
        s: Stride.
        p: Padding (must autopad to 1).
        g: Groups.
        act: Activation spec.
        deploy: If True, build only the reparameterized conv (no BN branches).

    Reference: Ding et al., "RepVGG" (https://arxiv.org/abs/2101.03697).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        act: bool | nn.Module = True,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert p is None or p == 1
        pad = 1 if p is None else p
        padding_11 = pad - k // 2

        self.act = (
            nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        )

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, pad, groups=g, bias=True)
        else:
            self.rbr_identity: nn.BatchNorm2d | None = (
                nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None
            )
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, pad, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1: torch.Tensor) -> torch.Tensor:
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(
        self, branch: nn.Sequential | nn.BatchNorm2d | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if branch is None:
            return torch.zeros(1), torch.zeros(1)
        if isinstance(branch, nn.Sequential):
            conv0 = branch[0]
            bn1 = branch[1]
            assert isinstance(conv0, nn.Conv2d)
            assert isinstance(bn1, nn.BatchNorm2d)
            kernel = conv0.weight
            running_mean = bn1.running_mean
            running_var = bn1.running_var
            gamma = bn1.weight
            beta = bn1.bias
            eps = bn1.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, 3, 3),
                    dtype=torch.float32,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        assert running_mean is not None
        assert running_var is not None
        assert beta is not None
        assert gamma is not None
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse(self) -> None:
        """Merge the 3x3, 1x1, and identity branches into ``rbr_reparam``."""
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,  # original RepConv only supports s=1 in practice
            padding=1,
            dilation=1,
            groups=self.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        assert self.rbr_reparam.bias is not None
        self.rbr_reparam.bias.data = bias
        for attr in ("rbr_identity", "rbr_dense", "rbr_1x1"):
            if hasattr(self, attr):
                delattr(self, attr)


class RepVGGDW(nn.Module):
    """Depthwise RepVGG block (YOLOv10 CIB with ``lk=True``).

    Training: 7x7 dw Conv+BN + 3x3 dw Conv+BN, summed and SiLU-activated.
    Deployment: call :meth:`fuse` to merge into the 7x7 branch.

    Args:
        ed: Input/output channels (depthwise, so ``c1 == c2 == ed``).
    """

    def __init__(self, ed: int) -> None:
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class SCDown(nn.Module):
    """Spatial-Channel Downsample (YOLOv10).

    Pointwise conv (channel adjustment) followed by depthwise strided conv
    (spatial downsampling). No activation.

    Args:
        c1: Input channels.
        c2: Output channels.
        k: Depthwise conv kernel size.
        s: Depthwise conv stride.
    """

    def __init__(self, c1: int, c2: int, k: int, s: int) -> None:
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class CIB(nn.Module):
    """Compact Inverted Block (YOLOv10).

    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: If True and ``c1 == c2``, add residual.
        e: Expansion ratio.
        lk: If True, use :class:`RepVGGDW` for the depthwise stage.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        e: float = 0.5,
        lk: bool = False,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2f with :class:`CIB` blocks instead of Bottlenecks (YOLOv10).

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of CIB blocks.
        shortcut: Pass-through to CIB.
        lk: Pass-through to CIB (large kernel).
        g: Groups (unused by CIB, kept for C2f API compat).
        e: Expansion ratio.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        lk: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Multi-head self-attention (YOLOv10 PSA).

    Args:
        dim: Input/output channel dimension.
        num_heads: Number of attention heads.
        attn_ratio: Ratio of key dim to head dim.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)


class PSA(nn.Module):
    """Position-Sensitive Attention block (YOLOv10).

    Splits input into 2 chunks; the second chunk passes through Attention +
    FFN, then both chunks are concatenated and fused.

    Args:
        c1: Input channels.
        c2: Output channels (must equal ``c1``).
        e: Expansion ratio (hidden = ``c1 * e``).
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5) -> None:
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))
