"""SAC-v2 and TR variants with expanded dilation and context (no modifications to sac.py).

Differences from sac.py:
- SAConv2d_v2:  dilation ratio 4 (was 3),  pre-context kernel 5 (was 1)
- BottleneckSAC_v2, C3SAC_v2:  drop-in replacements using SAConv2d_v2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from cracks_yolo.ops.sac import ConvAWS2d


class SAConv2dV2(nn.Module):
    """Switchable Atrous Convolution — v2 (dilation ratio 4, larger context).

    Unlike SAConv2d (ratio 3), this variant uses a 4x dilation ratio for
    the large-kernel branch and a 5x5 pre-context convolution for wider
    spatial context aggregation — beneficial for thin elongated cracks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        _ratio = 4
        pad = (kernel_size - 1) // 2 * d if p is None else p
        self._conv = ConvAWS2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=s,
            padding=pad,
            dilation=d,
            groups=g,
            bias=bias,
        )
        self.switch = nn.Conv2d(in_channels, 1, kernel_size=1, stride=s, bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)
        self.weight_diff = nn.Parameter(torch.zeros_like(self._conv.weight))
        self.pre_context = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)
        self.bn = nn.BatchNorm2d(out_channels)
        self._ratio = _ratio
        if act is True:
            self.act: nn.Module = nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.pre_context(x)
        x = x + avg
        avg = F.pad(x, (2, 2, 2, 2), mode="replicate")
        avg = F.avg_pool2d(avg, 5, stride=1, padding=0)
        switch = self.switch(avg)
        w = self._conv._get_weight(self._conv.weight)
        out_s = self._conv._conv_forward(x, w, None)
        op, od = self._conv.padding, self._conv.dilation
        self._conv.padding = tuple(self._ratio * p_ for p_ in self._conv.padding)
        self._conv.dilation = tuple(self._ratio * d_ for d_ in self._conv.dilation)
        out_l = self._conv._conv_forward(x, w + self.weight_diff, None)
        self._conv.padding, self._conv.dilation = op, od
        out = switch * out_s + (1 - switch) * out_l
        avg = F.adaptive_avg_pool2d(out, 1)
        out = out + self.post_context(avg).expand_as(out)
        return self.act(self.bn(out))


class BottleneckSACV2(nn.Module):
    """Bottleneck whose second conv is SAConv2dV2."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5) -> None:
        super().__init__()
        c_ = int(c2 * e)
        from ultralytics.nn.modules.conv import Conv as _UltraConv

        self.cv1 = _UltraConv(c1, c_, 1, 1)
        self.cv2 = SAConv2dV2(c_, c2, 3, s=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3SACV2(nn.Module):
    """CSP Bottleneck with 3 convs + SAC-v2 inner bottlenecks."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        from ultralytics.nn.modules.conv import Conv as _UltraConv

        self.cv1 = _UltraConv(c1, c_, 1, 1)
        self.cv2 = _UltraConv(c1, c_, 1, 1)
        self.cv3 = _UltraConv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[BottleneckSACV2(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


# ---------------------------------------------------------------------------
# Transformer v3 — identity-initialized attention (no random-init noise)
# ---------------------------------------------------------------------------


class TransformerLayerV3(nn.Module):
    """Transformer layer with identity-initialized QKV.

    Q and K start near zero → uniform softmax attention (no bias).
    V starts as identity → features pass through unchanged.
    MLP starts near-zero → residual dominates early training.
    The model gradually learns to deviate from identity as training progresses.
    """

    def __init__(self, c: int, num_heads: int = 4) -> None:
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        # Identity-init: Q, K near zero, V near identity
        nn.init.normal_(self.q.weight, std=1e-4)
        nn.init.normal_(self.k.weight, std=1e-4)
        nn.init.normal_(self.v.weight, std=0.0)
        self.v.weight.data += torch.eye(c)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)
        # MLP near-identity
        nn.init.normal_(self.fc1.weight, std=1e-4)
        nn.init.normal_(self.fc2.weight, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ma(self.q(x), self.k(x), self.v(x))[0]
        return x + self.fc2(self.fc1(x))


class TransformerBlockV3(nn.Module):
    """Identity-initialized transformer block — v3."""

    def __init__(self, c1: int, c2: int, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, c2, 1, 1))
        self.layers = nn.Sequential(*[TransformerLayerV3(c2, num_heads) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        b, c, h, w = x.shape
        p = x.flatten(2).permute(0, 2, 1)
        p = p + self.pos_embed.squeeze(-1).squeeze(-1).unsqueeze(0)
        p = self.layers(p)
        return p.permute(0, 2, 1).reshape(b, c, h, w)


class C3TRV3(nn.Module):
    """C3 with identity-initialized TransformerBlockV3."""

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
        _ = (n, shortcut, g)  # interface compat with C3
        c_ = int(c2 * e)
        from ultralytics.nn.modules.conv import Conv as _UltraConv

        self.cv1 = _UltraConv(c1, c_, 1, 1)
        self.cv2 = _UltraConv(c1, c_, 1, 1)
        self.m = TransformerBlockV3(c_, c_, num_heads=4, num_layers=2)
        self.cv3 = _UltraConv(2 * c_, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
