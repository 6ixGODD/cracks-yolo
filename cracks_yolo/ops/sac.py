"""SAC (Switchable Atrous Convolution) + TR (Transformer) operators.

These are the novel operators proposed in this work for tongue surface crack
detection. SAC lets the network choose its receptive field per-pixel — critical
for thin cracks that vary wildly in scale. TR adds global self-attention to the
backbone's deepest stage.

All classes are defined at **module top level** (not in factory closures) so
they can be pickled for checkpoint saving. The ultralytics adapter's
``apply_sac_tr()`` replaces C3/C2f/BottleneckCSP blocks at designated
backbone indices with these SAC variants at runtime, copying shared conv weights
so only the SAC-specific tensors (switches, context convs) stay randomly
initialized.

References:
    - Weight Standardization: Qiao et al., 2019
    - SAC: Wang et al., "Rethinking Receptive Fields for Small Object Detection", 2022
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# ---------------------------------------------------------------------------
# Weight-standardized convolution (basis for SAC)
# ---------------------------------------------------------------------------


class ConvAWS2d(nn.Conv2d):
    """Adaptive Weight Standardization Conv2d.

    Normalizes convolution weights to zero mean / unit variance per output
    channel before each forward, then applies a learned affine transform.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.register_buffer("weight_gamma", torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer("weight_beta", torch.zeros(self.out_channels, 1, 1, 1))

    def _get_weight(self, weight: torch.Tensor) -> torch.Tensor:
        wm = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        w = weight - wm
        std = torch.sqrt(w.view(w.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        return self.weight_gamma * (w / std) + self.weight_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super()._conv_forward(x, self._get_weight(self.weight), None)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        self.weight_gamma.data.fill_(-1)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if self.weight_gamma.data.mean() > 0:
            return
        w = self.weight.data
        wm = w.data.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        self.weight_beta.data.copy_(wm)
        std = torch.sqrt(w.view(w.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        self.weight_gamma.data.copy_(std)


# ---------------------------------------------------------------------------
# Switchable Atrous Convolution
# ---------------------------------------------------------------------------


class SAConv2d(nn.Module):
    """Switchable Atrous Convolution.

    Learns a soft spatial switch that fuses two parallel atrous convolutions
    (dilation d and 3d) on weight-standardized kernels, wrapped with pre/post
    context aggregation (global avg pool → 1×1 conv → broadcast → residual).
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
        self.pre_context = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)
        self.bn = nn.BatchNorm2d(out_channels)
        if act is True:
            self.act: nn.Module = nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-context: global avg pool → 1×1 conv → broadcast → residual.
        avg = F.adaptive_avg_pool2d(x, 1)
        avg = self.pre_context(avg).expand_as(x)
        x = x + avg
        # Switch prediction: reflect-pad → 5×5 avg pool → 1×1 conv.
        avg = F.pad(x, (2, 2, 2, 2), mode="replicate")
        avg = F.avg_pool2d(avg, 5, stride=1, padding=0)
        switch = self.switch(avg)
        # Two atrous convs on weight-standardized kernels.
        w = self._conv._get_weight(self._conv.weight)
        out_s = self._conv._conv_forward(x, w, None)
        op, od = self._conv.padding, self._conv.dilation
        self._conv.padding = tuple(3 * p_ for p_ in self._conv.padding)
        self._conv.dilation = tuple(3 * d_ for d_ in self._conv.dilation)
        out_l = self._conv._conv_forward(x, w + self.weight_diff, None)
        self._conv.padding, self._conv.dilation = op, od
        out = switch * out_s + (1 - switch) * out_l
        # Post-context.
        avg = F.adaptive_avg_pool2d(out, 1)
        out = out + self.post_context(avg).expand_as(out)
        return self.act(self.bn(out))


# ---------------------------------------------------------------------------
# Transformer block (TR enhancement)
# ---------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    """Transformer layer (QKV + MHA + MLP, no LayerNorm — matches ultralytics v5/v8)."""

    def __init__(self, c: int, num_heads: int = 8) -> None:
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ma(self.q(x), self.k(x), self.v(x))[0]
        return x + self.fc2(self.fc1(x))


class TransformerBlock(nn.Module):
    """Conv projection → learned positional embedding → N × TransformerLayer."""

    def __init__(self, c1: int, c2: int, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, c2, 1, 1))
        self.layers = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        b, c, h, w = x.shape
        p = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        p = p + self.pos_embed.squeeze(-1).squeeze(-1).unsqueeze(0)
        p = self.layers(p)
        return p.permute(0, 2, 1).reshape(b, c, h, w)


# ---------------------------------------------------------------------------
# SAC bottleneck — the inner block that replaces Bottleneck inside C3/C2f
# ---------------------------------------------------------------------------


class BottleneckSAC(nn.Module):
    """Bottleneck whose second conv is SAConv2d.

    Drop-in replacement for ultralytics' ``Bottleneck`` at SAC positions.
    The first 1×1 conv is the standard ultralytics ``Conv`` (imported lazily
    to avoid circular deps), the second 3×3 conv is our SAConv2d.
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
        from ultralytics.nn.modules.conv import Conv

        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = SAConv2d(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ---------------------------------------------------------------------------
# C3SAC — v5/v7 backbone C3 with SAC bottlenecks
# ---------------------------------------------------------------------------


class C3SAC(nn.Module):
    """CSP Bottleneck with 3 convs + SAC inner bottlenecks (v5/v7 style).

    Mirrors ultralytics' ``C3`` but replaces the inner ``Bottleneck`` sequence
    with ``BottleneckSAC(e=1.0)``.
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
        from ultralytics.nn.modules.conv import Conv

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(BottleneckSAC(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# ---------------------------------------------------------------------------
# C2fSAC — v8/v9/v10 backbone C2f with SAC bottlenecks
# ---------------------------------------------------------------------------


class C2fSAC(nn.Module):
    """CSP Bottleneck with 2 convs + SAC inner bottlenecks (v8/v9/v10 style).

    Mirrors ultralytics' ``C2f`` but replaces the inner ``Bottleneck`` sequence
    with ``BottleneckSAC(e=1.0)``.
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
        from ultralytics.nn.modules.conv import Conv

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            BottleneckSAC(self.c, self.c, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ---------------------------------------------------------------------------
# C3TR — C3 with Transformer block (TR enhancement, already in ultralytics but
# we provide our own for consistency)
# ---------------------------------------------------------------------------


class C3TR(nn.Module):
    """C3 module with TransformerBlock for enhanced feature extraction."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,  # noqa: ARG002
        g: int = 1,  # noqa: ARG002
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        from ultralytics.nn.modules.conv import Conv

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = TransformerBlock(c_, c_, 4, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
