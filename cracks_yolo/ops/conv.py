"""Convolution operators.

`Conv` and `DWConv` match ultralytics' attribute layout (`conv`, `bn`, `act`) so
that COCO pretrained state_dicts load with minimal remapping. `ConvAWS2d` and
`SAConv2d` are ported verbatim (with type annotations) from the original
`cracks_yolo/compat/ultralytics/conv.py`, which itself followed the SAC
implementation from Wang et al. 2022.
"""

from __future__ import annotations

import typing as t
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.common_types as ct
import torch.nn.functional as F  # noqa: N812

from cracks_yolo.ops.activation import parse_activation


def autopad(
    k: int | list[int],
    p: int | list[int] | None = None,
    d: int = 1,
) -> int | list[int]:
    """Compute "same"-output padding for a convolution.

    Effective kernel size accounting for dilation: ``d * (k - 1) + 1``. Padding
    is half the effective kernel size (floor).

    Args:
        k: Kernel size (int or 2-list).
        p: Explicit padding. If None, auto-computed.
        d: Dilation rate.

    Returns:
        Padding value(s) (int if ``k`` is int, list if ``k`` is list).
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard Conv2d + BatchNorm2d + activation.

    Attribute names ``conv``, ``bn``, ``act`` match upstream YOLO state_dicts
    (both ultralytics and yolov5). The Conv2d has ``bias=False`` (bias is
    absorbed by BN).

    Args:
        c1: Input channels.
        c2: Output channels.
        k: Kernel size. Default 1.
        s: Stride. Default 1.
        p: Padding. If None, autopad. Default None.
        g: Groups. Default 1.
        d: Dilation. Default 1.
        act: Activation spec. ``True`` (default SiLU), ``False`` (Identity),
            an ``nn.Module`` instance, or a registered name string
            (see :mod:`cracks_yolo.ops.activation`).
    """

    default_act: ClassVar[nn.Module] = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | str | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)  # type: ignore[arg-type]
        self.bn = nn.BatchNorm2d(c2)
        self.act: nn.Module = self._resolve_act(act)

    @staticmethod
    def _resolve_act(act: bool | str | nn.Module) -> nn.Module:
        if act is True:
            return Conv.default_act
        if isinstance(act, nn.Module):
            return act
        if isinstance(act, str):
            return parse_activation(act)
        return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward after fusing Conv + BN into the Conv2d."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depthwise Convolution (groups = c1 = c2)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        d: int = 1,
        act: bool | str | nn.Module = True,
    ) -> None:
        super().__init__(c1, c2, k, s, p, g=c1, d=d, act=act)


class ConvAWS2d(nn.Conv2d):
    """Adaptive Weight Standardization Convolution.

    Normalizes convolution weights to zero mean and unit variance per output
    channel before each forward pass, then applies a learned affine transform
    (``gamma * normalized_weight + beta``). Ported verbatim from the original
    ``cracks_yolo/compat/ultralytics/conv.py``.

    Reference: Qiao et al., "Weight Standardization" (2019).
    """

    weight_gamma: torch.Tensor
    weight_beta: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: ct._size_2_t,
        stride: ct._size_2_t = 1,
        padding: str | ct._size_2_t = 0,
        dilation: ct._size_2_t = 1,
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
        weight_mean = (
            weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        )
        weight = weight - weight_mean
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        weight = weight / std
        return self.weight_gamma * weight + self.weight_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._get_weight(self.weight)
        return super()._conv_forward(x, weight, None)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, t.Any],
        prefix: str,
        local_metadata: dict[str, t.Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Sentinel: detect whether gamma was loaded from state_dict.
        self.weight_gamma.data.fill_(-1)
        super()._load_from_state_dict(  # type: ignore[no-untyped-call]
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if self.weight_gamma.data.mean() > 0:
            return
        # Initialize gamma and beta from current weights when loading from a
        # state_dict trained without AWS.
        weight = self.weight.data
        weight_mean = (
            weight.data.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
        )
        self.weight_beta.data.copy_(weight_mean)
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        self.weight_gamma.data.copy_(std)


class SAConv2d(ConvAWS2d):
    """Switchable Atrous Convolution.

    Learns a soft spatial switch map that fuses two parallel atrous convolutions
    (dilation ``d`` and ``3d``) on weight-standardized kernels, wrapped with
    pre/post context aggregation (global avg pool -> 1x1 conv -> broadcast ->
    residual). Ported verbatim from the original
    ``cracks_yolo/compat/ultralytics/conv.py``.

    Reference: Wang et al., "Rethinking Receptive Fields for Small Object
    Detection" (2022).
    """

    default_act: ClassVar[nn.Module] = nn.SiLU()

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
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=s,
            padding=autopad(kernel_size, p, d),  # type: ignore[arg-type]
            dilation=d,
            groups=g,
            bias=bias,
        )
        self.switch = nn.Conv2d(self.in_channels, 1, kernel_size=1, stride=s, bias=True)
        self.switch.weight.data.fill_(0)
        assert self.switch.bias is not None
        self.switch.bias.data.fill_(1)
        self.weight_diff = nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
        self.pre_context = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=True)
        self.pre_context.weight.data.fill_(0)
        assert self.pre_context.bias is not None
        self.pre_context.bias.data.fill_(0)
        self.post_context = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=1, bias=True
        )
        self.post_context.weight.data.fill_(0)
        assert self.post_context.bias is not None
        self.post_context.bias.data.fill_(0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-context: global avg pool -> 1x1 conv -> broadcast -> residual.
        avg_x = F.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # Switch prediction: reflect-pad -> 5x5 avg pool -> 1x1 conv.
        avg_x = F.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = F.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # Two atrous convs on weight-standardized kernels.
        weight = self._get_weight(self.weight)
        out_s = super()._conv_forward(x, weight, None)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p_ for p_ in self.padding)  # type: ignore[misc]
        self.dilation = tuple(3 * d_ for d_ in self.dilation)
        weight = weight + self.weight_diff
        out_l = super()._conv_forward(x, weight, None)
        out = switch * out_s + (1 - switch) * out_l
        self.padding = ori_p
        self.dilation = ori_d
        # Post-context.
        avg_x = F.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
        return self.act(self.bn(out))
