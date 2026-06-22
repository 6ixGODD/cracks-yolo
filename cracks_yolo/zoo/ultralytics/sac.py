"""SAC (Switchable Atrous Convolution) + TR (Transformer) modules compatible
with Ultralytics' C3/C2f, plus a runtime module-replacement helper.

These let us build SAC/TR variants of any Ultralytics-loaded YOLO (v3/v5/v8/
v9/v10) WITHOUT reimplementing the architecture: load the official model via
``DetectionModel(cfg)``, then ``apply_sac_tr`` surgically swaps designated
C3/C2f blocks for SAC/TR versions, copying the shared conv weights so only
the SAC-specific tensors (SAConv2d switches) stay randomly initialized.

C3 layout (ultralytics): cv1 = Conv(c1, c_, 1), cv2 = Conv(c1, c_, 1),
cv3 = Conv(2*c_, c2, 1), m = Sequential(Bottleneck(c_, c_, ...)).
C2f layout: cv1 = Conv(c1, 2*c_, 1), cv2 = Conv(c_ + c_*n, c2, 1), m = ... .
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _autopad(k, p=None, d=1):
    return (k - 1) // 2 * d if p is None else p


class ConvAWS2d(nn.Conv2d):
    """Weight-standardized Conv2d (Qiao et al. 2019)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
        self.register_buffer("weight_gamma", torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer("weight_beta", torch.zeros(self.out_channels, 1, 1, 1))

    def _get_weight(self, weight):
        wm = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        w = weight - wm
        std = torch.sqrt(w.view(w.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        return self.weight_gamma * (w / std) + self.weight_beta

    def forward(self, x):
        return super()._conv_forward(x, self._get_weight(self.weight), None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.weight_gamma.data.fill_(-1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if self.weight_gamma.data.mean() > 0:
            return
        w = self.weight.data
        wm = w.data.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        self.weight_beta.data.copy_(wm)
        std = torch.sqrt(w.view(w.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        self.weight_gamma.data.copy_(std)


class SAConv2d(nn.Module):
    """Switchable Atrous Convolution (Wang et al. 2022)."""

    default_act = nn.SiLU()

    def __init__(self, in_channels, out_channels, kernel_size=3, s=1, p=None, g=1, d=1, act=True, bias=False):
        super().__init__()
        self._conv = ConvAWS2d(in_channels, out_channels, kernel_size, stride=s,
                               padding=_autopad(kernel_size, p, d), dilation=d, groups=g, bias=bias)
        self.switch = nn.Conv2d(in_channels, 1, 1, stride=s, bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)
        self.weight_diff = nn.Parameter(torch.zeros_like(self._conv.weight))
        self.pre_context = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = nn.Conv2d(out_channels, out_channels, 1, bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        avg = self.pre_context(avg).expand_as(x)
        x = x + avg
        avg = F.pad(x, (2, 2, 2, 2), mode="replicate")
        avg = F.avg_pool2d(avg, 5, stride=1, padding=0)
        switch = self.switch(avg)
        w = self._conv._get_weight(self._conv.weight)
        out_s = self._conv._conv_forward(x, w, None)
        op, od = self._conv.padding, self._conv.dilation
        self._conv.padding = tuple(3 * p_ for p_ in self._conv.padding)
        self._conv.dilation = tuple(3 * d_ for d_ in self._conv.dilation)
        out_l = self._conv._conv_forward(x, w + self.weight_diff, None)
        self._conv.padding, self._conv.dilation = op, od
        out = switch * out_s + (1 - switch) * out_l
        avg = F.adaptive_avg_pool2d(out, 1)
        out = out + self.post_context(avg).expand_as(out)
        return self.act(self.bn(out))


def _make_bottleneck_sac(ultralytics_conv):
    """Build a BottleneckSAC class whose cv1 uses the passed Conv class."""
    class BottleneckSAC(nn.Module):
        def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
            super().__init__()
            c_ = int(c2 * e)
            self.cv1 = ultralytics_conv(c1, c_, 1, 1)
            self.cv2 = SAConv2d(c_, c2, 3, 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

    return BottleneckSAC


def _make_c3_sac(ultralytics_c3, ultralytics_conv):
    """C3SAC = ultralytics C3 with BottleneckSAC inner sequence."""
    BottleneckSAC = _make_bottleneck_sac(ultralytics_conv)
    class C3SAC(ultralytics_c3):
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)
            self.m = nn.Sequential(*(BottleneckSAC(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    return C3SAC


def _make_c2f_sac(ultralytics_c2f, ultralytics_conv):
    """C2fSAC = ultralytics C2f with BottleneckSAC inner sequence."""
    BottleneckSAC = _make_bottleneck_sac(ultralytics_conv)
    class C2fSAC(ultralytics_c2f):
        def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
            super().__init__(c1, c2, n, shortcut, g, e)
            self.m = nn.Sequential(*(BottleneckSAC(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)))
    return C2fSAC


def _copy_layer_meta(new_layer, old_layer):
    """Copy ultralytics layer routing attrs (f=from, i=index, type, np) so the
    replaced layer plugs into the model's forward graph correctly."""
    for attr in ("f", "i", "type", "np"):
        if hasattr(old_layer, attr):
            try:
                setattr(new_layer, attr, getattr(old_layer, attr))
            except AttributeError:
                pass


def _copy_shared_weights(new_module, old_module):
    """Copy state_dict keys that exist in both + shape-match (SAC-only keys stay random)."""
    new_sd = new_module.state_dict()
    old_sd = old_module.state_dict()
    copied = 0
    for k, v in old_sd.items():
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v
            copied += 1
    new_module.load_state_dict(new_sd, strict=False)
    return copied


def apply_sac_tr(model, sac_indices=(), tr_indices=()):
    """Replace C3/C2f blocks at ``sac_indices`` with SAC variants, and at
    ``tr_indices`` with C3TR (Transformer) variants, in-place on ``model.model``.

    Args:
        model: ultralytics DetectionModel (has ``.model`` Sequential).
        sac_indices: layer indices to upgrade to SAC.
        tr_indices: layer indices to upgrade to TR (C3TR; v8 C2f→C2fTR falls
            back to keeping C2f since ultralytics has no C2fTR — TR is v5-only
            by convention).
    """
    from ultralytics.nn.modules import C3
    from ultralytics.nn.modules import C3TR
    from ultralytics.nn.modules import C2f
    from ultralytics.nn.modules import Conv

    C3SAC = _make_c3_sac(C3, Conv)
    C2fSAC = _make_c2f_sac(C2f, Conv)
    seq = model.model
    for i in sac_indices:
        old = seq[i]
        if isinstance(old, C3):
            c1 = old.cv1.conv.in_channels
            c2 = old.cv3.conv.out_channels
            n = len(old.m)
            new = C3SAC(c1, c2, n, shortcut=getattr(old, "cv2", None) is not None)
            _copy_shared_weights(new, old)
            _copy_layer_meta(new, old); seq[i] = new
        elif isinstance(old, C2f):
            c1 = old.cv1.conv.in_channels
            c2 = old.cv2.conv.out_channels
            n = len(old.m)
            new = C2fSAC(c1, c2, n)
            _copy_shared_weights(new, old)
            _copy_layer_meta(new, old); seq[i] = new
    for i in tr_indices:
        old = seq[i]
        if isinstance(old, C3):
            c1 = old.cv1.conv.in_channels
            c2 = old.cv3.conv.out_channels
            n = len(old.m)
            new = C3TR(c1, c2, n, shortcut=True)
            _copy_shared_weights(new, old)
            _copy_layer_meta(new, old); seq[i] = new
        # C2f TR: ultralytics has no C2fTR; skip (TR is v5-convention).
    return model


# Per-architecture SAC/TR insertion points (layer indices into model.model).
# v5 backbone C3 at 2,4,6 (P2/P3/P4) + TR at 8 (P5). v5 head C3TR at 23.
# v8 backbone C2f at 2,4,6,8 + (no TR convention).
SAC_TR_SPEC = {
    "v5": {"sac": (2, 4, 6), "tr": (8,)},
    "v8": {"sac": (2, 4, 6, 8), "tr": ()},
}


__all__ = ["SAC_TR_SPEC", "ConvAWS2d", "SAConv2d", "apply_sac_tr"]
