from __future__ import annotations

import contextlib

import torch.nn
import ultralytics.nn.modules
from ultralytics.nn.modules import AIFI
from ultralytics.nn.modules import C1
from ultralytics.nn.modules import C2
from ultralytics.nn.modules import C2PSA
from ultralytics.nn.modules import C3
from ultralytics.nn.modules import C3TR
from ultralytics.nn.modules import ELAN1
from ultralytics.nn.modules import OBB
from ultralytics.nn.modules import PSA
from ultralytics.nn.modules import SPP
from ultralytics.nn.modules import SPPELAN
from ultralytics.nn.modules import SPPF
from ultralytics.nn.modules import A2C2f
from ultralytics.nn.modules import AConv
from ultralytics.nn.modules import ADown
from ultralytics.nn.modules import Bottleneck
from ultralytics.nn.modules import BottleneckCSP
from ultralytics.nn.modules import C2f
from ultralytics.nn.modules import C2fAttn
from ultralytics.nn.modules import C2fCIB
from ultralytics.nn.modules import C2fPSA
from ultralytics.nn.modules import C3Ghost
from ultralytics.nn.modules import C3k2
from ultralytics.nn.modules import C3x
from ultralytics.nn.modules import CBFuse
from ultralytics.nn.modules import CBLinear
from ultralytics.nn.modules import Classify
from ultralytics.nn.modules import Concat
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules import ConvTranspose
from ultralytics.nn.modules import Detect
from ultralytics.nn.modules import DWConv
from ultralytics.nn.modules import DWConvTranspose2d
from ultralytics.nn.modules import Focus
from ultralytics.nn.modules import GhostBottleneck
from ultralytics.nn.modules import GhostConv
from ultralytics.nn.modules import HGBlock
from ultralytics.nn.modules import HGStem
from ultralytics.nn.modules import ImagePoolingAttn
from ultralytics.nn.modules import Index
from ultralytics.nn.modules import Pose
from ultralytics.nn.modules import RepC3
from ultralytics.nn.modules import RepNCSPELAN4
from ultralytics.nn.modules import ResNetLayer
from ultralytics.nn.modules import RTDETRDecoder
from ultralytics.nn.modules import SCDown
from ultralytics.nn.modules import Segment
from ultralytics.nn.modules import TorchVision
from ultralytics.nn.modules import WorldDetect
from ultralytics.nn.modules import YOLOEDetect
from ultralytics.nn.modules import YOLOESegment
from ultralytics.nn.modules import v10Detect
import ultralytics.nn.tasks
from ultralytics.utils import LOGGER
from ultralytics.utils import colorstr
from ultralytics.utils.ops import make_divisible

from cracks_yolo.compat.ultralytics import C3SAC


def parse_model(d, ch, verbose=True):
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        model (torch.nn.Sequential): PyTorch model.
        save (list): Sorted list of output layers.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (
        d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")
    )
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(
            act
        )  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    base_modules = frozenset({
        Classify,
        Conv,
        ConvTranspose,
        GhostConv,
        Bottleneck,
        GhostBottleneck,
        SPP,
        SPPF,
        C2fPSA,
        C2PSA,
        DWConv,
        Focus,
        BottleneckCSP,
        C1,
        C2,
        C2f,
        C3k2,
        RepNCSPELAN4,
        ELAN1,
        ADown,
        AConv,
        SPPELAN,
        C2fAttn,
        C3,
        C3SAC,
        C3TR,
        C3Ghost,
        torch.nn.ConvTranspose2d,
        DWConvTranspose2d,
        C3x,
        RepC3,
        PSA,
        SCDown,
        C2fCIB,
        A2C2f,
    })
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3SAC,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1)
                    if args[2] > 1
                    else args[2]
                )

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset({
            Detect,
            WorldDetect,
            YOLOEDetect,
            Segment,
            YOLOESegment,
            Pose,
            OBB,
            ImagePoolingAttn,
            v10Detect,
        }):
            args.append([ch[x] for x in f])
            if m is Segment or m is YOLOESegment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # print
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


ultralytics.nn.tasks.parse_model = parse_model
