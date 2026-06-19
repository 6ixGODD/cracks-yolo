"""Cracks-YOLO operator primitives.

Conv, CSP, transformer, detect head, and SAC/TR building blocks. These are
plain ``nn.Module`` / ``nn.Conv2d`` subclasses with no ultralytics coupling.

Ported (with type annotations) from:
- ``deps/yolov5/models/common.py`` and ``deps/yolov5/models/yolo.py``
- ``deps/yolov7/models/common.py`` and ``deps/yolov7/models/yolo.py``
- ``deps/ultralytics/ultralytics/nn/modules/`` and ``deps/ultralytics/ultralytics/utils/tal.py``
- The original ``cracks_yolo/compat/ultralytics/conv.py`` (now deleted) for SAC.
"""

from __future__ import annotations

from cracks_yolo.ops.activation import parse_activation
from cracks_yolo.ops.conv import Conv
from cracks_yolo.ops.conv import ConvAWS2d
from cracks_yolo.ops.conv import DWConv
from cracks_yolo.ops.conv import SAConv2d
from cracks_yolo.ops.conv import autopad
from cracks_yolo.ops.csp import C3
from cracks_yolo.ops.csp import C3SAC
from cracks_yolo.ops.csp import C3TR
from cracks_yolo.ops.csp import CIB
from cracks_yolo.ops.csp import PSA
from cracks_yolo.ops.csp import SPPCSPC
from cracks_yolo.ops.csp import SPPF
from cracks_yolo.ops.csp import Attention
from cracks_yolo.ops.csp import Bottleneck
from cracks_yolo.ops.csp import BottleneckSAC
from cracks_yolo.ops.csp import C2f
from cracks_yolo.ops.csp import C2fCIB
from cracks_yolo.ops.csp import C2fSAC
from cracks_yolo.ops.csp import Concat
from cracks_yolo.ops.csp import RepConv
from cracks_yolo.ops.csp import RepVGGDW
from cracks_yolo.ops.csp import SCDown
from cracks_yolo.ops.detect_heads import DFL
from cracks_yolo.ops.detect_heads import DetectAnchorBased
from cracks_yolo.ops.detect_heads import DetectAnchorFree
from cracks_yolo.ops.detect_heads import IAuxDetect
from cracks_yolo.ops.detect_heads import IDetect
from cracks_yolo.ops.detect_heads import bbox2dist
from cracks_yolo.ops.detect_heads import dist2bbox
from cracks_yolo.ops.detect_heads import make_anchors
from cracks_yolo.ops.detect_heads import v10Detect
from cracks_yolo.ops.implicit import ImplicitA
from cracks_yolo.ops.implicit import ImplicitM
from cracks_yolo.ops.transformer import TransformerBlock
from cracks_yolo.ops.transformer import TransformerLayer
from cracks_yolo.ops.yolov9 import SP
from cracks_yolo.ops.yolov9 import SPPELAN
from cracks_yolo.ops.yolov9 import ADown
from cracks_yolo.ops.yolov9 import RepConvN
from cracks_yolo.ops.yolov9 import RepNBottleneck
from cracks_yolo.ops.yolov9 import RepNCSP
from cracks_yolo.ops.yolov9 import RepNCSPELAN4
from cracks_yolo.ops.yolov9 import Silence

__all__ = [  # noqa: RUF022
    # activation
    "parse_activation",
    # conv
    "autopad",
    "Conv",
    "DWConv",
    "ConvAWS2d",
    "SAConv2d",
    # csp / blocks
    "Concat",
    "Bottleneck",
    "BottleneckSAC",
    "C3",
    "C3SAC",
    "C3TR",
    "C2f",
    "C2fSAC",
    "C2fCIB",
    "CIB",
    "SPPF",
    "SPPCSPC",
    "RepConv",
    "RepVGGDW",
    "SCDown",
    "Attention",
    "PSA",
    # transformer
    "TransformerLayer",
    "TransformerBlock",
    # implicit
    "ImplicitA",
    "ImplicitM",
    # detect heads
    "DFL",
    "DetectAnchorBased",
    "DetectAnchorFree",
    "IDetect",
    "IAuxDetect",
    "v10Detect",
    "make_anchors",
    "dist2bbox",
    "bbox2dist",
    # yolov9
    "ADown",
    "RepConvN",
    "RepNCSP",
    "RepNCSPELAN4",
    "RepNBottleneck",
    "Silence",
    "SP",
    "SPPELAN",
]
