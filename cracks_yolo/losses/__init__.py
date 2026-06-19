"""Loss functions for cracks_yolo model zoo.

Each module is decoupled from the upstream ``model.hyp`` / ``model.model[-1]``
coupling — config and detect-head metadata are passed explicitly so the losses
can be constructed standalone.
"""

from __future__ import annotations

from cracks_yolo.losses._common import BCEBlurWithLogitsLoss
from cracks_yolo.losses._common import DFLoss
from cracks_yolo.losses._common import FocalLoss
from cracks_yolo.losses._common import QFocalLoss
from cracks_yolo.losses._common import bbox_iou
from cracks_yolo.losses._common import smooth_BCE
from cracks_yolo.losses._common import xywh2xyxy
from cracks_yolo.losses._common import xyxy2xywh
from cracks_yolo.losses.yolov5 import ComputeLoss
from cracks_yolo.losses.yolov7 import ComputeLossOTA
from cracks_yolo.losses.yolov8 import BboxLoss
from cracks_yolo.losses.yolov8 import TaskAlignedAssigner
from cracks_yolo.losses.yolov8 import v8DetectionLoss
from cracks_yolo.losses.yolov10 import E2ELoss

__all__ = [  # noqa: RUF022
    "BCEBlurWithLogitsLoss",
    "DFLoss",
    "FocalLoss",
    "QFocalLoss",
    "bbox_iou",
    "smooth_BCE",
    "xywh2xyxy",
    "xyxy2xywh",
    "ComputeLoss",
    "ComputeLossOTA",
    "TaskAlignedAssigner",
    "BboxLoss",
    "v8DetectionLoss",
    "E2ELoss",
]
