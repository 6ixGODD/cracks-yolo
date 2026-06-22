"""Cracks-YOLO model zoo registry.

Re-exports zoo classes and provides a ``ZOO`` mapping from short names to
classes::

    from cracks_yolo.zoo import ZOO

    model = ZOO["yolov5s_sactr"](num_classes=1)

YOLO families v3/v5/v8/v9/v10 are loaded via Ultralytics with SAC/TR injected
by runtime module replacement (``cracks_yolo.zoo.ultralytics``). YOLOv7 is the
only YOLO NOT shipped by Ultralytics, so it keeps a cracks_yolo reimplementation.
torchvision detectors + DETR round out the cross-paradigm baselines.
"""

from __future__ import annotations

import torch.nn as nn

from cracks_yolo.zoo.base import DetectorModel
from cracks_yolo.zoo.base import PretrainedSpec
from cracks_yolo.zoo.base import default_optimizer
from cracks_yolo.zoo.detr import DETRR50
from cracks_yolo.zoo.detr import DETR_R50_CE_L1_GIoU_AdamW
from cracks_yolo.zoo.torchvision import FCOSR50
from cracks_yolo.zoo.torchvision import SSD300VGG16
from cracks_yolo.zoo.torchvision import FasterRCNN_R50_CEA_SmoothL1_SGD_SILU
from cracks_yolo.zoo.torchvision import FasterRCNNR50
from cracks_yolo.zoo.torchvision import FCOS_R50_FocalLoss_Centerness_SGD_SILU
from cracks_yolo.zoo.torchvision import MaskRCNN_R50_CEA_SmoothL1_BCE_SGD_SILU
from cracks_yolo.zoo.torchvision import MaskRCNNR50
from cracks_yolo.zoo.torchvision import RetinaNet_R50_FocalLoss_SGD_SILU
from cracks_yolo.zoo.torchvision import RetinaNetR50
from cracks_yolo.zoo.torchvision import SSD300_VGG16_FocalLoss_SmoothL1_SGD_ReLU
from cracks_yolo.zoo.torchvision import SSDlite320_MobileNetV3_FocalLoss_SmoothL1_SGD_Hardswish
from cracks_yolo.zoo.torchvision import SSDlite320MobileNetV3
from cracks_yolo.zoo.ultralytics import ULTRALYTICS_ZOO
# YOLOv7 is NOT shipped by Ultralytics — kept as a cracks_yolo reimplementation.
from cracks_yolo.zoo.yolov7 import YOLOv7w
from cracks_yolo.zoo.yolov7 import YOLOv7w_CIouOTA_BCEObj_BCECls_AdamW_SILU
from cracks_yolo.zoo.yolov7 import YOLOv7wSAC
from cracks_yolo.zoo.yolov7 import YOLOv7wSAC_CIouOTA_BCEObj_BCECls_AdamW_SILU

# Registry: short ergonomic names -> model classes (structurally satisfy the
# DetectorModel Protocol). v3/v5/v8/v9/v10 come from ULTRALYTICS_ZOO; v7 +
# torchvision + detr are registered explicitly below.
ZOO: dict[str, type[nn.Module]] = {
    "yolov7w": YOLOv7w,
    "yolov7w_sac": YOLOv7wSAC,
    "retinanet_r50": RetinaNetR50,
    "faster_rcnn_r50": FasterRCNNR50,
    "mask_rcnn_r50": MaskRCNNR50,
    "fcos_r50": FCOSR50,
    "ssd300_vgg16": SSD300VGG16,
    "ssdlite320_mobilenetv3": SSDlite320MobileNetV3,
    "detr_r50": DETRR50,
}
# All v3/v5/v8/v9/v10 (baselines + SAC/TR variants) from the Ultralytics path.
ZOO.update(ULTRALYTICS_ZOO)

__all__ = [  # noqa: RUF022
    "DetectorModel",
    "PretrainedSpec",
    "default_optimizer",
    "ZOO",
    "ULTRALYTICS_ZOO",
    # v7 (cracks_yolo reimpl — not Ultralytics).
    "YOLOv7w",
    "YOLOv7w_CIouOTA_BCEObj_BCECls_AdamW_SILU",
    "YOLOv7wSAC",
    "YOLOv7wSAC_CIouOTA_BCEObj_BCECls_AdamW_SILU",
    # torchvision.
    "RetinaNet_R50_FocalLoss_SGD_SILU",
    "FasterRCNN_R50_CEA_SmoothL1_SGD_SILU",
    "MaskRCNN_R50_CEA_SmoothL1_BCE_SGD_SILU",
    "FCOS_R50_FocalLoss_Centerness_SGD_SILU",
    "SSD300_VGG16_FocalLoss_SmoothL1_SGD_ReLU",
    "SSDlite320_MobileNetV3_FocalLoss_SmoothL1_SGD_Hardswish",
    "RetinaNetR50",
    "FasterRCNNR50",
    "MaskRCNNR50",
    "FCOSR50",
    "SSD300VGG16",
    "SSDlite320MobileNetV3",
    # DETR.
    "DETR_R50_CE_L1_GIoU_AdamW",
    "DETRR50",
]
