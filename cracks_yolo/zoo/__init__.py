"""Cracks-YOLO model zoo registry.

Re-exports every zoo class and provides a ``ZOO`` mapping from short names
to classes for ergonomic lookup::

    from cracks_yolo.zoo import ZOO

    model = ZOO["yolov5s_sactr"](num_classes=1)
"""

from __future__ import annotations

import torch.nn as nn

from cracks_yolo.zoo.base import DetectorModel
from cracks_yolo.zoo.base import PretrainedSpec
from cracks_yolo.zoo.base import default_optimizer
from cracks_yolo.zoo.torchvision_detectors import FCOSR50
from cracks_yolo.zoo.torchvision_detectors import SSD300VGG16
from cracks_yolo.zoo.torchvision_detectors import FasterRCNN_R50_CEA_SmoothL1_SGD_SILU
from cracks_yolo.zoo.torchvision_detectors import FasterRCNNR50
from cracks_yolo.zoo.torchvision_detectors import FCOS_R50_FocalLoss_Centerness_SGD_SILU
from cracks_yolo.zoo.torchvision_detectors import MaskRCNN_R50_CEA_SmoothL1_BCE_SGD_SILU
from cracks_yolo.zoo.torchvision_detectors import MaskRCNNR50
from cracks_yolo.zoo.torchvision_detectors import RetinaNet_R50_FocalLoss_SGD_SILU
from cracks_yolo.zoo.torchvision_detectors import RetinaNetR50
from cracks_yolo.zoo.torchvision_detectors import SSD300_VGG16_FocalLoss_SmoothL1_SGD_ReLU
from cracks_yolo.zoo.torchvision_detectors import (
    SSDlite320_MobileNetV3_FocalLoss_SmoothL1_SGD_Hardswish,
)
from cracks_yolo.zoo.torchvision_detectors import SSDlite320MobileNetV3
from cracks_yolo.zoo.yolov5 import YOLOv5s
from cracks_yolo.zoo.yolov5 import YOLOv5s_CIoU_BCEObj_BCECls_AdamW_SILU
from cracks_yolo.zoo.yolov5 import YOLOv5sSAC
from cracks_yolo.zoo.yolov5 import YOLOv5sSAC_CIoU_BCEObj_BCECls_AdamW_SILU
from cracks_yolo.zoo.yolov5 import YOLOv5sSACTR
from cracks_yolo.zoo.yolov5 import YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU
from cracks_yolo.zoo.yolov5 import YOLOv5sTR
from cracks_yolo.zoo.yolov5 import YOLOv5sTR_CIoU_BCEObj_BCECls_AdamW_SILU
from cracks_yolo.zoo.yolov7 import YOLOv7w
from cracks_yolo.zoo.yolov7 import YOLOv7w_CIouOTA_BCEObj_BCECls_AdamW_SILU
from cracks_yolo.zoo.yolov7 import YOLOv7wSAC
from cracks_yolo.zoo.yolov7 import YOLOv7wSAC_CIouOTA_BCEObj_BCECls_AdamW_SILU
from cracks_yolo.zoo.yolov8 import YOLOv8l
from cracks_yolo.zoo.yolov8 import YOLOv8l_CIoU_DFL_AdamW_SILU
from cracks_yolo.zoo.yolov8 import YOLOv8lSAC
from cracks_yolo.zoo.yolov8 import YOLOv8lSAC_CIoU_DFL_AdamW_SILU
from cracks_yolo.zoo.yolov8 import YOLOv8m
from cracks_yolo.zoo.yolov8 import YOLOv8m_CIoU_DFL_AdamW_SILU
from cracks_yolo.zoo.yolov8 import YOLOv8mSAC
from cracks_yolo.zoo.yolov8 import YOLOv8mSAC_CIoU_DFL_AdamW_SILU
from cracks_yolo.zoo.yolov8 import YOLOv8n
from cracks_yolo.zoo.yolov8 import YOLOv8n_CIoU_DFL_AdamW_SILU
from cracks_yolo.zoo.yolov8 import YOLOv8nSAC
from cracks_yolo.zoo.yolov8 import YOLOv8nSAC_CIoU_DFL_AdamW_SILU
from cracks_yolo.zoo.yolov8 import YOLOv8s
from cracks_yolo.zoo.yolov8 import YOLOv8s_CIoU_DFL_AdamW_SILU
from cracks_yolo.zoo.yolov8 import YOLOv8sSAC
from cracks_yolo.zoo.yolov8 import YOLOv8sSAC_CIoU_DFL_AdamW_SILU
from cracks_yolo.zoo.yolov8 import YOLOv8x
from cracks_yolo.zoo.yolov8 import YOLOv8x_CIoU_DFL_AdamW_SILU
from cracks_yolo.zoo.yolov8 import YOLOv8xSAC
from cracks_yolo.zoo.yolov8 import YOLOv8xSAC_CIoU_DFL_AdamW_SILU
from cracks_yolo.zoo.yolov9 import YOLOv9c
from cracks_yolo.zoo.yolov9 import YOLOv9c_CIoU_DFL_AdamW_SILU
from cracks_yolo.zoo.yolov9 import YOLOv9cSAC
from cracks_yolo.zoo.yolov9 import YOLOv9cSAC_CIoU_DFL_AdamW_SILU
from cracks_yolo.zoo.yolov10 import YOLOv10s
from cracks_yolo.zoo.yolov10 import YOLOv10s_CIoU_DFL_E2E_AdamW_SILU
from cracks_yolo.zoo.yolov10 import YOLOv10sSAC
from cracks_yolo.zoo.yolov10 import YOLOv10sSAC_CIoU_DFL_E2E_AdamW_SILU

# Registry mapping short ergonomic names to model classes. Concrete classes
# are nn.Module subclasses that structurally satisfy the DetectorModel
# Protocol — we type the registry as type[nn.Module] (the common supertype)
# and rely on the Protocol at call sites.
ZOO: dict[str, type[nn.Module]] = {
    "yolov5s": YOLOv5s,
    "yolov5s_sac": YOLOv5sSAC,
    "yolov5s_tr": YOLOv5sTR,
    "yolov5s_sactr": YOLOv5sSACTR,
    "yolov7w": YOLOv7w,
    "yolov7w_sac": YOLOv7wSAC,
    "yolov8n": YOLOv8n,
    "yolov8n_sac": YOLOv8nSAC,
    "yolov8s": YOLOv8s,
    "yolov8s_sac": YOLOv8sSAC,
    "yolov8m": YOLOv8m,
    "yolov8m_sac": YOLOv8mSAC,
    "yolov8l": YOLOv8l,
    "yolov8l_sac": YOLOv8lSAC,
    "yolov8x": YOLOv8x,
    "yolov8x_sac": YOLOv8xSAC,
    "yolov10s": YOLOv10s,
    "yolov10s_sac": YOLOv10sSAC,
    "yolov9c": YOLOv9c,
    "yolov9c_sac": YOLOv9cSAC,
    "retinanet_r50": RetinaNetR50,
    "faster_rcnn_r50": FasterRCNNR50,
    "mask_rcnn_r50": MaskRCNNR50,
    "fcos_r50": FCOSR50,
    "ssd300_vgg16": SSD300VGG16,
    "ssdlite320_mobilenetv3": SSDlite320MobileNetV3,
}

__all__ = [  # noqa: RUF022
    "DetectorModel",
    "PretrainedSpec",
    "default_optimizer",
    "ZOO",
    # Long-form class names (documentation-is-the-name).
    "YOLOv5s_CIoU_BCEObj_BCECls_AdamW_SILU",
    "YOLOv5sSAC_CIoU_BCEObj_BCECls_AdamW_SILU",
    "YOLOv5sTR_CIoU_BCEObj_BCECls_AdamW_SILU",
    "YOLOv5sSACTR_CIoU_BCEObj_BCECls_AdamW_SILU",
    "YOLOv7w_CIouOTA_BCEObj_BCECls_AdamW_SILU",
    "YOLOv7wSAC_CIouOTA_BCEObj_BCECls_AdamW_SILU",
    "YOLOv8s_CIoU_DFL_AdamW_SILU",
    "YOLOv8sSAC_CIoU_DFL_AdamW_SILU",
    "YOLOv8n_CIoU_DFL_AdamW_SILU",
    "YOLOv8nSAC_CIoU_DFL_AdamW_SILU",
    "YOLOv8m_CIoU_DFL_AdamW_SILU",
    "YOLOv8mSAC_CIoU_DFL_AdamW_SILU",
    "YOLOv8l_CIoU_DFL_AdamW_SILU",
    "YOLOv8lSAC_CIoU_DFL_AdamW_SILU",
    "YOLOv8x_CIoU_DFL_AdamW_SILU",
    "YOLOv8xSAC_CIoU_DFL_AdamW_SILU",
    "YOLOv10s_CIoU_DFL_E2E_AdamW_SILU",
    "YOLOv10sSAC_CIoU_DFL_E2E_AdamW_SILU",
    "YOLOv9c_CIoU_DFL_AdamW_SILU",
    "YOLOv9cSAC_CIoU_DFL_AdamW_SILU",
    "RetinaNet_R50_FocalLoss_SGD_SILU",
    "FasterRCNN_R50_CEA_SmoothL1_SGD_SILU",
    "MaskRCNN_R50_CEA_SmoothL1_BCE_SGD_SILU",
    "FCOS_R50_FocalLoss_Centerness_SGD_SILU",
    "SSD300_VGG16_FocalLoss_SmoothL1_SGD_ReLU",
    "SSDlite320_MobileNetV3_FocalLoss_SmoothL1_SGD_Hardswish",
    # Short aliases.
    "YOLOv5s",
    "YOLOv5sSAC",
    "YOLOv5sTR",
    "YOLOv5sSACTR",
    "YOLOv7w",
    "YOLOv7wSAC",
    "YOLOv8n",
    "YOLOv8nSAC",
    "YOLOv8s",
    "YOLOv8sSAC",
    "YOLOv8m",
    "YOLOv8mSAC",
    "YOLOv8l",
    "YOLOv8lSAC",
    "YOLOv8x",
    "YOLOv8xSAC",
    "YOLOv10s",
    "YOLOv10sSAC",
    "YOLOv9c",
    "YOLOv9cSAC",
    "RetinaNetR50",
    "FasterRCNNR50",
    "MaskRCNNR50",
    "FCOSR50",
    "SSD300VGG16",
    "SSDlite320MobileNetV3",
]
