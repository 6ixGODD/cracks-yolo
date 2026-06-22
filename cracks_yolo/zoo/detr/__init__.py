"""Official DETR model and cracks_yolo protocol adapter."""

from cracks_yolo.zoo.detr.adapter import DETRR50
from cracks_yolo.zoo.detr.adapter import DETR_R50_CE_L1_GIoU_AdamW

__all__ = ["DETRR50", "DETR_R50_CE_L1_GIoU_AdamW"]
