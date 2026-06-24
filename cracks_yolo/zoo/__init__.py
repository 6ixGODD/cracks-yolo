"""Cracks-YOLO model zoo registry.

ZOO maps short names to model classes. All YOLO families (v3/v5/v6/v8/v9/v10/
rtdetr + yolo11/12/26) come from the ultralytics adapter; 6 torchvision
detectors each have their own file.
"""

from cracks_yolo.zoo.torchvision import ZOO as _TV_ZOO
from cracks_yolo.zoo.ultralytics import ZOO as _UL_ZOO

ZOO: dict[str, type] = {}
ZOO.update(_UL_ZOO)
ZOO.update(_TV_ZOO)
