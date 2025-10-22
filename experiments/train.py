from __future__ import annotations

import ultralytics.nn.modules
from cracks_yolo.nn.modules import *  # noqa: F401

print(ultralytics.nn.modules.__all__)

model = ultralytics.YOLO("yolov5s-sac-tr.yaml")
print(model)
