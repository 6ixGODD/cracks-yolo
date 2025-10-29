from __future__ import annotations

# noinspection PyUnusedImports
from cracks_yolo.nn.modules import *  # noqa: F403
import ultralytics.nn.modules

print(ultralytics.nn.modules.__all__)

model = ultralytics.YOLO("yolov5s-sac-tr.yaml").load("yolov5su.pt")
print(model.model)
print(model.trainer)

model.train(data="../.chore/outputs/cracks-detection/data.yaml")
