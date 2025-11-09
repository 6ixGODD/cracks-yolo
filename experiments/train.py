from __future__ import annotations

import torch
import ultralytics.nn.modules

if __name__ == "__main__":
    from cracks_yolo.nn.modules import *  # noqa: F403

    torch.autograd.set_detect_anomaly(True)
    model = ultralytics.YOLO("yolov5s-sac-tr.yaml").load("yolov5s.pt")

    print(ultralytics.nn.modules.__all__)
    model.train(data="../.chore/outputs/cracks-detection/data.yaml", epochs=300, batch=12)
