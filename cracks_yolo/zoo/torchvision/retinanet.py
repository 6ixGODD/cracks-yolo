"""RetinaNet-R50 (torchvision) — own train/inference/save."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cracks_yolo.zoo.torchvision._base import TorchvisionBase

if TYPE_CHECKING:
    from loguru import Logger

    from cracks_yolo.zoo.base import InferenceResult
    from cracks_yolo.zoo.base import TrainConfig
    from cracks_yolo.zoo.base import TrainReport


class RetinaNetModel(TorchvisionBase):
    """torchvision RetinaNet-ResNet50-FPN."""

    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Logger = None) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, logger=logger)
        from torchvision.models import ResNet50_Weights
        from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
        from torchvision.models.detection import retinanet_resnet50_fpn
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead

        # Build with pretrained backbone + custom head
        self._inner = retinanet_resnet50_fpn(
            weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
            num_classes=91,
        )
        # Replace classification head for our num_classes (+1 for bg)
        num_anchors = self._inner.head.classification_head.num_anchors
        self._inner.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes + 1,
        )
        self._inner.score_thresh = 0.01  # lower for recall during val

    def train_model(self, config: TrainConfig) -> TrainReport:
        return self._run_train_loop(config, train_loader=None, val_loader=None, score_thresh=0.01)

    def inference(self, images: torch.Tensor) -> list[InferenceResult]:
        self._assert_state(self._trained_state(), "inference")
        self._inner.eval()
        with torch.no_grad():
            outs = self._inner(images)
        return self._tv_output_to_inference(outs)

    def _trained_state(self):
        from cracks_yolo.zoo.base import ModelState

        return ModelState.TRAINED
