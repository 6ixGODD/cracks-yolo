"""SSD300-VGG16 (torchvision) — own train/inference/save."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cracks_yolo.zoo.torchvision._base import TorchvisionBase

if TYPE_CHECKING:
    from loguru import Logger

    from cracks_yolo.zoo.base import InferenceResult
    from cracks_yolo.zoo.base import TrainConfig
    from cracks_yolo.zoo.base import TrainReport


class SSD300Model(TorchvisionBase):
    """torchvision SSD300-VGG16."""

    def __init__(self, num_classes: int = 1, input_size: int = 300, logger: Logger = None) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, logger=logger)
        from torchvision.models import VGG16_Weights
        from torchvision.models.detection import SSD300_VGG16_Weights
        from torchvision.models.detection import ssd300_vgg16
        from torchvision.models.detection.ssd import SSD
        from torchvision.models.detection.ssd import SSDHead

        self._inner: SSD = ssd300_vgg16(
            weights=SSD300_VGG16_Weights.DEFAULT,
            weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
            num_classes=91,
        )
        in_channels_list = list(self._inner.backbone.out_channels)
        num_anchors = self._inner.anchor_generator.num_anchors_per_location()
        self._inner.head = SSDHead(
            in_channels=in_channels_list,
            num_anchors=num_anchors,
            num_classes=num_classes + 1,
        )

    def train_model(self, config: TrainConfig) -> TrainReport:
        return self._run_train_loop(config, train_loader=None, val_loader=None, score_thresh=0.01)

    def inference(self, images: torch.Tensor) -> list[InferenceResult]:
        from cracks_yolo.zoo.base import ModelState

        self._assert_state(ModelState.TRAINED, "inference")
        self._inner.eval()
        with torch.no_grad():
            outs = self._inner(images)
        # SSD boxes are in 300×300 space → scale to original (640 assumed)
        scale = 640.0 / self.input_size if self.input_size != 640 else 1.0
        return self._tv_output_to_inference(outs, scale_x=scale, scale_y=scale)
