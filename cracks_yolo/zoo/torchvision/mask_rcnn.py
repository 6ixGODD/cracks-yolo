"""Mask R-CNN-R50 (torchvision) — own train/inference/save."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cracks_yolo.zoo.torchvision._base import TorchvisionBase

if TYPE_CHECKING:
    from loguru import Logger

    from cracks_yolo.zoo.base import InferenceResult
    from cracks_yolo.zoo.base import TrainConfig
    from cracks_yolo.zoo.base import TrainReport


class MaskRCNNModel(TorchvisionBase):
    """torchvision Mask R-CNN-ResNet50-FPN."""

    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Logger = None) -> None:
        super().__init__(num_classes=num_classes, input_size=input_size, logger=logger)
        from torchvision.models import ResNet50_Weights
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        self._inner = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
            num_classes=91,
        )
        in_features = self._inner.roi_heads.box_predictor.cls_score.in_features
        self._inner.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        in_features_mask = self._inner.roi_heads.mask_predictor.mask_fcn_logits.in_channels
        self._inner.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask,
            dim_reduced=256,
            num_classes=num_classes + 1,
        )
        self._needs_masks = True
        self._print_model_summary()

    def train_model(self, config: TrainConfig) -> TrainReport:
        return self._run_train_loop(config, train_loader=None, val_loader=None, score_thresh=0.01)

    def inference(self, images: torch.Tensor) -> list[InferenceResult]:
        from cracks_yolo.zoo.base import ModelState

        self._assert_state(ModelState.TRAINED, "inference")
        self._inner.eval()
        with torch.no_grad():
            outs = self._inner(images)
        return self._tv_output_to_inference(outs)
