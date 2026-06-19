"""Transforms pipeline for detection.

A detection transform takes ``(PIL_image, boxes_xyxy_normalized, labels)``
and returns ``(tensor_chw_normalized, boxes_xyxy_absolute_pixels, labels)``.
We avoid torchvision's ``transforms.v2`` API (still in flux) and write a
small callable class instead — keeps the type contract explicit.

Normalization: pixels scaled to [0, 1] (no mean/std subtraction — YOLO
convention). Resize: bilinear to ``input_size``. Train augmentation:
horizontal flip only (kept conservative so baseline comparisons aren't
confounded by heavy augmentation).
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from PIL import Image
import torch
from torchvision.transforms import functional as F  # noqa: N812


@dataclass
class DetectionTransform:
    """Resize + (optional) flip + ToTensor.

    Attributes:
        input_size: Target square size (e.g. 640).
        train: If True, apply random horizontal flip with prob 0.5.
        augment: If False, suppresses augmentation even when ``train=True``.
    """

    input_size: int
    train: bool = False
    augment: bool = True

    def __call__(
        self,
        image: Image.Image,
        boxes_norm: list[tuple[float, float, float, float]],
        labels: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the transform.

        Returns:
            (image_tensor (3, H, W) float in [0, 1],
             boxes_xyxy_abs (N, 4) float,
             labels (N,) long)
        """
        _orig_w, _orig_h = image.size
        image = image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        tensor = F.to_tensor(image)

        scale = torch.tensor(
            [self.input_size, self.input_size, self.input_size, self.input_size],
            dtype=torch.float32,
        )
        if not boxes_norm:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            lbl = torch.zeros((0,), dtype=torch.long)
            return tensor, boxes, lbl

        boxes = torch.tensor(boxes_norm, dtype=torch.float32) * scale
        lbl = torch.tensor(labels, dtype=torch.long)

        if self.train and self.augment and random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
            if boxes.numel() > 0:
                boxes[:, [0, 2]] = self.input_size - boxes[:, [2, 0]]

        return tensor, boxes, lbl


def build_transforms(
    input_size: int,
    train: bool = False,
    augment: bool = True,
) -> DetectionTransform:
    """Factory: returns a DetectionTransform with the given config."""
    return DetectionTransform(input_size=input_size, train=train, augment=augment)
