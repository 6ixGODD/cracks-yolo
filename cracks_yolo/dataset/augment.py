"""Mosaic + HSV augmentation for cracks_yolo training.

Mosaic stitches 4 images into a 2x2 grid of size 2*input_size, then random-crops
back to input_size — enriches small-object context (critical for thin cracks).
HSV jitters color so the model doesn't overfit illumination.

Designed as a wrapper around DetectionDataset: __getitem__ returns a
mosaic'd sample with probability ``p``, else a single augmented sample.
Boxes are in xyxy absolute pixels (post-resize to input_size).
"""

from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image
import torch

from cracks_yolo.dataset.torchadapter import DetectionDataset
from cracks_yolo.dataset.torchadapter import DetectionSample
from cracks_yolo.dataset.types import RawDetection


def augment_hsv(
    img: np.ndarray,
    hgain: float = 0.015,
    sgain: float = 0.7,
    vgain: float = 0.4,
) -> np.ndarray:
    """HSV color jitter (in-place on a uint8 HWC RGB array). Ported from yolov5."""
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * (1 + r[1]), 0, 255).astype(dtype)
        lut_val = np.clip(x * (1 + r[2]), 0, 255).astype(dtype)
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=img)
    return img


def _load_pil(record: RawDetection) -> np.ndarray:
    with Image.open(record.image_path) as im:
        return np.array(im.convert("RGB"))


def _resize_letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect, pad to size×size. Returns (img_padded, scale, dw, dh)."""  # noqa: RUF002
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))  # noqa: RUF046
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad = np.full((size, size, 3), 114, dtype=np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    pad[dh : dh + nh, dw : dw + nw] = img
    return pad, r, dw, dh


def _boxes_to_xyxy_abs(boxes_norm, scale, dw, dh, size) -> np.ndarray:
    if not boxes_norm:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.array(boxes_norm, dtype=np.float32)
    arr[:, [0, 2]] = arr[:, [0, 2]] * size * scale + dw
    arr[:, [1, 3]] = arr[:, [1, 3]] * size * scale + dh
    return arr


class MosaicDetectionDataset(DetectionDataset):
    """DetectionDataset + mosaic (4-img stitch) + HSV jitter.

    Args:
        records: list of RawDetection.
        input_size: target square size.
        mosaic_prob: probability of mosaic per sample.
        hsv: whether to apply HSV jitter.
    """

    def __init__(
        self,
        records: list[RawDetection],
        input_size: int = 640,
        mosaic_prob: float = 0.5,
        hsv: bool = True,
        augment: bool = True,
    ) -> None:
        # Bypass DetectionDataset.__init__ (we override getitem entirely).
        self.records = records
        self.input_size = input_size
        self.mosaic_prob = mosaic_prob if augment else 0.0
        self.hsv = hsv and augment
        self.augment = augment

    def _load_one(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (img_hw_bgr, boxes_xyxy_abs, labels)."""
        rec = self.records[idx]
        img = _load_pil(rec)[:, :, ::-1]  # RGB->BGR for cv2
        s = self.input_size
        img, scale, dw, dh = _resize_letterbox(img, s)
        boxes = _boxes_to_xyxy_abs(rec.boxes_norm, scale, dw, dh, s)
        labels = np.array(rec.labels, dtype=np.float32).reshape(-1, 1)
        return img, boxes, labels

    def _mosaic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self.input_size
        s2 = s * 2
        yc, xc = (int(random.uniform(s // 2, s2 - s // 2)) for _ in range(2))
        img4 = np.full((s2, s2, 3), 114, dtype=np.uint8)
        boxes4, labels4 = [], []
        indices = [random.randint(0, len(self.records) - 1) for _ in range(4)]
        for i, idx in enumerate(indices):
            img, boxes, labels = self._load_one(idx)
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - s, 0), max(yc - s, 0), xc, yc
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - s, 0), min(xc + s, s2), yc
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - s, 0), yc, xc, min(yc + s, s2)
            else:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + s, s2), min(yc + s, s2)
            # Source crop: image is s*s, place a s*s region.
            x1b = max(0, s - (x2a - x1a)) if i in (0, 2) else 0
            y1b = max(0, s - (y2a - y1a)) if i in (0, 1) else 0
            x2b, y2b = x1b + (x2a - x1a), y1b + (y2a - y1a)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            # Shift boxes into the mosaic frame.
            if boxes.shape[0]:
                dx = x1a - x1b
                dy = y1a - y1b
                boxes_m = boxes.copy()
                boxes_m[:, [0, 2]] += dx
                boxes_m[:, [1, 3]] += dy
                boxes4.append(boxes_m)
                labels4.append(labels)
        if boxes4:
            boxes4 = np.concatenate(boxes4, 0)
            labels4 = np.concatenate(labels4, 0)
        else:
            boxes4 = np.zeros((0, 4), dtype=np.float32)
            labels4 = np.zeros((0, 1), dtype=np.float32)
        # Random crop to s*s.
        img4, boxes4, labels4 = self._random_crop(img4, boxes4, labels4, s)
        return img4, boxes4, labels4

    def _random_crop(self, img: np.ndarray, boxes: np.ndarray, labels: np.ndarray, s: int):
        h, w = img.shape[:2]
        x = random.randint(0, max(0, w - s))
        y = random.randint(0, max(0, h - s))
        img = img[y : y + s, x : x + s]
        if boxes.shape[0]:
            boxes[:, [0, 2]] -= x
            boxes[:, [1, 3]] -= y
            # Clip + filter degenerate.
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, s)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, s)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes, labels = boxes[keep], labels[keep]
        return img, boxes, labels

    def __getitem__(self, idx: int) -> DetectionSample:
        s = self.input_size
        if self.augment and random.random() < self.mosaic_prob:
            img, boxes, labels = self._mosaic()
        else:
            img, boxes, labels = self._load_one(idx)
            if self.augment and random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])  # horizontal flip
                if boxes.shape[0]:
                    boxes[:, [0, 2]] = s - boxes[:, [2, 0]]
        if self.hsv:
            img = augment_hsv(img)
        # To tensor (HWC RGB uint8 -> CHW RGB float [0,1]).
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        boxes_t = torch.from_numpy(boxes).float()
        labels_t = torch.from_numpy(labels.reshape(-1)).long()
        return DetectionSample(
            image=tensor,
            boxes=boxes_t,
            labels=labels_t,
            image_id=self.records[idx].image_id,
        )

    def __len__(self) -> int:
        return len(self.records)
