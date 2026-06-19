"""Shared pytest fixtures for the cracks_yolo test suite."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import pytest
import torch


@pytest.fixture
def tmp_weights_dir(tmp_path: Path) -> Path:
    """Per-test ``weights/`` directory for pretrained-download tests."""
    d = tmp_path / "weights"
    d.mkdir()
    return d


@pytest.fixture
def deterministic_seed() -> int:
    """Seed python, numpy, and torch RNGs; return the seed used."""
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def tiny_batch() -> dict[str, Any]:
    """A tiny random batch for zoo smoke tests.

    Returns a dict with:
      - ``images``: ``(2, 3, 640, 640)`` float tensor.
      - ``targets``: list of per-image target tensors, each ``(N, 6)`` with
        columns ``(image_idx, class, x, y, w, h)`` in COCO xywh normalized.
    """
    images = torch.randn(2, 3, 640, 640)
    targets = [
        torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.2], [0, 0, 0.3, 0.7, 0.1, 0.1]], dtype=torch.float32),
        torch.tensor([[1, 0, 0.4, 0.4, 0.15, 0.25]], dtype=torch.float32),
    ]
    return {"images": images, "targets": targets}


@pytest.fixture
def small_feature_batch() -> list[torch.Tensor]:
    """3-scale feature pyramid for detect-head tests: P3/P4/P5 at 80/40/20."""
    return [
        torch.randn(2, 64, 80, 80),
        torch.randn(2, 128, 40, 40),
        torch.randn(2, 256, 20, 20),
    ]


# COCO v5 anchors — used by v5/v7 detect-head tests.
COCO_ANCHORS_V5: tuple[tuple[int, ...], ...] = (
    (10, 13, 16, 30, 33, 23),
    (30, 61, 62, 45, 59, 119),
    (116, 90, 156, 198, 373, 326),
)


@pytest.fixture
def coco_anchors_v5() -> tuple[tuple[int, ...], ...]:
    return COCO_ANCHORS_V5
