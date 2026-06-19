"""Tests for cracks_yolo.losses.yolov5.ComputeLoss."""

from __future__ import annotations

import pytest
import torch

from cracks_yolo.losses.yolov5 import ComputeLoss

# COCO v5 anchors as (nl, na, 2)
V5_ANCHORS_FLAT: tuple[tuple[int, ...], ...] = (
    (10, 13, 16, 30, 33, 23),
    (30, 61, 62, 45, 59, 119),
    (116, 90, 156, 198, 373, 326),
)


@pytest.fixture
def v5_anchors() -> torch.Tensor:
    return torch.tensor(V5_ANCHORS_FLAT, dtype=torch.float32).view(3, 3, 2)


@pytest.fixture
def v5_hyp() -> dict[str, float]:
    return {
        "box": 0.05,
        "obj": 0.7,
        "cls": 0.3,
        "cls_pw": 1.0,
        "obj_pw": 1.0,
        "anchor_t": 4.0,
        "fl_gamma": 0.0,
        "label_smoothing": 0.0,
    }


@pytest.fixture
def v5_predictions() -> list[torch.Tensor]:
    """3-scale YOLOv5 train-mode outputs: (B, na, ny, nx, nc+5)."""
    nc = 1
    no = nc + 5
    na = 3
    return [
        torch.randn(2, na, 80, 80, no, requires_grad=True),
        torch.randn(2, na, 40, 40, no, requires_grad=True),
        torch.randn(2, na, 20, 20, no, requires_grad=True),
    ]


@pytest.fixture
def v5_targets() -> torch.Tensor:
    """(N, 6) — (image_idx, class, x, y, w, h) normalized."""
    return torch.tensor(
        [
            [0, 0, 0.5, 0.5, 0.2, 0.2],
            [0, 0, 0.3, 0.7, 0.1, 0.1],
            [1, 0, 0.4, 0.4, 0.15, 0.25],
        ],
        dtype=torch.float32,
    )


def _make_loss(anchors: torch.Tensor, hyp: dict[str, float]) -> ComputeLoss:
    return ComputeLoss(
        nc=1,
        anchors=anchors,
        stride=torch.tensor([8.0, 16.0, 32.0]),
        hyp=hyp,
        device=torch.device("cpu"),
    )


class TestComputeLoss:
    def test_finite_loss(
        self,
        v5_anchors: torch.Tensor,
        v5_hyp: dict[str, float],
        v5_predictions: list[torch.Tensor],
        v5_targets: torch.Tensor,
    ) -> None:
        loss_fn = _make_loss(v5_anchors, v5_hyp)
        total, parts = loss_fn(v5_predictions, v5_targets)
        assert torch.isfinite(total)
        assert parts.shape == (3,)
        assert torch.isfinite(parts).all()

    def test_grad_flow(
        self,
        v5_anchors: torch.Tensor,
        v5_hyp: dict[str, float],
        v5_predictions: list[torch.Tensor],
        v5_targets: torch.Tensor,
    ) -> None:
        loss_fn = _make_loss(v5_anchors, v5_hyp)
        total, _ = loss_fn(v5_predictions, v5_targets)
        total.backward()  # type: ignore[no-untyped-call]
        for p in v5_predictions:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_returns_correct_parts_shape(
        self,
        v5_anchors: torch.Tensor,
        v5_hyp: dict[str, float],
        v5_predictions: list[torch.Tensor],
        v5_targets: torch.Tensor,
    ) -> None:
        loss_fn = _make_loss(v5_anchors, v5_hyp)
        _, parts = loss_fn(v5_predictions, v5_targets)
        # parts = cat((lbox, lobj, lcls))
        assert parts.shape == (3,)

    def test_zero_targets_gives_only_obj_loss(
        self,
        v5_anchors: torch.Tensor,
        v5_hyp: dict[str, float],
        v5_predictions: list[torch.Tensor],
    ) -> None:
        """With no targets, only objectness loss should be non-zero (boxes/cls are 0)."""
        loss_fn = _make_loss(v5_anchors, v5_hyp)
        targets = torch.zeros(0, 6, dtype=torch.float32)
        total, parts = loss_fn(v5_predictions, targets)
        assert torch.isfinite(total)
        # box and cls should be 0
        assert parts[0].item() == pytest.approx(0.0, abs=1e-6)
        assert parts[2].item() == pytest.approx(0.0, abs=1e-6)
        # obj loss should be positive
        assert parts[1].item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
