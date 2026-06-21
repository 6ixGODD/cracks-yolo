"""Tests for cracks_yolo.losses.yolov7.ComputeLossOTA."""

from __future__ import annotations

import pytest
import torch

from cracks_yolo.losses.yolov7 import ComputeLossOTA

V5_ANCHORS_FLAT: tuple[tuple[int, ...], ...] = (
    (10, 13, 16, 30, 33, 23),
    (30, 61, 62, 45, 59, 119),
    (116, 90, 156, 198, 373, 326),
)


@pytest.fixture
def v7_anchors() -> torch.Tensor:
    return torch.tensor(V5_ANCHORS_FLAT, dtype=torch.float32).view(3, 3, 2)


@pytest.fixture
def v7_hyp() -> dict[str, float]:
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
def v7_predictions() -> list[torch.Tensor]:
    nc = 1
    no = nc + 5
    na = 3
    return [
        torch.randn(2, na, 80, 80, no, requires_grad=True),
        torch.randn(2, na, 40, 40, no, requires_grad=True),
        torch.randn(2, na, 20, 20, no, requires_grad=True),
    ]


@pytest.fixture
def v7_imgs() -> torch.Tensor:
    return torch.randn(2, 3, 640, 640)


@pytest.fixture
def v7_targets() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 0.5, 0.5, 0.2, 0.2],
            [0, 0, 0.3, 0.7, 0.1, 0.1],
            [1, 0, 0.4, 0.4, 0.15, 0.25],
        ],
        dtype=torch.float32,
    )


def _make_loss(anchors: torch.Tensor, hyp: dict[str, float]) -> ComputeLossOTA:
    return ComputeLossOTA(
        nc=1,
        anchors=anchors,
        stride=torch.tensor([8.0, 16.0, 32.0]),
        hyp=hyp,
        device=torch.device("cpu"),
    )


class TestComputeLossOTA:
    def test_finite_loss(
        self,
        v7_anchors: torch.Tensor,
        v7_hyp: dict[str, float],
        v7_predictions: list[torch.Tensor],
        v7_imgs: torch.Tensor,
        v7_targets: torch.Tensor,
    ) -> None:
        loss_fn = _make_loss(v7_anchors, v7_hyp)
        total, parts = loss_fn(v7_predictions, v7_targets, v7_imgs)
        assert torch.isfinite(total)
        # parts = cat((lbox, lcls, lobj)) — matches loss_parts_schema ("box","cls","obj").
        assert parts.shape == (3,)

    def test_grad_flow(
        self,
        v7_anchors: torch.Tensor,
        v7_hyp: dict[str, float],
        v7_predictions: list[torch.Tensor],
        v7_imgs: torch.Tensor,
        v7_targets: torch.Tensor,
    ) -> None:
        loss_fn = _make_loss(v7_anchors, v7_hyp)
        total, _ = loss_fn(v7_predictions, v7_targets, v7_imgs)
        total.backward()  # type: ignore[no-untyped-call]
        for p in v7_predictions:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
