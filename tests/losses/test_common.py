"""Tests for cracks_yolo.losses._common: bbox_iou, xywh2xyxy, smooth_BCE,
FocalLoss, QFocalLoss, DFLoss, BCEBlurWithLogitsLoss.
"""

from __future__ import annotations

import pytest
import torch

from cracks_yolo.losses._common import BCEBlurWithLogitsLoss
from cracks_yolo.losses._common import DFLoss
from cracks_yolo.losses._common import FocalLoss
from cracks_yolo.losses._common import QFocalLoss
from cracks_yolo.losses._common import bbox_iou
from cracks_yolo.losses._common import smooth_BCE
from cracks_yolo.losses._common import xywh2xyxy
from cracks_yolo.losses._common import xyxy2xywh


class TestBboxIoU:
    def test_perfect_overlap_returns_one(self) -> None:
        a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        b = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        assert torch.allclose(bbox_iou(a, b, xywh=False), torch.tensor([[1.0]]), atol=1e-5)

    def test_no_overlap_returns_zero(self) -> None:
        a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        b = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
        assert torch.allclose(bbox_iou(a, b, xywh=False), torch.tensor([[0.0]]), atol=1e-5)

    def test_partial_overlap(self) -> None:
        # a = 10x10 = 100, b = 10x10 = 100, inter = 5*5 = 25
        # union = 100 + 100 - 25 = 175, iou = 25/175 = 1/7
        a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        b = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        assert torch.allclose(
            bbox_iou(a, b, xywh=False),
            torch.tensor([[25.0 / 175.0]]),
            atol=1e-5,
        )

    def test_ciou_returns_finite(self) -> None:
        a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        b = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        out = bbox_iou(a, b, xywh=False, CIoU=True)
        assert torch.isfinite(out).all()
        # CIoU <= IoU <= 1, CIoU can be negative
        assert (out <= 1.0 + 1e-5).all()

    def test_xywh_input_matches_xyxy(self) -> None:
        xywh = torch.tensor([[5.0, 5.0, 10.0, 10.0]])  # cx, cy, w, h
        xyxy = torch.tensor([[0.0, 0.0, 10.0, 10.0]])  # same box as xyxy
        assert torch.allclose(
            bbox_iou(xywh, xywh, xywh=True),
            bbox_iou(xyxy, xyxy, xywh=False),
            atol=1e-5,
        )


class TestXYConvert:
    def test_xywh2xyxy_roundtrip(self) -> None:
        xywh = torch.tensor([[5.0, 7.0, 10.0, 6.0]])
        xyxy = xywh2xyxy(xywh)
        assert torch.allclose(xyxy, torch.tensor([[0.0, 4.0, 10.0, 10.0]]), atol=1e-5)
        back = xyxy2xywh(xyxy)
        assert torch.allclose(back, xywh, atol=1e-5)


class TestSmoothBCE:
    def test_default_targets(self) -> None:
        cp, cn = smooth_BCE()
        assert cp == pytest.approx(0.95)
        assert cn == pytest.approx(0.05)

    def test_zero_eps(self) -> None:
        cp, cn = smooth_BCE(eps=0.0)
        assert cp == 1.0
        assert cn == 0.0


class TestBCEBlurWithLogitsLoss:
    def test_shape_and_finite(self) -> None:
        m = BCEBlurWithLogitsLoss()
        pred = torch.randn(4, 5)
        true = torch.zeros(4, 5)
        true[:, 0] = 1.0
        out = m(pred, true)
        assert torch.isfinite(out)
        assert out.ndim == 0


class TestFocalLoss:
    def test_finite_loss(self) -> None:
        bce = torch.nn.BCEWithLogitsLoss()
        fl = FocalLoss(bce, gamma=1.5, alpha=0.25)
        pred = torch.randn(4, 5)
        true = torch.zeros(4, 5)
        true[:, 0] = 1.0
        out = fl(pred, true)
        assert torch.isfinite(out)
        assert out.ndim == 0


class TestQFocalLoss:
    def test_finite_loss(self) -> None:
        bce = torch.nn.BCEWithLogitsLoss()
        fl = QFocalLoss(bce, gamma=1.5, alpha=0.25)
        pred = torch.randn(4, 5)
        true = torch.zeros(4, 5)
        true[:, 0] = 1.0
        out = fl(pred, true)
        assert torch.isfinite(out)
        assert out.ndim == 0


class TestDFLoss:
    def test_shape(self) -> None:
        dfl = DFLoss(reg_max=16)
        # pred_dist (N, 16), target (N,) continuous.
        pred = torch.randn(8, 16)
        target = torch.rand(8, 1) * 15
        out = dfl(pred, target)
        assert out.shape == (8, 1)

    def test_grad_flow(self) -> None:
        dfl = DFLoss(reg_max=16)
        pred = torch.randn(8, 16, requires_grad=True)
        target = torch.rand(8, 1) * 15
        dfl(pred, target).sum().backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
