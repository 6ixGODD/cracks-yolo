"""Tests for cracks_yolo.losses.yolov8: TaskAlignedAssigner, BboxLoss,
v8DetectionLoss, and cracks_yolo.losses.yolov10.E2ELoss.
"""

from __future__ import annotations

import pytest
import torch

from cracks_yolo.losses.yolov8 import TaskAlignedAssigner
from cracks_yolo.losses.yolov8 import v8DetectionLoss
from cracks_yolo.losses.yolov10 import E2ELoss

PredDict = dict[str, torch.Tensor | list[torch.Tensor]]


@pytest.fixture
def v8_hyp() -> dict[str, float]:
    return {"box": 7.5, "cls": 0.5, "dfl": 1.5}


@pytest.fixture
def v8_predictions() -> PredDict:
    """Anchor-free dict from DetectAnchorFree in train mode.

    boxes: (B, 4*reg_max, N_total), scores: (B, nc, N_total).
    """
    nc = 1
    reg_max = 16
    # 80*80 + 40*40 + 20*20 = 8400
    boxes = torch.randn(2, 4 * reg_max, 8400, requires_grad=True)
    scores = torch.randn(2, nc, 8400, requires_grad=True)
    feats = [
        torch.randn(2, 64, 80, 80),
        torch.randn(2, 128, 40, 40),
        torch.randn(2, 256, 20, 20),
    ]
    return {"boxes": boxes, "scores": scores, "feats": feats}


@pytest.fixture
def v8_batch() -> dict[str, torch.Tensor]:
    """3 targets across 2 images."""
    return {
        "batch_idx": torch.tensor([0, 0, 1]),
        "cls": torch.tensor([[0], [0], [0]], dtype=torch.float32),
        "bboxes": torch.tensor(
            [
                [0.5, 0.5, 0.7, 0.7],
                [0.3, 0.3, 0.4, 0.4],
                [0.4, 0.4, 0.6, 0.6],
            ],
            dtype=torch.float32,
        ),
    }


def _make_v8_loss(hyp: dict[str, float]) -> v8DetectionLoss:
    return v8DetectionLoss(
        nc=1,
        reg_max=16,
        stride=torch.tensor([8.0, 16.0, 32.0]),
        hyp=hyp,
        device=torch.device("cpu"),
    )


class TestTaskAlignedAssigner:
    def test_shape(self) -> None:
        a = TaskAlignedAssigner(topk=10, num_classes=1)
        bs, n_anc = 2, 100
        pd_scores = torch.rand(bs, n_anc, 1)
        pd_bboxes = torch.rand(bs, n_anc, 4) * 100
        anc_points = torch.rand(n_anc, 2) * 100
        gt_labels = torch.tensor([[[0], [0]], [[0], [0]]], dtype=torch.float32)
        gt_bboxes = torch.rand(bs, 2, 4) * 100
        mask_gt = torch.ones(bs, 2, 1, dtype=torch.bool)
        out = a(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        assert len(out) == 5
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = out
        assert target_labels.shape == (bs, n_anc)
        assert target_bboxes.shape == (bs, n_anc, 4)
        assert target_scores.shape == (bs, n_anc, 1)
        assert fg_mask.shape == (bs, n_anc)
        assert target_gt_idx.shape == (bs, n_anc)


class TestV8DetectionLoss:
    def test_finite_loss(
        self, v8_hyp: dict[str, float], v8_predictions: PredDict, v8_batch: dict[str, torch.Tensor]
    ) -> None:
        loss_fn = _make_v8_loss(v8_hyp)
        total, parts = loss_fn(v8_predictions, v8_batch)
        assert torch.isfinite(total).all()
        assert parts.shape == (3,)
        assert torch.isfinite(parts).all()

    def test_grad_flow(
        self, v8_hyp: dict[str, float], v8_predictions: PredDict, v8_batch: dict[str, torch.Tensor]
    ) -> None:
        loss_fn = _make_v8_loss(v8_hyp)
        total, _ = loss_fn(v8_predictions, v8_batch)
        total.sum().backward()  # type: ignore[no-untyped-call]
        boxes = v8_predictions["boxes"]
        scores = v8_predictions["scores"]
        assert isinstance(boxes, torch.Tensor)
        assert isinstance(scores, torch.Tensor)
        assert boxes.grad is not None
        assert scores.grad is not None


class TestE2ELoss:
    def test_finite_loss(
        self, v8_hyp: dict[str, float], v8_predictions: PredDict, v8_batch: dict[str, torch.Tensor]
    ) -> None:
        """E2ELoss expects a dict with 'one2many' and 'one2one' sub-dicts."""
        loss_fn = E2ELoss(
            nc=1,
            reg_max=16,
            stride=torch.tensor([8.0, 16.0, 32.0]),
            hyp=v8_hyp,
            device=torch.device("cpu"),
        )
        # Duplicate the prediction dict for the two heads.
        boxes = v8_predictions["boxes"]
        scores = v8_predictions["scores"]
        feats = v8_predictions["feats"]
        assert isinstance(boxes, torch.Tensor)
        assert isinstance(scores, torch.Tensor)
        assert isinstance(feats, list)
        o2m: PredDict = {"boxes": boxes, "scores": scores, "feats": feats}
        o2o: PredDict = {
            "boxes": boxes.detach().requires_grad_(True),
            "scores": scores.detach().requires_grad_(True),
            "feats": feats,
        }
        preds = {"one2many": o2m, "one2one": o2o}
        total, parts = loss_fn(preds, v8_batch)
        assert torch.isfinite(total).all()
        assert parts.shape == (3,)

    def test_decay_decreases_o2m(self, v8_hyp: dict[str, float]) -> None:
        loss_fn = E2ELoss(
            nc=1,
            reg_max=16,
            stride=torch.tensor([8.0, 16.0, 32.0]),
            hyp=v8_hyp,
            device=torch.device("cpu"),
        )
        initial = loss_fn.o2m
        for _ in range(100):
            loss_fn.update()
        assert loss_fn.o2m < initial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
