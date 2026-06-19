"""Tests for cracks_yolo.ops.detect_heads: DFL, DetectAnchorBased,
DetectAnchorFree, IDetect, IAuxDetect, v10Detect, plus the make_anchors /
dist2bbox / bbox2dist helpers.
"""

from __future__ import annotations

import pytest
import torch

from cracks_yolo.ops.detect_heads import DFL
from cracks_yolo.ops.detect_heads import DetectAnchorBased
from cracks_yolo.ops.detect_heads import DetectAnchorFree
from cracks_yolo.ops.detect_heads import IAuxDetect
from cracks_yolo.ops.detect_heads import IDetect
from cracks_yolo.ops.detect_heads import bbox2dist
from cracks_yolo.ops.detect_heads import dist2bbox
from cracks_yolo.ops.detect_heads import make_anchors
from cracks_yolo.ops.detect_heads import v10Detect

# COCO anchors (3 scales x 3 anchors x 2 dims).
V5_ANCHORS: tuple[tuple[int, ...], ...] = (
    (10, 13, 16, 30, 33, 23),
    (30, 61, 62, 45, 59, 119),
    (116, 90, 156, 198, 373, 326),
)


# ============================================================================
# Helper functions
# ============================================================================


class TestMakeAnchors:
    def test_three_scales_yields_8400_total_points(self) -> None:
        feats = [
            torch.randn(2, 64, 80, 80),
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 256, 20, 20),
        ]
        strides = torch.tensor([8.0, 16.0, 32.0])
        anchor_points, stride_tensor = make_anchors(feats, strides, 0.5)
        # Total: 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400
        assert anchor_points.shape == (8400, 2)
        assert stride_tensor.shape == (8400, 1)

    def test_grid_cell_offset_applied(self) -> None:
        feats = [torch.randn(1, 4, 2, 2)]
        strides = torch.tensor([1.0])
        anchor_points, _ = make_anchors(feats, strides, grid_cell_offset=0.5)
        # For a 2x2 grid with offset 0.5, points are at (0.5, 0.5), (1.5, 0.5),
        # (0.5, 1.5), (1.5, 1.5) — order is row-major (ij indexing).
        expected = torch.tensor(
            [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]], dtype=torch.float32
        )
        assert torch.allclose(anchor_points, expected, atol=1e-5)


class TestDist2BboxBbox2Dist:
    def test_dist2bbox_xywh_roundtrip(self) -> None:
        anchors = torch.tensor([[10.0, 10.0]])
        dist = torch.tensor([[2.0, 2.0, 4.0, 4.0]])  # lt, lt, rb, rb
        box = dist2bbox(dist, anchors, xywh=True, dim=-1)
        # xy = ((10-2) + (10+4)) / 2 = (8+14)/2 = 11
        # wh = (10+4) - (10-2) = 14 - 8 = 6
        expected = torch.tensor([[11.0, 11.0, 6.0, 6.0]])
        assert torch.allclose(box, expected, atol=1e-5)

    def test_bbox2dist_inverts_dist2bbox(self) -> None:
        anchors = torch.tensor([[5.0, 7.0]])
        dist_orig = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        box = dist2bbox(dist_orig, anchors, xywh=False, dim=-1)
        # box is (x1y1, x2y2) = ((5-1, 7-2), (5+3, 7+4)) = ((4,5),(8,11))
        dist_back = bbox2dist(anchors, box)
        assert torch.allclose(dist_orig, dist_back, atol=1e-5)


# ============================================================================
# DFL
# ============================================================================


class TestDFL:
    def test_shape(self) -> None:
        m = DFL(c1=16)
        # Input is (B, 4*16, A); output is (B, 4, A).
        x = torch.randn(2, 64, 100)
        assert m(x).shape == (2, 4, 100)

    def test_non_learnable(self) -> None:
        m = DFL(c1=16)
        for p in m.parameters():
            assert not p.requires_grad

    def test_fixed_conv_weight_is_arange(self) -> None:
        m = DFL(c1=16)
        # Conv weight should be a (1, 16, 1, 1) arange.
        expected = torch.arange(16, dtype=torch.float32).view(1, 16, 1, 1)
        assert torch.allclose(m.conv.weight.detach(), expected)

    def test_softmax_sums_to_one(self) -> None:
        """When the input distribution is uniform, the integral output equals
        the midpoint of the arange range = (c1 - 1) / 2."""
        m = DFL(c1=16)
        # Uniform logits over 16 bins, 4 box sides, 10 anchor points.
        x = torch.zeros(1, 64, 10)
        out = m(x)
        # For uniform input, softmax is uniform, integral = sum(arange)/c1.
        expected_val = sum(range(16)) / 16.0
        assert torch.allclose(out, torch.full((1, 4, 10), expected_val), atol=1e-5)


# ============================================================================
# Anchor-based heads
# ============================================================================


class TestDetectAnchorBased:
    def test_train_forward_shape(self, small_feature_batch: list[torch.Tensor]) -> None:
        nc = 1
        m = DetectAnchorBased(nc=nc, anchors=V5_ANCHORS, ch=(64, 128, 256))
        m.stride = torch.tensor([8, 16, 32])
        m.train()
        out = m([f.clone() for f in small_feature_batch])
        # Train returns the list of reshaped (B, na, ny, nx, nc+5) tensors.
        assert isinstance(out, list)
        assert len(out) == 3
        assert out[0].shape == (2, 3, 80, 80, nc + 5)
        assert out[1].shape == (2, 3, 40, 40, nc + 5)
        assert out[2].shape == (2, 3, 20, 20, nc + 5)

    def test_eval_forward_shape(self, small_feature_batch: list[torch.Tensor]) -> None:
        nc = 1
        m = DetectAnchorBased(nc=nc, anchors=V5_ANCHORS, ch=(64, 128, 256))
        m.stride = torch.tensor([8, 16, 32])
        m.eval()
        out = m([f.clone() for f in small_feature_batch])
        # Eval returns (decoded, raw_list).
        assert isinstance(out, tuple)
        decoded, raw = out
        assert decoded.shape == (2, 25200, nc + 5)  # 3*(80*80+40*40+20*20)=25200
        assert isinstance(raw, list)
        assert len(raw) == 3

    def test_grad_flow(self, small_feature_batch: list[torch.Tensor]) -> None:
        m = DetectAnchorBased(nc=1, anchors=V5_ANCHORS, ch=(64, 128, 256))
        m.stride = torch.tensor([8, 16, 32])
        m.train()
        feats = [torch.randn_like(f, requires_grad=True) for f in small_feature_batch]
        out = m(list(feats))
        total = sum(o.sum() for o in out)
        total.backward()
        for f in feats:
            assert f.grad is not None


class TestIDetect:
    def test_train_forward_shape(self, small_feature_batch: list[torch.Tensor]) -> None:
        nc = 1
        m = IDetect(nc=nc, anchors=V5_ANCHORS, ch=(64, 128, 256))
        m.stride = torch.tensor([8, 16, 32])
        m.train()
        out = m([f.clone() for f in small_feature_batch])
        assert isinstance(out, list)
        assert len(out) == 3
        assert out[0].shape == (2, 3, 80, 80, nc + 5)

    def test_has_implicit_a_and_m(self) -> None:
        m = IDetect(nc=1, anchors=V5_ANCHORS, ch=(64, 128, 256))
        assert len(m.ia) == 3
        assert len(m.im) == 3

    def test_fuse_preserves_forward(self, small_feature_batch: list[torch.Tensor]) -> None:
        m = IDetect(nc=1, anchors=V5_ANCHORS, ch=(64, 128, 256))
        m.stride = torch.tensor([8, 16, 32])
        m.eval()
        x = [f.clone() for f in small_feature_batch]
        out_before = m(x)[0]
        m.fuse()
        x = [f.clone() for f in small_feature_batch]
        out_after = m(x)[0]
        # ImplicitA/M fusion involves matmul + reshape over float32 weights;
        # allow small relative drift (some entries are O(100) so atol=1 alone
        # is too tight).
        assert torch.allclose(out_before, out_after, rtol=2e-2, atol=1e-1)


class TestIAuxDetect:
    def test_train_forward_shape(self) -> None:
        nc = 1
        # IAuxDetect requires ch length == 2*nl (main + aux).
        ch = (64, 128, 256, 64, 128, 256)
        m = IAuxDetect(nc=nc, anchors=V5_ANCHORS, ch=ch)
        m.stride = torch.tensor([8, 16, 32])
        m.train()
        # The forward expects the input list to contain both main and aux feats.
        # small_feature_batch has 3 feats; for IAux we need 6.
        feats = [
            torch.randn(2, 64, 80, 80),
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 256, 20, 20),
            torch.randn(2, 64, 80, 80),
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 256, 20, 20),
        ]
        out = m([f.clone() for f in feats])
        assert isinstance(out, list)
        assert len(out) == 6


# ============================================================================
# Anchor-free heads
# ============================================================================


class TestDetectAnchorFree:
    def test_train_forward_shape(self, small_feature_batch: list[torch.Tensor]) -> None:
        nc = 1
        reg_max = 16
        m = DetectAnchorFree(nc=nc, reg_max=reg_max, ch=(64, 128, 256))
        m.stride = torch.tensor([8.0, 16.0, 32.0])
        m.bias_init()
        m.train()
        out = m([f.clone() for f in small_feature_batch])
        assert isinstance(out, dict)
        # boxes = 4 * reg_max = 64 channels, scores = nc channels.
        assert out["boxes"].shape == (2, 4 * reg_max, 8400)
        assert out["scores"].shape == (2, nc, 8400)

    def test_eval_forward_shape(self, small_feature_batch: list[torch.Tensor]) -> None:
        nc = 1
        m = DetectAnchorFree(nc=nc, reg_max=16, ch=(64, 128, 256))
        m.stride = torch.tensor([8.0, 16.0, 32.0])
        m.bias_init()
        m.eval()
        out = m([f.clone() for f in small_feature_batch])
        # (B, 4 + nc, N_total)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 4 + nc, 8400)

    def test_grad_flow(self, small_feature_batch: list[torch.Tensor]) -> None:
        m = DetectAnchorFree(nc=1, reg_max=16, ch=(64, 128, 256))
        m.stride = torch.tensor([8.0, 16.0, 32.0])
        m.bias_init()
        m.train()
        feats = [torch.randn_like(f, requires_grad=True) for f in small_feature_batch]
        out = m(list(feats))
        out["boxes"].sum().backward()
        for f in feats:
            assert f.grad is not None


class TestV10Detect:
    def test_train_forward_shape(self, small_feature_batch: list[torch.Tensor]) -> None:
        nc = 1
        reg_max = 16
        m = v10Detect(nc=nc, ch=(64, 128, 256))
        m.stride = torch.tensor([8.0, 16.0, 32.0])
        m.bias_init()
        m.train()
        out = m([f.clone() for f in small_feature_batch])
        assert isinstance(out, dict)
        assert "one2many" in out
        assert "one2one" in out
        # Each head has boxes (4*reg_max channels) and scores (nc channels).
        for key in ("one2many", "one2one"):
            head = out[key]
            assert isinstance(head, dict)
            assert head["boxes"].shape == (2, 4 * reg_max, 8400)
            assert head["scores"].shape == (2, nc, 8400)

    def test_eval_forward_shape(self, small_feature_batch: list[torch.Tensor]) -> None:
        nc = 1
        m = v10Detect(nc=nc, ch=(64, 128, 256))
        m.stride = torch.tensor([8.0, 16.0, 32.0])
        m.bias_init()
        m.eval()
        out = m([f.clone() for f in small_feature_batch])
        # v10 eval forward only runs the one2one head, output shape (B, 4+nc, N).
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 4 + nc, 8400)

    def test_fuse_drops_one2many_head(self) -> None:
        m = v10Detect(nc=1, ch=(64, 128, 256))
        m.stride = torch.tensor([8.0, 16.0, 32.0])
        m.bias_init()
        m.fuse()
        # After fuse the one2many cv2/cv3 are set to None.
        assert m.cv2 is None
        assert m.cv3 is None  # type: ignore[unreachable]

    def test_grad_flow(self, small_feature_batch: list[torch.Tensor]) -> None:
        m = v10Detect(nc=1, ch=(64, 128, 256))
        m.stride = torch.tensor([8.0, 16.0, 32.0])
        m.bias_init()
        m.train()
        feats = [torch.randn_like(f, requires_grad=True) for f in small_feature_batch]
        out = m(list(feats))
        out["one2many"]["boxes"].sum().backward()
        for f in feats:
            assert f.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
