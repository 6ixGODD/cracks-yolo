"""Tests for cracks_yolo.ops.csp: Bottleneck, C3, C3SAC, C3TR, C2f, C2fCIB,
SPPF, SPPCSPC, RepConv, SCDown, PSA, Attention, Concat.
"""

from __future__ import annotations

import pytest
import torch

from cracks_yolo.ops.csp import C3
from cracks_yolo.ops.csp import C3SAC
from cracks_yolo.ops.csp import C3TR
from cracks_yolo.ops.csp import PSA
from cracks_yolo.ops.csp import SPPCSPC
from cracks_yolo.ops.csp import SPPF
from cracks_yolo.ops.csp import Attention
from cracks_yolo.ops.csp import Bottleneck
from cracks_yolo.ops.csp import BottleneckSAC
from cracks_yolo.ops.csp import C2f
from cracks_yolo.ops.csp import C2fCIB
from cracks_yolo.ops.csp import Concat
from cracks_yolo.ops.csp import RepConv
from cracks_yolo.ops.csp import RepVGGDW
from cracks_yolo.ops.csp import SCDown


class TestBottleneck:
    def test_shape(self) -> None:
        m = Bottleneck(16, 16)
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 16, 16, 16)

    def test_residual_when_c1_eq_c2_and_shortcut(self) -> None:
        """With shortcut=True and c1==c2, forward must add the input."""
        m = Bottleneck(16, 16, shortcut=True)
        m.eval()
        x = torch.randn(2, 16, 16, 16)
        # Manually compute the non-residual path and compare.
        without_residual = m.cv2(m.cv1(x))
        # In eval mode BN is deterministic; residual path is x + without_residual.
        assert torch.allclose(m(x), x + without_residual, atol=1e-5)

    def test_no_residual_when_c1_neq_c2(self) -> None:
        m = Bottleneck(16, 32, shortcut=True)  # c1 != c2, no residual
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 32, 16, 16)

    def test_no_residual_when_shortcut_false(self) -> None:
        m = Bottleneck(16, 16, shortcut=False)
        m.eval()
        x = torch.randn(2, 16, 16, 16)
        without_residual = m.cv2(m.cv1(x))
        assert torch.allclose(m(x), without_residual, atol=1e-5)

    def test_grad_flow(self) -> None:
        m = Bottleneck(16, 16)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestBottleneckSAC:
    def test_shape(self) -> None:
        m = BottleneckSAC(16, 16)
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 16, 16, 16)

    def test_residual_path_uses_sac(self) -> None:
        m = BottleneckSAC(16, 16, shortcut=True)
        from cracks_yolo.ops.conv import SAConv2d

        assert isinstance(m.cv2, SAConv2d)

    def test_grad_flow(self) -> None:
        m = BottleneckSAC(16, 16)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestC3:
    def test_shape(self) -> None:
        m = C3(16, 32, n=2)
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 32, 16, 16)

    def test_residual_correctness_with_n_zero(self) -> None:
        """With n=1 bottleneck, m path == identity bottleneck(x_in); the
        C3 forward concatenates m(cv1(x)) and cv2(x) then applies cv3."""
        m = C3(16, 16, n=1, shortcut=False, e=0.5)
        m.eval()
        x = torch.randn(2, 16, 16, 16)
        y_expected = m.cv3(torch.cat((m.m(m.cv1(x)), m.cv2(x)), dim=1))
        assert torch.allclose(m(x), y_expected, atol=1e-5)

    def test_grad_flow(self) -> None:
        m = C3(16, 32, n=2)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestC3SAC:
    def test_shape(self) -> None:
        m = C3SAC(16, 32, n=2)
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 32, 16, 16)

    def test_grad_flow(self) -> None:
        m = C3SAC(16, 32, n=2)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestC3TR:
    def test_shape(self) -> None:
        m = C3TR(16, 32, n=2)
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 32, 16, 16)

    def test_grad_flow(self) -> None:
        m = C3TR(16, 32, n=2)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestC2f:
    def test_shape(self) -> None:
        m = C2f(16, 32, n=2)
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 32, 16, 16)

    def test_grad_flow(self) -> None:
        m = C2f(16, 32, n=2)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestC2fCIB:
    def test_shape(self) -> None:
        m = C2fCIB(16, 32, n=2)
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 32, 16, 16)

    def test_grad_flow(self) -> None:
        m = C2fCIB(16, 32, n=2)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestSPPF:
    def test_shape(self) -> None:
        m = SPPF(16, 32, k=5)
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 32, 16, 16)

    def test_grad_flow(self) -> None:
        m = SPPF(16, 32)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestSPPCSPC:
    def test_shape(self) -> None:
        m = SPPCSPC(16, 32, k=(5, 9, 13))
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 32, 16, 16)

    def test_grad_flow(self) -> None:
        m = SPPCSPC(16, 32)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestRepConv:
    def test_shape_train_mode(self) -> None:
        m = RepConv(16, 32, deploy=False)
        m.train()
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 32, 16, 16)

    def test_shape_deploy_mode(self) -> None:
        m = RepConv(16, 32, deploy=True)
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 32, 16, 16)

    def test_fuse_produces_single_reparam_conv(self) -> None:
        m = RepConv(16, 32, deploy=False)
        m.eval()  # eval mode so BN uses running stats consistently
        x = torch.randn(2, 16, 16, 16)
        y_pre = m(x)
        m.fuse()
        assert hasattr(m, "rbr_reparam")
        # After fuse, deploy-mode forward (single conv) should produce
        # close output to the pre-fuse eval forward (3-branch sum) for the
        # same weights, since fuse just combines them.
        y_post = m(x)
        assert torch.allclose(y_pre, y_post, atol=1e-4)

    def test_grad_flow(self) -> None:
        m = RepConv(16, 32, deploy=False)
        m.train()
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestRepVGGDW:
    def test_shape(self) -> None:
        m = RepVGGDW(16)
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 16, 16, 16)

    def test_grad_flow(self) -> None:
        m = RepVGGDW(16)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestSCDown:
    def test_shape_stride2(self) -> None:
        m = SCDown(16, 32, k=3, s=2)
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 32, 8, 8)

    def test_grad_flow(self) -> None:
        m = SCDown(16, 32, k=3, s=2)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestAttention:
    def test_shape(self) -> None:
        # dim=64, num_heads=8 -> head_dim=8, key_dim=4
        m = Attention(64, num_heads=8)
        x = torch.randn(2, 64, 16, 16)
        assert m(x).shape == (2, 64, 16, 16)

    def test_grad_flow(self) -> None:
        m = Attention(64, num_heads=8)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestPSA:
    def test_shape(self) -> None:
        m = PSA(16, 16)
        x = torch.randn(2, 16, 16, 16)
        assert m(x).shape == (2, 16, 16, 16)

    def test_requires_c1_eq_c2(self) -> None:
        with pytest.raises(AssertionError):
            PSA(16, 32)

    def test_grad_flow(self) -> None:
        m = PSA(16, 16)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestConcat:
    def test_shape_default_dim1(self) -> None:
        m = Concat()
        a = torch.randn(2, 8, 16, 16)
        b = torch.randn(2, 16, 16, 16)
        assert m([a, b]).shape == (2, 24, 16, 16)

    def test_other_dim(self) -> None:
        m = Concat(dimension=0)
        a = torch.randn(2, 8, 16, 16)
        b = torch.randn(3, 8, 16, 16)
        assert m([a, b]).shape == (5, 8, 16, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
