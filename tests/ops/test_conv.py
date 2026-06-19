"""Tests for cracks_yolo.ops.conv: autopad, Conv, DWConv, ConvAWS2d, SAConv2d."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cracks_yolo.ops.conv import Conv
from cracks_yolo.ops.conv import ConvAWS2d
from cracks_yolo.ops.conv import DWConv
from cracks_yolo.ops.conv import SAConv2d
from cracks_yolo.ops.conv import autopad


class TestAutopad:
    def test_int_kernel_no_dilation(self) -> None:
        assert autopad(1) == 0
        assert autopad(3) == 1
        assert autopad(5) == 2
        assert autopad(7) == 3

    def test_int_kernel_with_dilation(self) -> None:
        # Effective kernel: d*(k-1)+1
        # k=3, d=2 -> 5 -> pad 2
        assert autopad(3, d=2) == 2
        # k=3, d=3 -> 7 -> pad 3
        assert autopad(3, d=3) == 3

    def test_explicit_padding_passthrough(self) -> None:
        assert autopad(3, p=7) == 7
        assert autopad(5, p=0) == 0

    def test_list_kernel(self) -> None:
        assert autopad([3, 3]) == [1, 1]
        assert autopad([3, 5]) == [1, 2]


class TestConv:
    def test_shape_default(self) -> None:
        """1x1 conv preserves spatial dims; output channels match arg."""
        m = Conv(8, 16, k=1)
        x = torch.randn(2, 8, 16, 16)
        y = m(x)
        assert y.shape == (2, 16, 16, 16)

    def test_shape_3x3_same_padding(self) -> None:
        m = Conv(8, 16, k=3)
        x = torch.randn(2, 8, 16, 16)
        assert m(x).shape == (2, 16, 16, 16)

    def test_stride(self) -> None:
        m = Conv(8, 16, k=3, s=2)
        x = torch.randn(2, 8, 16, 16)
        assert m(x).shape == (2, 16, 8, 8)

    def test_act_default_silu(self) -> None:
        m = Conv(8, 16)
        assert isinstance(m.act, nn.SiLU)

    def test_act_false_identity(self) -> None:
        m = Conv(8, 16, act=False)
        assert isinstance(m.act, nn.Identity)

    def test_act_named(self) -> None:
        m = Conv(8, 16, act="relu")
        assert isinstance(m.act, nn.ReLU)

    def test_grad_flow(self) -> None:
        m = Conv(8, 16, k=3)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in m.parameters())

    def test_determinism_same_input_same_output(self) -> None:
        m = Conv(8, 16, k=3)
        m.eval()  # turn off BN randomness (BN still deterministic in eval)
        x = torch.randn(2, 8, 16, 16)
        y1 = m(x)
        y2 = m(x)
        assert torch.equal(y1, y2)

    def test_bn_running_stats_update_in_train(self) -> None:
        m = Conv(8, 16, k=3)
        assert m.bn.running_mean is not None
        assert m.bn.running_var is not None
        before_mean = m.bn.running_mean.clone()
        before_var = m.bn.running_var.clone()
        m.train()
        m(torch.randn(4, 8, 16, 16))
        assert m.bn.running_mean is not None
        assert m.bn.running_var is not None
        after_mean = m.bn.running_mean
        after_var = m.bn.running_var
        # In train mode BN updates running stats via momentum; expect change.
        assert not torch.equal(before_mean, after_mean)
        assert not torch.equal(before_var, after_var)


class TestDWConv:
    def test_shape_and_groups(self) -> None:
        m = DWConv(8, 8, k=3)
        x = torch.randn(2, 8, 16, 16)
        assert m(x).shape == (2, 8, 16, 16)
        # groups must equal in_channels for depthwise
        assert m.conv.groups == 8

    def test_grad_flow(self) -> None:
        m = DWConv(8, 8, k=3)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestConvAWS2d:
    def test_shape(self) -> None:
        m = ConvAWS2d(8, 16, 3, padding=1, bias=False)
        x = torch.randn(2, 8, 16, 16)
        assert m(x).shape == (2, 16, 16, 16)

    def test_has_weight_gamma_and_beta_buffers(self) -> None:
        m = ConvAWS2d(8, 16, 3, padding=1, bias=False)
        assert hasattr(m, "weight_gamma")
        assert hasattr(m, "weight_beta")
        assert m.weight_gamma.shape == (16, 1, 1, 1)
        assert m.weight_beta.shape == (16, 1, 1, 1)
        # Initialized to ones / zeros.
        assert torch.allclose(m.weight_gamma, torch.ones_like(m.weight_gamma))
        assert torch.allclose(m.weight_beta, torch.zeros_like(m.weight_beta))

    def test_grad_flow(self) -> None:
        m = ConvAWS2d(8, 16, 3, padding=1, bias=False)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None
        assert m.weight.grad is not None

    def test_load_from_non_aws_state_dict_initializes_gamma_beta(self) -> None:
        """Loading a state_dict without gamma/beta should derive them from weight."""
        src = ConvAWS2d(8, 16, 3, padding=1, bias=False)
        sd = src.state_dict()
        # Strip gamma/beta to simulate loading from a non-AWS checkpoint.
        sd.pop("weight_gamma")
        sd.pop("weight_beta")

        dst = ConvAWS2d(8, 16, 3, padding=1, bias=False)
        # Reset gamma to a sentinel so we can detect the post-load reinit.
        dst.weight_gamma.data.fill_(-99.0)
        dst.weight_beta.data.fill_(-99.0)
        missing: list[str] = []
        unexpected: list[str] = []
        dst._load_from_state_dict(sd, "", {}, False, missing, unexpected, [])
        # After load, gamma/beta should be re-derived (positive gamma, real beta).
        assert dst.weight_gamma.data.mean() > 0
        assert dst.weight_beta.data.mean() != -99.0


class TestSAConv2d:
    def test_shape(self) -> None:
        m = SAConv2d(8, 16, 3, p=1)
        x = torch.randn(2, 8, 16, 16)
        assert m(x).shape == (2, 16, 16, 16)

    def test_has_switch_and_context_layers(self) -> None:
        m = SAConv2d(8, 16, 3, p=1)
        assert hasattr(m, "switch")
        assert hasattr(m, "pre_context")
        assert hasattr(m, "post_context")
        assert hasattr(m, "weight_diff")
        assert hasattr(m, "bn")

    def test_grad_flow(self) -> None:
        m = SAConv2d(8, 16, 3, p=1)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None
        # switch, pre_context, post_context all have params; weight_diff too.
        assert m.switch.weight.grad is not None
        assert m.pre_context.weight.grad is not None
        assert m.post_context.weight.grad is not None
        assert m.weight_diff.grad is not None

    def test_dilation_restored_after_forward(self) -> None:
        """Forward must restore padding/dilation to their pre-forward values."""
        m = SAConv2d(8, 16, 3, p=1, d=1)
        before_pad = m.padding
        before_dil = m.dilation
        m(torch.randn(2, 8, 16, 16))
        assert m.padding == before_pad
        assert m.dilation == before_dil

    def test_determinism_eval(self) -> None:
        m = SAConv2d(8, 16, 3, p=1)
        m.eval()
        x = torch.randn(2, 8, 16, 16)
        y1 = m(x)
        y2 = m(x)
        assert torch.equal(y1, y2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
