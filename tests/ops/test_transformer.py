"""Tests for cracks_yolo.ops.transformer: TransformerLayer, TransformerBlock."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cracks_yolo.ops.transformer import TransformerBlock
from cracks_yolo.ops.transformer import TransformerLayer


class TestTransformerLayer:
    def test_shape(self) -> None:
        m = TransformerLayer(c=64, num_heads=4)
        # nn.MultiheadAttention expects (seq, batch, embed)
        x = torch.randn(16, 2, 64)
        assert m(x).shape == (16, 2, 64)

    def test_residual_correctness(self) -> None:
        """The layer adds residual connections at both attention and MLP."""
        m = TransformerLayer(c=64, num_heads=4)
        m.eval()
        x = torch.randn(16, 2, 64)
        # Manually compute: x + MHA(QKV) then + MLP(...)
        attn_out, _ = m.ma(m.q(x), m.k(x), m.v(x), need_weights=False)
        after_attn = attn_out + x
        mlp_out = m.fc2(m.fc1(after_attn))
        expected = mlp_out + after_attn
        assert torch.allclose(m(x), expected, atol=1e-5)

    def test_has_multihead_attention(self) -> None:
        m = TransformerLayer(c=64, num_heads=4)
        assert isinstance(m.ma, nn.MultiheadAttention)
        assert m.ma.embed_dim == 64
        assert m.ma.num_heads == 4

    def test_grad_flow(self) -> None:
        m = TransformerLayer(c=64, num_heads=4)
        x = torch.randn(16, 2, 64, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None
        for name, p in m.named_parameters():
            assert p.grad is not None, f"no grad on {name}"


class TestTransformerBlock:
    def test_shape_same_channels(self) -> None:
        m = TransformerBlock(64, 64, num_heads=4, num_layers=2)
        x = torch.randn(2, 64, 16, 16)
        assert m(x).shape == (2, 64, 16, 16)

    def test_shape_different_channels(self) -> None:
        m = TransformerBlock(32, 64, num_heads=4, num_layers=2)
        x = torch.randn(2, 32, 16, 16)
        assert m(x).shape == (2, 64, 16, 16)

    def test_has_conv_when_c1_neq_c2(self) -> None:
        m = TransformerBlock(32, 64, num_heads=4, num_layers=2)
        assert m.conv is not None

    def test_no_conv_when_c1_eq_c2(self) -> None:
        m = TransformerBlock(64, 64, num_heads=4, num_layers=2)
        assert m.conv is None

    def test_grad_flow(self) -> None:
        m = TransformerBlock(32, 64, num_heads=4, num_layers=2)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_determinism_eval(self) -> None:
        m = TransformerBlock(64, 64, num_heads=4, num_layers=2)
        m.eval()
        x = torch.randn(2, 64, 16, 16)
        y1 = m(x)
        y2 = m(x)
        assert torch.equal(y1, y2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
