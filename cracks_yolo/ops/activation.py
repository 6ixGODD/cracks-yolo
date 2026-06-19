"""Activation function registry and factory.

Used by model constructors that expose `act: str | nn.Module = "silu"` so the
activation is configurable without subclassing.
"""

from __future__ import annotations

import torch.nn as nn

_ACT_REGISTRY: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "leakyrelu": nn.LeakyReLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "identity": nn.Identity,
    "none": nn.Identity,
}


def parse_activation(spec: str | nn.Module) -> nn.Module:
    """Resolve an activation spec into an `nn.Module` instance.

    Args:
        spec: Either an `nn.Module` instance (returned as-is) or a registered
            activation name. Case-insensitive. Registered names: silu, relu,
            relu6, leakyrelu, mish, gelu, elu, identity, none.

    Returns:
        A fresh activation module instance.

    Raises:
        ValueError: If `spec` is a string not in the registry.
    """
    if isinstance(spec, nn.Module):
        return spec
    try:
        cls = _ACT_REGISTRY[spec.lower()]
    except KeyError as e:
        raise ValueError(f"Unknown activation {spec!r}. Registered: {sorted(_ACT_REGISTRY)}") from e
    return cls()
