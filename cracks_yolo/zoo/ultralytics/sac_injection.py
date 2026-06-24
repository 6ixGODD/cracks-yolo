"""Runtime SAC/TR module replacement for ultralytics models.

Replaces C3/C2f blocks at designated backbone indices with SAC/TR variants
from ``cracks_yolo.ops.sac_ops``. Copies shared conv weights so only
SAC-specific tensors (switches, context convs) stay randomly initialized.
"""

from __future__ import annotations

import contextlib

import torch.nn as nn

from cracks_yolo.ops.sac import C3SAC
from cracks_yolo.ops.sac import C3TR
from cracks_yolo.ops.sac import C2fSAC


def _copy_layer_meta(new_layer: nn.Module, old_layer: nn.Module) -> None:
    """Copy ultralytics routing attrs (f, i, type, np) so the replaced layer
    plugs into the model's forward graph correctly."""
    for attr in ("f", "i", "type", "np"):
        if hasattr(old_layer, attr):
            with contextlib.suppress(AttributeError):
                setattr(new_layer, attr, getattr(old_layer, attr))


def _copy_shared_weights(new_module: nn.Module, old_module: nn.Module) -> int:
    """Copy state_dict keys that exist in both + shape-match."""
    new_sd = new_module.state_dict()
    old_sd = old_module.state_dict()
    copied = 0
    for k, v in old_sd.items():
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v
            copied += 1
    new_module.load_state_dict(new_sd, strict=False)
    return copied


def apply_sac_tr(
    model: nn.Module,
    sac_indices: tuple[int, ...] = (),
    tr_indices: tuple[int, ...] = (),
) -> None:
    """Replace C3/C2f blocks at ``sac_indices`` with SAC variants, and at
    ``tr_indices`` with C3TR, in-place on ``model.model`` Sequential.
    """
    from ultralytics.nn.modules.block import C3
    from ultralytics.nn.modules.block import C2f

    seq = model.model
    for i in sac_indices:
        old = seq[i]
        if isinstance(old, C3):
            c1 = old.cv1.conv.in_channels
            c2 = old.cv3.conv.out_channels
            n = len(old.m)
            new = C3SAC(c1, c2, n, shortcut=True)
            _copy_shared_weights(new, old)
            _copy_layer_meta(new, old)
            seq[i] = new
        elif isinstance(old, C2f):
            c1 = old.cv1.conv.in_channels
            c2 = old.cv2.conv.out_channels
            n = len(old.m)
            new = C2fSAC(c1, c2, n)
            _copy_shared_weights(new, old)
            _copy_layer_meta(new, old)
            seq[i] = new
    for i in tr_indices:
        old = seq[i]
        if isinstance(old, C3):
            c1 = old.cv1.conv.in_channels
            c2 = old.cv3.conv.out_channels
            n = len(old.m)
            new = C3TR(c1, c2, n, shortcut=True)
            _copy_shared_weights(new, old)
            _copy_layer_meta(new, old)
            seq[i] = new
