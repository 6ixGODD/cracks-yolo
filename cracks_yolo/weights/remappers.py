"""Per-architecture remappers: rewrite raw COCO checkpoint state_dict keys
into the cracks_yolo model layout.

Official YOLO checkpoints (ultralytics/v5/v7) store every layer under a
single top-level ``model.<global_idx>.<rest>`` Sequential. Our zoo models
split the layers into ``backbone./neck./head.`` subsections (and v8/v9/v10
add 1x1 Conv layers the upstream neck lacks). A simple prefix remap cannot
recover the per-section local index, so each remapper walks the model's
own layer sequence and the checkpoint's layer sequence together, pairing
layers by structural signature, then rebuilds keys as
``<section>.<local_idx>.<rest>``.

All remappers are pure functions registered on :class:`PretrainedSpec`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _submodule_keys(prefix: str, module: nn.Module) -> set[str]:
    """Return the set of leaf state_dict keys under ``prefix`` for ``module``."""
    return {f"{prefix}{k}" for k in module.state_dict().keys()}


def _layer_sections(model: nn.Module) -> list[tuple[str, nn.Sequential | nn.Module]]:
    """Ordered (section_name, layer_iterable) pairs covering the full forward.

    Detects ``backbone``/``neck`` Sequential attributes plus a ``head`` module
    (the YOLO zoo layout). Falls back to a single ``model`` section if absent.
    """
    sections: list[tuple[str, nn.Sequential | nn.Module]] = []
    for name in ("backbone", "neck"):
        seq = getattr(model, name, None)
        if isinstance(seq, nn.Sequential):
            sections.append((name, seq))
    head = getattr(model, "head", None)
    if head is not None:
        sections.append(("head", head))
    if not sections:
        sections.append(("model", model))
    return sections


def _ckpt_layer_index_groups(raw: dict[str, torch.Tensor]) -> dict[int, list[tuple[str, str]]]:
    """Group raw checkpoint keys by their ``model.<g>.`` global layer index.

    Returns ``{g: [(rest, full_key), ...]}``. Keys not starting with
    ``model.<int>.`` are returned under index ``-1`` (carried through as-is).
    """
    groups: dict[int, list[tuple[str, str]]] = {}
    for k in raw:
        parts = k.split(".", 2)
        if parts[0] == "model" and len(parts) >= 2 and parts[1].lstrip("-").isdigit():
            g = int(parts[1])
            rest = parts[2] if len(parts) > 2 else ""
            groups.setdefault(g, []).append((rest, k))
        else:
            groups.setdefault(-1, []).append(("", k))
    return groups


def _greedy_align(
    raw: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Pair checkpoint layers to model layers by key-name signature, then
    load per-key (shape-checked), rebuilding keys.

    Two-phase strategy:

    1. **Layer pairing by key-name set.** Flatten the model into an ordered
       list of ``(section, local_idx, module)`` triples and walk it alongside
       the checkpoint's sorted global layer indices. A checkpoint layer pairs
       with the next model layer whose state_dict **key-name set** (the per-
       layer rest suffixes, e.g. ``{conv.weight, bn.weight, ...}``) matches.
       Pairing on key names (not shapes) is essential: our v8/v9/v10 neck
       adds 1x1 Conv layers that change downstream channel counts, so a C2f
       layer's ``cv1.conv.weight`` shape differs between checkpoint and model.
       Name-set pairing still aligns the structurally-identical C2f blocks,
       while the extra Conv layers (no checkpoint counterpart) are skipped.

    2. **Per-key load with shape check.** Within a paired layer, rewrite each
       ``model.<g>.<rest>`` key to ``<section>.<local_idx>.<rest>`` and copy
       the tensor **only if shapes match**. Keys whose shapes differ (channel
       changes from the extra Convs) are dropped — they stay randomly
       initialized, exactly as ``strict=False`` intends.
    """
    sections = _layer_sections(model)
    # (section, idx_or_None, module). idx=None for non-Sequential sections
    # (the head) so keys map to ``<section>.<rest>`` not ``<section>.0.<rest>``.
    flat: list[tuple[str, int | None, nn.Module]] = []
    for sec_name, seq in sections:
        if isinstance(seq, nn.Sequential):
            for i, layer in enumerate(seq):
                flat.append((sec_name, i, layer))
        else:
            flat.append((sec_name, None, seq))

    groups = _ckpt_layer_index_groups(raw)
    non_layer_keys = groups.pop(-1, [])
    ckpt_order = sorted(groups.keys())

    # Per-layer key-name sets (rest suffixes), for pairing.
    def _rest_names(module: nn.Module) -> frozenset[str]:
        return frozenset(module.state_dict().keys())

    flat_names = [(sec, idx, mod, _rest_names(mod)) for sec, idx, mod in flat]

    out: dict[str, torch.Tensor] = {}
    for _, full_k in non_layer_keys:
        out[full_k] = raw[full_k]

    mi = 0
    for g in ckpt_order:
        ck_rests = {rest: full_k for rest, full_k in groups[g]}
        ck_names = frozenset(ck_rests.keys())
        ck_shapes = {rest: tuple(raw[full_k].shape) for rest, full_k in groups[g]}
        matched = False
        while mi < len(flat_names):
            sec, idx, mod, mod_names = flat_names[mi]
            mod_shapes = {k: tuple(v.shape) for k, v in mod.state_dict().items()}
            common = ck_names & mod_names
            # Pair when the key-name overlap is substantial (>= half of the
            # checkpoint layer's keys). This aligns structurally-similar blocks
            # even when one side has extra keys (C3TR vs C3; C2f with/without
            # the extra 1x1 Conv's channel change). Below the threshold the
            # model layer is treated as having no checkpoint counterpart.
            if common and len(common) >= max(1, len(ck_names) // 2):
                for rest in common:
                    if ck_shapes[rest] == mod_shapes.get(rest):
                        if idx is None:
                            new_key = f"{sec}." + rest if rest else sec
                        else:
                            new_key = f"{sec}.{idx}." + rest if rest else f"{sec}.{idx}"
                        out[new_key] = raw[ck_rests[rest]]
                mi += 1
                matched = True
                break
            mi += 1  # model layer has no checkpoint counterpart -> skip
        # If unmatched, the checkpoint layer is dropped (no model counterpart).
    return out


def yolo_remapper(
    raw: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Generic remapper for YOLOv5/v7/v8/v9/v10 checkpoints.

    Used by every YOLO zoo class whose checkpoint stores layers under a single
    ``model.<g>`` Sequential. The greedy shape-based alignment handles both
    faithful ports (v5: ~97% match) and ports with extra layers (v8/v9/v10:
    extra 1x1 Convs are skipped, remaining layers matched).
    """
    return _greedy_align(raw, model)


__all__ = ["yolo_remapper"]
