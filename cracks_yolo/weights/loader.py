"""Pretrained weight loader: download, key-remap, partial-load + LoadReport.

Decoupled from the model zoo — given an ``nn.Module`` and a
:class:`~cracks_yolo.zoo.base.PretrainedSpec`, fetches the ``.pt`` file
(streams to ``weights/`` if missing), remaps state_dict keys, and loads
with ``strict=False``. Returns a :class:`LoadReport` describing what
matched / was missing / was unexpected.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
from typing import TYPE_CHECKING
from typing import Any

import requests
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from cracks_yolo.zoo.base import PretrainedSpec

logger = logging.getLogger(__name__)


@dataclass
class LoadReport:
    """Result of a pretrained-weight load operation.

    Attributes:
        model: The model instance with weights loaded.
        matched: Keys successfully loaded from the checkpoint.
        missing: Keys present in the model but absent from the checkpoint
            (e.g. SAC/TR layers when loading COCO weights — random init).
        unexpected: Keys present in the checkpoint but absent from the model.
        key: The :attr:`PretrainedSpec.key` used for the cache filename.
        url: The URL the weights were fetched from.
        cached: True if the weights file was already on disk before this call.
    """

    model: nn.Module
    matched: list[str]
    missing: list[str]
    unexpected: list[str]
    key: str
    url: str
    cached: bool


def _default_weights_dir() -> Path:
    return Path.cwd() / "weights"


def _download(url: str, dest: Path) -> None:
    """Stream-download ``url`` to ``dest`` with progress logging.

    Uses ``requests`` (a runtime dependency). Raises if the request fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    logger.info(
                        "downloading %s: %d/%d (%.1f%%)",
                        url,
                        downloaded,
                        total,
                        100.0 * downloaded / total,
                    )
                else:
                    logger.info("downloading %s: %d bytes", url, downloaded)


class _StateCapture(nn.Module):
    """Stand-in for unpickling checkpoint model objects whose real modules
    (``models.*``, ``ultralytics.*``) aren't importable.

    Subclasses :class:`nn.Module` so that ``state_dict()`` recurses the
    captured ``_modules`` / ``_parameters`` / ``_buffers`` and produces a
    flat tensor dict (matching the ultralytics/YOLOv5 layer naming). A plain
    object returning ``self.__dict__`` would instead expose nn.Module's
    internal bookkeeping dicts, not the weights.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Don't call super().__init__ here — the pickled __setstate__ payload
        # restores the full nn.Module internal state (including hook dicts
        # whose presence the torch version on disk may differ from ours).
        # We run nn.Module.__init__ inside __setstate__ to guarantee every
        # expected internal attribute exists before the payload overwrites it.
        pass

    def __setstate__(self, state: Any) -> None:
        nn.Module.__init__(self)
        if isinstance(state, dict):
            self.__dict__.update(state)


class _LenientUnpickler(pickle.Unpickler):
    """Unpickler that resolves ``models.*`` / ``utils.*`` / ``ultralytics.*``
    references to :class:`_StateCapture` so YOLO checkpoints saved by the
    upstream repos unpickle without those packages installed.

    Unlike injecting dummy modules into ``sys.modules`` (which leaks and can
    break ``inspect`` for unrelated imports), this only affects this one
    ``torch.load`` call via the ``pickle_module`` parameter.
    """

    _TOP_LEVELS = ("models", "utils", "ultralytics")

    def find_class(self, module: str, name: str) -> Any:
        if module.split(".", 1)[0] in self._TOP_LEVELS:
            return _StateCapture
        return super().find_class(module, name)


class _PickleShim:
    """Shim passed as ``torch.load(pickle_module=...)`` to use
    :class:`_LenientUnpickler` without polluting ``sys.modules``."""

    Unpickler = _LenientUnpickler


def _torch_load_lenient(path: Path) -> Any:
    """``torch.load`` that unpickles YOLOv5/v7/v8/v9/v10 checkpoints without
    requiring ``models``/``utils``/``ultralytics`` to be importable."""
    try:
        # Plain state-dict checkpoints (including official DETR)
        # need no pickle globals and should use PyTorch's restricted loader.
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_PickleShim)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a ``.pt`` checkpoint and return the inner ``state_dict``.

    Handles both raw ``state_dict`` checkpoints and Ultralytics/YOLOv5-style
    ``{"model": nn.Module, ...}`` checkpoints (which pickle the full model
    object — see :func:`_torch_load_lenient`).
    """
    ckpt = _torch_load_lenient(path)
    if isinstance(ckpt, dict) and "model" in ckpt:
        inner = ckpt["model"]
        if hasattr(inner, "state_dict"):
            return inner.state_dict()  # type: ignore[no-any-return]
        if isinstance(inner, dict):
            return inner
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]  # type: ignore[no-any-return]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unrecognized checkpoint format at {path}")


def _remap_keys(
    state_dict: dict[str, torch.Tensor], key_map: dict[str, str]
) -> dict[str, torch.Tensor]:
    if not key_map:
        return state_dict
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        for src, dst in key_map.items():
            if k.startswith(src):
                new_k = dst + k[len(src) :]
                break
        out[new_k] = v
    return out


def load_pretrained(
    model: nn.Module,
    spec: PretrainedSpec,
    weights_dir: Path | None = None,
    strict: bool = False,
) -> LoadReport:
    """Load COCO pretrained weights into ``model``.

    Args:
        model: Target model (mutated in-place — its ``load_state_dict`` is
            called with ``strict=strict``).
        spec: Where to fetch the weights and how to remap keys.
        weights_dir: Directory to cache ``{key}.pt``. Defaults to ``./weights``.
        strict: If True, missing/unexpected keys raise. Default False so
            SAC/TR layers load partially.
    """
    weights_dir = weights_dir or _default_weights_dir()
    dest = weights_dir / f"{spec.key}.pt"
    cached = dest.exists()
    if not cached:
        logger.info("downloading %s -> %s", spec.url, dest)
        _download(spec.url, dest)
    else:
        logger.info("using cached weights at %s", dest)

    raw = _load_state_dict(dest)
    remapped = _remap_keys(raw, spec.state_dict_key_map)
    if spec.remapper is not None:
        remapped = spec.remapper(remapped, model)

    model_state = model.state_dict()
    # ``strict=False`` still raises for shape mismatches.  Detection
    # checkpoints commonly have a COCO-sized classification head while the
    # target model has a project-specific class count, so only pass compatible
    # tensors to PyTorch and leave the new head randomly initialized.
    compatible = {
        k: v for k, v in remapped.items() if k in model_state and model_state[k].shape == v.shape
    }
    shape_mismatches = [
        k for k, v in remapped.items() if k in model_state and model_state[k].shape != v.shape
    ]
    if strict and shape_mismatches:
        raise RuntimeError(f"pretrained tensor shape mismatch: {shape_mismatches}")
    result = model.load_state_dict(compatible, strict=strict)
    n_matched = len(compatible)
    n_total = len(model_state)
    logger.info(
        "pretrained load %s: matched %d/%d keys, missing %d, unexpected %d",
        spec.key,
        n_matched,
        n_total,
        len(result.missing_keys),
        len(result.unexpected_keys),
    )
    return LoadReport(
        model=model,
        matched=list(compatible),
        missing=list(result.missing_keys),
        unexpected=list(result.unexpected_keys)
        + [k for k in remapped if k not in model_state]
        + shape_mismatches,
        key=spec.key,
        url=spec.url,
        cached=cached,
    )
