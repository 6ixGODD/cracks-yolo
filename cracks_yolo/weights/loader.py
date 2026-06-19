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
import sys
import types
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


class _StateCapture:
    """Stand-in class for unpickling checkpoint objects whose real module
    isn't importable.

    YOLOv5/v7 checkpoints pickle the full ``DetectionModel`` object, which
    references ``models.yolo.DetectionModel``, ``models.common.Conv``, etc.
    We don't have those modules (they live in ``deps/`` and we never import
    from there at runtime). This dummy accepts any ``__setstate__`` payload
    into ``__dict__`` and exposes ``state_dict()`` so the caller can extract
    the weight tensors.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.__dict__


class _DummyModule(types.ModuleType):
    """Module that returns :class:`_StateCapture` for any attribute access.

    Installed temporarily in ``sys.modules`` so :func:`torch.load` can
    unpickle YOLOv5/v7 checkpoint objects whose real classes we don't
    import. After unpickling, we extract ``.state_dict()`` and discard the
    object — we never call any of its methods.
    """

    def __getattr__(self, name: str) -> type[_StateCapture]:
        return _StateCapture


_YOLOV5_PICKLE_MODULES = (
    "models",
    "models.yolo",
    "models.common",
    "models.experimental",
    "utils",
    "utils.general",
    "utils.torch_utils",
    "utils.loss",
    "utils.metrics",
    "utils.autoanchor",
    "utils.plots",
    "utils.dataloaders",
    "utils.augmentations",
    "utils.downloads",
)


def _torch_load_lenient(path: Path) -> Any:
    """``torch.load`` with dummy modules for YOLOv5/v7 checkpoint unpickling.

    Restores ``sys.modules`` exactly as it was on exit.
    """
    saved: dict[str, types.ModuleType | None] = {}
    installed: list[str] = []
    try:
        for mod_name in _YOLOV5_PICKLE_MODULES:
            if mod_name in sys.modules:
                saved[mod_name] = sys.modules[mod_name]
            else:
                saved[mod_name] = None
                sys.modules[mod_name] = _DummyModule(mod_name)
                installed.append(mod_name)
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        for mod_name, original in saved.items():
            if original is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = original


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

    result = model.load_state_dict(remapped, strict=strict)
    return LoadReport(
        model=model,
        matched=[k for k in remapped if k in model.state_dict()],
        missing=list(result.missing_keys),
        unexpected=list(result.unexpected_keys),
        key=spec.key,
        url=spec.url,
        cached=cached,
    )
