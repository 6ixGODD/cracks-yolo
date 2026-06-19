"""Tests for :mod:`cracks_yolo.weights.loader`.

The download path is mocked — no network access required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from cracks_yolo.weights.loader import LoadReport
from cracks_yolo.weights.loader import _load_state_dict
from cracks_yolo.weights.loader import _remap_keys
from cracks_yolo.weights.loader import load_pretrained
from cracks_yolo.zoo.base import PretrainedSpec


class _DummyModel(nn.Module):
    """Tiny model with two named params for partial-load tests."""

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(3, 4)
        self.lin2 = nn.Linear(4, 2)


def _make_fake_state_dict() -> dict[str, torch.Tensor]:
    """A state_dict matching _DummyModel exactly."""
    m = _DummyModel()
    return dict(m.state_dict())


def _make_fake_checkpoint(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    """Write a state_dict to ``path`` as a raw .pt file (dict format)."""
    torch.save(state_dict, path)


class _FakeResponse:
    """A minimal stand-in for requests.Response for stream-download tests."""

    def __init__(self, content: bytes) -> None:
        self._content = content
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 1 << 16) -> list[bytes]:  # noqa: ARG002
        # Split into 2 chunks for realism.
        mid = len(self._content) // 2
        return [self._content[:mid], self._content[mid:]]

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_remap_keys_no_map() -> None:
    """Empty key_map returns the state_dict unchanged."""
    sd = {"a": torch.zeros(1), "b": torch.zeros(2)}
    out = _remap_keys(sd, {})
    assert set(out.keys()) == {"a", "b"}


def test_remap_keys_prefix_substitution() -> None:
    """Prefix-substitution remaps matching keys, leaves others untouched."""
    sd = {
        "model.0.weight": torch.zeros(1),
        "model.1.bias": torch.zeros(2),
        "other.weight": torch.zeros(3),
    }
    out = _remap_keys(sd, {"model.": "backbone."})
    assert "backbone.0.weight" in out
    assert "backbone.1.bias" in out
    assert "other.weight" in out
    assert "model.0.weight" not in out


def test_load_state_dict_dict_format(tmp_path: Path) -> None:
    """_load_state_dict returns a raw dict checkpoint as-is."""
    sd = _make_fake_state_dict()
    p = tmp_path / "raw.pt"
    _make_fake_checkpoint(p, sd)
    out = _load_state_dict(p)
    assert isinstance(out, dict)
    assert "lin1.weight" in out


def test_load_state_dict_with_state_dict_key(tmp_path: Path) -> None:
    """_load_state_dict unwraps ``{"state_dict": ...}`` checkpoints."""
    sd = _make_fake_state_dict()
    p = tmp_path / "wrapped.pt"
    torch.save({"state_dict": sd, "epoch": 10}, p)
    out = _load_state_dict(p)
    assert "lin1.weight" in out


def test_load_state_dict_with_model_key(tmp_path: Path) -> None:
    """_load_state_dict extracts state_dict from ``{"model": nn.Module}``."""
    p = tmp_path / "ultralytics.pt"
    torch.save({"model": _DummyModel(), "epoch": 10}, p)
    out = _load_state_dict(p)
    assert "lin1.weight" in out


def test_load_pretrained_cache_hit(
    tmp_weights_dir: Path,
) -> None:
    """load_pretrained uses the cached .pt file when present (no download)."""
    sd = _make_fake_state_dict()
    dest = tmp_weights_dir / "dummy.pt"
    _make_fake_checkpoint(dest, sd)

    spec = PretrainedSpec(
        key="dummy",
        url="https://example.com/dummy.pt",
        state_dict_key_map={},
    )
    model = _DummyModel()
    report = load_pretrained(model=model, spec=spec, weights_dir=tmp_weights_dir)
    assert isinstance(report, LoadReport)
    assert report.cached is True
    assert report.key == "dummy"
    assert "lin1.weight" in report.matched
    assert len(report.missing) == 0
    assert len(report.unexpected) == 0


def test_load_pretrained_download(
    tmp_weights_dir: Path,
) -> None:
    """load_pretrained downloads the .pt when missing and writes to cache."""
    sd = _make_fake_state_dict()
    ckpt_bytes = _serialize_state_dict(sd)

    spec = PretrainedSpec(
        key="dummy_dl",
        url="https://example.com/dummy_dl.pt",
        state_dict_key_map={},
    )

    # Patch requests.get to return our fake response without network access.
    fake_resp = _FakeResponse(ckpt_bytes)
    with patch("cracks_yolo.weights.loader.requests.get", return_value=fake_resp):
        model = _DummyModel()
        report = load_pretrained(model=model, spec=spec, weights_dir=tmp_weights_dir)

    assert report.cached is False
    assert (tmp_weights_dir / "dummy_dl.pt").exists()
    assert "lin1.weight" in report.matched


def test_load_pretrained_strict_false_reports_missing(
    tmp_weights_dir: Path,
) -> None:
    """With strict=False, missing keys are reported (not raised)."""
    # State dict with only lin1 — lin2 should be reported as missing.
    full_sd = _make_fake_state_dict()
    partial_sd = {k: v for k, v in full_sd.items() if k.startswith("lin1")}
    dest = tmp_weights_dir / "partial.pt"
    _make_fake_checkpoint(dest, partial_sd)

    spec = PretrainedSpec(
        key="partial",
        url="https://example.com/partial.pt",
        state_dict_key_map={},
    )
    model = _DummyModel()
    report = load_pretrained(model=model, spec=spec, weights_dir=tmp_weights_dir)
    assert any("lin2" in k for k in report.missing)
    assert all("lin1" in k for k in report.matched)


def test_load_pretrained_strict_true_raises_on_missing(
    tmp_weights_dir: Path,
) -> None:
    """With strict=True, missing keys raise RuntimeError."""
    full_sd = _make_fake_state_dict()
    partial_sd = {k: v for k, v in full_sd.items() if k.startswith("lin1")}
    dest = tmp_weights_dir / "partial_strict.pt"
    _make_fake_checkpoint(dest, partial_sd)

    spec = PretrainedSpec(
        key="partial_strict",
        url="https://example.com/partial_strict.pt",
        state_dict_key_map={},
    )
    model = _DummyModel()
    with pytest.raises(RuntimeError):
        load_pretrained(model=model, spec=spec, weights_dir=tmp_weights_dir, strict=True)


def test_load_pretrained_unexpected_keys(
    tmp_weights_dir: Path,
) -> None:
    """Keys in the checkpoint but not the model are reported as unexpected."""
    sd = _make_fake_state_dict()
    sd["lin_unknown.weight"] = torch.zeros(5)
    dest = tmp_weights_dir / "extra.pt"
    _make_fake_checkpoint(dest, sd)

    spec = PretrainedSpec(
        key="extra",
        url="https://example.com/extra.pt",
        state_dict_key_map={},
    )
    model = _DummyModel()
    report = load_pretrained(model=model, spec=spec, weights_dir=tmp_weights_dir)
    assert "lin_unknown.weight" in report.unexpected


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Serialize a state_dict to bytes (as torch.save would)."""
    import io

    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getvalue()
