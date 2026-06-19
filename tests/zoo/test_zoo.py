"""End-to-end smoke tests for every class in :data:`cracks_yolo.zoo.ZOO`.

Each test instantiates a model with ``num_classes=1``, runs a forward pass
on a small batch, computes the loss against random targets, runs backward,
and asserts that every trainable parameter received a gradient. This is the
contract that pipelines depend on.
"""

from __future__ import annotations

from typing import Any
from typing import cast

import pytest
import torch
import torch.nn as nn

from cracks_yolo.zoo import ZOO
from cracks_yolo.zoo.base import DetectorModel


def _make_targets(num_classes: int = 1) -> torch.Tensor:
    """Return a small ``(N, 6)`` target tensor — (img_idx, cls, x, y, w, h)."""
    return torch.tensor(
        [
            [0, 0 % num_classes, 0.50, 0.50, 0.20, 0.20],
            [0, 0 % num_classes, 0.30, 0.70, 0.10, 0.10],
            [1, 0 % num_classes, 0.40, 0.40, 0.15, 0.25],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def small_image_batch() -> torch.Tensor:
    """A ``(2, 3, 640, 640)`` random image batch."""
    return torch.randn(2, 3, 640, 640)


def _instantiate(cls: type[nn.Module], num_classes: int = 1) -> nn.Module:
    """Instantiate a ZOO class. Returns nn.Module; cast to DetectorModel
    where protocol-specific attribute access is needed.

    The registry is typed as ``dict[str, type[nn.Module]]`` (the common
    supertype) because Protocol classes can't be used as concrete class types.
    """
    return cls(num_classes=num_classes)


def _as_detector(model: nn.Module) -> DetectorModel:
    """Cast an nn.Module to the DetectorModel Protocol for attribute access."""
    return cast(DetectorModel, model)


def _forward_loss_backward(cls: type[nn.Module], num_classes: int = 1) -> dict[str, Any]:
    """Run the full forward → loss → backward contract for ``cls``."""
    torch.manual_seed(0)
    model = _instantiate(cls, num_classes=num_classes)
    detector = _as_detector(model)

    # Train-mode forward → loss → backward.
    model.train()
    x = torch.randn(2, 3, 640, 640)
    preds = model(x)
    targets = _make_targets(num_classes=num_classes)

    # v7 needs the image batch passed to compute_loss (OTA assignment).
    if cls.__name__.startswith("YOLOv7"):
        loss, parts = detector.compute_loss(preds, targets, imgs=x)
    else:
        loss, parts = detector.compute_loss(preds, targets)

    assert torch.isfinite(loss).all(), f"{cls.__name__}: loss is not finite"
    assert loss.requires_grad, f"{cls.__name__}: loss does not require grad"
    loss.backward()  # type: ignore[no-untyped-call]

    no_grad_params = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    return {
        "loss": float(loss.detach()),
        "parts": parts.detach().tolist(),
        "no_grad_params": no_grad_params,
        "preds": preds,
    }


@pytest.mark.parametrize("key", list(ZOO.keys()))
def test_zoo_forward_loss_backward(key: str) -> None:
    """Every ZOO class: forward → loss → backward → all params have grad."""
    cls = ZOO[key]
    result = _forward_loss_backward(cls, num_classes=1)
    assert not result["no_grad_params"], (
        f"{cls.__name__}: parameters with no gradient: {result['no_grad_params'][:5]}"
    )


@pytest.mark.parametrize("key", list(ZOO.keys()))
def test_zoo_eval_forward(key: str) -> None:
    """Every ZOO class: eval-mode forward + decode produces a final tensor.

    Anchor-based (v5/v7): forward returns a ``(decoded, raw)`` tuple; decode
    extracts the decoded tensor ``(B, N, nc+5)``.
    Anchor-free (v8/v10): forward returns the decoded tensor directly
    ``(B, 4+nc, N)``.
    """
    cls = ZOO[key]
    torch.manual_seed(0)
    model = _instantiate(cls, num_classes=1)
    detector = _as_detector(model)
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 640, 640)
        out = model(x)
    decoded = detector.decode(out) if not isinstance(out, torch.Tensor) else out
    assert isinstance(decoded, torch.Tensor), f"{cls.__name__}: decode did not return a tensor"
    assert decoded.dim() == 3, f"{cls.__name__}: expected 3D output, got {decoded.dim()}D"
    assert decoded.shape[0] == 2, f"{cls.__name__}: expected batch=2, got {decoded.shape[0]}"
    cls_name = cls.__name__
    if cls_name.startswith(("YOLOv5", "YOLOv7")):
        # (B, N, nc+5) — 25200 anchors, last dim 6 (4+1+1).
        assert decoded.shape[-1] == 6, (
            f"{cls_name}: expected last dim 6 (4+1+1), got {decoded.shape[-1]}"
        )
        assert decoded.shape[1] == 25200, (
            f"{cls_name}: expected 25200 anchors, got {decoded.shape[1]}"
        )
    elif cls_name.startswith(("RetinaNet", "FasterRCNN", "MaskRCNN", "FCOS", "SSD")):
        # torchvision wrapper: (B, N_max, 6) where N_max is variable per batch.
        # Just check the last-dim contract (4 bbox + 1 score + 1 cls = 6).
        assert decoded.shape[-1] == 6, (
            f"{cls_name}: expected last dim 6 (4+1+1), got {decoded.shape[-1]}"
        )
    else:  # v8/v10/v9
        # (B, 4+nc, N) — 5 channels, 8400 grid cells.
        assert decoded.shape[1] == 5, (
            f"{cls_name}: expected middle dim 5 (4+1), got {decoded.shape[1]}"
        )
        assert decoded.shape[-1] == 8400, (
            f"{cls_name}: expected 8400 grid cells, got {decoded.shape[-1]}"
        )


@pytest.mark.parametrize("key", list(ZOO.keys()))
def test_zoo_stride(key: str) -> None:
    """Every ZOO class: stride is the expected [8, 16, 32]."""
    cls = ZOO[key]
    model = _instantiate(cls, num_classes=1)
    stride = _as_detector(model).stride
    assert isinstance(stride, torch.Tensor)
    assert stride.tolist() == [8.0, 16.0, 32.0], (
        f"{cls.__name__}: stride = {stride.tolist()}, expected [8, 16, 32]"
    )


@pytest.mark.parametrize("key", list(ZOO.keys()))
def test_zoo_build_optimizer(key: str) -> None:
    """Every ZOO class: build_optimizer returns a working optimizer."""
    cls = ZOO[key]
    model = _instantiate(cls, num_classes=1)
    opt = _as_detector(model).build_optimizer()
    assert isinstance(opt, torch.optim.Optimizer)
    # Optimizer must reference at least one param group with parameters.
    assert len(opt.param_groups) >= 1
    assert sum(len(g["params"]) for g in opt.param_groups) > 0


@pytest.mark.parametrize("key", list(ZOO.keys()))
def test_zoo_class_satisfies_protocol(key: str) -> None:
    """Every ZOO class structurally satisfies the DetectorModel Protocol.

    Structural check: instantiate and verify the protocol-required attributes
    are present. We don't use isinstance (Protocol is not runtime_checkable
    by default here); we check attribute presence directly.
    """
    cls = ZOO[key]
    model = _instantiate(cls, num_classes=1)
    for attr in (
        "input_size",
        "num_classes",
        "class_names",
        "stride",
        "pretrained_spec",
        "forward",
        "compute_loss",
        "decode",
        "build_optimizer",
    ):
        assert hasattr(model, attr), f"{cls.__name__}: missing {attr!r}"
    # from_pretrained is a classmethod.
    assert hasattr(cls, "from_pretrained"), f"{cls.__name__}: missing from_pretrained"


def test_zoo_registry_keys() -> None:
    """ZOO has the expected entries (20 YOLO + 6 torchvision = 26 total)."""
    expected = {
        "yolov5s",
        "yolov5s_sac",
        "yolov5s_tr",
        "yolov5s_sactr",
        "yolov7w",
        "yolov7w_sac",
        "yolov8n",
        "yolov8n_sac",
        "yolov8s",
        "yolov8s_sac",
        "yolov8m",
        "yolov8m_sac",
        "yolov8l",
        "yolov8l_sac",
        "yolov8x",
        "yolov8x_sac",
        "yolov10s",
        "yolov10s_sac",
        "yolov9c",
        "yolov9c_sac",
        "retinanet_r50",
        "faster_rcnn_r50",
        "mask_rcnn_r50",
        "fcos_r50",
        "ssd300_vgg16",
        "ssdlite320_mobilenetv3",
    }
    assert set(ZOO.keys()) == expected, f"ZOO keys = {set(ZOO.keys())}"


def test_zoo_pretrained_specs() -> None:
    """Only the baseline variants (no SAC/TR) declare COCO pretrained specs."""
    from cracks_yolo.zoo import YOLOv5s
    from cracks_yolo.zoo import YOLOv5sSAC
    from cracks_yolo.zoo import YOLOv5sSACTR
    from cracks_yolo.zoo import YOLOv5sTR
    from cracks_yolo.zoo import YOLOv7w
    from cracks_yolo.zoo import YOLOv7wSAC
    from cracks_yolo.zoo import YOLOv8s
    from cracks_yolo.zoo import YOLOv8sSAC
    from cracks_yolo.zoo import YOLOv9c
    from cracks_yolo.zoo import YOLOv9cSAC
    from cracks_yolo.zoo import YOLOv10s
    from cracks_yolo.zoo import YOLOv10sSAC

    for baseline in (YOLOv5s, YOLOv7w, YOLOv8s, YOLOv10s, YOLOv9c):
        assert baseline.pretrained_spec is not None, (  # type: ignore[attr-defined]
            f"{baseline.__name__} should have a pretrained_spec"
        )
    for variant in (
        YOLOv5sSAC,
        YOLOv5sTR,
        YOLOv5sSACTR,
        YOLOv7wSAC,
        YOLOv8sSAC,
        YOLOv10sSAC,
        YOLOv9cSAC,
    ):
        assert variant.pretrained_spec is None, (  # type: ignore[attr-defined]
            f"{variant.__name__} should have pretrained_spec=None "
            "(SAC/TR layers have no COCO weights)"
        )


# DetectorModel Protocol re-exported from the zoo package.
def test_detector_model_protocol_exported() -> None:
    """The Protocol is importable from cracks_yolo.zoo."""
    assert DetectorModel is not None
    # Protocol structural check via runtime attributes (skip isinstance since
    # the Protocol isn't decorated @runtime_checkable).
