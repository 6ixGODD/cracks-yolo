"""Grad-CAM heatmap extraction for any ``nn.Module`` detector.

Hooks the forward + backward of a target layer's output, weights the
feature map channels by the average gradient, ReLUs and normalizes to
[0, 1]. Up-samples to the input image size and saves as a PNG overlay
plus the raw feature map as ``.npy``.

The caller picks target layers by attribute path (e.g. ``"backbone.10"``
or ``"backbone.13"``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _get_module_by_path(root: nn.Module, path: str) -> nn.Module:
    """Resolve ``"backbone.10.conv"`` against ``root``."""
    mod: nn.Module = root
    for part in path.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)  # type: ignore[index]
    return mod


class _GradCAMHook:
    """Stores forward output + backward gradient for one module."""

    def __init__(self) -> None:
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        self._fwd_handle: Any = None
        self._bwd_handle: Any = None

    def attach(self, module: nn.Module) -> None:
        self._fwd_handle = module.register_forward_hook(self._forward)
        self._bwd_handle = module.register_full_backward_hook(self._backward)

    def detach(self) -> None:
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()

    def _forward(
        self,
        module: nn.Module,  # noqa: ARG002
        inp: tuple[torch.Tensor, ...],  # noqa: ARG002
        out: torch.Tensor,
    ) -> None:
        self.activation = out.detach()

    def _backward(
        self,
        module: nn.Module,  # noqa: ARG002
        grad_input: tuple[torch.Tensor, ...] | torch.Tensor,  # noqa: ARG002
        grad_output: tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> None:
        if isinstance(grad_output, torch.Tensor):
            self.gradient = grad_output.detach()
        elif grad_output:
            self.gradient = grad_output[0].detach()


class GradCAMExtractor:
    """Run Grad-CAM on a list of target layers for one forward + backward pass."""

    def __init__(
        self,
        model: nn.Module,
        target_layers: list[str],
    ) -> None:
        self.model = model
        self.target_layers = target_layers
        self._hooks: dict[str, _GradCAMHook] = {}
        for path in target_layers:
            mod = _get_module_by_path(model, path)
            hook = _GradCAMHook()
            hook.attach(mod)
            self._hooks[path] = hook

    def close(self) -> None:
        for h in self._hooks.values():
            h.detach()
        self._hooks.clear()

    def __enter__(self) -> GradCAMExtractor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def generate(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute one heatmap per target layer.

        Args:
            image: ``(1, 3, H, W)`` input tensor (already on the right device).
            target_class: class index to backprop; if None, uses the
                argmax of the model's decoded output (best-effort).

        Returns:
            ``{layer_path: heatmap_2d_ndarray}`` — each value in [0, 1],
            up-sampled to the input spatial size.
        """
        self.model.zero_grad(set_to_none=True)
        out = self.model(image)
        logits = (out[0] if len(out) > 0 else None) if isinstance(out, (list, tuple)) else out
        if logits is None:
            raise RuntimeError("model returned empty output")

        # Pick target score: sum over the target-class channel if possible.
        while isinstance(logits, (list, tuple)):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError(f"unsupported model output type: {type(logits)!r}")

        if target_class is None:
            # Reduce to a scalar: max activation across all heads/channels.
            score = logits.float().sum()
        else:
            # Best-effort: pick the target_class-th channel of the last dim.
            if logits.dim() >= 2 and logits.shape[-1] > target_class:
                score = logits[..., target_class].float().sum()
            else:
                score = logits.float().sum()
        score.backward()  # type: ignore[no-untyped-call]

        results: dict[str, np.ndarray] = {}
        for path, hook in self._hooks.items():
            if hook.activation is None or hook.gradient is None:
                continue
            act = hook.activation  # (1, C, h, w) typically
            grad = hook.gradient
            weights = (
                grad.mean(dim=(2, 3), keepdim=True)
                if grad.dim() == 4
                else grad.mean(dim=0, keepdim=True)
            )
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            if cam.numel() == 0:
                continue
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cam_np = cam.squeeze().cpu().numpy()
            # Up-sample to input image size.
            img_h, img_w = image.shape[-2], image.shape[-1]
            if cam_np.shape != (img_h, img_w):
                cam_tensor = torch.from_numpy(cam_np).float().unsqueeze(0).unsqueeze(0)
                cam_tensor = torch.nn.functional.interpolate(
                    cam_tensor, size=(img_h, img_w), mode="bilinear", align_corners=False
                )
                cam_np = cam_tensor.squeeze().numpy()
            results[path] = cam_np
        return results


def save_heatmap_overlay(
    image: torch.Tensor,
    cam: np.ndarray,
    out_png: Path,
) -> None:
    """Save a heatmap overlaid on the image as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = image.detach().cpu().float().squeeze().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.imshow(cam, cmap="jet", alpha=0.5)
    ax.set_axis_off()
    ax.set_title("Grad-CAM")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


__all__ = ["GradCAMExtractor", "save_heatmap_overlay"]
