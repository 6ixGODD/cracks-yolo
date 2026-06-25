"""BaseModel — the unified detector base class with state machine.

Every model in the ZOO inherits from BaseModel. It provides:
- A 3-state state machine (UNINITIALIZED → PRETRAINED → TRAINED).
- Abstract methods: train_model, inference, save, from_pretrained.
- Concrete methods: analyze (efficiency + structure), load, state assertions.

Subclasses (UltralyticsAdapter, TorchvisionBase) implement the abstract
methods with their own training loops and inference decoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import enum
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class ModelState(enum.Enum):
    """Lifecycle state of a BaseModel instance."""

    UNINITIALIZED = "uninitialized"  # freshly built, no weights loaded
    PRETRAINED = "pretrained"  # COCO pretrained weights loaded, not trained on our data
    TRAINED = "trained"  # trained (or fine-tuned), ready for inference


# ---------------------------------------------------------------------------
# Inference result — unified output across all model families
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Detections for a single image, in original image coordinates.

    Attributes:
        boxes: (N, 4) xyxy in pixel coordinates of the original image.
        scores: (N,) confidence scores in [0, 1].
        labels: (N,) class indices (0-indexed).
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor


# ---------------------------------------------------------------------------
# Analysis report
# ---------------------------------------------------------------------------


@dataclass
class LayerInfo:
    """One node in the model structure tree."""

    name: str
    type: str
    n_params: int
    children: list[LayerInfo] = field(default_factory=list)


@dataclass
class ModelAnalysisReport:
    """Full analysis of a model: efficiency + structure."""

    model_name: str
    n_parameters: int
    n_trainable_parameters: int
    macs: float
    gflops: float
    fps_mean: float
    fps_p50: float
    fps_p95: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    peak_vram_bytes: int
    input_size: int
    device: str
    structure: list[LayerInfo]


# ---------------------------------------------------------------------------
# Train configuration (used by train_model)
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training configuration passed to BaseModel.train_model.

    Not pydantic — a plain dataclass so subclasses can extend freely.
    """

    output_dir: Path
    dataset: str = ""
    data_yaml: str = ""
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_epochs: int = 3
    warmup_lr: float = 1e-4
    momentum: float = 0.937
    amp: bool = True
    clip_grad_norm: float | None = None
    early_stopping_patience: int | None = None
    cosine_lr: bool = True
    cosine_lrf: float = 0.01
    use_ema: bool = True
    ema_decay: float = 0.9999
    optimizer: str = "adamw"
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 8
    pretrained: bool = True
    close_mosaic: int | None = None
    degrees: float = 0.0
    # For ultralytics: passed through to YOLO().train()
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainReport:
    """Result of a training run."""

    output_dir: Path
    best_epoch: int
    best_map50: float
    final_train_loss: float
    final_val_map50: float
    final_val_map5095: float
    total_epochs: int
    elapsed_sec: float
    early_stopped: bool = False
    checkpoint_path: Path | None = None


# ---------------------------------------------------------------------------
# BaseModel
# ---------------------------------------------------------------------------


class BaseModel(nn.Module):
    """Unified detector base class.

    Subclasses must implement:
        - ``train_model(config)``: run training, return TrainReport.
        - ``inference(images)``: return list[InferenceResult].
        - ``save(path, torchscript, onnx)``: persist weights.
        - ``from_pretrained(cls, ...)``: classmethod to load COCO weights.
        - ``load(path)``: load a trained checkpoint.

    The state machine enforces:
        - ``inference()`` only callable when state == TRAINED.
        - ``train_model()`` transitions to TRAINED.
        - ``from_pretrained()`` transitions to PRETRAINED.
    """

    def __init__(self, num_classes: int = 1, input_size: int = 640, logger: Any = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self._logger = logger
        self._state = ModelState.UNINITIALIZED

    @property
    def state(self) -> ModelState:
        return self._state

    def _set_state(self, state: ModelState) -> None:
        self._state = state
        if self._logger:
            self._logger.info(f"model state → {state.value}")

    def _assert_state(self, required: ModelState, action: str) -> None:
        if self._state != required:
            raise RuntimeError(
                f"cannot {action}: model state is {self._state.value}, expected {required.value}"
            )

    def train_model(self, config: TrainConfig) -> TrainReport:
        raise NotImplementedError

    def inference(self, images: torch.Tensor) -> list[InferenceResult]:
        raise NotImplementedError

    def save(self, path: Path, torchscript: bool = False, onnx: bool = False) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, num_classes: int = 1, **kwargs: Any) -> BaseModel:
        raise NotImplementedError

    def analyze(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_warmup: int = 5,
        n_runs: int = 20,
    ) -> ModelAnalysisReport:
        """Compute efficiency metrics + model structure tree."""
        dev = torch.device(device)
        self.to(dev).eval()
        isize = self.input_size
        model_name = type(self).__name__

        # Params
        n_params = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # MACs / GFLOPs
        macs = 0.0
        gflops = 0.0
        try:
            from thop import profile as thop_profile

            x = torch.zeros(1, 3, isize, isize, device=dev)
            macs, _ = thop_profile(self, inputs=(x,), verbose=False)
            gflops = 2 * macs / 1e9
        except Exception:
            pass

        # FPS / latency
        times: list[float] = []
        with torch.no_grad():
            x = torch.zeros(1, 3, isize, isize, device=dev)
            for _ in range(n_warmup):
                self(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            for _ in range(n_runs):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                self(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        fps_mean = n_runs / sum(times) if times else 0.0
        lat_mean = 1000 * sum(times) / max(n_runs, 1)
        lat_sorted = sorted(times)
        lat_p50 = 1000 * lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0
        lat_p95 = 1000 * lat_sorted[int(len(lat_sorted) * 0.95)] if lat_sorted else 0.0
        fps_p50 = 1000 / lat_p50 if lat_p50 > 0 else 0.0
        fps_p95 = 1000 / lat_p95 if lat_p95 > 0 else 0.0
        vram = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0

        # Structure tree
        structure = _build_structure_tree(self)

        return ModelAnalysisReport(
            model_name=model_name,
            n_parameters=n_params,
            n_trainable_parameters=n_train,
            macs=float(macs),
            gflops=round(gflops, 4),
            fps_mean=round(fps_mean, 1),
            fps_p50=round(fps_p50, 1),
            fps_p95=round(fps_p95, 1),
            latency_mean_ms=round(lat_mean, 2),
            latency_p50_ms=round(lat_p50, 2),
            latency_p95_ms=round(lat_p95, 2),
            peak_vram_bytes=vram,
            input_size=isize,
            device=device,
            structure=structure,
        )


def _build_structure_tree(
    module: nn.Module,
    prefix: str = "",
    max_depth: int = 3,
    depth: int = 0,
) -> list[LayerInfo]:
    """Walk module.named_children() recursively, return a tree of LayerInfo."""
    if depth >= max_depth:
        return []
    result: list[LayerInfo] = []
    for name, child in module.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        info = LayerInfo(
            name=name,
            type=type(child).__name__,
            n_params=n_params,
            children=_build_structure_tree(child, f"{prefix}.{name}", max_depth, depth + 1),
        )
        result.append(info)
    return result
