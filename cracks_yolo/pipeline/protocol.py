"""TrainPipeline / TestPipeline Protocols + pydantic configs.

Pipelines own no model knowledge — they call only the ``DetectorModel``
Protocol methods (forward, compute_loss, decode, build_optimizer). The
concrete pipeline implementation lands in a later pass; this module fixes
the *contract* so future code slots in without re-architecting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from cracks_yolo.metrics.schemas import MetricReport
from cracks_yolo.zoo.base import DetectorModel


class TrainConfig(BaseModel):
    """Configuration for a training run."""

    output_dir: Path
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=16, ge=1)
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    warmup_epochs: int = Field(default=3, ge=0)
    warmup_lr: float = Field(default=1e-4, gt=0.0)
    momentum: float = Field(default=0.937, ge=0.0, le=1.0)
    amp: bool = True
    grad_accum_steps: int = Field(default=1, ge=1)
    clip_grad_norm: float | None = None
    early_stopping_patience: int | None = None
    log_every_n_steps: int = Field(default=50, ge=1)
    val_interval: int = Field(default=1, ge=1)
    save_top_k: int = Field(default=3, ge=0)
    seed: int = 42
    num_workers: int = Field(default=4, ge=0)
    device: str = "cuda"
    # Cosine annealing: lr decays from cfg.lr to cfg.lr*cosine_lrf over epochs.
    cosine_lr: bool = False
    cosine_lrf: float = Field(default=0.01, gt=0.0, le=1.0)
    # Exponential moving average of weights (improves generalization ~0.5 mAP).
    use_ema: bool = False
    ema_decay: float = Field(default=0.9999, gt=0.0, lt=1.0)
    # Optimizer: "adamw" (default, v8+) or "sgd" (v7).
    optimizer: str = "adamw"


class TestConfig(BaseModel):
    """Configuration for a test/eval run."""

    output_dir: Path
    batch_size: int = 16
    iou_thr: float = 0.6
    conf_thr: float = 0.001
    max_dets: int = 300
    device: str = "cuda"
    num_workers: int = 4
    # When True, the test pipeline measures end-to-end FPS/latency (forward +
    # decode + NMS) over the real test loader and reports params/MACs/GFLOPs/
    # peak-VRAM via analyze_model. Disable to skip the extra forward passes.
    measure_efficiency: bool = True


class EfficiencyReport(BaseModel):
    """End-to-end efficiency metrics measured during a test run.

    ``fps_*`` and ``latency_*`` are measured on the real test loader (forward
    + decode + NMS, batch=``TestConfig.batch_size``), so they reflect actual
    inference throughput — not a synthetic single-image forward. ``n_*``,
    ``macs``, ``gflops`` come from :func:`cracks_yolo.analysis.analyze_model`
    on a single-image input; ``peak_vram_bytes`` is the high-water mark over
    the real inference loop (falls back to the synthetic value on CPU).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_images: int = 0
    # Real end-to-end inference throughput (images/sec).
    fps_mean: float = 0.0
    fps_p50: float = 0.0
    fps_p95: float = 0.0
    # Per-image latency (ms) = batch time / batch size.
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    # Structural / compute-budget metrics (single-image analysis).
    n_parameters: int = 0
    n_trainable_parameters: int = 0
    macs: float = 0.0
    gflops: float = 0.0
    peak_vram_bytes: int = 0
    input_size: int = 0
    device: str = ""
    batch_size: int = 1


class TrainReport(BaseModel):
    """Result of a training run."""

    output_dir: Path
    best_epoch: int
    best_map50: float
    final_train_loss: float
    final_val_map50: float
    final_val_map5095: float
    total_steps: int
    total_epochs: int
    elapsed_sec: float
    early_stopped: bool = False
    checkpoint_paths: list[Path] = Field(default_factory=list)


class TestReport(BaseModel):
    """Result of a test run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_dir: Path
    metrics: MetricReport
    elapsed_sec: float
    efficiency: EfficiencyReport | None = None


@runtime_checkable
class TrainPipeline(Protocol):
    """Training pipeline contract.

    The pipeline does NOT branch on model type — it only calls
    :class:`DetectorModel` Protocol methods. ``run`` returns a structured
    :class:`TrainReport`.
    """

    def run(
        self,
        model: DetectorModel,
        train_loader: Any,
        val_loader: Any | None,
        cfg: TrainConfig,
    ) -> TrainReport:
        """Train ``model`` on ``train_loader``, validate on ``val_loader``."""
        ...


@runtime_checkable
class TestPipeline(Protocol):
    """Test/eval pipeline contract.

    Calls ``model.forward`` + ``model.decode`` and feeds detections to a
    :class:`MetricsCalculator`. Returns a structured :class:`TestReport`.
    """

    def run(
        self,
        model: DetectorModel,
        test_loader: Any,
        cfg: TestConfig,
    ) -> TestReport:
        """Evaluate ``model`` on ``test_loader`` and return metrics."""
        ...
