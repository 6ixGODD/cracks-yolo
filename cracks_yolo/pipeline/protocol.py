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
    log_every_n_steps: int = Field(default=50, ge=1)
    val_interval: int = Field(default=1, ge=1)
    save_top_k: int = Field(default=3, ge=0)
    seed: int = 42
    num_workers: int = Field(default=4, ge=0)
    device: str = "cuda"


class TestConfig(BaseModel):
    """Configuration for a test/eval run."""

    output_dir: Path
    batch_size: int = 16
    iou_thr: float = 0.6
    conf_thr: float = 0.001
    max_dets: int = 300
    device: str = "cuda"
    num_workers: int = 4


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
    checkpoint_paths: list[Path] = Field(default_factory=list)


class TestReport(BaseModel):
    """Result of a test run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_dir: Path
    metrics: MetricReport
    elapsed_sec: float


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
