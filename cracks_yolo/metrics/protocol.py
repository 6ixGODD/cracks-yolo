"""MetricsCalculator Protocol.

Pipelines depend on this Protocol — never on a concrete implementation. The
real implementation (pycocotools + torchvision) lands in a later pass.
"""

from __future__ import annotations

from typing import Protocol
from typing import runtime_checkable

from cracks_yolo.metrics.schemas import MetricReport
from cracks_yolo.metrics.schemas import PerImageDetection


@runtime_checkable
class MetricsCalculator(Protocol):
    """Compute detection metrics from a list of per-image detections.

    Train-side usage (light metrics): the calculator is fed incrementally
    per-batch and emits a small summary at epoch end. Test-side usage (full
    metrics): all per-image detections are collected, then ``run`` produces
    the full :class:`MetricReport` (mAP, AR, per-class AP, performance,
    statistical tests).
    """

    def update(self, batch: list[PerImageDetection]) -> None:
        """Accumulate one batch of per-image detections."""
        ...

    def run(self) -> MetricReport:
        """Compute and return the full metric report."""
        ...
