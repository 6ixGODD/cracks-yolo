"""Train/Test/CrossVal/Compare pipeline contracts + concrete implementations.

Protocols + pydantic configs in :mod:`cracks_yolo.pipeline.protocol`.
Concrete loops in :mod:`cracks_yolo.pipeline.train`, :mod:`...test`,
:mod:`...crossval`, :mod:`...compare`.
"""

from __future__ import annotations

from cracks_yolo.pipeline.compare import ComparisonReport
from cracks_yolo.pipeline.compare import compare_models_cross_val
from cracks_yolo.pipeline.crossval import CrossValReport
from cracks_yolo.pipeline.crossval import run_cross_validation
from cracks_yolo.pipeline.protocol import TestConfig
from cracks_yolo.pipeline.protocol import TestPipeline
from cracks_yolo.pipeline.protocol import TestReport
from cracks_yolo.pipeline.protocol import TrainConfig
from cracks_yolo.pipeline.protocol import TrainPipeline
from cracks_yolo.pipeline.protocol import TrainReport
from cracks_yolo.pipeline.test import TestPipelineImpl
from cracks_yolo.pipeline.train import TrainPipelineImpl

__all__ = [
    "ComparisonReport",
    "CrossValReport",
    "TestConfig",
    "TestPipeline",
    "TestPipelineImpl",
    "TestReport",
    "TrainConfig",
    "TrainPipeline",
    "TrainPipelineImpl",
    "TrainReport",
    "compare_models_cross_val",
    "run_cross_validation",
]
