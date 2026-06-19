"""Smoke tests for the real pipeline implementation (train/test/crossval/compare).

Uses a tiny in-memory YOLOv5 model + a 4-image synthetic dataset to verify
the pipelines run end-to-end and emit the expected artifacts.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cracks_yolo.dataset.torchadapter import DetectionDataset
from cracks_yolo.dataset.torchadapter import detection_collate
from cracks_yolo.dataset.types import RawDetection
from cracks_yolo.pipeline import TestConfig
from cracks_yolo.pipeline import TrainConfig
from cracks_yolo.pipeline import TrainPipelineImpl
from cracks_yolo.pipeline._utils import detections_to_per_image
from cracks_yolo.pipeline._utils import is_anchor_free_model
from cracks_yolo.pipeline._utils import is_v7_model
from cracks_yolo.pipeline._utils import pick_device
from cracks_yolo.pipeline._utils import set_seed
from cracks_yolo.pipeline._utils import targets_to_yolo
from cracks_yolo.pipeline.compare import compare_models_cross_val
from cracks_yolo.pipeline.crossval import run_cross_validation
from cracks_yolo.pipeline.test import TestPipelineImpl
from cracks_yolo.zoo import ZOO


def _make_records(tmp_path: Path, n: int = 4, size: int = 64) -> list[RawDetection]:
    """Write n synthetic PNG images + return RawDetection list."""
    from PIL import Image

    records: list[RawDetection] = []
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img_path = img_dir / f"img_{i:03d}.png"
        Image.new("RGB", (size, size), color=(i * 50, 100, 200)).save(img_path)
        records.append(
            RawDetection(
                image_path=img_path,
                image_id=i,
                width=size,
                height=size,
                boxes_norm=[(0.1, 0.1, 0.4, 0.4)],
                labels=[0],
            )
        )
    return records


def _build_loader(
    records: list[RawDetection], input_size: int, batch_size: int = 2
) -> DataLoader[tuple[torch.Tensor, list[dict[str, torch.Tensor]]]]:
    """Build a tiny DataLoader from RawDetection records."""
    from cracks_yolo.dataset.transforms import build_transforms

    ds = DetectionDataset(
        records,
        transform=build_transforms(input_size, train=False, augment=False),
    )
    return DataLoader(
        ds,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )


def test_utils_set_seed_deterministic() -> None:
    """set_seed makes two consecutive torch.rand calls identical."""
    set_seed(0)
    a = torch.rand(4)
    set_seed(0)
    b = torch.rand(4)
    assert torch.equal(a, b)


def test_utils_pick_device_cpu() -> None:
    """pick_device returns cpu when cuda is unavailable or preferred='cpu'."""
    dev = pick_device("cpu")
    assert dev.type == "cpu"


def test_utils_targets_to_yolo_shape() -> None:
    """targets_to_yolo produces (N, 6) normalized YOLO target tensor."""
    targets: list[dict[str, torch.Tensor]] = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
            "image_id": torch.tensor(0),
        },
        {
            "boxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
            "image_id": torch.tensor(1),
        },
    ]
    out = targets_to_yolo(targets, image_size=64)
    assert out.shape == (2, 6)
    # All coords in [0, 1] since normalized.
    assert (out[:, 2:] >= 0).all() and (out[:, 2:] <= 1).all()


def test_utils_detections_to_per_image_empty() -> None:
    """A 2D decoded tensor returns an empty list (defensive)."""
    decoded = torch.zeros((4, 5))
    targets: list[dict[str, torch.Tensor]] = [
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.long),
            "image_id": torch.tensor(0),
        }
    ]
    out = detections_to_per_image(decoded, targets, image_size=64, conf_thr=0.5)
    assert out == []


def test_utils_model_class_helpers() -> None:
    """is_anchor_free_model / is_v7_model identify YOLO variants by class name."""

    class YOLOv7w:
        pass

    class YOLOv8s:
        pass

    class YOLOv5s:
        pass

    assert is_v7_model(YOLOv7w())
    assert not is_v7_model(YOLOv5s())
    assert is_anchor_free_model(YOLOv8s())
    assert not is_anchor_free_model(YOLOv5s())


def test_train_pipeline_smoke(tmp_path: Path) -> None:
    """TrainPipelineImpl trains YOLOv5s for 1 epoch on 4 synthetic images."""
    records = _make_records(tmp_path, n=4, size=64)
    cls = ZOO["yolov5s"]
    model = cls(num_classes=1, input_size=64)
    train_loader = _build_loader(records, input_size=64, batch_size=2)
    val_loader = _build_loader(records, input_size=64, batch_size=2)

    cfg = TrainConfig(
        output_dir=tmp_path / "train_run",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        amp=False,
        log_every_n_steps=1,
        val_interval=1,
        device="cpu",
    )
    report = TrainPipelineImpl().run(model, train_loader, val_loader, cfg)
    assert report.total_epochs == 1
    assert report.total_steps >= 1
    assert (cfg.output_dir / "best.pt").exists()
    assert (cfg.output_dir / "metrics.csv").exists()
    assert (cfg.output_dir / "config.yaml").exists()


def test_test_pipeline_smoke(tmp_path: Path) -> None:
    """TestPipelineImpl emits metrics.csv + per_image JSON + curves dir."""
    records = _make_records(tmp_path, n=4, size=64)
    cls = ZOO["yolov5s"]
    model = cls(num_classes=1, input_size=64)
    test_loader = _build_loader(records, input_size=64, batch_size=2)
    cfg = TestConfig(
        output_dir=tmp_path / "test_run",
        batch_size=2,
        device="cpu",
        conf_thr=0.001,
    )
    report = TestPipelineImpl().run(model, test_loader, cfg)
    assert (cfg.output_dir / "metrics.csv").exists()
    assert (cfg.output_dir / "per_image").is_dir()
    assert (cfg.output_dir / "curves").is_dir()
    assert report.elapsed_sec >= 0.0


def test_crossval_smoke(tmp_path: Path) -> None:
    """run_cross_validation runs 2 folds and emits cv_summary.csv + cv_report.json."""

    from cracks_yolo.zoo.base import DetectorModel

    records = _make_records(tmp_path, n=6, size=64)
    cls = ZOO["yolov5s"]

    def factory() -> DetectorModel:
        return cls(num_classes=1, input_size=64)

    train_cfg = TrainConfig(
        output_dir=tmp_path / "cv_run",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        amp=False,
        log_every_n_steps=1,
        val_interval=1,
        device="cpu",
    )
    report = run_cross_validation(
        model_factory=factory,
        records=records,
        input_size=64,
        train_cfg=train_cfg,
        n_folds=2,
        seed=0,
        batch_size=2,
        num_workers=0,
    )
    assert report.n_folds == 2
    assert len(report.per_fold_test) == 2
    assert (train_cfg.output_dir / "cv_summary.csv").exists()
    assert (train_cfg.output_dir / "cv_report.json").exists()
    agg = report.aggregated()
    assert "map50" in agg
    assert "mean" in agg["map50"]


def test_compare_smoke(tmp_path: Path) -> None:
    """compare_models_cross_val runs CV for 2 models and emits comparison.csv."""

    from cracks_yolo.zoo.base import DetectorModel

    records = _make_records(tmp_path, n=6, size=64)
    cls_a = ZOO["yolov5s"]
    cls_b = ZOO["yolov5s_sac"]

    def factory_a() -> DetectorModel:
        return cls_a(num_classes=1, input_size=64)

    def factory_b() -> DetectorModel:
        return cls_b(num_classes=1, input_size=64)

    train_cfg = TrainConfig(
        output_dir=tmp_path / "compare_run",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        amp=False,
        log_every_n_steps=1,
        val_interval=1,
        device="cpu",
    )
    report = compare_models_cross_val(
        model_factories={"yolov5s": factory_a, "yolov5s_sac": factory_b},
        records=records,
        input_size=64,
        train_cfg=train_cfg,
        n_folds=2,
        seed=0,
        metric="map50",
        batch_size=2,
        num_workers=0,
    )
    assert len(report.model_names) == 2
    tests = report.pairwise_tests()
    assert len(tests) == 1  # one pair
    assert (train_cfg.output_dir / "comparison.csv").exists()
    assert (train_cfg.output_dir / "paired_t_test.csv").exists()
    assert (train_cfg.output_dir / "comparison_report.json").exists()
