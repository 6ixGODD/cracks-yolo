"""Tests for cracks_yolo.dataset (YOLO source + transforms + torch adapter)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest
import torch

from cracks_yolo.dataset import DetectionDataset
from cracks_yolo.dataset import YOLOSource
from cracks_yolo.dataset import build_dataloader
from cracks_yolo.dataset import build_transforms
from cracks_yolo.dataset.convert import coco_to_yolo
from cracks_yolo.dataset.convert import yolo_to_coco


@pytest.fixture
def tiny_yolo_dataset(tmp_path: Path) -> Path:
    """Create a 3-image YOLO-format dataset under tmp_path."""
    root = tmp_path / "tiny"
    for split in ("train", "valid", "test"):
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        for i in range(3):
            img_path = images_dir / f"img_{i}.jpg"
            Image.new("RGB", (640, 480), color=(i * 30, 0, 0)).save(img_path)
            label_path = labels_dir / f"img_{i}.txt"
            label_path.write_text(f"0 0.{i + 1} 0.5 0.2 0.3\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['cracks']\n",
        encoding="utf-8",
    )
    return root


def test_yolo_source_reads_data_yaml(tiny_yolo_dataset: Path) -> None:
    """YOLOSource parses data.yaml for class names + count."""
    src = YOLOSource(tiny_yolo_dataset)
    assert src.num_classes == 1
    assert src.class_names == ["cracks"]


def test_yolo_source_lists_splits(tiny_yolo_dataset: Path) -> None:
    """list_splits returns train/valid/test when present."""
    src = YOLOSource(tiny_yolo_dataset)
    assert set(src.list_splits()) == {"train", "valid", "test"}


def test_yolo_source_load_split(tiny_yolo_dataset: Path) -> None:
    """load_split returns one RawDetection per image, with normalized boxes."""
    src = YOLOSource(tiny_yolo_dataset)
    records = src.load_split("train")
    assert len(records) == 3
    for rec in records:
        assert rec.width == 640
        assert rec.height == 480
        assert len(rec.boxes_norm) == 1
        assert rec.labels == [0]
        x1, y1, x2, y2 = rec.boxes_norm[0]
        assert 0.0 <= x1 < x2 <= 1.0
        assert 0.0 <= y1 < y2 <= 1.0


def test_detection_dataset_from_yolo(tiny_yolo_dataset: Path) -> None:
    """DetectionDataset.from_yolo returns a torch Dataset of expected size."""
    ds = DetectionDataset.from_yolo(tiny_yolo_dataset, split="train", input_size=320, train=False)
    assert len(ds) == 3
    sample = ds[0]
    assert sample.image.shape == (3, 320, 320)
    assert sample.image.dtype == torch.float32
    assert sample.image.min() >= 0.0 and sample.image.max() <= 1.0
    assert sample.boxes.shape[1] == 4
    assert sample.labels.dtype == torch.long


def test_build_dataloader(tiny_yolo_dataset: Path) -> None:
    """build_dataloader returns batches of (images, list-of-targets)."""
    ds = DetectionDataset.from_yolo(tiny_yolo_dataset, split="train", input_size=320, train=False)
    loader = build_dataloader(ds, batch_size=2, shuffle=False)
    images, targets = next(iter(loader))
    assert images.shape == (2, 3, 320, 320)
    assert isinstance(targets, list)
    assert len(targets) == 2
    assert set(targets[0].keys()) == {"boxes", "labels", "image_id"}


def test_build_transforms_train_applies_flip() -> None:
    """Train transform is a DetectionTransform with train=True."""
    t = build_transforms(input_size=640, train=True, augment=True)
    assert t.train is True
    assert t.input_size == 640


def test_yolo_to_coco_round_trip(tiny_yolo_dataset: Path, tmp_path: Path) -> None:
    """yolo_to_coco writes one JSON per split with images+annotations."""
    out_dir = tmp_path / "coco"
    result = yolo_to_coco(tiny_yolo_dataset, out_dir)
    assert set(result.keys()) == {"train", "valid", "test"}
    import json

    with result["train"].open(encoding="utf-8") as f:
        coco = json.load(f)
    assert len(coco["images"]) == 3
    assert len(coco["annotations"]) == 3
    assert coco["categories"] == [{"id": 1, "name": "cracks", "supercategory": "cracks"}]


def test_coco_to_yolo_round_trip(tiny_yolo_dataset: Path, tmp_path: Path) -> None:
    """coco_to_yolo regenerates YOLO label files from a COCO JSON."""
    out_dir = tmp_path / "coco"
    yolo_to_coco(tiny_yolo_dataset, out_dir)

    labels_out = tmp_path / "labels_back"
    n = coco_to_yolo(
        out_dir / "instances_train.json",
        tiny_yolo_dataset / "train" / "images",
        labels_out,
    )
    assert n == 3
    # Each label file should have at least one line.
    label_files = list(labels_out.glob("*.txt"))
    assert len(label_files) == 3
    for lf in label_files:
        assert lf.read_text(encoding="utf-8").strip() != ""
