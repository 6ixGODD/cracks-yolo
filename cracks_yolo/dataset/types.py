from __future__ import annotations

import pathlib
import typing as t

from cracks_yolo.exceptions import ValidationError


class BBox(t.NamedTuple):
    x_min: float
    """Minimum X coordinate of the bounding box."""

    y_min: float
    """Minimum Y coordinate of the bounding box."""

    x_max: float
    """Maximum X coordinate of the bounding box."""

    y_max: float
    """Maximum Y coordinate of the bounding box."""

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Return bounding box in XYXY format (x_min, y_min, x_max, y_max)."""
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """Convert to XYWH format (x, y, width, height)."""
        return self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min

    @property
    def cxcywh(self) -> tuple[float, float, float, float]:
        """Convert to center format (center_x, center_y, width, height)."""
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        return self.x_min + width / 2, self.y_min + height / 2, width, height

    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BBox:
        """Create BBox from XYWH format."""
        return cls(x, y, x + w, y + h)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BBox:
        """Create BBox from center format."""
        return cls(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class Annotation(t.NamedTuple):
    bbox: BBox
    """Bounding box of the annotation."""

    category_id: int
    """Category ID of the annotation."""

    category_name: str
    """Category name of the annotation."""

    image_id: str
    """ID of the image the annotation belongs to."""

    annotation_id: str | None = None
    """Unique ID of the annotation."""

    area: float | None = None
    """Area of the annotation, if available."""

    iscrowd: int = 0
    """Crowd annotation flag (0 or 1)."""

    def get_area(self) -> float:
        """Get annotation area, calculate from bbox if not provided."""
        return self.area if self.area is not None else self.bbox.area


class ImageInfo(t.NamedTuple):
    image_id: str
    """Unique identifier for the image."""

    file_name: str
    """Filename of the image."""

    width: int
    """Width of the image."""

    height: int
    """Height of the image."""

    path: pathlib.Path | None = None
    """Optional filesystem path to the image."""


class DatasetStatistics(t.TypedDict):
    num_images: int
    """Number of images in the dataset."""

    num_annotations: int
    """Number of annotations in the dataset."""

    num_categories: int
    """Number of unique categories in the dataset."""

    category_distribution: dict[str, int]
    """Distribution of annotations per category."""

    avg_annotations_per_image: float
    """Average number of annotations per image."""

    std_annotations_per_image: float
    """Standard deviation of annotations per image."""

    min_annotations_per_image: int
    """Minimum number of annotations in any image."""

    max_annotations_per_image: int
    """Maximum number of annotations in any image."""

    avg_bbox_area: float
    """Average bounding box area."""

    median_bbox_area: float
    """Median bounding box area."""


class SplitRatio(t.NamedTuple):
    train: float = 0.8
    """Training set ratio."""

    val: float = 0.1
    """Validation set ratio."""

    test: float = 0.1
    """Test set ratio."""

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if not abs(total - 1.0) < 1e-6:
            raise ValidationError(f"Split ratios must sum to 1.0, got {total}")
