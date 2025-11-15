from __future__ import annotations

import csv
import datetime
import json
import pathlib
import typing as t

from cracks_yolo.dataset.types import Annotation


class WorkspaceData(t.TypedDict):
    """Workspace data structure."""

    version: str
    created_at: str
    modified_at: str
    dataset_path: str
    dataset_format: str
    loaded_splits: list[t.Literal["train", "val", "test"]]
    current_split: str
    current_index: int
    original_annotations: dict[str, list[dict[str, t.Any]]]  # 原始的 annotations
    modified_annotations: dict[str, list[dict[str, t.Any]]]
    audit_mode: bool
    audit_status: dict[str, str]  # image_id -> "approved" | "rejected" | "pending"


class Workspace:
    """Manage annotation workspace for save/load sessions."""

    VERSION = "1.0"

    def __init__(self):
        """Initialize workspace."""
        self.data: WorkspaceData = {
            "version": self.VERSION,
            "created_at": datetime.datetime.now().isoformat(),
            "modified_at": datetime.datetime.now().isoformat(),
            "dataset_path": "",
            "dataset_format": "",
            "loaded_splits": [],
            "current_split": "",
            "current_index": 0,
            "original_annotations": {},
            "modified_annotations": {},
            "audit_mode": False,
            "audit_status": {},
        }

    def save(self, filepath: str | pathlib.Path) -> None:
        """Save workspace to file.

        Args:
            filepath: Path to save workspace file
        """
        self.data["modified_at"] = datetime.datetime.now().isoformat()

        filepath = pathlib.Path(filepath)
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str | pathlib.Path) -> Workspace:
        """Load workspace from file.

        Args:
            filepath: Path to workspace file

        Returns:
            Loaded workspace instance
        """
        workspace = cls()

        filepath = pathlib.Path(filepath)
        with filepath.open(encoding="utf-8") as f:
            loaded_data = json.load(f)

        # Validate version
        if loaded_data.get("version") != cls.VERSION:
            raise ValueError(f"Incompatible workspace version: {loaded_data.get('version')}")

        workspace.data = loaded_data
        return workspace

    def set_dataset_info(
        self,
        path: str,
        format_type: str,
        splits: list[t.Literal["train", "val", "test"]],
    ) -> None:
        """Set dataset information."""
        self.data["dataset_path"] = path
        self.data["dataset_format"] = format_type
        self.data["loaded_splits"] = splits

    def set_current_position(self, split: str, index: int) -> None:
        """Set current viewing position."""
        self.data["current_split"] = split
        self.data["current_index"] = index

    def set_original_annotations(self, image_id: str, annotations: list[Annotation]) -> None:
        """Store original annotations for an image (when first loaded)."""
        # Only store if not already stored (preserve original)
        if image_id in self.data["original_annotations"]:
            return

        ann_dicts = []
        for ann in annotations:
            ann_dict = {
                "bbox": {
                    "x_min": ann.bbox.x_min,
                    "y_min": ann.bbox.y_min,
                    "x_max": ann.bbox.x_max,
                    "y_max": ann.bbox.y_max,
                },
                "category_id": ann.category_id,
                "category_name": ann.category_name,
                "image_id": ann.image_id,
                "annotation_id": ann.annotation_id,
                "area": ann.area,
                "iscrowd": ann.iscrowd,
            }
            ann_dicts.append(ann_dict)

        self.data["original_annotations"][image_id] = ann_dicts

    def set_modified_annotations(
        self,
        image_id: str,
        annotations: list[Annotation],
    ) -> None:
        """Store modified annotations for an image."""
        ann_dicts = []
        for ann in annotations:
            ann_dict = {
                "bbox": {
                    "x_min": ann.bbox.x_min,
                    "y_min": ann.bbox.y_min,
                    "x_max": ann.bbox.x_max,
                    "y_max": ann.bbox.y_max,
                },
                "category_id": ann.category_id,
                "category_name": ann.category_name,
                "image_id": ann.image_id,
                "annotation_id": ann.annotation_id,
                "area": ann.area,
                "iscrowd": ann.iscrowd,
            }
            ann_dicts.append(ann_dict)

        self.data["modified_annotations"][image_id] = ann_dicts

    def get_modified_annotations(self, image_id: str) -> list[Annotation] | None:
        """Get modified annotations for an image."""
        from cracks_yolo.dataset.types import Annotation
        from cracks_yolo.dataset.types import BBox

        if image_id not in self.data["modified_annotations"]:
            return None

        ann_dicts = self.data["modified_annotations"][image_id]
        annotations = []

        for ann_dict in ann_dicts:
            bbox_dict = ann_dict["bbox"]
            bbox = BBox(
                x_min=bbox_dict["x_min"],
                y_min=bbox_dict["y_min"],
                x_max=bbox_dict["x_max"],
                y_max=bbox_dict["y_max"],
            )

            ann = Annotation(
                bbox=bbox,
                category_id=ann_dict["category_id"],
                category_name=ann_dict["category_name"],
                image_id=ann_dict["image_id"],
                annotation_id=ann_dict["annotation_id"],
                area=ann_dict["area"],
                iscrowd=ann_dict["iscrowd"],
            )
            annotations.append(ann)

        return annotations

    def set_audit_mode(self, enabled: bool) -> None:
        """Enable/disable audit mode."""
        self.data["audit_mode"] = enabled

    def set_audit_status(self, image_id: str, status: str) -> None:
        """Set audit status for an image.

        Args:
            image_id: Image ID
            status: "approved", "rejected", or "pending"
        """
        self.data["audit_status"][image_id] = status

    def get_audit_status(self, image_id: str) -> str:
        """Get audit status for an image.

        Returns:
            "approved", "rejected", or "pending"
        """
        return self.data["audit_status"].get(image_id, "pending")

    def get_audit_statistics(self) -> dict[str, int]:
        """Get audit statistics.

        Returns:
            Dict with counts of approved, rejected, and pending images
        """
        stats = {"approved": 0, "rejected": 0, "pending": 0}

        for status in self.data["audit_status"].values():
            if status in stats:
                stats[status] += 1

        return stats

    def generate_audit_report_csv(
        self,
        output_path: str | pathlib.Path,
        image_info_map: dict[str, tuple[str, str]],  # image_id -> (filename, source)
    ) -> None:
        """Generate audit report as CSV file.

        Args:
            output_path: Path to save CSV report
            image_info_map: Mapping of image_id to (filename, source)
        """
        output_path = pathlib.Path(output_path)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "Image ID",
                "Filename",
                "Source",
                "Original Annotations",
                "Modified Annotations",
                "Has Changes",
                "Audit Status",
                "Timestamp",
            ])

            # Get all image IDs that were touched
            all_image_ids = set(self.data["original_annotations"].keys()) | set(
                self.data["audit_status"].keys()
            )

            for img_id in sorted(all_image_ids):
                filename, source = image_info_map.get(img_id, (img_id, "Unknown"))

                # Original annotations
                original_anns = self.data["original_annotations"].get(img_id, [])
                original_str = self._format_annotations_for_csv(original_anns)

                # Modified annotations
                modified_anns = self.data["modified_annotations"].get(img_id, original_anns)
                modified_str = self._format_annotations_for_csv(modified_anns)

                # Check if there are changes
                has_changes = "Yes" if modified_anns != original_anns else "No"

                # Audit status
                audit_status = self.data["audit_status"].get(img_id, "Pending")

                # Timestamp
                timestamp = self.data["modified_at"]

                writer.writerow([
                    img_id,
                    filename,
                    source,
                    original_str,
                    modified_str,
                    has_changes,
                    audit_status,
                    timestamp,
                ])

    def _format_annotations_for_csv(self, annotations: list[dict[str, t.Any]]) -> str:
        """Format annotations list as readable string for CSV.

        Args:
            annotations: List of annotation dictionaries

        Returns:
            Formatted string
        """
        if not annotations:
            return "[]"

        parts = []
        for ann in annotations:
            bbox = ann["bbox"]
            cat_name = ann["category_name"]
            bbox_str = (
                f"[{bbox['x_min']:.1f},{bbox['y_min']:.1f},{bbox['x_max']:.1f},{bbox['y_max']:.1f}]"
            )
            parts.append(f"{cat_name}:{bbox_str}")

        return "; ".join(parts)
