"""Visualization helpers — curves, confusion matrix, heatmap, dataset plots."""

from __future__ import annotations

from cracks_yolo.viz.confusion import plot_confusion_matrix
from cracks_yolo.viz.curves import plot_loss_curve
from cracks_yolo.viz.curves import plot_metric_curve
from cracks_yolo.viz.curves import plot_pr_curve
from cracks_yolo.viz.curves import plot_roc_curve
from cracks_yolo.viz.dataset import plot_bbox_position_heatmap
from cracks_yolo.viz.dataset import plot_bbox_size_distribution
from cracks_yolo.viz.dataset import plot_class_distribution
from cracks_yolo.viz.dataset import plot_image_size_distribution
from cracks_yolo.viz.heatmap import GradCAMExtractor
from cracks_yolo.viz.heatmap import save_heatmap_overlay

__all__ = [
    "GradCAMExtractor",
    "plot_bbox_position_heatmap",
    "plot_bbox_size_distribution",
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_image_size_distribution",
    "plot_loss_curve",
    "plot_metric_curve",
    "plot_pr_curve",
    "plot_roc_curve",
    "save_heatmap_overlay",
]
