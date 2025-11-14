"""Dataset visualization utilities."""

from __future__ import annotations

import collections
import contextlib
import pathlib
import random
import typing as t

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image

if t.TYPE_CHECKING:
    from cracks_yolo.dataset import Dataset


class DatasetVisualizer:
    """Comprehensive dataset visualization toolkit."""

    def __init__(
        self,
        datasets: dict[str, Dataset],
        output_dir: pathlib.Path,
        seed: int | None = None,
    ):
        """Initialize visualizer.

        Args:
            datasets: Dict mapping split names to Dataset objects
            output_dir: Output directory for visualizations
            seed: Random seed for reproducibility
        """
        self.datasets = datasets
        self.output_dir = output_dir
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_all(
        self,
        num_samples: int = 5,
        show_distribution: bool = True,
        show_source_dist: bool = False,
        show_heatmap: bool = False,
    ) -> dict[str, pathlib.Path]:
        """Generate all visualizations.

        Args:
            num_samples: Number of samples per split
            show_distribution: Show category distribution
            show_source_dist: Show source distribution (from filename prefixes)
            show_heatmap: Show annotation heatmap

        Returns:
            Dict mapping visualization names to file paths
        """
        outputs = {}

        # 1. Category distribution
        if show_distribution:
            dist_path = self.visualize_category_distribution()
            outputs["category_distribution"] = dist_path

        # 2. Split comparison
        if len(self.datasets) > 1:
            comparison_path = self.visualize_split_comparison()
            outputs["split_comparison"] = comparison_path

            # Save statistics table
            stats_path = self.save_statistics_table()
            outputs["statistics_table"] = stats_path

        # 3. Source distribution (from filename prefixes)
        if show_source_dist:
            source_path = self.visualize_source_distribution()
            if source_path:
                outputs["source_distribution"] = source_path

        # 4. Sample images
        sample_paths = self.visualize_samples(num_samples)
        outputs.update(sample_paths)

        # 5. Annotation heatmap
        if show_heatmap:
            heatmap_path = self.visualize_annotation_heatmap()
            if heatmap_path:
                outputs["annotation_heatmap"] = heatmap_path

        return outputs

    def _extract_source_from_filename(self, filename: str) -> str | None:
        """Extract source prefix from filename (before first underscore).

        Args:
            filename: Image filename

        Returns:
            Source prefix or None if no underscore found
        """
        # Remove extension
        stem = pathlib.Path(filename).stem

        # Split by underscore
        if "_" in stem:
            return stem.split("_")[0]

        return None

    def visualize_category_distribution(self) -> pathlib.Path:
        """Visualize category distribution across splits.

        Returns:
            Path to saved figure
        """
        _, axes = plt.subplots(1, len(self.datasets), figsize=(6 * len(self.datasets), 5))

        if len(self.datasets) == 1:
            axes = [axes]

        for ax, (split_name, dataset) in zip(axes, self.datasets.items(), strict=True):
            stats = dataset.get_statistics(by_source=False)
            cat_dist = stats["category_distribution"]

            categories = list(cat_dist.keys())
            counts = list(cat_dist.values())

            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))  # type: ignore
            bars = ax.bar(categories, counts, color=colors, edgecolor="black", alpha=0.8)  # type: ignore

            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()  # type: ignore
                ax.text(  # type: ignore
                    bar.get_x() + bar.get_width() / 2,  # type: ignore
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_title(f"{split_name.capitalize()} Split", fontsize=14, fontweight="bold")  # type: ignore
            ax.set_xlabel("Category", fontsize=12)  # type: ignore
            ax.set_ylabel("Count", fontsize=12)  # type: ignore
            ax.tick_params(axis="x", rotation=45)  # type: ignore
            ax.grid(axis="y", alpha=0.3)  # type: ignore

        plt.tight_layout()
        output_path = self.output_dir / "category_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def visualize_split_comparison(self) -> pathlib.Path:
        """Visualize comparison between splits.

        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Images per split (pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        split_names = list(self.datasets.keys())
        image_counts = [len(ds) for ds in self.datasets.values()]

        colors = plt.cm.Pastel1(np.linspace(0, 1, len(split_names)))  # type: ignore
        ax1.pie(  # type: ignore
            image_counts,
            labels=[s.capitalize() for s in split_names],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"},
        )
        ax1.set_title("Images per Split", fontsize=14, fontweight="bold")  # type: ignore

        # 2. Annotations per split (bar chart)
        ax2 = fig.add_subplot(gs[0, 1])
        ann_counts = [ds.num_annotations() for ds in self.datasets.values()]

        bars = ax2.bar(  # type: ignore
            [s.capitalize() for s in split_names],
            ann_counts,
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )

        for bar in bars:
            height = bar.get_height()  # type: ignore
            ax2.text(  # type: ignore
                bar.get_x() + bar.get_width() / 2,  # type: ignore
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax2.set_title("Annotations per Split", fontsize=14, fontweight="bold")  # type: ignore
        ax2.set_ylabel("Count", fontsize=12)  # type: ignore
        ax2.grid(axis="y", alpha=0.3)  # type: ignore

        # 3. Average annotations per image
        ax3 = fig.add_subplot(gs[1, 0])
        avg_anns = [
            ds.get_statistics(by_source=False)["avg_annotations_per_image"]
            for ds in self.datasets.values()
        ]

        bars = ax3.bar(  # type: ignore
            [s.capitalize() for s in split_names],
            avg_anns,
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )

        for bar in bars:
            height = bar.get_height()  # type: ignore
            ax3.text(  # type: ignore
                bar.get_x() + bar.get_width() / 2,  # type: ignore
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax3.set_title("Avg Annotations per Image", fontsize=14, fontweight="bold")  # type: ignore
        ax3.set_ylabel("Average", fontsize=12)  # type: ignore
        ax3.grid(axis="y", alpha=0.3)  # type: ignore

        # 4. Category distribution comparison (grouped bar)
        ax4 = fig.add_subplot(gs[1, 1])

        # Collect all categories
        all_categories: set[str] = set()
        for dataset in self.datasets.values():
            stats = dataset.get_statistics(by_source=False)
            all_categories.update(stats["category_distribution"].keys())

        all_categories_list = sorted(all_categories)
        x = np.arange(len(all_categories_list))
        width = 0.8 / len(self.datasets)

        for i, (split_name, dataset) in enumerate(self.datasets.items()):
            stats = dataset.get_statistics(by_source=False)
            cat_dist = stats["category_distribution"]
            counts = [cat_dist.get(cat, 0) for cat in all_categories_list]  # type: ignore

            ax4.bar(  # type: ignore
                x + i * width,
                counts,
                width,
                label=split_name.capitalize(),
                color=colors[i],
                edgecolor="black",
                alpha=0.8,
            )

        ax4.set_xlabel("Category", fontsize=12)  # type: ignore
        ax4.set_ylabel("Count", fontsize=12)  # type: ignore
        ax4.set_title("Category Distribution Comparison", fontsize=14, fontweight="bold")  # type: ignore
        ax4.set_xticks(x + width * (len(self.datasets) - 1) / 2)  # type: ignore
        ax4.set_xticklabels(all_categories_list, rotation=45, ha="right")  # type: ignore
        ax4.legend()  # type: ignore
        ax4.grid(axis="y", alpha=0.3)  # type: ignore

        plt.tight_layout()
        output_path = self.output_dir / "split_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def visualize_source_distribution(self) -> pathlib.Path | None:
        """Visualize distribution by source extracted from filename prefixes.

        Returns:
            Path to saved figure or None if no sources detected
        """
        # Collect source information from filenames
        source_counts: collections.Counter[str] = collections.Counter()
        source_annotations: collections.Counter[str] = collections.Counter()

        for dataset in self.datasets.values():
            for img_id, img_info in dataset.images.items():
                source = self._extract_source_from_filename(img_info.file_name)
                if source:
                    source_counts[source] += 1
                    source_annotations[source] += len(dataset.get_annotations(img_id))

        if not source_counts:
            return None

        # Create visualization
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        sources = list(source_counts.keys())
        image_counts = [source_counts[s] for s in sources]
        ann_counts = [source_annotations[s] for s in sources]

        colors = plt.cm.Set2(np.linspace(0, 1, len(sources)))  # type: ignore

        # Images by source
        bars1 = ax1.bar(sources, image_counts, color=colors, edgecolor="black", alpha=0.8)  # type: ignore
        for bar in bars1:
            height = bar.get_height()  # type: ignore
            ax1.text(  # type: ignore
                bar.get_x() + bar.get_width() / 2,  # type: ignore
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax1.set_title("Images by Source (Filename Prefix)", fontsize=14, fontweight="bold")  # type: ignore
        ax1.set_xlabel("Source", fontsize=12)  # type: ignore
        ax1.set_ylabel("Count", fontsize=12)  # type: ignore
        ax1.tick_params(axis="x", rotation=45)  # type: ignore
        ax1.grid(axis="y", alpha=0.3)  # type: ignore

        # Annotations by source
        bars2 = ax2.bar(sources, ann_counts, color=colors, edgecolor="black", alpha=0.8)  # type: ignore
        for bar in bars2:
            height = bar.get_height()  # type: ignore
            ax2.text(  # type: ignore
                bar.get_x() + bar.get_width() / 2,  # type: ignore
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax2.set_title("Annotations by Source (Filename Prefix)", fontsize=14, fontweight="bold")  # type: ignore
        ax2.set_xlabel("Source", fontsize=12)  # type: ignore
        ax2.set_ylabel("Count", fontsize=12)  # type: ignore
        ax2.tick_params(axis="x", rotation=45)  # type: ignore
        ax2.grid(axis="y", alpha=0.3)  # type: ignore

        plt.tight_layout()
        output_path = self.output_dir / "source_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def visualize_samples(self, num_samples: int = 5) -> dict[str, pathlib.Path]:
        """Visualize sample images from each split, combined into grids.

        Args:
            num_samples: Number of samples per split

        Returns:
            Dict mapping split names to saved figure paths
        """
        outputs = {}

        for split_name, dataset in self.datasets.items():
            # Select samples
            image_ids = list(dataset.images.keys())
            num_to_sample = min(num_samples, len(image_ids))

            if num_to_sample == 0:
                continue

            sampled_ids = random.sample(image_ids, num_to_sample)

            # Create grid
            cols = min(3, num_to_sample)
            rows = (num_to_sample + cols - 1) // cols

            _, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))

            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)

            for idx, img_id in enumerate(sampled_ids):
                row = idx // cols
                col = idx % cols
                ax = axes[row, col]

                img_info = dataset.get_image(img_id)
                if not img_info or not img_info.path or not img_info.path.exists():
                    ax.axis("off")  # type: ignore
                    continue

                # Load and display image
                img = Image.open(img_info.path)
                ax.imshow(img)  # type: ignore

                # Draw annotations
                anns = dataset.get_annotations(img_id)
                colors = plt.cm.rainbow(np.linspace(0, 1, dataset.num_categories()))  # type: ignore
                category_colors = {
                    cat_id: colors[i] for i, cat_id in enumerate(dataset.categories.keys())
                }

                for ann in anns:
                    bbox = ann.bbox
                    color = category_colors[ann.category_id]

                    rect = plt.Rectangle(
                        (bbox.x_min, bbox.y_min),
                        bbox.x_max - bbox.x_min,
                        bbox.y_max - bbox.y_min,
                        linewidth=2,
                        edgecolor=color,
                        facecolor="none",
                    )
                    ax.add_patch(rect)  # type: ignore

                    # Label
                    ax.text(  # type: ignore
                        bbox.x_min,
                        bbox.y_min - 5,
                        ann.category_name,
                        color="white",
                        fontsize=10,
                        bbox={"facecolor": color, "alpha": 0.7, "edgecolor": "none", "pad": 2},
                    )

                # Extract source from filename
                source = self._extract_source_from_filename(img_info.file_name)
                title = f"{img_info.file_name}\n{len(anns)} annotations"
                if source:
                    title += f"\nSource: {source}"

                ax.set_title(title, fontsize=10)  # type: ignore
                ax.axis("off")  # type: ignore

            # Hide empty subplots
            for idx in range(num_to_sample, rows * cols):
                row = idx // cols
                col = idx % cols
                axes[row, col].axis("off")  # type: ignore

            plt.tight_layout()
            output_path = self.output_dir / f"samples_{split_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            outputs[f"samples_{split_name}"] = output_path

        return outputs

    def visualize_annotation_heatmap(self) -> pathlib.Path | None:
        """Generate annotation density heatmap with blue-yellow colormap.

        Returns:
            Path to saved figure or None if no images
        """
        # Collect all bounding boxes
        all_boxes = []

        for dataset in self.datasets.values():
            for img_id, img_info in dataset.images.items():
                anns = dataset.get_annotations(img_id)
                for ann in anns:
                    # Normalize coordinates
                    bbox = ann.bbox
                    norm_box = (
                        bbox.x_min / img_info.width,
                        bbox.y_min / img_info.height,
                        bbox.x_max / img_info.width,
                        bbox.y_max / img_info.height,
                    )
                    all_boxes.append(norm_box)

        if not all_boxes:
            return None

        # Create heatmap
        resolution = 100
        heatmap = np.zeros((resolution, resolution))

        for x_min, y_min, x_max, y_max in all_boxes:
            # Convert to grid coordinates
            x_min_idx = int(x_min * resolution)
            y_min_idx = int(y_min * resolution)
            x_max_idx = int(x_max * resolution)
            y_max_idx = int(y_max * resolution)

            # Clip to bounds
            x_min_idx = max(0, min(x_min_idx, resolution - 1))
            y_min_idx = max(0, min(y_min_idx, resolution - 1))
            x_max_idx = max(0, min(x_max_idx, resolution - 1))
            y_max_idx = max(0, min(y_max_idx, resolution - 1))

            # Fill region
            heatmap[y_min_idx : y_max_idx + 1, x_min_idx : x_max_idx + 1] += 1

        # Plot with blue-yellow colormap
        _, ax = plt.subplots(figsize=(12, 10))

        # Use 'YlGnBu_r' (yellow-green-blue reversed) or 'RdYlBu_r' for blue-yellow
        im = ax.imshow(heatmap, cmap="YlOrBr", interpolation="bilinear", origin="upper")  # type: ignore
        ax.set_title(  # type: ignore
            "Annotation Density Heatmap\n(Normalized Image Coordinates)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Normalized X", fontsize=12)  # type: ignore
        ax.set_ylabel("Normalized Y", fontsize=12)  # type: ignore

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Annotation Density", rotation=270, labelpad=20, fontsize=12)

        plt.tight_layout()
        output_path = self.output_dir / "annotation_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def save_statistics_table(self) -> pathlib.Path:
        """Save detailed statistics as CSV and Excel files.

        Returns:
            Path to saved Excel file
        """
        # Collect statistics
        data = []

        for split_name, dataset in self.datasets.items():
            stats = dataset.get_statistics(by_source=False)

            row = {
                "Split": split_name.capitalize(),
                "Images": len(dataset),
                "Annotations": dataset.num_annotations(),
                "Categories": dataset.num_categories(),
                "Avg Annotations/Image": stats["avg_annotations_per_image"],
                "Std Annotations/Image": stats["std_annotations_per_image"],
                "Min Annotations/Image": stats["min_annotations_per_image"],
                "Max Annotations/Image": stats["max_annotations_per_image"],
                "Avg BBox Area": stats["avg_bbox_area"],
                "Median BBox Area": stats["median_bbox_area"],
            }

            # Add category counts
            for cat_name, count in stats["category_distribution"].items():
                row[f"Category_{cat_name}"] = count

            data.append(row)

        df = pd.DataFrame(data)

        # Save as CSV
        csv_path = self.output_dir / "statistics.csv"
        df.to_csv(csv_path, index=False)

        # Save as Excel with formatting
        excel_path = self.output_dir / "statistics.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Statistics", index=False)

            # Auto-adjust column widths
            worksheet = writer.sheets["Statistics"]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    with contextlib.suppress(Exception):
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        return excel_path
