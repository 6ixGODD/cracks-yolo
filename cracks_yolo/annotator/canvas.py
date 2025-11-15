from __future__ import annotations

import tkinter as tk
import typing as t

from PIL import Image
from PIL import ImageTk

from cracks_yolo.dataset.types import Annotation
from cracks_yolo.dataset.types import BBox


class AnnotationCanvas(tk.Canvas):
    def __init__(
        self,
        parent: tk.Widget,
        on_annotation_changed: t.Callable[[], None] | None = None,
        **kwargs,
    ):
        super().__init__(parent, bg="#2b2b2b", highlightthickness=0, **kwargs)

        self.on_annotation_changed = on_annotation_changed

        self.image: Image.Image | None = None
        self.photo_image: ImageTk.PhotoImage | None = None
        self.zoom_level: float = 1.0
        self.annotations: list[Annotation] = []
        self.categories: dict[int, str] = {}
        self.current_category: int | None = None

        # Image cache for different zoom levels
        self.image_cache: dict[float, ImageTk.PhotoImage] = {}
        self.max_cache_size: int = 10

        # Offset for image position
        self.offset_x: int = 0
        self.offset_y: int = 0

        # Drawing state
        self.drawing: bool = False
        self.draw_start_x: int = 0
        self.draw_start_y: int = 0
        self.current_rect: int | None = None

        # Selection state
        self.selected_annotation: int | None = None

        # Drag state for resizing
        self.dragging: bool = False
        self.drag_start_x: int = 0
        self.drag_start_y: int = 0
        self.drag_handle: str | None = None

        # Panning state
        self.panning: bool = False
        self.pan_start_x: int = 0
        self.pan_start_y: int = 0
        self.pan_offset_x: int = 0
        self.pan_offset_y: int = 0

        # Crosshair
        self.show_crosshair: bool = True
        self.crosshair_x: int = 0
        self.crosshair_y: int = 0

        # Edit mode
        self.edit_mode: bool = False

        # Undo stack
        self.undo_stack: list[list[Annotation]] = []
        self.max_undo_steps: int = 50

        # Performance optimization
        self.pending_render: t.Any = None  # After handle for debouncing

        # Bind mouse events
        self.bind("<ButtonPress-1>", self.on_mouse_press)
        self.bind("<B1-Motion>", self.on_mouse_drag)
        self.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.bind("<Button-2>", self.on_middle_press)
        self.bind("<B2-Motion>", self.on_middle_drag)
        self.bind("<ButtonRelease-2>", self.on_middle_release)
        self.bind("<Button-3>", self.on_right_click)
        self.bind("<Motion>", self.on_mouse_move)
        self.bind("<MouseWheel>", self.on_mouse_wheel)
        self.bind("<Leave>", self.on_mouse_leave)

    def set_edit_mode(self, enabled: bool) -> None:
        """Enable or disable edit mode."""
        self.edit_mode = enabled
        if not self.panning:
            self.config(cursor="crosshair" if enabled else "arrow")

    def set_current_category(self, category_id: int | None) -> None:
        """Set current category for drawing."""
        self.current_category = category_id

    def display_image(
        self,
        image: Image.Image,
        annotations: list[Annotation],
        categories: dict[int, str],
    ) -> None:
        """Display image with annotations."""
        self.image = image
        self.annotations = annotations.copy()
        self.categories = categories
        self.zoom_level = 1.0
        self.selected_annotation = None
        self.drawing = False
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        # Clear cache when loading new image
        self.image_cache.clear()
        self.undo_stack.clear()

        self.render()

    def get_annotations(self) -> list[Annotation]:
        return self.annotations.copy()

    def delete_selected(self) -> bool:
        """Delete selected annotation."""
        if self.selected_annotation is not None and self.edit_mode:
            self.save_to_undo_stack()
            del self.annotations[self.selected_annotation]
            self.selected_annotation = None
            self.render()

            if self.on_annotation_changed:
                self.on_annotation_changed()

            return True
        return False

    def undo(self) -> bool:
        """Undo last annotation change."""
        if self.undo_stack and self.edit_mode:
            self.annotations = self.undo_stack.pop()
            self.selected_annotation = None
            self.render()

            if self.on_annotation_changed:
                self.on_annotation_changed()

            return True
        return False

    def save_to_undo_stack(self) -> None:
        """Save current state to undo stack."""
        state = [
            Annotation(
                bbox=ann.bbox,
                category_id=ann.category_id,
                category_name=ann.category_name,
                image_id=ann.image_id,
                annotation_id=ann.annotation_id,
                area=ann.area,
                iscrowd=ann.iscrowd,
            )
            for ann in self.annotations
        ]

        self.undo_stack.append(state)

        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)

    def get_cached_image(self, zoom_level: float) -> ImageTk.PhotoImage:
        """Get or create cached resized image.

        Args:
            zoom_level: Current zoom level

        Returns:
            Cached PhotoImage
        """
        # Round zoom level to reduce cache size
        cache_key = round(zoom_level, 2)

        if cache_key in self.image_cache:
            return self.image_cache[cache_key]

        # Create new resized image
        if self.image:
            width = int(self.image.width * zoom_level)
            height = int(self.image.height * zoom_level)

            # Use faster resampling for large images
            if width * height > 4000000:  # > 4MP
                resample = Image.Resampling.BILINEAR
            else:
                resample = Image.Resampling.LANCZOS

            resized = self.image.resize((width, height), resample)
            photo = ImageTk.PhotoImage(resized)

            # Cache it
            self.image_cache[cache_key] = photo

            # Limit cache size
            if len(self.image_cache) > self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]

            return photo

        # Fallback
        return ImageTk.PhotoImage(Image.new("RGB", (1, 1)))

    def render(self, fast: bool = False) -> None:
        """Render canvas.

        Args:
            fast: If True, only update position without redrawing annotations
        """
        if self.image is None:
            return

        self.delete("all")

        # Get cached image
        self.photo_image = self.get_cached_image(self.zoom_level)

        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()

        width = int(self.image.width * self.zoom_level)
        height = int(self.image.height * self.zoom_level)

        self.offset_x = (canvas_width - width) // 2 + self.pan_offset_x
        self.offset_y = (canvas_height - height) // 2 + self.pan_offset_y

        self.create_image(
            self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo_image, tags="image"
        )

        if not fast:
            self.draw_annotations()

        self.draw_crosshair()

    def render_debounced(self, delay: int = 16) -> None:
        """Render with debouncing for smooth performance.

        Args:
            delay: Delay in milliseconds (default 16ms ≈ 60fps)
        """
        if self.pending_render:
            self.after_cancel(self.pending_render)

        self.pending_render = self.after(delay, self.render, True)

    def draw_crosshair(self) -> None:
        """Draw crosshair lines at mouse position."""
        if not self.show_crosshair or not self.edit_mode:
            return

        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()

        # Vertical line
        self.create_line(
            self.crosshair_x,
            0,
            self.crosshair_x,
            canvas_height,
            fill="#00FF00",
            width=1,
            dash=(4, 4),
            tags="crosshair",
        )

        # Horizontal line
        self.create_line(
            0,
            self.crosshair_y,
            canvas_width,
            self.crosshair_y,
            fill="#00FF00",
            width=1,
            dash=(4, 4),
            tags="crosshair",
        )

    def draw_annotations(self) -> None:
        if not self.annotations or not self.image:
            return

        num_categories = len(self.categories)
        colors = self.generate_colors(num_categories)
        category_colors = {cat_id: colors[i] for i, cat_id in enumerate(self.categories.keys())}

        for idx, ann in enumerate(self.annotations):
            bbox = ann.bbox

            x1 = int(bbox.x_min * self.zoom_level) + self.offset_x
            y1 = int(bbox.y_min * self.zoom_level) + self.offset_y
            x2 = int(bbox.x_max * self.zoom_level) + self.offset_x
            y2 = int(bbox.y_max * self.zoom_level) + self.offset_y

            color = category_colors.get(ann.category_id, "#FF0000")
            width = 3 if idx == self.selected_annotation else 2

            self.create_rectangle(
                x1, y1, x2, y2, outline=color, width=width, tags=("bbox", f"bbox_{idx}")
            )

            # Draw resize handles if selected
            if idx == self.selected_annotation and self.edit_mode:
                handle_size = 8

                # Corner handles
                corners = [
                    (x1, y1, "nw"),
                    (x2, y1, "ne"),
                    (x1, y2, "sw"),
                    (x2, y2, "se"),
                ]
                for hx, hy, handle_type in corners:
                    self.create_rectangle(
                        hx - handle_size // 2,
                        hy - handle_size // 2,
                        hx + handle_size // 2,
                        hy + handle_size // 2,
                        fill=color,
                        outline="white",
                        width=1,
                        tags=("handle", "corner_handle", f"handle_{idx}_{handle_type}"),
                    )

                # Edge handles
                edge_size = 6
                edges = [
                    ((x1 + x2) // 2, y1, "n"),
                    ((x1 + x2) // 2, y2, "s"),
                    (x1, (y1 + y2) // 2, "w"),
                    (x2, (y1 + y2) // 2, "e"),
                ]
                for hx, hy, handle_type in edges:
                    self.create_rectangle(
                        hx - edge_size // 2,
                        hy - edge_size // 2,
                        hx + edge_size // 2,
                        hy + edge_size // 2,
                        fill="white",
                        outline=color,
                        width=2,
                        tags=("handle", "edge_handle", f"handle_{idx}_{handle_type}"),
                    )

            # Draw label
            label = ann.category_name
            label_width = len(label) * 8
            self.create_rectangle(
                x1,
                y1 - 20,
                x1 + label_width,
                y1,
                fill=color,
                outline=color,
                tags=("label", f"label_{idx}"),
            )
            self.create_text(
                x1 + 2,
                y1 - 10,
                text=label,
                anchor="w",
                fill="white",
                tags=("label", f"label_{idx}"),
            )

    def generate_colors(self, num_colors: int) -> list[str]:
        if num_colors == 0:
            return []

        import colorsys

        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            hex_color = f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
            colors.append(hex_color)

        return colors

    def screen_to_image_coords(self, x: int, y: int) -> tuple[float, float]:
        img_x = int((x - self.offset_x) / self.zoom_level)
        img_y = int((y - self.offset_y) / self.zoom_level)

        if self.image:
            img_x = max(0, min(img_x, self.image.width))
            img_y = max(0, min(img_y, self.image.height))

        return img_x, img_y

    def find_annotation_at(self, x: int, y: int) -> int | None:
        img_x, img_y = self.screen_to_image_coords(x, y)

        for idx, ann in enumerate(self.annotations):
            bbox = ann.bbox
            if bbox.x_min <= img_x <= bbox.x_max and bbox.y_min <= img_y <= bbox.y_max:
                return idx

        return None

    def find_handle_at(self, x: int, y: int) -> tuple[int, str] | None:
        if self.selected_annotation is None:
            return None

        overlapping = self.find_overlapping(x - 4, y - 4, x + 4, y + 4)
        for item in overlapping:
            tags = self.gettags(item)
            for tag in tags:
                if tag.startswith("handle_"):
                    parts = tag.split("_")
                    if len(parts) == 3:
                        idx = int(parts[1])
                        handle_type = parts[2]
                        return idx, handle_type

        return None

    def on_middle_press(self, event: tk.Event) -> None:  # type: ignore
        """Handle middle mouse button press for panning."""
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.config(cursor="fleur")

    def on_middle_drag(self, event: tk.Event) -> None:  # type: ignore
        """Handle middle mouse button drag for panning."""
        if self.panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y

            self.pan_offset_x += dx
            self.pan_offset_y += dy

            self.pan_start_x = event.x
            self.pan_start_y = event.y

            # Fast render during panning (skip annotations redraw)
            self.delete("all")
            self.photo_image = self.get_cached_image(self.zoom_level)

            canvas_width = self.winfo_width()
            canvas_height = self.winfo_height()
            width = int(self.image.width * self.zoom_level) if self.image else 0
            height = int(self.image.height * self.zoom_level) if self.image else 0

            self.offset_x = (canvas_width - width) // 2 + self.pan_offset_x
            self.offset_y = (canvas_height - height) // 2 + self.pan_offset_y

            self.create_image(
                self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo_image, tags="image"
            )

    def on_middle_release(self, _event: tk.Event) -> None:  # type: ignore
        """Handle middle mouse button release."""
        self.panning = False
        # Full render after panning
        self.render()

        if self.edit_mode:
            self.config(cursor="crosshair")
        else:
            self.config(cursor="arrow")

    def on_mouse_press(self, event: tk.Event) -> None:  # type: ignore
        if self.panning:
            return

        if not self.edit_mode:
            ann_idx = self.find_annotation_at(event.x, event.y)
            if ann_idx is not None:
                self.selected_annotation = ann_idx
                self.render()
            return

        handle = self.find_handle_at(event.x, event.y)
        if handle:
            self.save_to_undo_stack()
            self.dragging = True
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.drag_handle = handle[1]
            return

        ann_idx = self.find_annotation_at(event.x, event.y)
        if ann_idx is not None:
            self.selected_annotation = ann_idx
            self.render()
            return

        if self.current_category is not None:
            self.save_to_undo_stack()
            self.drawing = True
            self.draw_start_x = event.x
            self.draw_start_y = event.y
            self.selected_annotation = None

    def on_mouse_drag(self, event: tk.Event) -> None:  # type: ignore
        if self.panning or not self.edit_mode:
            return

        if self.drawing:
            if self.current_rect:
                self.delete(self.current_rect)

            self.current_rect = self.create_rectangle(
                self.draw_start_x,
                self.draw_start_y,
                event.x,
                event.y,
                outline="#00FF00",
                width=2,
                dash=(5, 5),
            )

        elif self.dragging and self.drag_handle:
            self.resize_annotation(event.x, event.y)
            self.drag_start_x = event.x
            self.drag_start_y = event.y

    def on_mouse_release(self, event: tk.Event) -> None:  # type: ignore
        if self.panning or not self.edit_mode:
            return

        if self.drawing:
            if self.current_rect:
                self.delete(self.current_rect)
                self.current_rect = None

            x1, y1 = self.screen_to_image_coords(self.draw_start_x, self.draw_start_y)
            x2, y2 = self.screen_to_image_coords(event.x, event.y)

            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)

            if x_max - x_min > 5 and y_max - y_min > 5:
                bbox = BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

                cat_name = self.categories.get(self.current_category, "unknown")
                new_ann = Annotation(
                    bbox=bbox,
                    category_id=self.current_category,  # type: ignore
                    category_name=cat_name,
                    image_id="",
                    annotation_id=None,
                    area=bbox.area,
                    iscrowd=0,
                )

                self.annotations.append(new_ann)
                self.selected_annotation = len(self.annotations) - 1
                self.render()

                if self.on_annotation_changed:
                    self.on_annotation_changed()

            self.drawing = False

        elif self.dragging:
            self.dragging = False
            self.drag_handle = None

    def resize_annotation(self, x: int, y: int) -> None:
        if self.selected_annotation is None or not self.drag_handle:
            return

        img_x, img_y = self.screen_to_image_coords(x, y)
        ann = self.annotations[self.selected_annotation]
        bbox = ann.bbox

        x_min, y_min, x_max, y_max = bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max

        if self.drag_handle == "n":
            y_min = img_y
        elif self.drag_handle == "s":
            y_max = img_y
        elif self.drag_handle == "w":
            x_min = img_x
        elif self.drag_handle == "e":
            x_max = img_x
        elif "n" in self.drag_handle:
            y_min = img_y
        elif "s" in self.drag_handle:
            y_max = img_y

        if "w" in self.drag_handle:
            x_min = img_x
        elif "e" in self.drag_handle:
            x_max = img_x

        if x_max - x_min >= 5 and y_max - y_min >= 5:
            new_bbox = BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

            self.annotations[self.selected_annotation] = Annotation(
                bbox=new_bbox,
                category_id=ann.category_id,
                category_name=ann.category_name,
                image_id=ann.image_id,
                annotation_id=ann.annotation_id,
                area=new_bbox.area,
                iscrowd=ann.iscrowd,
            )

            self.render()

            if self.on_annotation_changed:
                self.on_annotation_changed()

    def on_right_click(self, event: tk.Event) -> None:  # type: ignore
        if not self.edit_mode or self.panning:
            return

        ann_idx = self.find_annotation_at(event.x, event.y)
        if ann_idx is not None:
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(
                label="Delete Annotation (Del)", command=lambda: self.delete_annotation(ann_idx)
            )
            menu.post(event.x_root, event.y_root)

    def delete_annotation(self, idx: int) -> None:
        """Delete annotation by index."""
        if 0 <= idx < len(self.annotations):
            self.save_to_undo_stack()
            del self.annotations[idx]

            if self.selected_annotation == idx:
                self.selected_annotation = None
            elif self.selected_annotation is not None and self.selected_annotation > idx:
                self.selected_annotation -= 1

            self.render()

            if self.on_annotation_changed:
                self.on_annotation_changed()

    def on_mouse_move(self, event: tk.Event) -> None:  # type: ignore
        # Update crosshair position
        self.crosshair_x = event.x
        self.crosshair_y = event.y

        if self.panning:
            return

        if not self.edit_mode or self.drawing or self.dragging:
            if self.edit_mode and not self.drawing and not self.dragging:
                # Only redraw crosshair
                self.delete("crosshair")
                self.draw_crosshair()
            return

        # Redraw crosshair
        self.delete("crosshair")
        self.draw_crosshair()

        handle = self.find_handle_at(event.x, event.y)
        if handle:
            handle_type = handle[1]
            if handle_type in ("nw", "se"):
                self.config(cursor="size_nw_se")
            elif handle_type in ("ne", "sw"):
                self.config(cursor="size_ne_sw")
            elif handle_type in ("n", "s"):
                self.config(cursor="sb_v_double_arrow")
            elif handle_type in ("w", "e"):
                self.config(cursor="sb_h_double_arrow")
            return

        ann_idx = self.find_annotation_at(event.x, event.y)
        if ann_idx is not None:
            self.config(cursor="hand2")
        else:
            self.config(cursor="crosshair")

    def on_mouse_leave(self, _event: tk.Event) -> None:  # type: ignore
        """Handle mouse leaving canvas."""
        self.delete("crosshair")

    def on_mouse_wheel(self, event: tk.Event) -> None:  # type: ignore
        """Handle mouse wheel for zooming."""
        if event.delta > 0:
            self.zoom_level = min(self.zoom_level * 1.2, 5.0)
        else:
            self.zoom_level = max(self.zoom_level / 1.2, 0.1)

        self.render()

    def zoom_in(self) -> None:
        """Zoom in."""
        self.zoom_level = min(self.zoom_level * 1.2, 5.0)
        self.render()

    def zoom_out(self) -> None:
        """Zoom out."""
        self.zoom_level = max(self.zoom_level / 1.2, 0.1)
        self.render()

    def reset_zoom(self) -> None:
        """Reset zoom and pan."""
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.render()
