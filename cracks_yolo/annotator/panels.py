from __future__ import annotations

import tkinter as tk
import tkinter.ttk as ttk
import typing as t

if t.TYPE_CHECKING:
    from cracks_yolo.dataset.types import Annotation
    from cracks_yolo.dataset.types import ImageInfo


class ImageListPanel(ttk.Frame):
    def __init__(self, parent: tk.Widget, on_select: t.Callable[[str], None], width: int = 250):
        super().__init__(parent, width=width)

        self.on_select = on_select

        # Title with modern styling
        title_frame = ttk.Frame(self)
        title_frame.pack(fill="x", pady=(5, 10))

        # Search box
        search_frame = ttk.Frame(self)
        search_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(search_frame, text="Search: ").pack(side="left", padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._on_search)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side="left", fill="x", expand=True)

        # Listbox with scrollbar
        list_frame = ttk.Frame(self)
        list_frame.pack(fill="both", expand=True, padx=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set, font=("Consolas", 9), selectmode=tk.SINGLE
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        self.all_images: list[tuple[str, str]] = []  # (id, filename)

    def set_images(self, images: list[tuple[str, str]]) -> None:
        """Set images to display.

        Args:
            images: List of (image_id, filename) tuples
        """
        self.all_images = images
        self._update_display()

    def _update_display(self) -> None:
        """Update listbox display."""
        search_term = self.search_var.get().lower()

        self.listbox.delete(0, tk.END)

        for _, filename in self.all_images:
            if search_term in filename.lower():
                self.listbox.insert(tk.END, filename)

    def _on_search(self, *_args: t.Any) -> None:
        """Handle search text change."""
        self._update_display()

    def _on_listbox_select(self, _event: tk.Event) -> None:
        """Handle listbox selection."""
        selection = self.listbox.curselection()
        if selection:
            index = selection[0]
            filename = self.listbox.get(index)

            for img_id, fname in self.all_images:
                if fname == filename:
                    self.on_select(img_id)
                    break

    def select_image(self, image_id: str) -> None:
        """Select image in list."""
        for _i, (img_id, filename) in enumerate(self.all_images):
            if img_id == image_id:
                # Find in current filtered list
                for j in range(self.listbox.size()):
                    if self.listbox.get(j) == filename:
                        self.listbox.selection_clear(0, tk.END)
                        self.listbox.selection_set(j)
                        self.listbox.see(j)
                        break
                break


class ControlPanel(ttk.Frame):
    """Panel for navigation and editing controls."""

    def __init__(
        self,
        parent: tk.Widget,
        on_prev: t.Callable[[], None],
        on_next: t.Callable[[], None],
        on_split_changed: t.Callable[[str], None],
        on_edit_mode_changed: t.Callable[[bool], None],
        on_category_changed: t.Callable[[int | None], None],
    ):
        """Initialize control panel."""
        super().__init__(parent)

        self.on_split_changed = on_split_changed
        self.on_edit_mode_changed = on_edit_mode_changed
        self.on_category_changed = on_category_changed

        # Left side: Split and navigation
        left_frame = ttk.Frame(self)
        left_frame.pack(side="left", padx=5)

        # Split selector
        ttk.Label(left_frame, text="Split:").pack(side="left", padx=(0, 5))
        self.split_var = tk.StringVar()
        self.split_combo = ttk.Combobox(
            left_frame, textvariable=self.split_var, state="readonly", width=10
        )
        self.split_combo.pack(side="left", padx=(0, 15))
        self.split_combo.bind("<<ComboboxSelected>>", self._on_split_select)

        # Navigation with counter
        ttk.Button(left_frame, text="◀ Prev", command=on_prev, width=8).pack(side="left", padx=2)

        self.counter_label = ttk.Label(left_frame, text="0 / 0", font=("Segoe UI", 10, "bold"))
        self.counter_label.pack(side="left", padx=10)

        ttk.Button(left_frame, text="Next ▶", command=on_next, width=8).pack(side="left", padx=2)

        # Right side: Edit controls
        right_frame = ttk.Frame(self)
        right_frame.pack(side="right", padx=5)

        # Category selector
        ttk.Label(right_frame, text="Category:").pack(side="left", padx=(0, 5))
        self.category_var = tk.StringVar()
        self.category_combo = ttk.Combobox(
            right_frame, textvariable=self.category_var, state="readonly", width=15
        )
        self.category_combo.pack(side="left", padx=(0, 15))
        self.category_combo.bind("<<ComboboxSelected>>", self._on_category_select)

        # Edit mode toggle
        self.edit_mode_var = tk.BooleanVar(value=False)
        edit_btn = ttk.Checkbutton(
            right_frame,
            text="📝 Edit Mode",
            variable=self.edit_mode_var,
            command=self._on_edit_mode_toggle,
            style="Toolbutton",
        )
        edit_btn.pack(side="left")

        self.categories: dict[int, str] = {}

    def set_splits(self, splits: list[str], current: str) -> None:
        """Set available splits."""
        self.split_combo["values"] = splits
        self.split_var.set(current)

    def set_categories(self, categories: dict[int, str]) -> None:
        """Set available categories."""
        self.categories = categories
        category_names = list(categories.values())
        self.category_combo["values"] = category_names

        if category_names:
            self.category_var.set(category_names[0])
            self._on_category_select(None)

    def update_counter(self, current: int, total: int) -> None:
        """Update image counter display."""
        self.counter_label.config(text=f"{current} / {total}")

    def _on_split_select(self, _event: tk.Event) -> None:
        """Handle split selection."""
        split = self.split_var.get()
        self.on_split_changed(split)

    def _on_edit_mode_toggle(self) -> None:
        """Handle edit mode toggle."""
        enabled = self.edit_mode_var.get()
        self.on_edit_mode_changed(enabled)

    def _on_category_select(self, _event: tk.Event | None) -> None:
        """Handle category selection."""
        cat_name = self.category_var.get()

        cat_id = None
        for cid, cname in self.categories.items():
            if cname == cat_name:
                cat_id = cid
                break

        self.on_category_changed(cat_id)


class InfoPanel(ttk.Frame):
    """Panel for displaying image and dataset information."""

    def __init__(self, parent: tk.Widget, width: int = 300):
        """Initialize info panel."""
        super().__init__(parent, width=width)

        # Dataset info
        self.dataset_info_frame = ttk.LabelFrame(self, text="Dataset Overview", padding=5)
        self.dataset_info_frame.pack(fill="x", padx=5, pady=5)

        self.dataset_text = tk.Text(
            self.dataset_info_frame,
            height=5,
            width=30,
            state="disabled",
            font=("Consolas", 9),
            wrap="word",
        )
        self.dataset_text.pack(fill="x")

        # Image info
        self.image_info_frame = ttk.LabelFrame(self, text="Current Image", padding=5)
        self.image_info_frame.pack(fill="x", padx=5, pady=5)

        self.image_text = tk.Text(
            self.image_info_frame,
            height=5,
            width=30,
            state="disabled",
            font=("Consolas", 9),
            wrap="word",
        )
        self.image_text.pack(fill="x")

        # Annotations
        self.annotations_frame = ttk.LabelFrame(self, text="Bounding Boxes", padding=5)
        self.annotations_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.annotations_text = tk.Text(
            self.annotations_frame, width=30, state="disabled", font=("Consolas", 9)
        )
        ann_scrollbar = ttk.Scrollbar(self.annotations_frame, command=self.annotations_text.yview)
        self.annotations_text.config(yscrollcommand=ann_scrollbar.set)

        self.annotations_text.pack(side="left", fill="both", expand=True)
        ann_scrollbar.pack(side="right", fill="y")

    def update_dataset_info(self, info: dict[str, int]) -> None:
        """Update dataset information."""
        self.dataset_text.config(state="normal")
        self.dataset_text.delete("1.0", tk.END)

        text = ""
        for key, value in info.items():
            text += f"{key.capitalize()}: {value}\n"

        self.dataset_text.insert("1.0", text)
        self.dataset_text.config(state="disabled")

    def update_image_info(
        self,
        image_info: ImageInfo,
        annotations: list[Annotation],
        source: str | None = None,
    ) -> None:
        """Update image information."""
        # Image info
        self.image_text.config(state="normal")
        self.image_text.delete("1.0", tk.END)

        text = f"Filename: {image_info.file_name}\n"
        text += f"Size: {image_info.width} x {image_info.height}\n"
        text += f"Annotations: {len(annotations)}\n"
        if source:
            text += f"Source: {source}\n"

        self.image_text.insert("1.0", text)
        self.image_text.config(state="disabled")

        # Annotations
        self.annotations_text.config(state="normal")
        self.annotations_text.delete("1.0", tk.END)

        for i, ann in enumerate(annotations, 1):
            self.annotations_text.insert(
                tk.END,
                f"{i}. {ann.category_name}\n   {ann.bbox.x_min:.0f}, {ann.bbox.y_min:.0f}, "
                f"{ann.bbox.x_max:.0f}, {ann.bbox.y_max:.0f}\n\n",
            )

        self.annotations_text.config(state="disabled")
