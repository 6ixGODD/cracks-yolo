from __future__ import annotations

import logging
import pathlib
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk

from cracks_yolo.annotator.canvas import AnnotationCanvas
from cracks_yolo.annotator.controller import AnnotationController
from cracks_yolo.annotator.dialogs import ExportDialog
from cracks_yolo.annotator.dialogs import ImportDialog
from cracks_yolo.annotator.panels import ControlPanel
from cracks_yolo.annotator.panels import ImageListPanel
from cracks_yolo.annotator.panels import InfoPanel

logger = logging.getLogger(__name__)


class AnnotatorApp:
    def __init__(self, root: tk.Tk | None = None):
        """Initialize the annotation application."""
        self.root = root or tk.Tk()
        self.root.title("Cracks YOLO Annotator")

        # Start maximized
        self.root.state("zoomed")

        # Modern theme
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")

        self.controller = AnnotationController()

        self._setup_menu()
        self._setup_layout()
        self._setup_bindings()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        logger.info("Annotator application initialized")

    def _setup_menu(self) -> None:
        """Setup menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Import Dataset...", command=self.import_dataset, accelerator="Ctrl+O"
        )
        file_menu.add_command(
            label="Export Dataset...", command=self.export_dataset, accelerator="Ctrl+S"
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Save Workspace...", command=self.save_workspace, accelerator="Ctrl+Shift+S"
        )
        file_menu.add_command(
            label="Load Workspace...", command=self.load_workspace, accelerator="Ctrl+Shift+O"
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing, accelerator="Alt+F4")

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Next Image", command=self.next_image, accelerator="→")
        view_menu.add_command(label="Previous Image", command=self.prev_image, accelerator="←")
        view_menu.add_separator()
        view_menu.add_command(label="Zoom In", command=self.zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self.zoom_out, accelerator="Ctrl+-")
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom, accelerator="Ctrl+0")

        # Audit menu
        audit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Audit", menu=audit_menu)
        audit_menu.add_command(
            label="Approve Current", command=self.approve_current, accelerator="A"
        )
        audit_menu.add_command(label="Reject Current", command=self.reject_current, accelerator="R")
        audit_menu.add_separator()
        audit_menu.add_command(label="View Audit Report", command=self.show_audit_report)
        audit_menu.add_command(label="Export Audit Report...", command=self.export_audit_report)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="About", command=self.show_about)

    # ... existing _setup_layout ...

    def _setup_layout(self) -> None:
        """Setup main layout with panels."""
        # Configure grid weights for responsiveness
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=0)

        # Left panel: Image list
        self.image_list_panel = ImageListPanel(
            self.root,  # type: ignore
            on_select=self.on_image_selected,
            width=280,
        )
        self.image_list_panel.grid(row=0, column=0, sticky="nsew", padx=(5, 2), pady=5)

        # Center: Canvas and controls
        center_frame = ttk.Frame(self.root)
        center_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=5)
        center_frame.grid_rowconfigure(0, weight=1)
        center_frame.grid_columnconfigure(0, weight=1)

        # Canvas
        self.canvas = AnnotationCanvas(
            center_frame,
            on_annotation_changed=self.on_annotations_changed,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        # Control panel (with audit callbacks)
        self.control_panel = ControlPanel(
            center_frame,
            on_prev=self.prev_image,
            on_next=self.next_image,
            on_split_changed=self.on_split_changed,
            on_edit_mode_changed=self.on_edit_mode_changed,
            on_category_changed=self.on_category_changed,
            on_audit_mode_changed=self.on_audit_mode_changed,
            on_approve=self.approve_current,
            on_reject=self.reject_current,
        )
        self.control_panel.grid(row=1, column=0, sticky="ew")

        # Right panel: Info
        self.info_panel = InfoPanel(self.root, width=320)  # type: ignore
        self.info_panel.grid(row=0, column=2, sticky="nsew", padx=(2, 5), pady=5)

        # Status bar
        status_frame = ttk.Frame(self.root, relief="sunken")
        status_frame.grid(row=1, column=0, columnspan=3, sticky="ew")

        self.status_var = tk.StringVar(value="Ready - Import a dataset to begin")
        ttk.Label(status_frame, textvariable=self.status_var, anchor="w", padding=(5, 2)).pack(
            side="left", fill="x", expand=True
        )

    def _setup_bindings(self) -> None:
        """Setup keyboard shortcuts."""
        self.root.bind("<Left>", lambda _: self.prev_image())
        self.root.bind("<Right>", lambda _: self.next_image())
        self.root.bind("<Control-o>", lambda _: self.import_dataset())
        self.root.bind("<Control-s>", lambda _: self.export_dataset())
        self.root.bind("<Control-Shift-S>", lambda _: self.save_workspace())
        self.root.bind("<Control-Shift-O>", lambda _: self.load_workspace())
        self.root.bind("<Control-plus>", lambda _: self.zoom_in())
        self.root.bind("<Control-equal>", lambda _: self.zoom_in())
        self.root.bind("<Control-minus>", lambda _: self.zoom_out())
        self.root.bind("<Control-0>", lambda _: self.reset_zoom())

        # Delete and Undo
        self.root.bind("<Delete>", lambda _: self.delete_selected())
        self.root.bind("<Control-z>", lambda _: self.undo())

        # Audit shortcuts
        self.root.bind("<a>", lambda _: self.approve_current())
        self.root.bind("<A>", lambda _: self.approve_current())
        self.root.bind("<r>", lambda _: self.reject_current())
        self.root.bind("<R>", lambda _: self.reject_current())

    def on_audit_mode_changed(self, enabled: bool) -> None:
        """Handle audit mode toggle."""
        self.controller.enable_audit_mode(enabled)

        # Update info panel with audit stats
        if enabled:
            audit_stats = self.controller.get_audit_statistics()
            self.info_panel.update_dataset_info(self.controller.get_dataset_info(), audit_stats)

            # Update current image audit status
            current_id = self.controller.get_current_image_id()
            if current_id:
                status = self.controller.get_audit_status(current_id)
                self.control_panel.update_audit_status(status)

        mode_text = "ON (🔍 Audit)" if enabled else "OFF"
        self.status_var.set(f"Audit mode: {mode_text}")

    def approve_current(self) -> None:
        """Approve current image."""
        if not self.controller.audit_mode:
            return

        current_id = self.controller.get_current_image_id()
        if current_id:
            self.controller.set_audit_status(current_id, "approved")
            self.control_panel.update_audit_status("approved")
            self.status_var.set("✓ Image approved")

            # Update info panel
            audit_stats = self.controller.get_audit_statistics()
            self.info_panel.update_dataset_info(self.controller.get_dataset_info(), audit_stats)

            # Auto advance to next
            self.next_image()

    def reject_current(self) -> None:
        """Reject current image."""
        if not self.controller.audit_mode:
            return

        current_id = self.controller.get_current_image_id()
        if current_id:
            self.controller.set_audit_status(current_id, "rejected")
            self.control_panel.update_audit_status("rejected")
            self.status_var.set("✗ Image rejected")

            # Update info panel
            audit_stats = self.controller.get_audit_statistics()
            self.info_panel.update_dataset_info(self.controller.get_dataset_info(), audit_stats)

            # Auto advance to next
            self.next_image()

    def save_workspace(self) -> None:
        """Save current workspace."""
        if not self.controller.has_dataset():
            messagebox.showwarning("Warning", "No dataset loaded")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".cyw",
            filetypes=[("Cracks YOLO Workspace", "*.cyw"), ("All Files", "*.*")],
            title="Save Workspace",
        )

        if filepath:
            try:
                self.controller.save_workspace(filepath)
                self.status_var.set(f"✓ Workspace saved: {filepath}")
                messagebox.showinfo("Success", f"Workspace saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save workspace:\n{e}")
                logger.error(f"Failed to save workspace: {e}", exc_info=True)

    def load_workspace(self) -> None:
        """Load workspace from file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Cracks YOLO Workspace", "*.cyw"), ("All Files", "*.*")],
            title="Load Workspace",
        )

        if filepath:
            try:
                self.controller.load_workspace(filepath)
                self.update_after_load()

                # Restore current image
                current_id = self.controller.get_current_image_id()
                if current_id:
                    self.load_image(current_id)

                self.status_var.set(f"✓ Workspace loaded: {filepath}")
                messagebox.showinfo("Success", f"Workspace loaded from:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load workspace:\n{e}")
                logger.error(f"Failed to load workspace: {e}", exc_info=True)

    def show_audit_report(self) -> None:
        """Show audit report in a dialog."""
        if not self.controller.audit_mode:
            messagebox.showinfo("Info", "Audit mode is not enabled")
            return

        report = self.controller.generate_audit_report()

        # Create report dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Audit Report")
        dialog.geometry("600x400")
        dialog.transient(self.root)  # type: ignore

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"+{x}+{y}")

        # Text widget with scrollbar
        text_frame = ttk.Frame(dialog, padding=10)
        text_frame.pack(fill="both", expand=True)

        text = tk.Text(text_frame, wrap="word", font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(text_frame, command=text.yview)
        text.config(yscrollcommand=scrollbar.set)

        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        text.insert("1.0", report)
        text.config(state="disabled")

        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy, width=12).pack(pady=10)

    def export_audit_report(self) -> None:
        """Export audit report to text file."""
        if not self.controller.audit_mode:
            messagebox.showinfo("Info", "Audit mode is not enabled")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Export Audit Report",
        )

        if filepath:
            try:
                report = self.controller.generate_audit_report()
                with pathlib.Path(filepath).open("w", encoding="utf-8") as f:
                    f.write(report)

                self.status_var.set(f"✓ Audit report exported: {filepath}")
                messagebox.showinfo("Success", f"Audit report exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report:\n{e}")
                logger.error(f"Failed to export report: {e}", exc_info=True)

    def export_dataset(self) -> None:
        """Show export dialog and save dataset."""
        if not self.controller.has_dataset():
            messagebox.showwarning("Warning", "No dataset loaded")
            return

        # Check for pending audits if in audit mode
        if self.controller.audit_mode:
            stats = self.controller.get_audit_statistics()
            if stats["pending"] > 0:
                response = messagebox.askyesno(
                    "Pending Audits",
                    f"There are {stats['pending']} images pending audit.\n\n"
                    "Do you want to continue exporting?",
                )
                if not response:
                    return

        dialog = ExportDialog(self.root)  # type: ignore
        result = dialog.show()

        if result:
            output_dir = result["output_dir"]
            format_type = result["format"]
            naming = result["naming"]

            self.status_var.set(f"Exporting to {format_type.upper()} format...")
            self.root.update()

            try:
                self.controller.export_dataset(output_dir, format_type, naming)

                self.status_var.set(f"✓ Exported to {output_dir}")

                # Export audit report if in audit mode
                if self.controller.audit_mode:
                    import pathlib

                    report_path = pathlib.Path(output_dir) / "audit_report.txt"
                    report = self.controller.generate_audit_report()
                    with report_path.open("w", encoding="utf-8") as f:
                        f.write(report)

                    messagebox.showinfo(
                        "Success",
                        f"Dataset exported successfully to:\n{output_dir}\n\n"
                        f"Audit report saved to:\n{report_path}",
                    )
                else:
                    messagebox.showinfo(
                        "Success", f"Dataset exported successfully to:\n{output_dir}"
                    )

            except Exception as e:
                messagebox.showerror("Error", f"Failed to export dataset:\n{e}")
                self.status_var.set("Ready")
                logger.error(f"Failed to export dataset: {e}", exc_info=True)

    def load_image(self, image_id: str) -> None:
        """Load and display image with annotations."""
        try:
            image_info = self.controller.get_image_info(image_id)
            annotations = self.controller.get_annotations(image_id)

            if image_info and image_info.path and image_info.path.exists():
                from PIL import Image

                img = Image.open(image_info.path)

                self.canvas.display_image(img, annotations, self.controller.get_categories())

                # Get audit status if in audit mode
                audit_status = None
                if self.controller.audit_mode:
                    audit_status = self.controller.get_audit_status(image_id)
                    self.control_panel.update_audit_status(audit_status)

                self.info_panel.update_image_info(
                    image_info=image_info,
                    annotations=annotations,
                    source=self.controller.get_image_source(image_id),
                    audit_status=audit_status,
                )

                self.status_var.set(f"📷 {image_info.file_name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            logger.error(f"Failed to load image {image_id}: {e}", exc_info=True)

    def show_shortcuts(self) -> None:
        """Show keyboard shortcuts dialog."""
        shortcuts = """
Keyboard Shortcuts:

Navigation:
  ←\t\t\tPrevious image
  →\t\t\tNext image

View:
  Ctrl + +\t\tZoom in (at mouse position)
  Ctrl + -\t\tZoom out (at mouse position)
  Ctrl + 0\t\tReset zoom
  Mouse Wheel\t\tZoom at cursor

Editing:
  Delete\t\tDelete selected annotation
  Ctrl + Z\t\tUndo last change
  Right Click\t\tShow delete menu
  Drag Corners\t\tResize (diagonal)
  Drag Edges\t\tResize (horizontal/vertical)
  Click BBox\t\tSelect annotation

Audit:
  A\t\t\tApprove current image
  R\t\t\tReject current image

File:
  Ctrl + O\t\tImport dataset
  Ctrl + S\t\tExport dataset
  Ctrl + Shift + S\tSave workspace
  Ctrl + Shift + O\tLoad workspace
"""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)

        # ... rest of existing methods ...

    def delete_selected(self) -> None:
        """Delete selected annotation."""
        if self.canvas.delete_selected():
            self.status_var.set("⚠️ Annotation deleted - Unsaved changes")

    def undo(self) -> None:
        """Undo last annotation change."""
        if self.canvas.undo():
            self.status_var.set("↶ Undo - Unsaved changes")

    def on_closing(self) -> None:
        """Handle window close event."""
        if self.controller.has_unsaved_changes():
            response = messagebox.askyesnocancel(
                "Unsaved Changes", "You have unsaved changes. Do you want to export before closing?"
            )
            if response is None:  # Cancel
                return
            if response:  # Yes
                self.export_dataset()
                if self.controller.has_unsaved_changes():  # Export was cancelled
                    return

        self.root.quit()

    def import_dataset(self) -> None:
        dialog = ImportDialog(self.root)  # type: ignore
        result = dialog.show()

        if result:
            format_type = result["format"]
            path = result["path"]
            splits = result["splits"]

            self.status_var.set(f"Loading {format_type.upper()} dataset...")
            self.root.update()

            try:
                self.controller.load_dataset(path, format_type, splits)
                self.update_after_load()

                self.status_var.set(
                    f"✓ Loaded {self.controller.total_images()} images from {len(splits)} split(s)"
                )

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset:\n{e}")
                self.status_var.set("Ready")
                logger.error(f"Failed to load dataset: {e}", exc_info=True)

    def update_after_load(self) -> None:
        splits = self.controller.get_splits()
        current_split = splits[0] if splits else None

        if current_split:
            self.control_panel.set_splits(splits, current_split)
            self.on_split_changed(current_split)

        categories = self.controller.get_categories()
        self.control_panel.set_categories(categories)

        self.info_panel.update_dataset_info(self.controller.get_dataset_info())

    def on_split_changed(self, split: str) -> None:
        self.controller.set_current_split(split)

        images = self.controller.get_images_in_split(split)
        self.image_list_panel.set_images(images)

        if images:
            self.load_image(images[0][0])

        self.update_counter()

    def on_image_selected(self, image_id: str) -> None:
        # Find index
        if self.controller.current_split:
            images = self.controller.image_ids_by_split.get(self.controller.current_split, [])
            try:
                self.controller.current_index = images.index(image_id)
                self.load_image(image_id)
                self.update_counter()
            except ValueError:
                pass

    def next_image(self) -> None:
        """Navigate to next image."""
        next_id = self.controller.next_image()
        if next_id:
            self.load_image(next_id)
            self.image_list_panel.select_image(next_id)
            self.update_counter()

    def prev_image(self) -> None:
        prev_id = self.controller.prev_image()
        if prev_id:
            self.load_image(prev_id)
            self.image_list_panel.select_image(prev_id)
            self.update_counter()

    def update_counter(self) -> None:
        current = self.controller.get_current_index()
        total = self.controller.get_split_size()
        self.control_panel.update_counter(current, total)

    def on_edit_mode_changed(self, enabled: bool) -> None:
        self.canvas.set_edit_mode(enabled)
        mode_text = "ON" if enabled else "OFF"
        self.status_var.set(f"Edit mode: {mode_text}")

    def on_category_changed(self, category_id: int | None) -> None:
        self.canvas.set_current_category(category_id)

    def on_annotations_changed(self) -> None:
        current_id = self.controller.get_current_image_id()
        if current_id:
            annotations = self.canvas.get_annotations()

            self.controller.update_annotations(current_id, annotations)

            image_info = self.controller.get_image_info(current_id)
            if image_info:
                self.info_panel.update_image_info(
                    image_info=image_info,
                    annotations=annotations,
                    source=self.controller.get_image_source(current_id),
                )

        self.status_var.set("⚠️ Unsaved changes - Export to save")

    def zoom_in(self) -> None:
        self.canvas.zoom_in()

    def zoom_out(self) -> None:
        self.canvas.zoom_out()

    def reset_zoom(self) -> None:
        self.canvas.reset_zoom()

    def show_about(self) -> None:
        """Show about dialog."""
        about_text = """
Cracks YOLO Annotator
Version 0.1.0

A simple tool for viewing and editing
object detection datasets in COCO and YOLO formats.

Features:
• Import COCO/YOLO datasets
• View and edit bounding boxes
• Multi-split support
• Export with flexible naming
"""
        messagebox.showinfo("About", about_text)

    def run(self) -> None:
        """Start the application."""
        logger.info("Starting annotator application")
        self.root.mainloop()


def main() -> None:
    app = AnnotatorApp()
    app.run()


if __name__ == "__main__":
    main()
