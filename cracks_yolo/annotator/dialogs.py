from __future__ import annotations

import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk
import typing as t


class ImportDialog:
    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.result: dict[str, t.Any] | None = None

    def show(self) -> dict[str, t.Any] | None:
        """Show dialog and return result."""
        dialog = tk.Toplevel(self.parent)
        dialog.title("Import Dataset")
        dialog.geometry("650x520")
        dialog.resizable(False, False)
        dialog.transient(self.parent)  # type: ignore
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (650 // 2)
        y = (dialog.winfo_screenheight() // 2) - (520 // 2)
        dialog.geometry(f"+{x}+{y}")

        # Main container
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Format selection
        format_frame = ttk.LabelFrame(main_frame, text="Source Type", padding=10)
        format_frame.pack(fill="x", pady=(0, 10))

        format_var = tk.StringVar(value="coco")
        ttk.Radiobutton(format_frame, text="COCO Dataset", variable=format_var, value="coco").pack(
            anchor="w", padx=5, pady=2
        )
        ttk.Radiobutton(format_frame, text="YOLO Dataset", variable=format_var, value="yolo").pack(
            anchor="w", padx=5, pady=2
        )
        ttk.Radiobutton(
            format_frame, text="Raw Images (Create New Dataset)", variable=format_var, value="raw"
        ).pack(anchor="w", padx=5, pady=2)

        # Path selection
        path_frame = ttk.LabelFrame(main_frame, text="Dataset Path", padding=10)
        path_frame.pack(fill="x", pady=(0, 10))

        path_var = tk.StringVar()
        path_entry = ttk.Entry(path_frame, textvariable=path_var, width=70)
        path_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)

        def browse():
            path = filedialog.askdirectory()
            if path:
                path_var.set(path)

        ttk.Button(path_frame, text="Browse...", command=browse, width=12).pack(side="left")

        # Splits selection (hidden for raw)
        splits_frame = ttk.LabelFrame(main_frame, text="Splits to Load", padding=10)
        splits_frame.pack(fill="x", pady=(0, 10))

        train_var = tk.BooleanVar(value=True)
        val_var = tk.BooleanVar(value=True)
        test_var = tk.BooleanVar(value=True)

        checks_frame = ttk.Frame(splits_frame)
        checks_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(checks_frame, text="Train", variable=train_var).pack(side="left", padx=15)
        ttk.Checkbutton(checks_frame, text="Val", variable=val_var).pack(side="left", padx=15)
        ttk.Checkbutton(checks_frame, text="Test", variable=test_var).pack(side="left", padx=15)

        # Categories frame (only shown for raw)
        categories_frame = ttk.LabelFrame(main_frame, text="Categories", padding=10)

        ttk.Label(
            categories_frame,
            text="Enter category names (one per line, press Enter to add new line):",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 5))

        categories_text = tk.Text(categories_frame, height=8, width=60, font=("Consolas", 9))
        categories_scrollbar = ttk.Scrollbar(categories_frame, command=categories_text.yview)
        categories_text.config(yscrollcommand=categories_scrollbar.set)

        categories_text.pack(side="left", fill="both", expand=True)
        categories_scrollbar.pack(side="right", fill="y")

        # Default categories
        default_categories = "crack\nspalling\nrebar_exposure"
        categories_text.insert("1.0", default_categories)

        # Toggle visibility based on format
        def on_format_change(*_args: t.Any) -> None:
            if format_var.get() == "raw":
                splits_frame.pack_forget()
                categories_frame.pack(fill="both", expand=True, pady=(0, 10))
            else:
                categories_frame.pack_forget()
                splits_frame.pack(fill="x", pady=(0, 10))

        format_var.trace("w", on_format_change)

        # Spacer
        ttk.Frame(main_frame).pack(fill="both", expand=True)

        # Buttons at bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))

        def on_ok():
            if not path_var.get():
                messagebox.showwarning("Warning", "Please select a dataset path")
                return

            format_type = format_var.get()

            if format_type == "raw":
                # Get categories
                categories_text_content = categories_text.get("1.0", "end-1c")
                categories = [
                    line.strip() for line in categories_text_content.split("\n") if line.strip()
                ]

                if not categories:
                    messagebox.showwarning("Warning", "Please enter at least one category")
                    return

                self.result = {
                    "format": "raw",
                    "path": path_var.get(),
                    "categories": categories,
                    "splits": ["train"],  # Default split for raw
                }
            else:
                # Existing COCO/YOLO logic
                splits = []
                if train_var.get():
                    splits.append("train")
                if val_var.get():
                    splits.append("val")
                if test_var.get():
                    splits.append("test")

                if not splits:
                    messagebox.showwarning("Warning", "Please select at least one split")
                    return

                self.result = {
                    "format": format_type,
                    "path": path_var.get(),
                    "splits": splits,
                }

            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ttk.Button(button_frame, text="Cancel", command=on_cancel, width=12).pack(
            side="right", padx=(5, 0)
        )
        ok_button = ttk.Button(button_frame, text="OK", command=on_ok, width=12)
        ok_button.pack(side="right")

        # Bind keys carefully
        # Only bind Return to OK for Entry widgets and buttons, not for Text widget
        path_entry.bind("<Return>", lambda _: on_ok())

        # Prevent Return from triggering OK when in Text widget
        def on_text_return(_event: tk.Event) -> None:
            # Allow normal Return behavior in Text widget
            return None  # Let default behavior happen

        categories_text.bind("<Return>", on_text_return)

        # Bind Escape to cancel everywhere
        dialog.bind("<Escape>", lambda _: on_cancel())

        # Bind Ctrl+Return to OK (alternative shortcut that works everywhere)
        dialog.bind("<Control-Return>", lambda _: on_ok())

        dialog.wait_window()
        return self.result


class ExportDialog:
    """Dialog for exporting datasets."""

    def __init__(self, parent: tk.Widget):
        """Initialize export dialog."""
        self.parent = parent
        self.result: dict[str, t.Any] | None = None

    def show(self) -> dict[str, t.Any] | None:
        """Show dialog and return result."""
        dialog = tk.Toplevel(self.parent)
        dialog.title("Export Dataset")
        dialog.geometry("600x420")
        dialog.resizable(False, False)
        dialog.transient(self.parent)  # type: ignore
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (420 // 2)
        dialog.geometry(f"+{x}+{y}")

        # Main container
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Output directory
        dir_frame = ttk.LabelFrame(main_frame, text="Output Directory", padding=10)
        dir_frame.pack(fill="x", pady=(0, 10))

        dir_var = tk.StringVar()
        dir_entry = ttk.Entry(dir_frame, textvariable=dir_var, width=60)
        dir_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)

        def browse() -> None:
            path = filedialog.askdirectory()
            if path:
                dir_var.set(path)

        ttk.Button(dir_frame, text="Browse...", command=browse, width=12).pack(side="left")

        # Format selection
        format_frame = ttk.LabelFrame(main_frame, text="Export Format", padding=10)
        format_frame.pack(fill="x", pady=(0, 10))

        format_var = tk.StringVar(value="coco")
        ttk.Radiobutton(format_frame, text="COCO", variable=format_var, value="coco").pack(
            anchor="w", padx=5, pady=5
        )
        ttk.Radiobutton(format_frame, text="YOLO", variable=format_var, value="yolo").pack(
            anchor="w", padx=5, pady=5
        )

        # Naming strategy
        naming_frame = ttk.LabelFrame(main_frame, text="File Naming Strategy", padding=10)
        naming_frame.pack(fill="x", pady=(0, 10))

        naming_var = tk.StringVar(value="original")

        naming_options = [
            ("original", "Original (keep original filenames)"),
            ("prefix", "Prefix (add source prefix)"),
            ("uuid", "UUID (generate unique IDs)"),
            ("uuid_prefix", "UUID + Prefix"),
            ("sequential", "Sequential (numbered)"),
            ("sequential_prefix", "Sequential + Prefix"),
        ]

        naming_combo = ttk.Combobox(
            naming_frame,
            textvariable=naming_var,
            values=[opt[0] for opt in naming_options],
            state="readonly",
            width=25,
        )
        naming_combo.pack(anchor="w", padx=5, pady=5)

        # Description label
        naming_desc = ttk.Label(
            naming_frame,
            text="Choose how to name exported files",
            foreground="gray",
            font=("Segoe UI", 8),
        )
        naming_desc.pack(anchor="w", padx=5, pady=(0, 5))

        # Spacer to push buttons to bottom
        ttk.Frame(main_frame).pack(fill="both", expand=True)

        # Buttons at bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))

        def on_ok() -> None:
            if not dir_var.get():
                messagebox.showwarning("Warning", "Please select an output directory")
                return

            self.result = {
                "output_dir": dir_var.get(),
                "format": format_var.get(),
                "naming": naming_var.get(),
            }
            dialog.destroy()

        def on_cancel() -> None:
            dialog.destroy()

        ttk.Button(button_frame, text="Cancel", command=on_cancel, width=12).pack(
            side="right", padx=(5, 0)
        )
        ttk.Button(button_frame, text="OK", command=on_ok, width=12).pack(side="right")

        # Bind Enter key to OK
        dialog.bind("<Return>", lambda _: on_ok())
        dialog.bind("<Escape>", lambda _: on_cancel())

        dialog.wait_window()
        return self.result
