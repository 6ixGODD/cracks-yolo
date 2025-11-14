# ruff: noqa: RUF001
from __future__ import annotations

import contextlib
import os
import pathlib
import sys
import typing as t

import halo

from cracks_yolo.cli.helper.ansi import AnsiFormatter as Ansi

# Initialize spinner
spinner = halo.Halo(spinner="dots")


# Box drawing characters
class Box:
    """Unicode box drawing characters."""

    TL = "┌"  # Top-left
    TR = "┐"  # Top-right
    BL = "└"  # Bottom-left
    BR = "┘"  # Bottom-right
    H = "─"  # Horizontal
    V = "│"  # Vertical
    VR = "├"  # Vertical-right
    VL = "┤"  # Vertical-left
    HU = "┴"  # Horizontal-up
    HD = "┬"  # Horizontal-down
    X = "┼"  # Cross


def success(message: str, /, prefix: str = "✓") -> None:
    """Print a success message."""
    print(f"{Ansi.success(prefix)} {message}")


def error(message: str, /, prefix: str = "✗") -> None:
    """Print an error message."""
    print(f"{Ansi.error(prefix)} {message}", file=sys.stderr)


def warning(message: str, /, prefix: str = "⚠") -> None:
    """Print a warning message."""
    print(f"{Ansi.warning(prefix)} {message}")


def info(message: str, /, prefix: str = "ℹ") -> None:
    """Print an info message."""
    print(f"{Ansi.info(prefix)} {message}")


def debug(message: str, /, prefix: str = "→") -> None:
    """Print a debug message (dimmed)."""
    print(Ansi.format(f"{prefix} {message}", Ansi.STYLE.DIM))


def step(message: str, /, step: int | None = None) -> None:
    """Print a step message in a process."""
    if step is not None:
        prefix = Ansi.format(f"[{step}]", Ansi.FG.CYAN, Ansi.STYLE.BOLD)
    else:
        prefix = Ansi.format("►", Ansi.FG.CYAN)
    print(f"{prefix} {message}")


def path(
    path: str | pathlib.Path | os.PathLike[str],
    /,
    label: str | None = None,
    exists: bool | None = None,
) -> None:
    """Print a formatted file path.

    Args:
        path: The path to display
        label: Optional label to show before the path
        exists: If provided, shows existence indicator (✓ or ✗)
    """
    path = pathlib.Path(path)

    # Determine path type and color
    if path.is_dir():
        colored_path = Ansi.format(str(path), Ansi.FG.BLUE, Ansi.STYLE.BOLD)
        icon = "📁"
    elif path.is_file():
        colored_path = Ansi.format(str(path), Ansi.FG.CYAN)
        icon = "📄"
    else:
        colored_path = Ansi.format(str(path), Ansi.FG.GRAY)
        icon = "📍"

    parts = []
    if label:
        parts.append(Ansi.format(label + ":", Ansi.STYLE.BOLD))

    if exists is not None:
        indicator = Ansi.success("✓") if exists else Ansi.error("✗")
        parts.append(indicator)

    parts.append(f"{icon}  {colored_path}")
    print(" ".join(parts))


def command(cmd: str, /) -> None:
    """Print a command being executed."""
    print(Ansi.format(f"$ {cmd}", Ansi.FG.GRAY, Ansi.STYLE.ITALIC))


def header(text: str, /, char: str = "═", width: int | None = None) -> None:
    """Print a section header with decorative border.

    Args:
        text: Header text
        char: Character to use for border
        width: Width of header (auto-detect if None)
    """
    if width is None:
        width = max(len(text) + 4, 60)

    padding = (width - len(text) - 2) // 2
    header_line = char * width

    print()
    print(Ansi.format(header_line, Ansi.FG.CYAN))
    print(Ansi.format(f"{' ' * padding}{text}{' ' * padding}", Ansi.FG.CYAN, Ansi.STYLE.BOLD))
    print(Ansi.format(header_line, Ansi.FG.CYAN))
    print()


def separator(char: str = "─", width: int = 60) -> None:
    """Print a separator line."""
    print(Ansi.format(char * width, Ansi.FG.GRAY, Ansi.STYLE.DIM))


def table(
    headers: list[str],
    rows: list[list[str]],
    /,
    colors: list[Ansi.FG | None] | None = None,
) -> None:
    """Print a formatted table with borders.

    Args:
        headers: Column headers
        rows: Table rows
        colors: Optional colors for each column
    """
    if not headers or not rows:
        return

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    # Apply colors if provided
    if colors is None:
        colors = [None] * len(headers)

    def format_row(cells: list[str], is_header: bool = False) -> str:
        formatted = []
        for i, cell in enumerate(cells):
            width = col_widths[i]
            padded = str(cell).ljust(width)
            if is_header:
                formatted.append(Ansi.format(padded, Ansi.STYLE.BOLD))
            elif colors[i]:
                formatted.append(Ansi.format(padded, colors[i]))
            else:
                formatted.append(padded)
        return f"{Box.V} {f' {Box.V} '.join(formatted)} {Box.V}"

    # Top border
    top = Box.TL + Box.HD.join(Box.H * (w + 2) for w in col_widths) + Box.TR
    print(Ansi.format(top, Ansi.FG.GRAY))

    # Header
    print(format_row(headers, is_header=True))

    # Header separator
    sep = Box.VR + Box.X.join(Box.H * (w + 2) for w in col_widths) + Box.VL
    print(Ansi.format(sep, Ansi.FG.GRAY))

    # Rows
    for row in rows:
        print(format_row(row))

    # Bottom border
    bottom = Box.BL + Box.HU.join(Box.H * (w + 2) for w in col_widths) + Box.BR
    print(Ansi.format(bottom, Ansi.FG.GRAY))


def list_items(
    items: list[str],
    /,
    bullet: str = "•",
    color: Ansi.FG | None = None,
    indent: int = 0,
) -> None:
    """Print a bulleted list of items.

    Args:
        items: List items to display
        bullet: Bullet character
        color: Optional color for bullets
        indent: Indentation level
    """
    indent_str = "  " * indent
    colored_bullet = Ansi.format(bullet, color) if color else bullet

    for item in items:
        print(f"{indent_str}{colored_bullet} {item}")


def key_value(
    data: dict[str, t.Any],
    /,
    indent: int = 0,
    key_color: Ansi.FG = Ansi.FG.CYAN,
) -> None:
    """Print key-value pairs in a formatted style.

    Args:
        data: Dictionary of key-value pairs
        indent: Indentation level
        key_color: Color for keys
    """
    if not data:
        return

    max_key_len = max(len(str(k)) for k in data)
    indent_str = "  " * indent

    for key, value in data.items():
        colored_key = Ansi.format(str(key).ljust(max_key_len), key_color, Ansi.STYLE.BOLD)
        print(f"{indent_str}{colored_key} : {value}")


def progress_bar(
    current: int,
    total: int,
    /,
    width: int = 40,
    label: str = "",
    show_percent: bool = True,
) -> None:
    """Print a progress bar.

    Args:
        current: Current progress value
        total: Total/max value
        width: Width of the progress bar
        label: Optional label
        show_percent: Show percentage
    """
    percent = current / total if total > 0 else 0
    filled = int(width * percent)

    colored_bar = Ansi.format("█" * filled, Ansi.FG.GREEN) + Ansi.format(
        "░" * (width - filled), Ansi.FG.GRAY
    )

    parts = []
    if label:
        parts.append(label)
    parts.append(f"[{colored_bar}]")

    if show_percent:
        percent_str = f"{percent * 100:>5.1f}%"
        parts.append(Ansi.format(percent_str, Ansi.FG.CYAN, Ansi.STYLE.BOLD))

    parts.append(f"({current}/{total})")

    # Use \r to overwrite the line
    print("\r" + " ".join(parts), end="", flush=True)

    if current >= total:
        print()  # New line when complete


@contextlib.contextmanager
def loading(
    text: str = "Loading",
    /,
    success_text: str | None = None,
    error_text: str | None = None,
) -> t.Generator[halo.Halo, None, None]:
    """Context manager for showing a loading spinner.

    Args:
        text: Loading message
        success_text: Message to show on success
        error_text: Message to show on error

    Example:
        with loading("Fetching data"):
            fetch_data()
    """
    spinner.text = text
    spinner.start()

    try:
        yield spinner
        if success_text:
            spinner.succeed(success_text)
        else:
            spinner.succeed()
    except Exception as e:
        if error_text:
            spinner.fail(error_text)
        else:
            spinner.fail(f"{text} failed: {e}")
        raise
    finally:
        spinner.stop()


@contextlib.contextmanager
def section(title: str, /) -> t.Generator[None, None, None]:
    """Context manager for a named section with visual boundaries.

    Args:
        title: Section title

    Example:
        with section("Configuration"):
            key_value(config)
    """
    # Print section header
    width = 60
    border = Box.H * (width - len(title) - 4)
    header = f"{Box.TL}{Box.H * 2} {title} {border}{Box.TR}"
    print()
    print(Ansi.format(header, Ansi.FG.CYAN, Ansi.STYLE.BOLD))
    print()

    try:
        yield
    finally:
        # Print section footer
        footer = Box.BL + Box.H * (width - 2) + Box.BR
        print()
        print(Ansi.format(footer, Ansi.FG.CYAN))
        print()


def banner(text: str, /, subtitle: str | None = None, version: str | None = None) -> None:
    """Print an application banner/splash screen.

    Args:
        text: Main banner text
        subtitle: Optional subtitle
        version: Optional version string
    """
    width = 70

    print()
    print(Ansi.format(Box.TL + Box.H * (width - 2) + Box.TR, Ansi.FG.CYAN))

    # Main text - centered
    padding = (width - len(text) - 2) // 2
    line = f"{Box.V}{' ' * padding}{text}{' ' * (width - padding - len(text) - 2)}{Box.V}"
    print(Ansi.format(line, Ansi.FG.CYAN, Ansi.STYLE.BOLD))

    # Subtitle
    if subtitle:
        padding = (width - len(subtitle) - 2) // 2
        line = (
            f"{Box.V}{' ' * padding}{subtitle}{' ' * (width - padding - len(subtitle) - 2)}{Box.V}"
        )
        print(Ansi.format(line, Ansi.FG.GRAY))

    # Version in bottom right
    if version:
        version_text = f"v{version}"
        padding = width - len(version_text) - 4
        line = f"{Box.V}{' ' * padding}{version_text}  {Box.V}"
        print(Ansi.format(line, Ansi.FG.GRAY, Ansi.STYLE.DIM))

    print(Ansi.format(Box.BL + Box.H * (width - 2) + Box.BR, Ansi.FG.CYAN))
    print()


def tree(data: dict[str, t.Any], /, prefix: str = "") -> None:
    """Print a tree structure.

    Args:
        data: Nested dictionary to display as tree
        prefix: Internal use for recursion
    """
    items = list(data.items())

    for i, (key, value) in enumerate(items):
        is_last_item = i == len(items) - 1

        # Choose the right connector
        if is_last_item:
            connector = "└── "
            extension = "    "
        else:
            connector = "├── "
            extension = "│   "

        # Format and print the key
        colored_connector = Ansi.format(connector, Ansi.FG.GRAY)
        colored_key = Ansi.format(str(key), Ansi.FG.CYAN, Ansi.STYLE.BOLD)

        if isinstance(value, dict):
            print(f"{prefix}{colored_connector}{colored_key}")
            # Recurse for nested dicts
            tree(value, prefix + extension)
        elif isinstance(value, list):
            print(f"{prefix}{colored_connector}{colored_key}")
            for item in value:
                item_connector = Ansi.format("    • ", Ansi.FG.GRAY)
                print(f"{prefix}{extension}{item_connector}{item}")
        else:
            print(f"{prefix}{colored_connector}{colored_key}: {value}")


def confirm(prompt: str, /, default: bool = False) -> bool:
    """Ask for user confirmation with a yes/no prompt.

    Args:
        prompt: Question to ask
        default: Default answer if user just presses Enter

    Returns:
        True if user confirms, False otherwise
    """
    suffix = " [Y/n]" if default else " [y/N]"
    colored_prompt = Ansi.format(prompt + suffix + " ", Ansi.FG.YELLOW, Ansi.STYLE.BOLD)

    while True:
        response = input(colored_prompt).strip().lower()

        if not response:
            return default

        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        error("Please answer 'y' or 'n'")
