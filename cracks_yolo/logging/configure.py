"""loguru logger setup: JSONL file sink + stderr sink.

Usage:
    from cracks_yolo.logging.configure import configure_logger
    configure_logger(output_dir=Path("output/run1"))
    logger.bind(**<TrainStepLog-shaped dict>).info("step done")
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger
    from loguru import Message


def _make_jsonl_sink(path: Path) -> Callable[[Message], None]:
    """Return a sink callable that writes one JSON line per record to ``path``."""
    file = path.open("a", encoding="utf-8")

    def sink(message: Message) -> None:
        record = message.record
        payload: dict[str, Any] = {
            "level": record["level"].name,
            "message": record["message"],
            "timestamp": record["time"].isoformat(),
        }
        payload.update(record.get("extra", {}))
        file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        file.flush()

    sink._file = file  # type: ignore[attr-defined]
    return sink


def configure_logger(
    output_dir: Path,
    level: str = "INFO",
    *,
    stderr: bool = True,
) -> Logger:
    """Configure loguru with a JSONL file sink and (optional) stderr sink.

    Args:
        output_dir: Directory to write ``run.log.jsonl``. Created if missing.
        level: Minimum log level (e.g. ``"INFO"``, ``"DEBUG"``).
        stderr: If True, also install a human-readable stderr sink.

    Returns:
        The configured loguru logger (for chaining).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sink=_make_jsonl_sink(output_dir / "run.log.jsonl"),
        level=level,
        enqueue=False,
    )
    if stderr:
        logger.add(
            sink=sys.stderr,
            level=level,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{message}</cyan>"
            ),
            enqueue=False,
        )
    return logger
