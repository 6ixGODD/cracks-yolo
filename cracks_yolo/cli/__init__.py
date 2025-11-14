"""Main CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from cracks_yolo import __version__
from cracks_yolo.cli.helper import display
from cracks_yolo.exceptions import CracksYoloError


def main() -> int:
    """Main entry point."""
    try:
        args = parse_args()

        if hasattr(args, "verbose") and args.verbose:
            logging.basicConfig(level=logging.DEBUG)  # Enable debug logging

        # Execute command
        args.func(args)
        return 0

    except KeyboardInterrupt:
        print()
        display.warning("Cancelled by user")
        return 130

    except CracksYoloError as e:
        display.show_error(e, verbose="--verbose" in sys.argv or "-v" in sys.argv)
        return e.code

    except Exception as e:
        display.show_error(e, verbose="--verbose" in sys.argv or "-v" in sys.argv)
        return 1


def parse_args() -> argparse.Namespace:
    from cracks_yolo.cli import dataset

    parser = argparse.ArgumentParser(
        prog="cracks-yolo",
        description="Cracks YOLO - Dataset management toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed error messages",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True,
    )

    # Register commands
    dataset.register(subparsers)

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
