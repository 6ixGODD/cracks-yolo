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


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    from cracks_yolo.cli import annotator
    from cracks_yolo.cli import dataset

    dataset.register(subparsers)
    annotator.register(subparsers)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cracks YOLO Command Line Interface",
        prog="cracks-yolo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="For more information, visit: https://github.com/6ixGODD/cracks-yolo",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed tracebacks",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands",
        required=True,
    )

    register_commands(subparsers)

    def print_help(args: argparse.Namespace) -> None:
        parser.print_help()
        if args.command is None:
            print()
            display.warning("Please specify a command. Use -h for help.")

    parser.set_defaults(func=print_help)

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
