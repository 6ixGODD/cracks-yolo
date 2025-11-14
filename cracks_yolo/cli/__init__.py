from __future__ import annotations

import argparse

from cracks_yolo.exceptions import CracksYoloError


def main() -> int:
    try:
        _main()
    except KeyboardInterrupt:
        return 130
    except CracksYoloError as e:
        return e.code
    except Exception:
        return 1
    return 0


def _main() -> None:
    args = parse_args()
    args.func(args)


def parse_args() -> argparse.Namespace:
    from cracks_yolo.cli import dataset

    parser = argparse.ArgumentParser(
        description="Cracks YOLO Command Line Interface",
        prog="python -m cracks_yolo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    dataset.register(subparsers)

    def print_help(args: argparse.Namespace) -> None:
        parser.print_help()
        if args.subcommand is None:
            print(r"\Please specify a subcommand. Use -h for help.")

    parser.set_defaults(func=print_help)
    return parser.parse_args()
