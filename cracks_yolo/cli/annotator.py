from __future__ import annotations

import argparse
import typing as t

from cracks_yolo.cli.args import BaseArgs

if t.TYPE_CHECKING:
    from argparse import _SubParsersAction


class Args(BaseArgs):
    """Arguments for annotator command."""

    def run(self) -> None:
        """Launch the annotator application."""
        from cracks_yolo.annotator import AnnotatorApp

        app = AnnotatorApp()
        app.run()

    @classmethod
    def build_args(cls, parser: argparse.ArgumentParser) -> None:
        """Build command-line arguments."""
        pass  # No arguments needed for annotator


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the annotator subcommand."""
    parser = subparser.add_parser(
        "annotator",
        help="Launch the dataset annotation viewer",
        description="Interactive GUI tool for viewing and managing datasets",
    )
    Args.build_args(parser)
    parser.set_defaults(func=Args.func)
