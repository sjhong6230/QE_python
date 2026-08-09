"""Command-line options shared by QE-compatible executables."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


INPUT_FILE_OPTIONS = ("-i", "-in", "-inp", "-input", "--input")
INPUT_FILE_OPTION_SET = frozenset(INPUT_FILE_OPTIONS)


def add_input_file_argument(parser: argparse.ArgumentParser) -> None:
    """Add every input-file spelling accepted by QE command-line tools."""
    parser.add_argument(*INPUT_FILE_OPTIONS, dest="input_file")
