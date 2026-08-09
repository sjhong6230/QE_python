from __future__ import annotations

import argparse

import pytest

from qepy_pw.cli_options import INPUT_FILE_OPTIONS, add_input_file_argument
from qepy_pw.pw.cli import build_parser


@pytest.mark.parametrize("option", INPUT_FILE_OPTIONS)
@pytest.mark.parametrize("joined", [False, True])
def test_every_executable_input_file_alias(option: str, joined: bool) -> None:
    arguments = [f"{option}=calculation.in"] if joined else [option, "calculation.in"]

    parser = argparse.ArgumentParser()
    add_input_file_argument(parser)
    assert parser.parse_args(arguments).input_file == "calculation.in"
    assert build_parser().parse_args(arguments).input_file == "calculation.in"
