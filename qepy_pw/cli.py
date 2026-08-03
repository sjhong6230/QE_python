"""Command-line interface retaining the familiar ``pw.x -in file`` shape."""

from __future__ import annotations

import argparse
import io
import sys

from .errors import QEInputError, UnsupportedFeatureError
from .input import read_pw_input
from .mpi import MPIContext
from .output import format_footer, format_header, format_iteration, format_setup
from .save import write_qe_save
from .scf import SCFIteration, SCFSetup, run_scf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pw.py", description="Python scalar-SCF port of QE pw.x")
    parser.add_argument("-in", "-inp", "--input", dest="input_file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mpi = MPIContext.world()
        if args.input_file:
            pw = read_pw_input(args.input_file)
        else:
            input_text = sys.stdin.read() if mpi.is_root else None
            input_text = mpi.broadcast(input_text)
            pw = read_pw_input(io.StringIO(str(input_text)))
        if mpi.is_root:
            print(format_header(pw), end="", flush=True)

        def report(kind: str, payload: SCFSetup | SCFIteration) -> None:
            if kind == "setup":
                text = format_setup(payload)  # type: ignore[arg-type]
            else:
                text = format_iteration(payload)  # type: ignore[arg-type]
            if mpi.is_root:
                print(text, end="", flush=True)

        result = run_scf(pw, progress=report, mpi=mpi)
        save_directory = write_qe_save(pw, result, mpi=mpi)
        if mpi.is_root:
            if save_directory is not None:
                print(
                    f"\n     Writing XML/HDF5 data to output data dir "
                    f"{save_directory} :\n",
                    flush=True,
                )
            print(format_footer(pw, result), end="", flush=True)
        return 0 if result.converged else 2
    except (QEInputError, UnsupportedFeatureError, OSError) as exc:
        rank = locals().get("mpi", MPIContext()).rank
        if rank == 0:
            print(
                f"\n     Error in routine pw.py:\n     {exc}\n",
                file=sys.stderr,
            )
        return 1
