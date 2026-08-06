"""Command-line interface retaining the familiar ``pw.x -in file`` shape."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from .input import PWInput
    from .mpi import MPIContext
    from .scf import ProgressCallback


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(prog="pw.py", description="Python scalar-SCF port of QE pw.x")
    parser.add_argument("-in", "-inp", "--input", dest="input_file")
    return parser


def _read_distributed_input(
    input_file: str | None,
    mpi: MPIContext,
    stdin: TextIO | None = None,
) -> PWInput:
    """Parse one shared input without reading replicated MPI stdin streams."""
    from .input import read_pw_input

    if input_file:
        return read_pw_input(Path(input_file))
    stream = sys.stdin if stdin is None else stdin
    input_text = stream.read() if mpi.is_root else None
    input_text = mpi.broadcast(input_text)
    return read_pw_input(io.StringIO(str(input_text)))


def _root_progress_reporter() -> ProgressCallback:
    """Build the rank-zero callback used for incremental QE-shaped output."""
    from .output import format_progress

    def report(kind, payload) -> None:
        print(format_progress(kind, payload), end="", flush=True)

    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .errors import QEInputError, UnsupportedFeatureError
    from .memory import trim_allocator
    from .mpi import MPIContext

    mpi: MPIContext | None = None
    try:
        mpi = MPIContext.world()
        pw = _read_distributed_input(args.input_file, mpi)
        if mpi.is_root:
            from .output import format_header

            print(format_header(pw), end="", flush=True)
            # Header formatting reads UPF metadata before the SCF owns its
            # persistent pseudopotentials. Return those temporary XML/radial
            # allocations to glibc before the calculation begins.
            trim_allocator()

        from .scf import run_scf

        progress = _root_progress_reporter() if mpi.is_root else None
        result = run_scf(pw, progress=progress, mpi=mpi)
        save_directory = None
        if str(pw.control.get("disk_io", "low")).strip().lower() != "none":
            from .save import write_qe_save

            save_directory = write_qe_save(pw, result, mpi=mpi)
        if mpi.is_root:
            from .output import format_footer

            if save_directory is not None:
                print(
                    f"\n     Writing XML/HDF5 data to output data dir "
                    f"{save_directory} :\n",
                    flush=True,
                )
            print(format_footer(pw, result), end="", flush=True)
        return 0 if result.converged else 2
    except (QEInputError, UnsupportedFeatureError, OSError) as exc:
        rank = mpi.rank if mpi is not None else 0
        if rank == 0:
            print(
                f"\n     Error in routine pw.py:\n     {exc}\n",
                file=sys.stderr,
            )
        return 1
