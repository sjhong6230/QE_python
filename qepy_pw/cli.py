"""Command-line interface retaining the familiar ``pw.x -in file`` shape."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from .input import PWInput
    from .mpi import MPIContext
    from .scf import ProgressCallback


class _Arguments:
    __slots__ = ("input_file",)

    def __init__(self, input_file: str | None) -> None:
        self.input_file = input_file


class _Parser:
    """The complete pw.py CLI grammar without argparse's per-rank import."""

    _usage = "usage: pw.py [-h] [-in INPUT_FILE]"

    def parse_args(self, argv: list[str] | None = None) -> _Arguments:
        tokens = list(sys.argv[1:] if argv is None else argv)
        input_file: str | None = None
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in {"-h", "--help"}:
                print(
                    self._usage
                    + "\n\nPython scalar pw.x port (SCF, NSCF, bands)\n\n"
                    + "options:\n"
                    + "  -h, --help            show this help message and exit\n"
                    + "  -in, -inp, --input INPUT_FILE"
                )
                raise SystemExit(0)
            if token.startswith("--input="):
                input_file = token.split("=", 1)[1]
                index += 1
                continue
            if token in {"-in", "-inp", "--input"}:
                if index + 1 >= len(tokens):
                    print(
                        self._usage
                        + f"\npw.py: error: argument {token}: expected one argument",
                        file=sys.stderr,
                    )
                    raise SystemExit(2)
                input_file = tokens[index + 1]
                index += 2
                continue
            print(
                self._usage + f"\npw.py: error: unrecognized arguments: {token}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return _Arguments(input_file)


def build_parser() -> _Parser:
    return _Parser()


def _read_distributed_input(
    input_file: str | None,
    mpi: MPIContext,
    stdin: TextIO | None = None,
) -> PWInput:
    """Parse and symmetry-reduce one shared input once on the root rank."""
    payload: tuple[bool, object] | None = None
    if mpi.is_root:
        try:
            import io
            from pathlib import Path

            from .input import read_pw_input

            if input_file:
                parsed = read_pw_input(Path(input_file))
            else:
                stream = sys.stdin if stdin is None else stdin
                parsed = read_pw_input(io.StringIO(stream.read()))
            payload = (True, parsed)
        except Exception as exc:
            # Broadcast failures too, otherwise non-root ranks would remain
            # blocked in bcast while rank zero exits through main().
            payload = (False, exc)
    payload = mpi.broadcast(payload)
    assert isinstance(payload, tuple) and len(payload) == 2
    succeeded, value = payload
    if not succeeded:
        assert isinstance(value, Exception)
        raise value
    return value


def _root_progress_reporter() -> ProgressCallback:
    """Build the rank-zero callback used for incremental QE-shaped output."""
    from .output import format_progress

    def report(kind, payload) -> None:
        print(format_progress(kind, payload), end="", flush=True)

    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .errors import QEInputError, UnsupportedFeatureError, format_qe_error
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
            print(format_qe_error(exc), end="", file=sys.stderr)
        return 1
