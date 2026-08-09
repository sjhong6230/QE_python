"""Quantum ESPRESSO-style working wavefunction buffers.

QE keeps processor-local, direct-access wavefunction records in ``wfcdir``
during a calculation.  These files are deliberately separate from the
portable, collected ``wfcN.hdf5`` files written below ``outdir/prefix.save``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import os
from pathlib import Path

import numpy as np

from ..errors import QEInputError
from ..mpi import MPIContext
from .input import PWInput


def resolve_prefix(pw: PWInput) -> str:
    """Return a validated QE file prefix."""

    prefix = str(pw.control.get("prefix", "pwscf")).strip() or "pwscf"
    if (
        prefix in {".", ".."}
        or "/" in prefix
        or "\\" in prefix
        or Path(prefix).name != prefix
    ):
        raise QEInputError(
            "CONTROL prefix must be a file-name prefix, not a path"
        )
    return prefix


def _resolve_directory(value: object) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    directory = Path(expanded)
    if not directory.is_absolute():
        directory = Path.cwd() / directory
    return directory.resolve()


def resolve_outdir(pw: PWInput) -> Path:
    """Resolve QE's temporary/output directory for an input."""

    raw_outdir = pw.control.get("outdir")
    if raw_outdir is None or not str(raw_outdir).strip():
        raw_outdir = os.environ.get("ESPRESSO_TMPDIR", ".")
    return _resolve_directory(raw_outdir)


def resolve_wfcdir(pw: PWInput) -> Path:
    """Resolve ``CONTROL/wfcdir``, which defaults to ``outdir`` in QE."""

    raw_wfcdir = pw.control.get("wfcdir")
    if raw_wfcdir is None or not str(raw_wfcdir).strip():
        return resolve_outdir(pw)
    return _resolve_directory(raw_wfcdir)


def prepare_io_directories(
    pw: PWInput, mpi: MPIContext | None = None
) -> tuple[Path, Path]:
    """Create ``outdir`` and ``wfcdir`` at calculation initialization."""

    mpi = mpi if mpi is not None else MPIContext()
    outdir = resolve_outdir(pw)
    wfcdir = resolve_wfcdir(pw)
    local_error: OSError | None = None
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        wfcdir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        local_error = exc
    failed = bool(mpi.max_scalar(float(local_error is not None)))
    if failed:
        detail = (
            str(local_error)
            if local_error is not None
            else "a temporary directory is unavailable on another MPI rank"
        )
        raise QEInputError(
            f"cannot create temporary directory: {detail}",
            routine="check_tempdir",
        ) from local_error
    mpi.barrier()
    return outdir, wfcdir


def uses_disk_wavefunction_buffer(disk_io: str, local_kpoints: int) -> bool:
    """Match QE's ``io_level`` rule for the working wavefunction buffer."""

    level = str(disk_io).strip().lower()
    return level == "high" or (level == "medium" and local_kpoints > 1)


class WavefunctionBuffer(Sequence[np.ndarray]):
    """Fixed-record binary storage for rank-local wavefunction matrices.

    Each record is stored band-major as ``complex128`` and padded to the
    largest local plane-wave row count.  This is the same direct-record model
    as QE's unformatted ``prefix.wfc[rank]`` working file; the file is not a
    portable interchange format and intentionally has no HDF5 dependency.
    """

    def __init__(
        self,
        directory: Path,
        prefix: str,
        row_counts: Sequence[int],
        number_of_bands: int,
        *,
        rank: int = 0,
        processes: int = 1,
        truncate: bool = True,
    ) -> None:
        self.row_counts = tuple(int(value) for value in row_counts)
        self.number_of_bands = int(number_of_bands)
        if not self.row_counts or min(self.row_counts) < 0:
            raise ValueError("wavefunction row counts must be nonnegative")
        if self.number_of_bands < 1:
            raise ValueError("number_of_bands must be positive")
        self.record_rows = max(self.row_counts)
        self.record_bytes = (
            self.record_rows
            * self.number_of_bands
            * np.dtype(np.complex128).itemsize
        )
        suffix = "" if processes == 1 else str(rank + 1)
        self.path = Path(directory) / f"{prefix}.wfc{suffix}"
        self._written = [False] * len(self.row_counts)
        if truncate:
            try:
                with self.path.open("w+b") as stream:
                    stream.truncate(self.record_bytes * len(self.row_counts))
            except OSError as exc:
                raise QEInputError(
                    f"cannot open wavefunction buffer {self.path}: {exc}",
                    routine="open_buffer",
                ) from exc

    def __len__(self) -> int:
        return len(self.row_counts)

    def __getitem__(self, index: int | slice) -> np.ndarray | list[np.ndarray]:
        if isinstance(index, slice):
            return [
                self[position]
                for position in range(*index.indices(len(self)))
            ]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if not self._written[index]:
            raise QEInputError(
                f"wavefunction record {index + 1} has not been written",
                routine="get_buffer",
            )
        try:
            with self.path.open("rb") as stream:
                stream.seek(index * self.record_bytes)
                payload = stream.read(self.record_bytes)
        except OSError as exc:
            raise QEInputError(
                f"cannot read wavefunction buffer {self.path}: {exc}",
                routine="get_buffer",
            ) from exc
        if len(payload) != self.record_bytes:
            raise QEInputError(
                f"short wavefunction record in {self.path}", routine="get_buffer"
            )
        stored = np.frombuffer(payload, dtype=np.complex128).reshape(
            self.number_of_bands, self.record_rows
        )
        return np.ascontiguousarray(stored[:, : self.row_counts[index]].T)

    def __iter__(self) -> Iterator[np.ndarray]:
        for index in range(len(self)):
            yield self[index]

    def has_record(self, index: int) -> bool:
        return self._written[index]

    def write(self, index: int, coefficients: np.ndarray) -> None:
        if not 0 <= index < len(self):
            raise IndexError(index)
        matrix = np.asarray(coefficients, dtype=np.complex128)
        expected = (self.row_counts[index], self.number_of_bands)
        if matrix.shape != expected:
            raise QEInputError(
                f"wavefunction record has shape {matrix.shape}, expected {expected}",
                routine="save_buffer",
            )
        record = np.zeros(
            (self.number_of_bands, self.record_rows), dtype=np.complex128
        )
        record[:, : self.row_counts[index]] = matrix.T
        try:
            with self.path.open("r+b") as stream:
                stream.seek(index * self.record_bytes)
                stream.write(record.tobytes(order="C"))
                stream.flush()
        except OSError as exc:
            raise QEInputError(
                f"cannot write wavefunction buffer {self.path}: {exc}",
                routine="save_buffer",
            ) from exc
        self._written[index] = True


__all__ = [
    "WavefunctionBuffer",
    "prepare_io_directories",
    "resolve_outdir",
    "resolve_prefix",
    "resolve_wfcdir",
    "uses_disk_wavefunction_buffer",
]
