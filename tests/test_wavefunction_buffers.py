from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qepy_pw.pw.buffers import (
    WavefunctionBuffer,
    prepare_io_directories,
    resolve_outdir,
    resolve_wfcdir,
    uses_disk_wavefunction_buffer,
)


def _pw(control: dict[str, object]):
    return SimpleNamespace(control=control)


def test_io_directories_are_created_and_wfcdir_defaults_to_outdir(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "out" / "nested"
    defaulted = _pw({
        "outdir": str(outdir), "prefix": "si", "disk_io": "none"
    })
    resolved_outdir, resolved_wfcdir = prepare_io_directories(defaulted)
    assert resolved_outdir == outdir.resolve()
    assert resolved_wfcdir == outdir.resolve()
    assert resolve_outdir(defaulted).is_dir()

    separate = tmp_path / "scratch" / "wfc"
    explicit = _pw({
        "outdir": str(outdir), "wfcdir": str(separate), "prefix": "si"
    })
    prepare_io_directories(explicit)
    assert resolve_wfcdir(explicit) == separate.resolve()
    assert separate.is_dir()


def test_medium_and_high_follow_qe_working_buffer_policy() -> None:
    assert not uses_disk_wavefunction_buffer("none", 4)
    assert not uses_disk_wavefunction_buffer("low", 4)
    assert not uses_disk_wavefunction_buffer("medium", 1)
    assert uses_disk_wavefunction_buffer("medium", 2)
    assert uses_disk_wavefunction_buffer("high", 1)


def test_wavefunction_buffer_round_trips_fixed_binary_records(
    tmp_path: Path,
) -> None:
    buffer = WavefunctionBuffer(tmp_path, "silicon", [2, 3], 2)
    first = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    second = (
        np.arange(6).reshape(3, 2)
        + 1j * np.arange(6, 12).reshape(3, 2)
    )

    buffer.write(0, first)
    buffer.write(1, second)

    assert buffer.path == tmp_path / "silicon.wfc"
    assert buffer.path.stat().st_size == 2 * 3 * 2 * 16
    assert buffer.path.read_bytes()[:8] != b"\x89HDF\r\n\x1a\n"
    np.testing.assert_array_equal(buffer[0], first)
    np.testing.assert_array_equal(buffer[1], second)


def test_parallel_working_file_uses_qe_rank_suffix(tmp_path: Path) -> None:
    buffer = WavefunctionBuffer(
        tmp_path, "pwscf", [1], 1, rank=2, processes=4
    )
    assert buffer.path.name == "pwscf.wfc3"
