from __future__ import annotations

import numpy as np
import pytest

import qepy_pw.timing as timing_module
from qepy_pw.errors import QEInputError
from qepy_pw.memory import format_bytes
from qepy_pw.mpi import MPIContext
from qepy_pw.threads import hybrid_thread_count, thread_environment_warning
from qepy_pw.timing import TimingRegistry


def test_serial_mpi_context_collectives_preserve_values_and_shapes() -> None:
    mpi = MPIContext()
    grid = np.arange(24, dtype=float).reshape(2, 3, 4)
    assert mpi.slab(7) == slice(0, 7)
    np.testing.assert_array_equal(mpi.sum_array(grid), grid)
    np.testing.assert_array_equal(mpi.gather_z_slabs(grid, grid.shape), grid)
    np.testing.assert_array_equal(mpi.gather_z_slabs_root(grid, grid.shape), grid)
    np.testing.assert_array_equal(mpi.scatter_z_slabs_root(grid, grid.shape), grid)
    shared = mpi.shared_readonly(grid)
    assert not shared.flags.writeable
    np.testing.assert_array_equal(shared, grid)


@pytest.mark.parametrize(
    ("value", "formatted"),
    [(0, "0.00 B"), (1023, "1023.00 B"), (1024, "1.00 KiB"), (3 * 1024**2, "3.00 MiB")],
)
def test_memory_byte_formatting(value: int, formatted: str) -> None:
    assert format_bytes(value) == formatted


def test_thread_count_honors_qepy_precedence_and_openmp_lists(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8,4")
    assert hybrid_thread_count() == 8
    monkeypatch.setenv("QEPY_NUM_THREADS", "3")
    assert hybrid_thread_count() == 3
    monkeypatch.setenv("QEPY_NUM_THREADS", "0")
    with pytest.raises(QEInputError, match="positive integer"):
        hybrid_thread_count()


def test_thread_environment_warning_exposes_conflicting_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("QEPY_NUM_THREADS", "1")
    assert thread_environment_warning() == (
        "QEPY_NUM_THREADS=1 overrides OMP_NUM_THREADS=4; "
        "using 1 thread(s) per MPI process"
    )
    monkeypatch.setenv("QEPY_NUM_THREADS", "4")
    assert thread_environment_warning() is None


def test_timing_registry_accumulates_calls_and_snapshot_is_independent(monkeypatch) -> None:
    cpu = iter([1.0, 1.25, 2.0, 2.5])
    wall = iter([10.0, 10.5, 20.0, 21.0])
    monkeypatch.setattr(timing_module.time, "process_time", lambda: next(cpu))
    monkeypatch.setattr(timing_module.time, "perf_counter", lambda: next(wall))
    registry = TimingRegistry()
    with registry.measure("fft"):
        pass
    started = registry.start()
    registry.stop("fft", started, calls=3)
    snapshot = registry.snapshot()
    assert snapshot["fft"].cpu_seconds == pytest.approx(0.75)
    assert snapshot["fft"].wall_seconds == pytest.approx(1.5)
    assert snapshot["fft"].calls == 4
    registry.entries["fft"].calls += 1
    assert snapshot["fft"].calls == 4


def test_timing_registry_counts_batched_logical_operations(monkeypatch) -> None:
    cpu = iter([3.0, 3.5])
    wall = iter([30.0, 31.25])
    monkeypatch.setattr(timing_module.time, "process_time", lambda: next(cpu))
    monkeypatch.setattr(timing_module.time, "perf_counter", lambda: next(wall))
    registry = TimingRegistry()

    with registry.measure("fftw", calls=8):
        pass

    assert registry.entries["fftw"].cpu_seconds == pytest.approx(0.5)
    assert registry.entries["fftw"].wall_seconds == pytest.approx(1.25)
    assert registry.entries["fftw"].calls == 8

