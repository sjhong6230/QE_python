"""Correctness checks for QE-style stick ownership and FFTW-MPI support."""

from __future__ import annotations

import numpy as np
import pytest

from qepy_pw import _native_fft
from qepy_pw.basis import (
    FFTGridDescriptor,
    FFTScratchPool,
    GammaWavefunctionMap,
    LocalPotentialWorkspace,
)
from qepy_pw.mpi import MPIContext


def _sphere_indices(shape: tuple[int, int, int], radius: float) -> np.ndarray:
    axes = [np.rint(np.fft.fftfreq(n) * n).astype(np.int32) for n in shape]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij", sparse=True)
    mask = gx * gx + gy * gy + gz * gz <= radius * radius
    slots = np.column_stack(np.nonzero(mask))
    return np.column_stack(
        [axes[axis][slots[:, axis]] for axis in range(3)]
    ).astype(np.int32, copy=False)


def test_stick_descriptor_has_one_persistent_owner() -> None:
    shape = (24, 24, 24)
    indices = _sphere_indices(shape, 8.0)
    descriptor = FFTGridDescriptor.build(indices, shape, 4)

    owned = np.concatenate(descriptor.sticks_by_rank)
    np.testing.assert_array_equal(np.sort(owned), np.arange(len(descriptor.sticks)))
    for rank, sticks in enumerate(descriptor.sticks_by_rank):
        np.testing.assert_array_equal(
            descriptor.stick_owners[sticks],
            np.full(len(sticks), rank, dtype=np.int32),
        )

    pairs = np.column_stack(
        (indices[:, 0] % shape[0], indices[:, 1] % shape[1])
    )
    np.testing.assert_array_equal(
        descriptor.sticks[descriptor.row_stick_indices], pairs
    )


def test_fftw_mpi_collective_plan_roundtrip() -> None:
    if not _native_fft.fftw_mpi_available():
        pytest.skip("extension was built without FFTW-MPI")
    mpi = MPIContext.world()
    dimensions = (8, 6, 4)
    allocation, local_n0, local_start = _native_fft.fftw_mpi_layout(
        dimensions, mpi.comm
    )
    active = local_n0 * dimensions[1] * dimensions[2]
    values = np.zeros(allocation, dtype=np.complex128)
    global_offset = local_start * dimensions[1] * dimensions[2]
    points = np.arange(active, dtype=np.float64) + global_offset
    values[:active] = np.sin(0.17 * points) + 1j * np.cos(0.11 * points)
    reference = values[:active].copy()

    plan = _native_fft.NativeFFTWMPIPlan(
        values, dimensions, mpi.comm, 0, 64, 1
    )
    plan.execute("forward")
    plan.execute("backward")
    values[:active] /= np.prod(dimensions)
    local_error = float(np.max(np.abs(values[:active] - reference)))
    assert mpi.max_scalar(local_error) < 5.0e-13


def test_distributed_band_tiles_bound_stick_scratch() -> None:
    mpi = MPIContext.world()
    if mpi.size == 1:
        pytest.skip("distributed band tiling requires more than one rank")
    shape = (12, 10, 8)
    if mpi.size > shape[2]:
        pytest.skip("legacy slab comparison requires ranks <= Nz")
    indices = _sphere_indices(shape, 3.5)
    descriptor = FFTGridDescriptor.build(
        indices, shape, mpi.size, local_rank=mpi.rank
    )
    tiled_pool = FFTScratchPool()
    full_pool = FFTScratchPool()
    tiled = LocalPotentialWorkspace(
        indices,
        shape,
        mpi=mpi,
        descriptor=descriptor,
        scratch_pool=tiled_pool,
        distributed_fft_batch_size=1,
    )
    full = LocalPotentialWorkspace(
        indices,
        shape,
        mpi=mpi,
        descriptor=descriptor,
        scratch_pool=full_pool,
    )
    bands = 5
    rows = tiled.local_plane_wave_indices.astype(np.float64)
    band = np.arange(bands, dtype=np.float64)
    coefficients = np.asfortranarray(
        np.sin(0.11 * rows[:, None] + band[None, :])
        + 1j * np.cos(0.07 * rows[:, None] - band[None, :])
    )
    slab = mpi.slab(shape[2])
    potential = np.ones(
        (slab.stop - slab.start, shape[0], shape[1]), dtype=np.float64
    )
    tiled_result = tiled.apply(
        potential, coefficients, native_potential_layout=True
    )
    full_result = full.apply(
        potential, coefficients, native_potential_layout=True
    )
    np.testing.assert_allclose(tiled_result, full_result, atol=8.0e-13)
    assert tiled_pool.nbytes < full_pool.nbytes


def test_distributed_gamma_half_map_roundtrip_metric_and_density() -> None:
    mpi = MPIContext.world()
    shape = (10, 8, 6)
    indices = _sphere_indices(shape, 3.5)
    descriptor = FFTGridDescriptor.build(
        indices, shape, mpi.size, local_rank=mpi.rank
    )
    workspace = LocalPotentialWorkspace(
        indices, shape, mpi=mpi, descriptor=descriptor
    )
    gamma = GammaWavefunctionMap(indices, workspace)
    workspace.gamma_map = gamma
    bands = 3
    rows = gamma.local_half_indices.astype(float)
    columns = np.arange(bands, dtype=float)
    half = np.asfortranarray(
        np.sin(0.17 * rows[:, None] + 0.3 * columns[None, :])
        + 1j * np.cos(0.13 * rows[:, None] - 0.2 * columns[None, :])
    )
    half[gamma.self_conjugate] = half[gamma.self_conjugate].real
    full = gamma.expand(half)
    np.testing.assert_allclose(gamma.compress(full), half, atol=2.0e-14)
    expected_gram = mpi.sum_array(full.conj().T @ full)
    np.testing.assert_allclose(
        gamma.gram(half, half), np.real(expected_gram), atol=2.0e-12
    )

    first, second = gamma.unpack_two(gamma.pack_two(half[:, 0], half[:, 1]))
    np.testing.assert_allclose(first[:, 0], half[:, 0], atol=2.0e-14)
    np.testing.assert_allclose(second[:, 0], half[:, 1], atol=2.0e-14)

    slab = mpi.slab(shape[2])
    density = np.zeros((shape[0], shape[1], slab.stop - slab.start))
    weights = np.asarray([0.4, 0.7, 0.2])
    gamma.accumulate_density(density, half, weights)
    real = workspace.coefficients_to_grid(full)
    expected_density = np.sum(
        np.abs(real) ** 2 * weights[None, None, None, :], axis=-1
    )
    np.testing.assert_allclose(density, expected_density, atol=5.0e-11)


def test_distributed_gamma_half_save_is_qe_compatible(
    tmp_path, monkeypatch
) -> None:
    mpi = MPIContext.world()
    if mpi.size == 1:
        pytest.skip("distributed Gamma save requires multiple ranks")
    from pathlib import Path

    import h5py

    from qepy_pw.input import read_pw_input
    from qepy_pw.pw.save import write_qe_save
    from qepy_pw.scf import run_scf

    root_directory = mpi.comm.bcast(
        str(tmp_path) if mpi.is_root else None, root=0
    )
    source = Path("tests/qe_reference/upstream/pw_scf/scf-gamma.in").resolve()
    pw = read_pw_input(source)
    pw.control.update(
        {
            "outdir": root_directory,
            "prefix": "mpi-gamma-half",
            "pseudo_dir": str((source.parents[1] / "pseudo").resolve()),
            "disk_io": "medium",
            "tstress": False,
        }
    )
    monkeypatch.delenv("QEPY_GAMMA_MODE", raising=False)
    result = run_scf(pw)
    save = write_qe_save(pw, result, mpi=mpi)
    mpi.comm.Barrier()
    if mpi.is_root:
        assert save is not None
        with h5py.File(save / "wfc1.hdf5", "r") as h5:
            assert str(h5.attrs["gamma_only"]).upper() == ".TRUE."
            assert (
                len(h5["MillerIndices"]) * 2 - 1
                == result.plane_waves_per_k[0]
            )
