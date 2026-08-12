"""MPI correctness tests for pencil decomposition and FFT task groups."""

from __future__ import annotations

import numpy as np

from qepy_pw.basis import FFTGridDescriptor, LocalPotentialWorkspace
from qepy_pw.fft_engine import FFTTaskTopology, PencilFFT3D
from qepy_pw.mpi import MPIContext


def _global_coefficients(
    shape: tuple[int, int, int], bands: int, task_group: int
) -> np.ndarray:
    points = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    band = np.arange(bands, dtype=np.float64)
    return (
        np.sin(0.071 * points[..., None] + 0.13 * band + task_group)
        + 1j * np.cos(0.053 * points[..., None] - 0.17 * band + task_group)
    )


def _local_z(global_values: np.ndarray, fft: PencilFFT3D) -> np.ndarray:
    return np.ascontiguousarray(
        global_values[fft.x_initial, fft.y_initial, :, :].transpose(2, 0, 1, 3)
    )


def _local_x(global_values: np.ndarray, fft: PencilFFT3D) -> np.ndarray:
    return np.ascontiguousarray(
        global_values[:, fft.y_final, fft.z_final, :]
    )


def test_pencil_roundtrip_and_local_potential() -> None:
    mpi = MPIContext.world()
    shape = (8, 10, 12)
    topology = FFTTaskTopology.build(mpi.comm, shape)
    fft = PencilFFT3D(shape, topology, threads=1)
    bands = 3
    coefficients = _global_coefficients(shape, bands, topology.task_group)
    local_coefficients = _local_z(coefficients, fft)

    roundtrip = fft.forward(fft.backward(local_coefficients)).copy()
    local_error = float(np.max(np.abs(roundtrip - local_coefficients)))
    assert mpi.max_scalar(local_error) < 8.0e-13

    x, y, z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )
    potential = 0.7 + 0.01 * x + 0.02 * y - 0.015 * z
    local_potential = potential[:, fft.y_final, fft.z_final]
    observed = fft.apply_local_potential(
        local_coefficients, local_potential, band_tile=1
    )
    real = np.fft.ifftn(coefficients, axes=(0, 1, 2)) * np.prod(shape)
    expected = np.fft.fftn(
        real * potential[..., None], axes=(0, 1, 2)
    ) / np.prod(shape)
    local_expected = _local_z(expected, fft)
    local_error = float(np.max(np.abs(observed - local_expected)))
    assert mpi.max_scalar(local_error) < 2.0e-11


def test_z_slab_x_pencil_field_redistribution_roundtrip() -> None:
    mpi = MPIContext.world()
    shape = (7, 10, 9)
    topology = FFTTaskTopology.build(mpi.comm, shape)
    fft = PencilFFT3D(shape, topology, threads=1)
    z_slab = mpi.slab(shape[2])
    x, y, z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(z_slab.start, z_slab.stop),
        indexing="ij",
    )
    local = np.ascontiguousarray(0.3 * x - 0.2 * y + 0.7 * z)
    pencil = fft.slab_to_x_pencil(local)
    restored = fft.x_pencil_to_slab(pencil)
    np.testing.assert_array_equal(restored, local)


def test_production_workspace_uses_sparse_pencil_boundary() -> None:
    mpi = MPIContext.world()
    shape = (6, 8, 10)
    topology = FFTTaskTopology.build(mpi.comm, shape)
    fft = PencilFFT3D(shape, topology, threads=1)
    indices = np.indices(shape, dtype=np.int32).reshape(3, -1).T
    descriptor = FFTGridDescriptor.build(
        indices,
        shape,
        mpi.size,
        local_rank=mpi.rank,
        decomposition="pencil",
        process_grid=topology.process_grid,
    )
    workspace = LocalPotentialWorkspace(
        indices,
        shape,
        mpi=mpi,
        descriptor=descriptor,
        pencil_fft=fft,
        distributed_fft_batch_size=2,
    )
    bands = 3
    reciprocal = _global_coefficients(shape, bands, 0)
    flat = reciprocal.reshape(-1, bands)
    local_coefficients = np.asfortranarray(
        flat[workspace.local_plane_wave_indices]
    )
    real = np.fft.ifftn(reciprocal, axes=(0, 1, 2)) * np.prod(shape)
    slab = mpi.slab(shape[2])

    observed_real = workspace.coefficients_to_grid(local_coefficients)
    np.testing.assert_allclose(
        observed_real,
        real[:, :, slab, :],
        rtol=2.0e-12,
        atol=2.0e-11,
    )
    observed_coefficients = workspace.grid_to_coefficients(
        np.ascontiguousarray(real[:, :, slab, :])
    )
    np.testing.assert_allclose(
        observed_coefficients,
        local_coefficients,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    x, y, z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )
    potential = 0.8 + 0.015 * x - 0.01 * y + 0.02 * z
    potential_x = fft.slab_to_x_pencil(
        np.ascontiguousarray(potential[:, :, slab])
    )
    observed_applied = workspace.apply(potential_x, local_coefficients)
    expected_applied = np.fft.fftn(
        real * potential[..., None], axes=(0, 1, 2)
    ) / np.prod(shape)
    np.testing.assert_allclose(
        observed_applied,
        expected_applied.reshape(-1, bands)[
            workspace.local_plane_wave_indices
        ],
        rtol=3.0e-12,
        atol=3.0e-11,
    )

    weights = np.asarray([0.4, 0.7, 0.2])
    observed_density = np.zeros(
        (shape[0], shape[1], slab.stop - slab.start)
    )
    workspace.accumulate_density(
        observed_density, local_coefficients, weights
    )
    expected_density = np.sum(
        np.abs(real) ** 2 * weights[None, None, None, :], axis=-1
    )
    np.testing.assert_allclose(
        observed_density,
        expected_density[:, :, slab],
        rtol=3.0e-12,
        atol=3.0e-10,
    )


def test_task_groups_execute_independent_band_sets() -> None:
    mpi = MPIContext.world()
    task_groups = 2 if mpi.size >= 2 and mpi.size % 2 == 0 else 1
    shape = (8, 8, 8)
    topology = FFTTaskTopology.build(
        mpi.comm, shape, task_groups=task_groups
    )
    fft = PencilFFT3D(shape, topology, threads=1)
    owned = topology.band_slice(11)
    assert owned.start == 11 * topology.task_group // task_groups
    assert owned.stop == 11 * (topology.task_group + 1) // task_groups

    coefficients = _global_coefficients(shape, 2, topology.task_group)
    local = _local_z(coefficients, fft)
    observed = fft.forward(fft.backward(local)).copy()
    local_error = float(np.max(np.abs(observed - local)))
    assert mpi.max_scalar(local_error) < 8.0e-13


def test_pencil_can_exceed_the_slab_z_rank_limit() -> None:
    mpi = MPIContext.world()
    if mpi.size <= 3:
        return
    shape = (8, 8, 3)
    topology = FFTTaskTopology.build(mpi.comm, shape)
    fft = PencilFFT3D(shape, topology, threads=1)
    coefficients = _global_coefficients(shape, 2, topology.task_group)
    local = _local_z(coefficients, fft)
    observed = fft.forward(fft.backward(local)).copy()
    local_error = float(np.max(np.abs(observed - local)))
    assert mpi.max_scalar(local_error) < 8.0e-13
