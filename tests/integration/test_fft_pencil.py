"""MPI correctness tests for pencil decomposition and FFT task groups."""

from __future__ import annotations

import numpy as np

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
