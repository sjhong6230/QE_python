from __future__ import annotations

import numpy as np
import pytest

from qepy_pw.mpi import MPIContext
from qepy_pw.pw.scf import _density_from_states, _symmetrize_density_components


class _ConstantDensityWorkspace:
    def accumulate_density(self, destination, coefficients, weights) -> None:
        del coefficients
        destination += float(np.sum(weights))


class _SpinorGridWorkspace:
    def __init__(self, grid: np.ndarray) -> None:
        self.grid = grid

    def coefficients_to_grid(self, coefficients) -> np.ndarray:
        assert coefficients.shape[1] == self.grid.shape[-1]
        return self.grid.copy()


class _MeanProjector:
    def apply(self, density: np.ndarray) -> np.ndarray:
        return np.full_like(density, np.mean(density))


def test_density_from_states_routes_and_normalizes_each_spin_block() -> None:
    shape = (2, 2, 2)
    volume = 8.0
    weights = np.asarray([0.4, 0.6, 0.4, 0.6])
    occupations = [
        np.asarray([1.0, 0.5]),
        np.asarray([1.0, 0.0]),
        np.asarray([0.5, 0.0]),
        np.asarray([1.0, 0.25]),
    ]
    density = _density_from_states(
        [np.ones((1, 2), dtype=complex) for _ in range(4)],
        [object() for _ in range(4)],
        weights,
        occupations,
        shape,
        volume,
        nelec=2.15,
        workspaces=[_ConstantDensityWorkspace() for _ in range(4)],
        kpoint_spins=(1, 1, 2, 2),
    )
    scale = volume / np.prod(shape)
    expected_up = 0.4 * 1.5 + 0.6 * 1.0
    expected_down = 0.4 * 0.5 + 0.6 * 1.25
    assert scale * np.sum(density[0]) == pytest.approx(expected_up)
    assert scale * np.sum(density[1]) == pytest.approx(expected_down)


def test_spatial_density_symmetry_does_not_mix_spin_channels() -> None:
    density = np.asarray(
        [
            np.arange(8, dtype=float).reshape(2, 2, 2) + 2.0,
            np.arange(8, dtype=float).reshape(2, 2, 2) + 0.5,
        ]
    )
    volume = 8.0
    nelec = float(np.sum(density) * volume / 8.0)
    result = _symmetrize_density_components(
        density,
        _MeanProjector(),
        nelec,
        (2, 2, 2),
        volume,
        MPIContext(),
    )
    assert np.all(result[0] == np.mean(density[0]))
    assert np.all(result[1] == np.mean(density[1]))
    assert np.sum(result[0] - result[1]) == pytest.approx(
        np.sum(density[0] - density[1])
    )


def test_spinor_density_builds_charge_and_three_magnetization_components() -> None:
    theta, phi = np.deg2rad([63.0, -28.0])
    spinor = np.asarray(
        [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)]
    )
    grid = np.empty((1, 1, 1, 2), dtype=complex)
    grid[0, 0, 0] = spinor
    density = _density_from_states(
        [np.ones((2, 1), dtype=complex)],
        [object()],
        np.asarray([1.0]),
        [np.asarray([1.0])],
        (1, 1, 1),
        1.0,
        nelec=1.0,
        workspaces=[_SpinorGridWorkspace(grid)],
        spinor=True,
        domag=True,
    )
    np.testing.assert_allclose(
        density[:, 0, 0, 0],
        [
            1.0,
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        atol=2e-15,
    )

