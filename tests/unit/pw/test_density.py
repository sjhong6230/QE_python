from __future__ import annotations

import numpy as np
import pytest

from qepy_pw.mpi import MPIContext
from qepy_pw.pw.scf import _density_from_states, _symmetrize_density_components


class _ConstantDensityWorkspace:
    def accumulate_density(self, destination, coefficients, weights) -> None:
        del coefficients
        destination += float(np.sum(weights))


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

