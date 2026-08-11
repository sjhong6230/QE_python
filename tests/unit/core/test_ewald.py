from __future__ import annotations

import numpy as np
import pytest

from qepy_pw.ewald import ewald_energy, ewald_forces, ewald_stress


def _ionic_cell() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lattice = np.asarray(
        [[6.1, 0.0, 0.0], [0.2, 5.7, 0.0], [0.1, -0.3, 6.4]]
    )
    positions = np.asarray([[1.1, 1.3, 1.7], [3.4, 3.1, 3.8]])
    charges = np.asarray([1.0, 1.4])
    return lattice, positions, charges


def test_ewald_energy_is_invariant_under_rigid_translation() -> None:
    lattice, positions, charges = _ionic_cell()
    reference = ewald_energy(lattice, positions, charges, 28.0)
    translated = ewald_energy(
        lattice, positions + np.asarray([0.37, -0.22, 0.41]), charges, 28.0
    )
    assert translated == pytest.approx(reference, abs=2.0e-13)


def test_ewald_forces_are_energy_derivatives_and_obey_sum_rule() -> None:
    lattice, positions, charges = _ionic_cell()
    analytic = ewald_forces(lattice, positions, charges, 28.0)
    numerical = np.zeros_like(positions)
    step = 2.0e-5
    for atom in range(len(positions)):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numerical[atom, axis] = -(
                ewald_energy(lattice, plus, charges, 28.0)
                - ewald_energy(lattice, minus, charges, 28.0)
            ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numerical, atol=2.0e-8)
    np.testing.assert_allclose(np.sum(analytic, axis=0), 0.0, atol=2.0e-13)


def test_ewald_stress_is_homogeneous_strain_derivative() -> None:
    lattice, positions, charges = _ionic_cell()
    stress = ewald_stress(lattice, positions, charges, 28.0)
    volume = abs(np.linalg.det(lattice))
    step = 2.0e-5
    plus_scale = 1.0 + step
    minus_scale = 1.0 - step
    derivative = (
        ewald_energy(lattice * plus_scale, positions * plus_scale, charges, 28.0)
        - ewald_energy(lattice * minus_scale, positions * minus_scale, charges, 28.0)
    ) / (2.0 * step)
    np.testing.assert_allclose(stress, stress.T, atol=1.0e-15)
    assert -derivative / volume == pytest.approx(np.trace(stress), abs=3.0e-8)
