from __future__ import annotations

import numpy as np

from qepy_pw.spinor import (
    aligned_spinors,
    apply_local_potential,
    density_components,
    magnetization_direction,
    time_reverse,
)
from qepy_pw.xc import noncollinear_lda


def test_angle_spinor_has_requested_magnetization_direction() -> None:
    parallel, antiparallel = aligned_spinors(71.0, -23.0)
    expected = magnetization_direction(71.0, -23.0)
    np.testing.assert_allclose(density_components(parallel)[1:], expected, atol=2e-15)
    np.testing.assert_allclose(density_components(antiparallel)[1:], -expected, atol=2e-15)


def test_spin_half_time_reversal_is_antiunitary_and_squares_to_minus_one() -> None:
    spinor = np.asarray([0.3 + 0.2j, -0.7 + 0.1j])
    np.testing.assert_allclose(time_reverse(time_reverse(spinor)), -spinor)
    np.testing.assert_allclose(
        density_components(time_reverse(spinor))[1:],
        -density_components(spinor)[1:],
    )


def test_pauli_potential_action_matches_explicit_matrix() -> None:
    potential = np.asarray([1.2, -0.3, 0.4, 0.7])
    spinor = np.asarray([0.2 + 0.8j, -0.5 + 0.1j])
    matrix = np.asarray(
        [[1.9, -0.3 - 0.4j], [-0.3 + 0.4j, 0.5]], dtype=complex
    )
    np.testing.assert_allclose(apply_local_potential(potential, spinor), matrix @ spinor)


def test_noncollinear_lda_is_spin_rotation_covariant() -> None:
    charge = np.asarray([0.8, 1.1, 0.35])
    magnitude = np.asarray([0.2, 0.4, 0.1])
    first = magnetization_direction(36.0, 14.0)
    second = magnetization_direction(117.0, -51.0)
    density_first = np.concatenate((charge[None, :], first[:, None] * magnitude))
    density_second = np.concatenate((charge[None, :], second[:, None] * magnitude))
    energy_first, potential_first = noncollinear_lda(density_first, "pz")
    energy_second, potential_second = noncollinear_lda(density_second, "pz")
    np.testing.assert_allclose(energy_first, energy_second, atol=2e-15)
    np.testing.assert_allclose(potential_first[0], potential_second[0], atol=2e-15)
    np.testing.assert_allclose(
        np.linalg.norm(potential_first[1:], axis=0),
        np.linalg.norm(potential_second[1:], axis=0),
        atol=2e-15,
    )
