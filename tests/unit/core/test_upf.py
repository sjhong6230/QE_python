from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qepy_pw.upf import (
    _qe_cubic_interpolate_with_derivative,
    _qe_real_spherical_harmonics_with_gradient,
    _qe_simpson,
    read_upf,
)


PSEUDO_DIR = Path(__file__).resolve().parents[2] / "qe_reference" / "upstream" / "pseudo"


@pytest.mark.parametrize(
    ("filename", "element", "valence"),
    [("H.pz-vbc.UPF", "H", 1.0), ("Si.pz-vbc.UPF", "Si", 4.0)],
)
def test_upf_reader_recovers_normalized_nc_metadata(
    filename: str, element: str, valence: float
) -> None:
    pseudo = read_upf(PSEUDO_DIR / filename)
    assert pseudo.element.strip() == element
    assert pseudo.z_valence == pytest.approx(valence)
    assert pseudo.pseudo_type == "NC"
    assert pseudo.r is not None and len(pseudo.r) == pseudo.mesh_size
    assert pseudo.rab is not None and len(pseudo.rab) == pseudo.mesh_size
    assert np.all(np.diff(pseudo.r) > 0.0)


def test_qe_simpson_integrates_quadratic_on_uniform_grid() -> None:
    x = np.arange(5, dtype=float)
    integral = _qe_simpson(x * x, np.ones(5))
    assert integral == pytest.approx(64.0 / 3.0, abs=2.0e-15)


def test_qe_cubic_interpolation_is_exact_for_cubic_polynomial() -> None:
    dq = 0.2
    grid = np.arange(20) * dq
    table = grid**3 - 2.0 * grid**2 + 0.5 * grid - 1.0
    q = np.asarray([0.27, 0.93, 2.41])
    values, derivatives = _qe_cubic_interpolate_with_derivative(table, q, dq)
    np.testing.assert_allclose(values, q**3 - 2.0 * q**2 + 0.5 * q - 1.0, atol=3e-15)
    np.testing.assert_allclose(derivatives, 3.0 * q**2 - 4.0 * q + 0.5, atol=2e-14)


@pytest.mark.parametrize("angular_momentum", [0, 1, 2, 3])
def test_real_spherical_harmonic_gradients_match_finite_difference(
    angular_momentum: int,
) -> None:
    vectors = np.asarray([[0.7, -0.4, 1.1], [-0.8, 0.9, 0.5]])
    values, gradients = _qe_real_spherical_harmonics_with_gradient(
        angular_momentum, vectors
    )
    step = 1.0e-6
    numerical = np.empty_like(gradients)
    for axis in range(3):
        plus = vectors.copy()
        minus = vectors.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        plus_values, _ = _qe_real_spherical_harmonics_with_gradient(
            angular_momentum, plus
        )
        minus_values, _ = _qe_real_spherical_harmonics_with_gradient(
            angular_momentum, minus
        )
        numerical[:, :, axis] = (plus_values - minus_values) / (2.0 * step)
    np.testing.assert_allclose(values.shape, (2, 2 * angular_momentum + 1))
    np.testing.assert_allclose(gradients, numerical, atol=2.0e-9)


def test_local_pseudopotential_radial_derivative_matches_finite_difference() -> None:
    pseudo = read_upf(PSEUDO_DIR / "Si.pz-vbc.UPF")
    q = np.asarray([0.31, 0.84, 1.27])
    volume = 125.0
    values, derivative = pseudo.fourier_with_derivative(q, volume)
    step = 1.0e-6
    numerical = (
        pseudo.fourier(q + step, volume) - pseudo.fourier(q - step, volume)
    ) / (2.0 * step)
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(derivative, numerical, rtol=3.0e-6, atol=2.0e-10)

