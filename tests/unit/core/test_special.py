from __future__ import annotations

import numpy as np
import pytest
from scipy.special import spherical_jn as scipy_spherical_jn

from qepy_pw.special import erf, erfc, spherical_harmonic, spherical_jn


def test_error_functions_preserve_shapes_and_complementarity() -> None:
    values = np.asarray([[-8.0, -0.5, 0.0], [0.25, 2.0, 8.0]])
    assert erf(values).shape == values.shape
    assert erfc(values).shape == values.shape
    np.testing.assert_allclose(erf(values) + erfc(values), 1.0, atol=2.0e-16)
    assert erf(np.asarray(0.0)).shape == ()


@pytest.mark.parametrize("order", range(7))
def test_spherical_bessel_matches_scipy_across_series_and_recurrence(order: int) -> None:
    values = np.asarray([-9.0, -1.0e-7, 0.0, 0.3, 2.5, 11.0])
    expected = scipy_spherical_jn(order, np.abs(values))
    expected[values < 0.0] *= (-1.0) ** order
    np.testing.assert_allclose(spherical_jn(order, values), expected, atol=3.0e-14)


def test_spherical_bessel_rejects_negative_order() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        spherical_jn(-1, np.asarray([0.0]))


def test_spherical_harmonics_obey_normalization_and_negative_m_relation() -> None:
    theta = np.asarray([0.2, 0.9, 2.1])
    phi = np.asarray([-0.4, 1.2, 2.8])
    for angular_momentum in range(5):
        polar = spherical_harmonic(angular_momentum, 0, np.zeros(1), np.zeros(1))
        assert polar[0] == pytest.approx(
            np.sqrt((2 * angular_momentum + 1) / (4.0 * np.pi)), abs=2.0e-15
        )
        for magnetic_number in range(1, angular_momentum + 1):
            positive = spherical_harmonic(
                angular_momentum, magnetic_number, theta, phi
            )
            negative = spherical_harmonic(
                angular_momentum, -magnetic_number, theta, phi
            )
            np.testing.assert_allclose(
                negative,
                (-1) ** magnetic_number * np.conjugate(positive),
                atol=2.0e-15,
            )


def test_spherical_harmonic_rejects_invalid_indices() -> None:
    with pytest.raises(ValueError, match="indices"):
        spherical_harmonic(1, 2, np.asarray([0.0]), np.asarray([0.0]))
