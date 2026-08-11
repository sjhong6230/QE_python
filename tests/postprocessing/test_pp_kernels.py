from __future__ import annotations

import numpy as np
import pytest

from qepy_pw.errors import QEInputError
from qepy_pw.pp.pp import (
    _miller_and_g,
    _range_option,
    _spectral_gradient,
    _spectral_gradient_hessian,
    _spectral_hessian,
)


def test_miller_grid_follows_numpy_fft_order() -> None:
    shape = (4, 3, 2)
    reciprocal = np.diag([2.0, 3.0, 5.0])

    miller, vectors = _miller_and_g(shape, reciprocal)

    expected_axes = [
        np.rint(np.fft.fftfreq(size) * size).astype(np.int64)
        for size in shape
    ]
    expected = np.stack(np.meshgrid(*expected_axes, indexing="ij"), axis=-1)
    np.testing.assert_array_equal(miller, expected)
    np.testing.assert_allclose(vectors, expected @ reciprocal)


def test_spectral_derivatives_are_exact_for_resolved_periodic_modes() -> None:
    shape = (8, 10, 12)
    lattice = np.diag([2.0, 3.0, 4.0])
    fractions = np.meshgrid(
        *(np.arange(size) / size for size in shape), indexing="ij"
    )
    x, y, _z = fractions
    values = np.sin(2.0 * np.pi * x) + 0.25 * np.cos(4.0 * np.pi * y)
    expected_gradient = np.zeros((3,) + shape)
    expected_gradient[0] = np.pi * np.cos(2.0 * np.pi * x)
    expected_gradient[1] = -(np.pi / 3.0) * np.sin(4.0 * np.pi * y)
    expected_hessian = np.zeros((3, 3) + shape)
    expected_hessian[0, 0] = -(np.pi**2) * np.sin(2.0 * np.pi * x)
    expected_hessian[1, 1] = -(4.0 * np.pi**2 / 9.0) * np.cos(4.0 * np.pi * y)

    gradient = _spectral_gradient(values, lattice)
    hessian = _spectral_hessian(values, lattice)
    combined_gradient, combined_hessian = _spectral_gradient_hessian(values, lattice)

    np.testing.assert_allclose(gradient, expected_gradient, atol=2.0e-13)
    np.testing.assert_allclose(hessian, expected_hessian, atol=2.0e-12)
    np.testing.assert_allclose(combined_gradient, gradient, atol=1.0e-14)
    np.testing.assert_allclose(combined_hessian, hessian, atol=1.0e-14)


def test_range_option_uses_one_based_inclusive_bounds() -> None:
    assert _range_option({"kband(1)": 2, "kband(2)": 4}, "kband", 5) == (2, 4)
    assert _range_option({"kband": 3}, "kband", 5) == (3, 3)

    with pytest.raises(QEInputError, match="wrong kband"):
        _range_option({"kband(1)": 0, "kband(2)": 2}, "kband", 5)
