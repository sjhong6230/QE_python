"""Small special functions used on the memory-critical SCF import path.

The production pseudopotentials need only real arguments and low angular
momenta.  Keeping these compact implementations here avoids loading the much
larger SciPy special-function extension in every MPI rank.
"""

from __future__ import annotations

from math import erf as _scalar_erf
from math import erfc as _scalar_erfc
from math import exp, lgamma

import numpy as np


def erf(values: np.ndarray | float) -> np.ndarray:
    """Vectorized :func:`math.erf` with NumPy shape/scalar semantics."""
    array = np.asarray(values, dtype=float)
    flat = np.fromiter(
        (_scalar_erf(float(value)) for value in array.flat),
        dtype=float,
        count=array.size,
    )
    return flat.reshape(array.shape)


def erfc(values: np.ndarray | float) -> np.ndarray:
    """Vectorized :func:`math.erfc` with NumPy shape/scalar semantics."""
    array = np.asarray(values, dtype=float)
    flat = np.fromiter(
        (_scalar_erfc(float(value)) for value in array.flat),
        dtype=float,
        count=array.size,
    )
    return flat.reshape(array.shape)


def spherical_jn(order: int, values: np.ndarray | float) -> np.ndarray:
    """Real spherical Bessel function for nonnegative integer ``order``.

    A power series is used where upward recurrence would lose accuracy near
    the origin.  Reciprocal-space UPF tables have nonnegative arguments, but
    the parity extension also makes the helper well defined for real inputs.
    """
    if order < 0:
        raise ValueError("spherical Bessel order must be nonnegative")
    x = np.asarray(values, dtype=float)
    absolute = np.abs(x)
    result = np.empty_like(x)
    use_series = absolute < max(1.0, float(order + 1))

    if np.any(use_series):
        xs = x[use_series]
        # First coefficient 1/(2l+1)!!, then update each term without
        # factorials: t_k/t_(k-1) = -x^2/[2k(2l+2k+1)].
        odd_factorial = 1.0
        for odd in range(1, 2 * order + 2, 2):
            odd_factorial *= odd
        term = np.power(xs, order) / odd_factorial
        summed = term.copy()
        for index in range(1, 80):
            term *= -xs * xs / (
                2.0 * index * (2.0 * order + 2.0 * index + 1.0)
            )
            summed += term
            if np.all(np.abs(term) <= 2.0e-16 * (1.0 + np.abs(summed))):
                break
        result[use_series] = summed

    if np.any(~use_series):
        xl = x[~use_series]
        j0 = np.sin(xl) / xl
        if order == 0:
            value = j0
        else:
            j1 = j0 / xl - np.cos(xl) / xl
            if order == 1:
                value = j1
            else:
                previous, value = j0, j1
                for angular_momentum in range(1, order):
                    previous, value = (
                        value,
                        (2 * angular_momentum + 1) * value / xl - previous,
                    )
        result[~use_series] = value
    return result


def spherical_harmonic(
    angular_momentum: int,
    magnetic_number: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Condon--Shortley complex spherical harmonic ``Y_l^m``."""
    l = int(angular_momentum)
    m = int(magnetic_number)
    if l < 0 or abs(m) > l:
        raise ValueError("spherical-harmonic indices require l >= 0, |m| <= l")
    if m < 0:
        positive = spherical_harmonic(l, -m, theta, phi)
        return (-1.0 if (-m) % 2 else 1.0) * np.conjugate(positive)

    cosine = np.cos(np.asarray(theta, dtype=float))
    azimuth = np.asarray(phi, dtype=float)
    if m == 0:
        p_mm = np.ones_like(cosine)
    else:
        coefficient = 1.0
        for odd in range(1, 2 * m, 2):
            coefficient *= -odd
        p_mm = coefficient * np.maximum(0.0, 1.0 - cosine**2) ** (0.5 * m)
    if l == m:
        associated = p_mm
    else:
        p_m1m = (2 * m + 1) * cosine * p_mm
        if l == m + 1:
            associated = p_m1m
        else:
            previous, associated = p_mm, p_m1m
            for degree in range(m + 2, l + 1):
                previous, associated = (
                    associated,
                    (
                        (2 * degree - 1) * cosine * associated
                        - (degree + m - 1) * previous
                    )
                    / (degree - m),
                )
    normalization = exp(
        0.5
        * (
            np.log((2 * l + 1) / (4.0 * np.pi))
            + lgamma(l - m + 1)
            - lgamma(l + m + 1)
        )
    )
    return normalization * associated * np.exp(1j * m * azimuth)
