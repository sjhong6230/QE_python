"""QE 7.5 regression tests for non-Gaussian smearing functions."""

from __future__ import annotations

from math import erf, erfc, exp, log, pi, sqrt

import numpy as np
import pytest

from qepy_pw.basis import _load_native_fft
from qepy_pw.occupations import (
    _integrated_tetra_fraction,
    _linear_tetra_moments,
    _linear_tetra_moments_block,
    smeared_occupations,
    smearing_order,
    tetrahedron_occupations,
    w1gauss,
    wgauss,
)


def _qe_wgauss_scalar(x: float, order: int) -> float:
    """Direct transcription of QE 7.5 ``Modules/wgauss.f90``."""
    if order == 1:
        return 0.5 * erfc(-x) + x * exp(-min(200.0, x * x)) / (2.0 * sqrt(pi))
    if order == -1:
        shifted = x - 1.0 / sqrt(2.0)
        return (
            0.5 * erf(shifted)
            + exp(-min(200.0, shifted * shifted)) / sqrt(2.0 * pi)
            + 0.5
        )
    if order == -99:
        if x < -200.0:
            return 0.0
        if x > 200.0:
            return 1.0
        return 1.0 / (1.0 + exp(-x))
    raise AssertionError(order)


def _qe_w1gauss_scalar(x: float, order: int) -> float:
    """Direct transcription of QE 7.5 ``Modules/w1gauss.f90``."""
    if order == 1:
        argument = min(200.0, x * x)
        return (2.0 * x * x - 1.0) * exp(-argument) / (4.0 * sqrt(pi))
    if order == -1:
        shifted = x - 1.0 / sqrt(2.0)
        return shifted * exp(-min(200.0, shifted * shifted)) / sqrt(2.0 * pi)
    if order == -99:
        if abs(x) > 36.0:
            return 0.0
        occupation = 1.0 / (1.0 + exp(-x))
        return occupation * log(occupation) + (1.0 - occupation) * log(
            1.0 - occupation
        )
    raise AssertionError(order)


@pytest.mark.parametrize("order", [1, -1, -99])
def test_non_gaussian_wgauss_and_w1gauss_match_qe_75(order: int) -> None:
    arguments = np.asarray([-40.0, -2.0, -0.5, 0.0, 0.75, 2.0, 40.0])
    expected_weights = np.asarray(
        [_qe_wgauss_scalar(float(value), order) for value in arguments]
    )
    expected_energy = np.asarray(
        [_qe_w1gauss_scalar(float(value), order) for value in arguments]
    )
    assert np.allclose(wgauss(arguments, order), expected_weights, atol=2.0e-15)
    assert np.allclose(w1gauss(arguments, order), expected_energy, atol=2.0e-15)


@pytest.mark.parametrize(
    ("name", "order"),
    [("mp", 1), ("methfessel-paxton", 1), ("cold", -1),
     ("marzari-vanderbilt", -1), ("fd", -99), ("fermi-dirac", -99)],
)
def test_non_gaussian_smearing_aliases_match_qe(name: str, order: int) -> None:
    assert smearing_order(name) == order


@pytest.mark.parametrize("order", [1, -1, -99])
def test_non_gaussian_fermi_weights_and_demet_are_consistent(order: int) -> None:
    eigenvalues = [
        np.asarray([-0.70, -0.20, 0.15, 0.55]),
        np.asarray([-0.60, -0.10, 0.25, 0.80]),
    ]
    weights = np.asarray([0.25, 0.75])
    broadening = 0.12
    fermi, occupations, demet = smeared_occupations(
        eigenvalues, weights, 3.0, broadening, order
    )
    assert sum(
        weight * float(np.sum(values))
        for weight, values in zip(weights, occupations)
    ) == pytest.approx(3.0, abs=2.0e-12)
    expected_occupations = [
        2.0
        * np.asarray(
            [
                _qe_wgauss_scalar((fermi - float(value)) / broadening, order)
                for value in band
            ]
        )
        for band in eigenvalues
    ]
    for actual, expected in zip(occupations, expected_occupations):
        assert np.allclose(actual, expected, atol=2.0e-15)
    expected_demet = 2.0 * broadening * sum(
        weight
        * sum(
            _qe_w1gauss_scalar((fermi - float(value)) / broadening, order)
            for value in band
        )
        for weight, band in zip(weights, eigenvalues)
    )
    assert demet == pytest.approx(expected_demet, abs=2.0e-15)


def test_vectorized_tetrahedron_moments_match_scalar_qe_path() -> None:
    rng = np.random.default_rng(9182)
    energies = np.sort(rng.normal(size=(7, 13, 4)), axis=-1)
    # Include fully and partially degenerate tetrahedron vertices.
    energies[0, 0] = 0.25
    energies[1, 0, 1] = energies[1, 0, 0]
    energies[2, 0, 2] = energies[2, 0, 1]
    for fermi in (-2.0, -0.25, 0.0, 0.8, 2.0):
        moments, dos = _linear_tetra_moments_block(energies, fermi)
        for index in np.ndindex(energies.shape[:-1]):
            expected_moments, expected_dos = _linear_tetra_moments(
                energies[index], fermi
            )
            assert np.allclose(moments[index], expected_moments, atol=2.0e-15)
            assert dos[index] == pytest.approx(expected_dos, abs=2.0e-14)


def test_native_tetrahedron_fermi_sum_matches_numpy_formula() -> None:
    rng = np.random.default_rng(481)
    energies = np.ascontiguousarray(
        np.sort(rng.normal(size=(97, 11, 4)), axis=-1)
    )
    energies[0, 0] = 0.25
    native = _load_native_fft()
    for fermi in (-3.0, -0.6, 0.0, 0.9, 3.0):
        expected = float(np.sum(_integrated_tetra_fraction(energies, fermi)))
        actual = float(native.tetrahedron_integrated_sum(energies, fermi))
        assert actual == pytest.approx(expected, abs=2.0e-12)


@pytest.mark.parametrize(
    "method", ["tetrahedra", "tetrahedra_lin", "tetrahedra_opt"]
)
def test_all_tetrahedron_methods_preserve_electron_count(method: str) -> None:
    grid = (2, 2, 2)
    eigenvalues = [
        np.asarray([-0.8 + 0.07 * index, 0.3 + 0.05 * index])
        for index in range(8)
    ]
    weights = np.full(8, 1.0 / 8.0)
    _fermi, occupations = tetrahedron_occupations(
        eigenvalues,
        weights,
        1.0,
        grid,
        np.arange(8, dtype=np.int32),
        np.eye(3),
        method,
    )
    electron_count = sum(
        weight * float(np.sum(values))
        for weight, values in zip(weights, occupations)
    )
    assert electron_count == pytest.approx(1.0, abs=2.0e-10)
