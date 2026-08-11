"""QE 7.5 regression tests for non-Gaussian smearing functions."""

from __future__ import annotations

from math import erf, erfc, exp, log, pi, sqrt

import numpy as np
import pytest

import qepy_pw.occupations as occupations_module
from qepy_pw.basis import _load_native_fft
from qepy_pw.occupations import (
    _integrated_tetra_fraction,
    _linear_tetra_moments,
    _linear_tetra_moments_block,
    smearing_density,
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


def _qe_w0gauss_scalar(x: float, order: int) -> float:
    """Direct transcription of QE 7.5 ``Modules/w0gauss.f90``."""
    if order == 1:
        return exp(-min(200.0, x * x)) / sqrt(pi) * (1.5 - x * x)
    if order == -1:
        shifted = x - 1.0 / sqrt(2.0)
        return (
            exp(-min(200.0, shifted * shifted))
            / sqrt(pi)
            * (2.0 - sqrt(2.0) * x)
        )
    if order == -99:
        return 0.0 if abs(x) > 36.0 else 1.0 / (
            2.0 + exp(-x) + exp(x)
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
    expected_density = np.asarray(
        [_qe_w0gauss_scalar(float(value), order) for value in arguments]
    )
    assert np.allclose(wgauss(arguments, order), expected_weights, atol=2.0e-15)
    assert np.allclose(w1gauss(arguments, order), expected_energy, atol=2.0e-15)
    assert np.allclose(
        smearing_density(arguments, order), expected_density, atol=2.0e-15
    )


@pytest.mark.parametrize("order", [0, 1, -1, -99])
def test_analytic_smearing_density_matches_broadened_step_derivative(
    order: int,
) -> None:
    arguments = np.linspace(-5.0, 5.0, 101)
    step = 1.0e-5
    numerical = (
        wgauss(arguments + step, order)
        - wgauss(arguments - step, order)
    ) / (2.0 * step)
    assert smearing_density(arguments, order) == pytest.approx(
        numerical, abs=2.0e-9
    )


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


@pytest.mark.parametrize("order", [1, -1])
def test_mp_and_cold_fermi_search_does_not_scan_a_dense_grid(
    monkeypatch: pytest.MonkeyPatch, order: int
) -> None:
    """QE's local efermig refinement must stay O(iterations), not O(2001)."""
    eigenvalues = [
        np.linspace(-1.2 + 0.01 * ik, 1.5 + 0.01 * ik, 12)
        for ik in range(432)
    ]
    weights = np.full(432, 1.0 / 216.0)
    original = occupations_module.wgauss
    calls = 0

    def counted_wgauss(x: np.ndarray | float, ngauss: int) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(x, ngauss)

    monkeypatch.setattr(occupations_module, "wgauss", counted_wgauss)
    _fermi, values, _demet = smeared_occupations(
        eigenvalues,
        weights,
        nelec=12.0,
        degauss_ha=0.005,
        order=order,
        spin_degeneracy=1.0,
    )
    electron_count = sum(
        weight * float(np.sum(row))
        for weight, row in zip(weights, values)
    )
    assert electron_count == pytest.approx(12.0, abs=1.0e-9)
    assert calls < 100


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


def test_native_tetrahedron_dos_matches_qe_moments() -> None:
    rng = np.random.default_rng(9821)
    energies = np.ascontiguousarray(
        np.sort(rng.normal(size=(37, 9, 4)), axis=-1)
    )
    energies[0, 0] = 0.25
    energies[1, 0, 1] = energies[1, 0, 0]
    energies[2, 0, 2] = energies[2, 0, 1]
    grid = np.ascontiguousarray(np.linspace(-2.5, 2.5, 51))
    expected_density = []
    expected_integrated = []
    for energy in grid:
        moments, density = _linear_tetra_moments_block(
            energies, float(energy)
        )
        expected_density.append(float(np.sum(density)))
        expected_integrated.append(float(np.sum(moments)))

    native = _load_native_fft()
    density, integrated = native.tetrahedron_dos_sums(energies, grid)
    assert density == pytest.approx(expected_density, abs=2.0e-12)
    assert integrated == pytest.approx(expected_integrated, abs=2.0e-12)


def test_native_tetrahedron_accumulation_matches_numpy_scatter() -> None:
    rng = np.random.default_rng(7204)
    connectivity = np.ascontiguousarray(
        rng.integers(0, 17, size=(83, 20), dtype=np.int32)
    )
    vertex_weights = np.ascontiguousarray(rng.normal(size=(83, 7, 4)))
    interpolation = np.ascontiguousarray(rng.normal(size=(4, 20)))
    weights = np.einsum(
        "cp,tbc->tbp", interpolation, vertex_weights
    )
    expected = np.zeros((17, 7))
    for corner in range(connectivity.shape[1]):
        np.add.at(
            expected,
            connectivity[:, corner],
            weights[:, :, corner],
        )

    native = _load_native_fft()
    actual = native.tetrahedron_accumulate(
        connectivity, vertex_weights, interpolation, 17
    )
    assert actual == pytest.approx(expected, abs=1.0e-13)


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
