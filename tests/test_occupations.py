import numpy as np
import pytest

from qepy_pw.errors import QEInputError, UnsupportedFeatureError
from qepy_pw.occupations import (
    default_number_of_bands,
    smeared_occupations,
    smearing_order,
    w1gauss,
    wgauss,
)


def test_qe_smearing_names_and_default_metal_bands():
    assert smearing_order("gaussian") == 0
    assert smearing_order("mp") == 1
    assert smearing_order("Methfessel-Paxton") == 1
    assert smearing_order("cold") == -1
    assert smearing_order("Marzari-Vanderbilt") == -1
    assert smearing_order("fermi-dirac") == -99
    assert default_number_of_bands(8.0, "fixed") == 4
    assert default_number_of_bands(8.0, "smearing") == 8
    with pytest.raises(UnsupportedFeatureError):
        smearing_order("not-a-smearing")


def test_wgauss_and_w1gauss_reference_values():
    assert wgauss(0.0, 0) == pytest.approx(0.5)
    assert wgauss(0.0, 1) == pytest.approx(0.5)
    assert wgauss(0.0, -99) == pytest.approx(0.5)
    assert wgauss(1.0 / np.sqrt(2.0), -1) == pytest.approx(
        0.5 + 1.0 / np.sqrt(2.0 * np.pi)
    )
    assert w1gauss(0.0, 0) == pytest.approx(
        -0.5 / np.sqrt(np.pi)
    )
    assert w1gauss(0.0, -99) == pytest.approx(-np.log(2.0))


@pytest.mark.parametrize("order", [0, 1, -1, -99])
def test_smeared_occupations_conserve_electron_number(order):
    eigenvalues = [
        np.array([-0.8, -0.1, 0.2, 0.9]),
        np.array([-0.7, 0.0, 0.35, 1.1]),
    ]
    weights = np.array([0.25, 0.75])
    fermi, occupations, correction = smeared_occupations(
        eigenvalues, weights, nelec=3.2, degauss_ha=0.15, order=order
    )
    electron_count = sum(
        weight * np.sum(values)
        for weight, values in zip(weights, occupations)
    )
    assert np.isfinite(fermi)
    assert np.isfinite(correction)
    assert electron_count == pytest.approx(3.2, abs=2.0e-11)


def test_smeared_occupations_require_broadening_and_empty_bands():
    eigenvalues = [np.array([-0.5, 0.5])]
    with pytest.raises(QEInputError, match="degauss"):
        smeared_occupations(eigenvalues, np.ones(1), 2.0, 0.0, 0)
    with pytest.raises(QEInputError, match="add unoccupied bands"):
        smeared_occupations(eigenvalues, np.ones(1), 4.0, 0.1, 0)
