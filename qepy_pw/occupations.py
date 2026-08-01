"""QE-compatible scalar occupations and Gaussian broadening functions."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, erfc

from .errors import QEInputError, UnsupportedFeatureError


_SMEARING_ORDERS = {
    "gauss": 0,
    "gaussian": 0,
    "mp": 1,
    "m-p": 1,
    "methfessel-paxton": 1,
    "methfesselpaxton": 1,
    "mv": -1,
    "m-v": -1,
    "marzari-vanderbilt": -1,
    "marzarivanderbilt": -1,
    "cold": -1,
    "fd": -99,
    "f-d": -99,
    "fermi-dirac": -99,
    "fermidirac": -99,
}


def smearing_order(name: object) -> int:
    """Translate QE's ``smearing`` names to its ``ngauss`` value."""
    normalized = str(name).strip().lower().replace("_", "-")
    try:
        return _SMEARING_ORDERS[normalized]
    except KeyError as exc:
        raise UnsupportedFeatureError(
            f"smearing={name!r} is not ported; use 'gaussian', "
            "'methfessel-paxton', 'marzari-vanderbilt'/'cold', or "
            "'fermi-dirac'"
        ) from exc


def default_number_of_bands(nelec: float, occupations: object) -> int:
    """Return QE's insulating or metallic default number of bands."""
    filled = int(np.ceil(0.5 * nelec - 1.0e-12))
    if str(occupations).strip().lower() != "smearing":
        return filled
    return max(filled + 4, int(np.ceil(1.2 * 0.5 * nelec)))


def wgauss(x: np.ndarray | float, order: int) -> np.ndarray:
    """QE ``wgauss``: broadened step function for a given ``ngauss``."""
    values = np.asarray(x, dtype=float)
    if order == -99:
        result = np.empty_like(values)
        result[values < -200.0] = 0.0
        result[values > 200.0] = 1.0
        middle = (values >= -200.0) & (values <= 200.0)
        result[middle] = 1.0 / (1.0 + np.exp(-values[middle]))
        return result
    if order == -1:
        shifted = values - 1.0 / np.sqrt(2.0)
        argument = np.minimum(200.0, shifted * shifted)
        return (
            0.5 * erf(shifted)
            + np.exp(-argument) / np.sqrt(2.0 * np.pi)
            + 0.5
        )
    if order < 0:
        raise QEInputError(f"invalid Methfessel-Paxton order {order}")

    argument = np.minimum(200.0, values * values)
    result = 0.5 * erfc(-values)
    if order == 0:
        return result
    hd = np.zeros_like(values)
    hp = np.exp(-argument)
    hermite_order = 0
    coefficient = 1.0 / np.sqrt(np.pi)
    for index in range(1, order + 1):
        hd = 2.0 * values * hp - 2.0 * hermite_order * hd
        hermite_order += 1
        coefficient = -coefficient / (4.0 * index)
        result = result - coefficient * hd
        hp = 2.0 * values * hd - 2.0 * hermite_order * hp
        hermite_order += 1
    return result


def w1gauss(x: np.ndarray | float, order: int) -> np.ndarray:
    """QE ``w1gauss`` variational smearing/free-energy correction."""
    values = np.asarray(x, dtype=float)
    if order == -99:
        result = np.zeros_like(values)
        middle = np.abs(values) <= 36.0
        fermi = 1.0 / (1.0 + np.exp(-values[middle]))
        complement = 1.0 - fermi
        result[middle] = (
            fermi * np.log(fermi)
            + complement * np.log(complement)
        )
        return result
    if order == -1:
        shifted = values - 1.0 / np.sqrt(2.0)
        argument = np.minimum(200.0, shifted * shifted)
        return shifted * np.exp(-argument) / np.sqrt(2.0 * np.pi)
    if order < 0:
        raise QEInputError(f"invalid Methfessel-Paxton order {order}")

    argument = np.minimum(200.0, values * values)
    result = -0.5 * np.exp(-argument) / np.sqrt(np.pi)
    if order == 0:
        return result
    hd = np.zeros_like(values)
    hp = np.exp(-argument)
    hermite_order = 0
    coefficient = 1.0 / np.sqrt(np.pi)
    for index in range(1, order + 1):
        hd = 2.0 * values * hp - 2.0 * hermite_order * hd
        hermite_order += 1
        previous_hp = hp
        hp = 2.0 * values * hd - 2.0 * hermite_order * hp
        hermite_order += 1
        coefficient = -coefficient / (4.0 * index)
        result = result - coefficient * (
            0.5 * hp + hermite_order * previous_hp
        )
    return result


def smeared_occupations(
    eigenvalues: list[np.ndarray],
    weights: np.ndarray,
    nelec: float,
    degauss_ha: float,
    order: int,
) -> tuple[float, list[np.ndarray], float]:
    """Find the Fermi level, occupations, and QE ``demet`` in Hartree.

    Returned occupations include the scalar spin degeneracy and therefore
    approach two below the Fermi level. K-point weights remain separate.
    """
    if degauss_ha <= 0.0:
        raise QEInputError("degauss must be positive for smeared occupations")
    if not eigenvalues or any(len(values) == 0 for values in eigenvalues):
        raise QEInputError("cannot determine occupations without bands")
    normalized_weights = np.asarray(weights, dtype=float)
    if normalized_weights.shape != (len(eigenvalues),):
        raise QEInputError("k-point weights do not match eigenvalue arrays")
    normalized_weights = normalized_weights / np.sum(normalized_weights)
    flat_eigenvalues = np.concatenate(
        [np.asarray(values, dtype=float) for values in eigenvalues]
    )
    flat_weights = np.concatenate(
        [
            np.full(len(values), weight, dtype=float)
            for weight, values in zip(normalized_weights, eigenvalues)
        ]
    )
    capacity = 2.0 * sum(
        weight * len(values)
        for weight, values in zip(normalized_weights, eigenvalues)
    )
    if nelec <= 0.0 or nelec >= capacity - 1.0e-12:
        raise QEInputError(
            f"nbnd provides {capacity:.6g} smeared states for {nelec:.6g} "
            "electrons; add unoccupied bands"
        )

    def electron_count(fermi: float, broadening_order: int = order) -> float:
        return 2.0 * float(
            np.dot(
                flat_weights,
                wgauss(
                    (fermi - flat_eigenvalues) / degauss_ha,
                    broadening_order,
                ),
            )
        )

    minimum = min(float(values[0]) for values in eigenvalues)
    maximum = max(float(values[-1]) for values in eigenvalues)
    lower = minimum - 10.0 * degauss_ha
    upper = maximum + 10.0 * degauss_ha
    while electron_count(lower) > nelec:
        lower -= 10.0 * degauss_ha
    while electron_count(upper) < nelec:
        upper += 10.0 * degauss_ha

    # QE first obtains a Gaussian estimate for MP/cold smearing. Select the
    # actual-function root nearest that estimate when the broadened electron
    # count is locally non-monotonic.
    gaussian_fermi = brentq(
        lambda value: electron_count(value, 0) - nelec,
        lower,
        upper,
        xtol=1.0e-13,
        rtol=1.0e-14,
    )
    if order in {0, -99}:
        fermi = brentq(
            lambda value: electron_count(value) - nelec,
            lower,
            upper,
            xtol=1.0e-13,
            rtol=1.0e-14,
        )
    else:
        grid = np.linspace(lower, upper, 2001)
        residual = np.array([electron_count(value) - nelec for value in grid])
        crossings = np.flatnonzero(residual[:-1] * residual[1:] <= 0.0)
        if crossings.size == 0:
            raise QEInputError("cannot bracket the Fermi energy")
        index = int(
            crossings[
                np.argmin(np.abs(grid[crossings] - gaussian_fermi))
            ]
        )
        fermi = brentq(
            lambda value: electron_count(value) - nelec,
            float(grid[index]),
            float(grid[index + 1]),
            xtol=1.0e-13,
            rtol=1.0e-14,
        )

    occupations = [
        2.0 * wgauss((fermi - np.asarray(values)) / degauss_ha, order)
        for values in eigenvalues
    ]
    smearing_energy = 2.0 * degauss_ha * float(
        np.dot(
            flat_weights,
            w1gauss(
                (fermi - flat_eigenvalues) / degauss_ha,
                order,
            ),
        )
    )
    return float(fermi), occupations, float(smearing_energy)
