"""QE-compatible classical Ewald energy for periodic point ions."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.special import erfc


def _integer_vectors_within_cutoff(
    reciprocal: np.ndarray, cutoff_g2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return complete Miller/G-vector sets with |G|^2 <= cutoff_g2."""
    minimum_stretch = float(
        np.linalg.svd(reciprocal, compute_uv=False)[-1]
    )
    bound = int(np.ceil(np.sqrt(cutoff_g2) / minimum_stretch)) + 1
    miller = np.array(
        list(itertools.product(range(-bound, bound + 1), repeat=3)),
        dtype=int,
    )
    vectors = miller @ reciprocal
    g2 = np.einsum("ij,ij->i", vectors, vectors)
    keep = g2 <= cutoff_g2 + 1.0e-12
    return vectors[keep], g2[keep]


def ewald_energy(
    lattice: np.ndarray,
    positions: np.ndarray,
    charges: np.ndarray,
    ecutrho_ry: float,
) -> float:
    """Return QE ``ewald`` energy in Hartree.

    This follows ``PW/src/ewald.f90`` for the ordinary three-dimensional,
    non-gamma-distributed case. Coordinates and lattice vectors are in bohr;
    the charge-density cutoff is in Ry.
    """
    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    total_charge = float(np.sum(charges))

    # QE decreases alpha from 2.9 in steps of 0.1 until its upper bound for
    # the omitted reciprocal contribution is at most 1e-7 Ry.
    alpha = 2.9
    while True:
        alpha -= 0.1
        upper_bound = (
            2.0
            * total_charge**2
            * np.sqrt(alpha / np.pi)
            * erfc(np.sqrt(ecutrho_ry / (4.0 * alpha)))
        )
        if upper_bound <= 1.0e-7:
            break
        if alpha <= 0.0:
            raise RuntimeError("optimal Ewald alpha not found")

    g_vectors, g2 = _integer_vectors_within_cutoff(
        reciprocal, ecutrho_ry
    )
    nonzero = g2 > 1.0e-14
    g_vectors = g_vectors[nonzero]
    g2 = g2[nonzero]
    phases = positions @ g_vectors.T
    structure = np.sum(
        charges[:, None] * np.exp(-1j * phases), axis=0
    )

    # QE's ewaldg before multiplication by 0.5*e2. With e2=2 Ry, the
    # resulting expression below is in Ry; divide by two on return.
    reciprocal_sum = -total_charge**2 / (4.0 * alpha)
    reciprocal_sum += float(
        np.sum(
            np.abs(structure) ** 2
            * np.exp(-g2 / (4.0 * alpha))
            / g2
        )
    )
    reciprocal_sum *= 4.0 * np.pi / volume
    reciprocal_sum -= float(
        np.sum(charges**2) * 2.0 * np.sqrt(alpha / np.pi)
    )

    real_sum = 0.0
    radius = 4.0 / np.sqrt(alpha)
    minimum_stretch = float(
        np.linalg.svd(lattice, compute_uv=False)[-1]
    )
    for left, left_charge in zip(positions, charges):
        for right, right_charge in zip(positions, charges):
            displacement0 = left - right
            bound = int(
                np.ceil(
                    (radius + np.linalg.norm(displacement0))
                    / minimum_stretch
                )
            ) + 1
            for image in itertools.product(
                range(-bound, bound + 1), repeat=3
            ):
                displacement = (
                    displacement0 + np.asarray(image) @ lattice
                )
                distance = float(np.linalg.norm(displacement))
                if distance < 1.0e-14 or distance > radius:
                    continue
                real_sum += (
                    left_charge
                    * right_charge
                    * erfc(np.sqrt(alpha) * distance)
                    / distance
                )

    energy_ry = reciprocal_sum + 0.5 * real_sum
    return 0.5 * energy_ry


def ewald_forces(
    lattice: np.ndarray,
    positions: np.ndarray,
    charges: np.ndarray,
    ecutrho_ry: float,
) -> np.ndarray:
    """Return the analytic derivative ``-dE_ewald/dR`` in Ha/bohr.

    The reciprocal and real-space prefactors deliberately differentiate the
    exact energy expression in :func:`ewald_energy`, including its QE cutoff
    and alpha selection, so energy and force cannot drift apart.
    """
    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    total_charge = float(np.sum(charges))

    alpha = 2.9
    while True:
        alpha -= 0.1
        upper_bound = (
            2.0
            * total_charge**2
            * np.sqrt(alpha / np.pi)
            * erfc(np.sqrt(ecutrho_ry / (4.0 * alpha)))
        )
        if upper_bound <= 1.0e-7:
            break
        if alpha <= 0.0:
            raise RuntimeError("optimal Ewald alpha not found")

    forces = np.zeros_like(positions)
    g_vectors, g2 = _integer_vectors_within_cutoff(
        reciprocal, ecutrho_ry
    )
    nonzero = g2 > 1.0e-14
    g_vectors = g_vectors[nonzero]
    g2 = g2[nonzero]
    phases = np.exp(-1j * (positions @ g_vectors.T))
    structure = np.sum(charges[:, None] * phases, axis=0)
    reciprocal_weight = (
        2.0
        * np.pi
        / volume
        * np.exp(-g2 / (4.0 * alpha))
        / g2
    )
    for atom, charge in enumerate(charges):
        derivative_structure = (
            -1j
            * charge
            * phases[atom, :, None]
            * g_vectors
        )
        derivative_norm = 2.0 * np.real(
            np.conjugate(structure)[:, None]
            * derivative_structure
        )
        forces[atom] -= np.sum(
            reciprocal_weight[:, None] * derivative_norm, axis=0
        )

    radius = 4.0 / np.sqrt(alpha)
    minimum_stretch = float(
        np.linalg.svd(lattice, compute_uv=False)[-1]
    )
    root_alpha = np.sqrt(alpha)
    for left_index, (left, left_charge) in enumerate(
        zip(positions, charges)
    ):
        for right_index, (right, right_charge) in enumerate(
            zip(positions, charges)
        ):
            displacement0 = left - right
            bound = int(
                np.ceil(
                    (radius + np.linalg.norm(displacement0))
                    / minimum_stretch
                )
            ) + 1
            for image in itertools.product(
                range(-bound, bound + 1), repeat=3
            ):
                displacement = displacement0 + np.asarray(image) @ lattice
                distance = float(np.linalg.norm(displacement))
                if distance < 1.0e-14 or distance > radius:
                    continue
                radial_derivative = (
                    -erfc(root_alpha * distance) / distance**2
                    - 2.0
                    * root_alpha
                    / np.sqrt(np.pi)
                    * np.exp(-alpha * distance**2)
                    / distance
                )
                energy_gradient = (
                    0.25
                    * left_charge
                    * right_charge
                    * radial_derivative
                    * displacement
                    / distance
                )
                forces[left_index] -= energy_gradient
                forces[right_index] += energy_gradient
    return forces
