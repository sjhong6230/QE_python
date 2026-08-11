"""Small, independently testable kernels for two-component spinors."""

from __future__ import annotations

import numpy as np


def magnetization_direction(angle1: float, angle2: float) -> np.ndarray:
    """Return QE's unit magnetization direction from angles in degrees."""
    theta, phi = np.deg2rad([float(angle1), float(angle2)])
    return np.asarray(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def aligned_spinors(angle1: float, angle2: float) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized spinors parallel and antiparallel to a direction."""
    theta, phi = np.deg2rad([float(angle1), float(angle2)])
    parallel = np.asarray(
        [np.cos(0.5 * theta), np.exp(1j * phi) * np.sin(0.5 * theta)],
        dtype=np.complex128,
    )
    antiparallel = np.asarray(
        [-np.exp(-1j * phi) * np.sin(0.5 * theta), np.cos(0.5 * theta)],
        dtype=np.complex128,
    )
    return parallel, antiparallel


def time_reverse(spinor: np.ndarray, spin_axis: int = 0) -> np.ndarray:
    """Apply spin-1/2 time reversal, ``i sigma_y K``, along ``spin_axis``."""
    values = np.asarray(spinor, dtype=np.complex128)
    if values.shape[spin_axis] != 2:
        raise ValueError("spinor axis must have length two")
    components = np.moveaxis(values, spin_axis, 0)
    reversed_components = np.stack(
        (np.conjugate(components[1]), -np.conjugate(components[0]))
    )
    return np.moveaxis(reversed_components, 0, spin_axis)


def density_components(spinor: np.ndarray, spin_axis: int = 0) -> np.ndarray:
    """Return charge and Cartesian magnetization of one or more spinors."""
    values = np.asarray(spinor, dtype=np.complex128)
    if values.shape[spin_axis] != 2:
        raise ValueError("spinor axis must have length two")
    up, down = np.moveaxis(values, spin_axis, 0)
    coherence = np.conjugate(up) * down
    up_density = np.real(np.conjugate(up) * up)
    down_density = np.real(np.conjugate(down) * down)
    return np.stack(
        (
            up_density + down_density,
            2.0 * np.real(coherence),
            2.0 * np.imag(coherence),
            up_density - down_density,
        )
    )


def apply_local_potential(
    potential: np.ndarray, spinor: np.ndarray, spin_axis: int = 0
) -> np.ndarray:
    """Apply ``V0 I + Vx sx + Vy sy + Vz sz`` pointwise."""
    fields = np.asarray(potential)
    if fields.shape[0] != 4:
        raise ValueError("potential must contain scalar, x, y, z components")
    values = np.moveaxis(np.asarray(spinor, dtype=np.complex128), spin_axis, 0)
    if values.shape[0] != 2:
        raise ValueError("spinor axis must have length two")
    up, down = values
    scalar, x_field, y_field, z_field = fields
    result = np.stack(
        (
            (scalar + z_field) * up + (x_field - 1j * y_field) * down,
            (x_field + 1j * y_field) * up + (scalar - z_field) * down,
        )
    )
    return np.moveaxis(result, 0, spin_axis)


def eigenchannel_densities(
    density: np.ndarray,
    quantization_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a local spin density, including QE's fixed-axis GGA labels.

    With ``quantization_axis`` omitted, channel 0 is the local majority
    channel.  QE instead gives the two GGA channels a continuous global label
    when all starting moments are collinear: the local eigenchannels are
    swapped wherever ``m . ux < 0``.  ``direction`` includes the corresponding
    sign so rotating the two channel potentials back remains invariant.
    """
    components = np.asarray(density, dtype=np.float64)
    if components.shape[0] != 4:
        raise ValueError("density must contain charge, mx, my, mz components")
    magnetization = components[1:]
    raw_magnitude = np.sqrt(
        np.einsum("i...,i...->...", magnetization, magnetization)
    )
    # A representable 2x2 density matrix obeys |m| <= rho.  Clipping tiny
    # numerical violations is the same positivity repair used by QE's
    # rho2zeta path before evaluating an LSDA/GGA functional.
    charge = np.maximum(components[0], 0.0)
    magnitude = np.minimum(raw_magnitude, charge)
    direction = np.zeros_like(magnetization)
    active = raw_magnitude > 1.0e-14
    direction[:, active] = magnetization[:, active] / raw_magnitude[active]
    sign = np.ones_like(magnitude)
    if quantization_axis is not None:
        axis = np.asarray(quantization_axis, dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("quantization axis must be a finite three-vector")
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-14:
            raise ValueError("quantization axis must be nonzero")
        projection = np.einsum("i...,i->...", magnetization, axis / norm)
        sign[projection < 0.0] = -1.0
        direction *= sign[None, ...]
    signed_magnitude = sign * magnitude
    channels = np.stack(
        (0.5 * (charge + signed_magnitude), 0.5 * (charge - signed_magnitude))
    )
    return channels, direction


def potential_components(
    channel_potential: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Rotate majority/minority XC potentials back to Pauli components."""
    potentials = np.asarray(channel_potential, dtype=np.float64)
    axis = np.asarray(direction, dtype=np.float64)
    if potentials.shape[0] != 2 or axis.shape != (3, *potentials.shape[1:]):
        raise ValueError("channel potentials and magnetization direction disagree")
    scalar = 0.5 * (potentials[0] + potentials[1])
    splitting = 0.5 * (potentials[0] - potentials[1])
    return np.concatenate((scalar[None, ...], axis * splitting[None, ...]))
