"""Numba-compiled scalar kernels.

This module is imported lazily: importing qepy_pw must not require Numba when
the optional JIT extra is not installed.
"""

from __future__ import annotations

import math

from numba import njit
import numpy as np


@njit(cache=True, nogil=True)
def scatter_scaled(
    destination: np.ndarray,
    linear_slots: np.ndarray,
    vectors: np.ndarray,
    scale: float,
) -> None:
    """Clear an FFT block and scatter plane-wave coefficients into it."""
    destination[:] = 0.0
    for plane in range(linear_slots.size):
        slot = linear_slots[plane]
        for vector in range(vectors.shape[1]):
            destination[slot, vector] = vectors[plane, vector] * scale


@njit(cache=True, nogil=True)
def multiply_real_complex(
    real_potential: np.ndarray,
    wavefunctions: np.ndarray,
    product: np.ndarray,
) -> None:
    """Fuse the local-potential multiplication without array temporaries."""
    for point in range(real_potential.size):
        value = real_potential[point]
        for vector in range(wavefunctions.shape[1]):
            product[point, vector] = value * wavefunctions[point, vector]


@njit(cache=True, nogil=True)
def gather_scaled(
    source: np.ndarray,
    linear_slots: np.ndarray,
    destination: np.ndarray,
    scale: float,
) -> None:
    """Gather retained plane waves from an FFT block."""
    for plane in range(linear_slots.size):
        slot = linear_slots[plane]
        for vector in range(destination.shape[1]):
            destination[plane, vector] = source[slot, vector] * scale


@njit(cache=True, nogil=True)
def accumulate_density(
    density: np.ndarray,
    wavefunction: np.ndarray,
    factor: float,
) -> None:
    """Accumulate factor*|psi|^2 without materializing an absolute-value array."""
    for point in range(density.size):
        value = wavefunction[point]
        density[point] += factor * (
            value.real * value.real + value.imag * value.imag
        )


@njit(cache=True, nogil=True)
def accumulate_density_bands(
    density: np.ndarray,
    wavefunctions: np.ndarray,
    factor: float,
) -> None:
    """Accumulate all occupied bands in one allocation-free kernel call."""
    for point in range(density.size):
        value = 0.0
        for band in range(wavefunctions.shape[1]):
            coefficient = wavefunctions[point, band]
            value += (
                coefficient.real * coefficient.real
                + coefficient.imag * coefficient.imag
            )
        density[point] += factor * value


@njit(cache=True, nogil=True)
def accumulate_density_weighted_bands(
    density: np.ndarray,
    wavefunctions: np.ndarray,
    occupations: np.ndarray,
    kpoint_weight: float,
) -> None:
    """Accumulate weighted fractional-band occupations without temporaries."""
    for point in range(density.size):
        value = 0.0
        for band in range(wavefunctions.shape[1]):
            coefficient = wavefunctions[point, band]
            value += occupations[band] * (
                coefficient.real * coefficient.real
                + coefficient.imag * coefficient.imag
            )
        density[point] += kpoint_weight * value


@njit(cache=True, nogil=True)
def coulomb_metric(
    left: np.ndarray,
    right: np.ndarray,
    g2: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Real part of QE's reciprocal-space Coulomb inner product."""
    value = 0.0
    for point in range(g2.size):
        if mask[point]:
            product = left[point].conjugate() * right[point]
            value += product.real / g2[point]
    return value


@njit(cache=True, nogil=True)
def coulomb_metric_packed(
    left: np.ndarray,
    right: np.ndarray,
    g2: np.ndarray,
) -> float:
    """Coulomb metric for histories packed to active reciprocal vectors."""
    value = 0.0
    for point in range(g2.size):
        product = left[point].conjugate() * right[point]
        value += product.real / g2[point]
    return value


@njit(cache=True, nogil=True)
def pz81_unpolarized(
    rho: np.ndarray,
    epsilon_xc: np.ndarray,
    potential_xc: np.ndarray,
) -> None:
    """One-pass PZ81 LDA evaluation matching QE's XClib driver."""
    density_threshold = 1.0e-10
    pi34 = 0.6203504908994
    exchange = -0.687247939924714 * (2.0 / 3.0)
    for point in range(rho.size):
        density = abs(rho[point])
        if density <= density_threshold:
            epsilon_xc[point] = 0.0
            potential_xc[point] = 0.0
            continue
        rs = pi34 / density ** (1.0 / 3.0)
        ex = exchange / rs
        vx = (4.0 / 3.0) * ex
        if rs < 1.0:
            a = 0.0311
            b = -0.048
            c = 0.0020
            d = -0.0116
            log_rs = math.log(rs)
            ec = a * log_rs + b + c * rs * log_rs + d * rs
            derivative = a / rs + c * (log_rs + 1.0) + d
        else:
            gamma = -0.1423
            beta1 = 1.0529
            beta2 = 0.3334
            denominator = 1.0 + beta1 * math.sqrt(rs) + beta2 * rs
            ec = gamma / denominator
            derivative = (
                -gamma
                * (0.5 * beta1 / math.sqrt(rs) + beta2)
                / (denominator * denominator)
            )
        vc = ec - rs * derivative / 3.0
        epsilon_xc[point] = ex + ec
        potential_xc[point] = vx + vc
