"""Local and semilocal exchange-correlation functionals."""

from __future__ import annotations

import numpy as np


def pw92_unpolarized(
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PW92 ``(epsilon_c, v_c)`` in Hartree.

    The decimal constants and density threshold follow Quantum ESPRESSO's
    native ``pw`` routine.  PW92 is the local correlation component used by
    PBE; it is intentionally separate from the PZ81 correlation used by the
    LDA path below.
    """
    density = np.abs(np.asarray(rho, dtype=float))
    epsilon = np.zeros_like(density)
    potential = np.zeros_like(density)
    active = density > 1.0e-10
    if not np.any(active):
        return epsilon, potential

    n = density[active]
    rs = 0.6203504908994 / np.cbrt(n)
    root_rs = np.sqrt(rs)
    rs32 = rs * root_rs
    rs2 = rs * rs
    a = 0.031091
    a1 = 0.21370
    omega = 2.0 * a * (
        7.5957 * root_rs
        + 3.5876 * rs
        + 1.6382 * rs32
        + 0.49294 * rs2
    )
    domega = 2.0 * a * (
        0.5 * 7.5957 * root_rs
        + 3.5876 * rs
        + 1.5 * 1.6382 * rs32
        + 2.0 * 0.49294 * rs2
    )
    logarithm = np.log1p(1.0 / omega)
    ec = -2.0 * a * (1.0 + a1 * rs) * logarithm
    vc = (
        -2.0 * a * (1.0 + (2.0 / 3.0) * a1 * rs) * logarithm
        - (2.0 / 3.0)
        * a
        * (1.0 + a1 * rs)
        * domega
        / (omega * (omega + 1.0))
    )
    epsilon[active] = ec
    potential[active] = vc
    return epsilon, potential


def pbe_unpolarized(
    rho: np.ndarray,
    gradient: np.ndarray,
    divergence,
) -> tuple[np.ndarray, np.ndarray]:
    """Return spin-unpolarized PBE ``(epsilon_xc, v_xc)`` in Hartree.

    ``gradient`` has shape ``(3, *rho.shape)`` and contains Cartesian
    derivatives in bohr^-1. ``divergence`` must apply the matching periodic
    divergence to a vector field.  Splitting the pointwise PBE kernel from
    the spectral differential operators lets the SCF driver use either its
    serial FFT or distributed stick decomposition without gathering grids.
    """
    density = np.abs(np.asarray(rho, dtype=float))
    gradient = np.asarray(gradient, dtype=float)
    if gradient.shape != (3, *density.shape):
        raise ValueError("gradient must have shape (3, *rho.shape)")

    epsilon = np.zeros_like(density)
    local_potential = np.zeros_like(density)
    lda_active = density > 1.0e-10
    if np.any(lda_active):
        n = density[lda_active]
        rs = 0.6203504908994 / np.cbrt(n)
        exchange = (-0.687247939924714 * (2.0 / 3.0)) / rs
        epsilon[lda_active] = exchange
        local_potential[lda_active] = (4.0 / 3.0) * exchange
    ec, vc = pw92_unpolarized(density)
    epsilon += ec
    local_potential += vc

    sigma = np.sum(gradient * gradient, axis=0)
    gradient_active = (density > 1.0e-6) & (sigma > 1.0e-10)
    coefficient = np.zeros_like(density)
    if np.any(gradient_active):
        n = density[gradient_active]
        grad = np.sqrt(sigma[gradient_active])

        # PBE exchange gradient correction (QE XClib ``pbex``, iflag=1).
        kf = 3.093667726280136 * np.cbrt(n)
        dsg = 0.5 / kf
        reduced_gradient = grad * dsg / n
        kappa = 0.804
        mu = 0.2195149727645171
        denominator = 1.0 + mu * reduced_gradient**2 / kappa
        enhancement_correction = kappa - kappa / denominator
        exchange_uniform = -(0.75 / np.pi) * kf
        exchange_per_particle = exchange_uniform * enhancement_correction
        derivative_enhancement = (
            2.0 * mu * reduced_gradient / denominator**2
        )
        v1_exchange = (
            exchange_per_particle
            + (exchange_uniform / 3.0) * enhancement_correction
            - (4.0 / 3.0)
            * exchange_uniform
            * derivative_enhancement
            * reduced_gradient
        )
        v2_exchange = (
            exchange_uniform
            * derivative_enhancement
            * dsg
            / grad
        )
        exchange_energy_density = n * exchange_per_particle

        # PBE correlation gradient correction (QE XClib ``pbec``, iflag=1).
        rs = 0.6203504908994 / np.cbrt(n)
        kf_correlation = 1.919158292677513 / rs
        ks = 1.128379167095513 * np.sqrt(kf_correlation)
        t = grad / (2.0 * ks * n)
        gamma = 0.0310906908696548950
        beta = 0.06672455060314922
        ec_active = ec[gradient_active]
        vc_active = vc[gradient_active]
        exponential = np.exp(-ec_active / gamma)
        af = (beta / gamma) / (exponential - 1.0)
        bf = exponential * (vc_active - ec_active)
        y = af * t * t
        y_denominator = 1.0 + y + y * y
        xy = (1.0 + y) / y_denominator
        qy = y * y * (2.0 + y) / y_denominator**2
        logarithm_argument = 1.0 + (beta / gamma) * t * t * xy
        h0 = gamma * np.log(logarithm_argument)
        dh0 = beta * t * t / logarithm_argument * (
            -(7.0 / 3.0) * xy
            - qy * (af * bf / beta - 7.0 / 3.0)
        )
        v2_correlation = (
            beta
            / (2.0 * ks * ks * n)
            * (xy - qy)
            / logarithm_argument
        )
        correlation_energy_density = n * h0

        correction_energy_density = (
            exchange_energy_density + correlation_energy_density
        )
        epsilon[gradient_active] += correction_energy_density / n
        local_potential[gradient_active] += (
            v1_exchange + h0 + dh0
        )
        coefficient[gradient_active] = v2_exchange + v2_correlation

    flux = coefficient[None, ...] * gradient
    potential = local_potential - np.asarray(divergence(flux), dtype=float)
    return epsilon, potential


def pz81_unpolarized(
    rho: np.ndarray, *, use_numba: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return (epsilon_xc, v_xc) in Hartree for an unpolarized density."""
    if use_numba:
        from .acceleration import numba_kernels

        density = np.asarray(rho, dtype=float)
        epsilon_xc = np.empty_like(density)
        potential_xc = np.empty_like(density)
        numba_kernels().pz81_unpolarized(
            density.ravel(), epsilon_xc.ravel(), potential_xc.ravel()
        )
        return epsilon_xc, potential_xc
    # Match QE's XClib LDA driver: it evaluates the functional at |rho|,
    # and returns exactly zero below rho_threshold_lda.
    density = np.abs(np.asarray(rho, dtype=float))
    epsilon_xc = np.zeros_like(density)
    potential_xc = np.zeros_like(density)
    active = density > 1.0e-10
    if not np.any(active):
        return epsilon_xc, potential_xc

    active_density = density[active]
    # These deliberately retain the decimal constants used by QE rather than
    # recomputing them from pi, which changes the last few XC digits.
    rs = 0.6203504908994 / np.cbrt(active_density)
    ex = (-0.687247939924714 * (2.0 / 3.0)) / rs
    vx = (4.0 / 3.0) * ex
    ec = np.empty_like(rs)
    vc = np.empty_like(rs)
    high = rs >= 1.0
    low = ~high
    if np.any(low):
        r = rs[low]
        a, b, c, d = 0.0311, -0.048, 0.0020, -0.0116
        ec[low] = a * np.log(r) + b + c * r * np.log(r) + d * r
        derivative = a / r + c * (np.log(r) + 1.0) + d
        vc[low] = ec[low] - r * derivative / 3.0
    if np.any(high):
        r = rs[high]
        gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
        denominator = 1.0 + beta1 * np.sqrt(r) + beta2 * r
        ec[high] = gamma / denominator
        derivative = -gamma * (0.5 * beta1 / np.sqrt(r) + beta2) / denominator**2
        vc[high] = ec[high] - r * derivative / 3.0
    epsilon_xc[active] = ex + ec
    potential_xc[active] = vx + vc
    return epsilon_xc, potential_xc
