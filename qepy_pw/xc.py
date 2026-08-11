"""Local and semilocal exchange-correlation functionals."""

from __future__ import annotations

import numpy as np

from .spinor import eigenchannel_densities, potential_components


LDA_FUNCTIONALS = frozenset({"pz", "pw"})
GGA_FUNCTIONALS = frozenset({"pbe", "pbesol", "revpbe", "rpbe"})
SUPPORTED_XC_FUNCTIONALS = LDA_FUNCTIONALS | GGA_FUNCTIONALS


def noncollinear_lda(
    density: np.ndarray, functional: str = "pz"
) -> tuple[np.ndarray, np.ndarray]:
    """Return rotationally invariant noncollinear LSDA energy and 4-potential."""
    channels, direction = eigenchannel_densities(density)
    epsilon, channel_potential = lsda_lda(channels, functional)
    return epsilon, potential_components(channel_potential, direction)


def canonical_xc_name(name: object) -> str | None:
    """Return the supported QE functional represented by ``name``.

    In addition to the ordinary ``input_dft`` names, this recognizes the
    four-component strings stored in UPF headers.
    """
    normalized = "".join(
        character
        for character in str(name).lower()
        if character.isalnum()
    )
    aliases = {
        "lda": "pz",
        "pz": "pz",
        "pz81": "pz",
        "perdewzunger": "pz",
        "slapznogxnogc": "pz",
        "pw": "pw",
        "pw92": "pw",
        "perdewwang": "pw",
        "ldapw": "pw",
        "slapwnogxnogc": "pw",
        "pbe": "pbe",
        "perdewburkeernzerhof": "pbe",
        "slapwpbxpbc": "pbe",
        "pbesol": "pbesol",
        "pbesolid": "pbesol",
        "slapwpsxpsc": "pbesol",
        "revpbe": "revpbe",
        "revisedpbe": "revpbe",
        "slapwrevxpbc": "revpbe",
        "rpbe": "rpbe",
        "slapwhhnxpbc": "rpbe",
    }
    return aliases.get(normalized)


_PBE_FAMILY_PARAMETERS = {
    "pbe": (0.804, 0.2195149727645171, 0.06672455060314922, "rational"),
    "pbesol": (0.804, 10.0 / 81.0, 0.046, "rational"),
    "revpbe": (1.245, 0.2195149727645171, 0.06672455060314922, "rational"),
    "rpbe": (0.804, 0.2195149727645171, 0.06672455060314922, "exponential"),
}

# Pointwise GGA algebra needs several same-shaped work arrays.  Keeping those
# arrays proportional to the entire FFT slab made the transient XC high-water
# mark scale with the charge grid even though every operation is local in real
# space.  A 32k-point tile fits the active doubles comfortably in cache while
# remaining large enough that NumPy ufunc dispatch is negligible.
GGA_POINT_BLOCK_SIZE = 32_768


def pw92_unpolarized(
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PW92 ``(epsilon_c, v_c)`` in Hartree.

    The decimal constants and density threshold follow Quantum ESPRESSO's
    native ``pw`` routine.  PW92 is the local correlation component used by
    PBE; it is intentionally separate from the PZ81 correlation used by the
    LDA path below.
    """
    density = np.abs(np.asarray(rho, dtype=np.float64))
    active = density > 1.0e-10
    np.maximum(density, 1.0e-10, out=density)

    # Evaluate QE's PW92 rational form with a bounded set of reusable arrays.
    # The names follow the published formula; ``work`` is deliberately reused
    # between terms so a grid-sized temporary is not created per expression.
    rs = np.cbrt(density)
    np.divide(0.6203504908994, rs, out=rs)
    root_rs = np.sqrt(rs)
    work = np.empty_like(density)
    epsilon = np.empty_like(density)
    potential = np.empty_like(density)
    a = 0.031091
    a1 = 0.21370

    np.multiply(root_rs, 7.5957, out=epsilon)
    np.multiply(rs, 3.5876, out=work)
    epsilon += work
    np.multiply(rs, root_rs, out=work)
    work *= 1.6382
    epsilon += work
    np.square(rs, out=work)
    work *= 0.49294
    epsilon += work
    epsilon *= 2.0 * a  # omega

    np.multiply(root_rs, 0.5 * 7.5957, out=potential)
    np.multiply(rs, 3.5876, out=work)
    potential += work
    np.multiply(rs, root_rs, out=work)
    work *= 1.5 * 1.6382
    potential += work
    np.square(rs, out=work)
    work *= 2.0 * 0.49294
    potential += work
    potential *= 2.0 * a  # d omega / d ln(rs)

    np.reciprocal(epsilon, out=root_rs)
    np.log1p(root_rs, out=root_rs)
    np.add(epsilon, 1.0, out=work)
    work *= epsilon
    np.divide(potential, work, out=potential)
    np.multiply(rs, a1, out=work)
    work += 1.0
    potential *= work
    potential *= -(2.0 / 3.0) * a
    np.multiply(rs, (2.0 / 3.0) * a1, out=work)
    work += 1.0
    work *= root_rs
    work *= -2.0 * a
    potential += work
    np.multiply(rs, a1, out=work)
    work += 1.0
    work *= root_rs
    work *= -2.0 * a
    epsilon[...] = work
    epsilon[~active] = 0.0
    potential[~active] = 0.0
    return epsilon, potential


def pbe_unpolarized(
    rho: np.ndarray,
    gradient: np.ndarray,
    divergence,
    functional: str = "pbe",
) -> tuple[np.ndarray, np.ndarray]:
    """Return a spin-unpolarized PBE-family XC energy and potential.

    ``gradient`` has shape ``(3, *rho.shape)`` and contains Cartesian
    derivatives in bohr^-1. ``divergence`` must apply the matching periodic
    divergence to a vector field.  Splitting the pointwise PBE kernel from
    the spectral differential operators lets the SCF driver use either its
    serial FFT or distributed stick decomposition without gathering grids.
    """
    gradient_array = np.asarray(gradient, dtype=float)
    epsilon, local_potential, coefficient = pbe_unpolarized_components(
        rho, gradient_array, functional=functional
    )
    flux = coefficient[None, ...] * gradient_array
    potential = local_potential - np.asarray(divergence(flux), dtype=float)
    return epsilon, potential


def pbe_unpolarized_components(
    rho: np.ndarray,
    gradient: np.ndarray,
    functional: str = "pbe",
    *,
    need_epsilon: bool = True,
    energy_density_out: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Return PBE-family energy, local potential, and gradient coefficient.

    If ``f_xc = rho * epsilon_xc``, the final array is the scalar
    coefficient ``c`` in ``d f_xc / d(grad rho) = c * grad rho``.  Keeping
    this pointwise quantity available lets the SCF driver form both
    ``-div(c grad rho)`` in the potential and QE's analytic GGA stress.
    If ``need_epsilon`` is false, the first result is ``None``.  Supplying
    ``energy_density_out`` writes ``rho * epsilon_xc`` directly into that
    array, allowing the SCF energy path to avoid a full epsilon owner.
    """
    density_array = np.asarray(rho, dtype=np.float64)
    gradient_array = np.asarray(gradient, dtype=np.float64)
    if gradient_array.shape != (3, *density_array.shape):
        raise ValueError("gradient must have shape (3, *rho.shape)")
    try:
        _PBE_FAMILY_PARAMETERS[functional]
    except KeyError as exc:
        raise ValueError(
            f"unsupported PBE-family functional {functional!r}"
        ) from exc

    energy_density = None
    if energy_density_out is not None:
        energy_density = np.asarray(energy_density_out)
        if energy_density.shape != density_array.shape:
            raise ValueError("energy_density_out must have rho.shape")
        if energy_density.dtype != np.float64:
            raise ValueError("energy_density_out must have dtype float64")
        if not energy_density.flags.writeable:
            raise ValueError("energy_density_out must be writeable")

    point_count = density_array.size
    if point_count <= GGA_POINT_BLOCK_SIZE:
        epsilon, local_potential, coefficient = (
            _pbe_unpolarized_components_block(
                density_array, gradient_array, functional
            )
        )
        if energy_density is not None:
            np.multiply(density_array, epsilon, out=energy_density)
        if not need_epsilon:
            epsilon = None
        return epsilon, local_potential, coefficient

    # MPI real slabs are frequently moveaxis views.  ``empty_like`` would
    # preserve those non-C strides, after which reshape(-1) may return a copy
    # and tile assignments would miss the actual output owner.  Explicit
    # C-order outputs are also the layout consumed by the following ufuncs.
    epsilon = (
        np.empty(density_array.shape, dtype=np.float64, order="C")
        if need_epsilon
        else None
    )
    local_potential = np.empty(
        density_array.shape, dtype=np.float64, order="C"
    )
    coefficient = np.empty(
        density_array.shape, dtype=np.float64, order="C"
    )
    density_flat = density_array.reshape(-1)
    gradient_flat = gradient_array.reshape(3, -1)
    epsilon_flat = None if epsilon is None else epsilon.reshape(-1)
    # ``flat`` writes through even when the MPI slab is a moveaxis view;
    # reshape(-1) could silently create a full-grid copy in that case.
    energy_density_flat = (
        None if energy_density is None else energy_density.flat
    )
    potential_flat = local_potential.reshape(-1)
    coefficient_flat = coefficient.reshape(-1)
    for first in range(0, point_count, GGA_POINT_BLOCK_SIZE):
        last = min(first + GGA_POINT_BLOCK_SIZE, point_count)
        block_epsilon, block_potential, block_coefficient = (
            _pbe_unpolarized_components_block(
                density_flat[first:last],
                gradient_flat[:, first:last],
                functional,
            )
        )
        if epsilon_flat is not None:
            epsilon_flat[first:last] = block_epsilon
        if energy_density_flat is not None:
            block_epsilon *= density_flat[first:last]
            energy_density_flat[first:last] = block_epsilon
        potential_flat[first:last] = block_potential
        coefficient_flat[first:last] = block_coefficient
    return epsilon, local_potential, coefficient


def _pbe_unpolarized_components_block(
    rho: np.ndarray,
    gradient: np.ndarray,
    functional: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the published PBE-family formula on one real-space tile."""
    density = np.abs(np.asarray(rho, dtype=np.float64))
    gradient = np.asarray(gradient, dtype=np.float64)
    kappa, mu, beta, exchange_form = _PBE_FAMILY_PARAMETERS[functional]

    # Start from Slater exchange plus PW92 local correlation.  Correlation is
    # retained until the PBE correction below needs ec and vc; the returned
    # energy and potential are independent copies.
    correlation, correlation_potential = pw92_unpolarized(density)
    epsilon = correlation.copy()
    local_potential = correlation_potential.copy()
    lda_active = density > 1.0e-10
    np.maximum(density, 1.0e-10, out=density)
    work0 = np.cbrt(density)
    work0 *= (
        (-0.687247939924714 * (2.0 / 3.0))
        / 0.6203504908994
    )
    np.add(epsilon, work0, out=epsilon, where=lda_active)
    work0 *= 4.0 / 3.0
    np.add(
        local_potential,
        work0,
        out=local_potential,
        where=lda_active,
    )

    # sigma and the active mask are formed without materializing
    # ``gradient * gradient``.  From here density and sigma are safe n and
    # |grad n| work arrays; the physical input arrays are not modified.
    sigma = np.einsum("i...,i...->...", gradient, gradient)
    gradient_active = density > 1.0e-6
    np.greater(sigma, 1.0e-10, out=lda_active)
    np.logical_and(
        gradient_active,
        lda_active,
        out=gradient_active,
    )
    np.logical_not(gradient_active, out=lda_active)
    # PW92 correctly returns zero below its LDA threshold.  PBE does not use
    # those points, but a benign finite ec keeps the full-grid, mask-free
    # algebra away from the removable A=beta/[gamma*(exp(-ec/gamma)-1)]
    # singularity before inactive outputs are reset below.
    correlation[lda_active] = -0.0310906908696548950
    correlation_potential[lda_active] = 0.0
    np.maximum(density, 1.0e-6, out=density)
    np.maximum(sigma, 1.0e-10, out=sigma)
    np.sqrt(sigma, out=sigma)  # |grad n|

    # PBE-family exchange correction (QE XClib ``pbex``).  Four reusable
    # arrays carry k_F, s, F_x-1, and the current expression.  This preserves
    # the published algebra while avoiding one allocation per named scalar.
    np.cbrt(density, out=work0)
    work0 *= 3.093667726280136  # k_F
    work1 = np.multiply(work0, density)
    work1 *= 2.0
    np.divide(sigma, work1, out=work1)  # reduced gradient s
    work2 = np.square(work1)
    work2 *= mu / kappa
    work3 = np.empty_like(density)
    coefficient = np.empty_like(density)
    if exchange_form == "rational":
        work2 += 1.0  # 1 + mu*s^2/kappa
        np.divide(kappa, work2, out=work3)
        np.subtract(kappa, work3, out=work3)  # F_x - 1
        np.multiply(work1, 2.0 * mu, out=coefficient)
        coefficient /= work2
        coefficient /= work2  # dF_x/ds
    else:
        np.negative(work2, out=work2)
        np.exp(work2, out=work2)
        np.subtract(1.0, work2, out=work3)
        work3 *= kappa  # F_x - 1
        np.multiply(work1, work2, out=coefficient)
        coefficient *= 2.0 * mu  # dF_x/ds

    # First form v2x, then use v2x*|grad n|^2/n to evaluate the last v1x
    # term.  This identity lets k_F and dF_x/ds share their buffers.
    coefficient *= 0.5
    coefficient /= work0
    coefficient /= sigma
    work0 *= -(0.75 / np.pi)  # uniform exchange per particle
    coefficient *= work0  # v2 exchange
    np.multiply(work0, work3, out=work2)  # gradient exchange energy / n
    np.add(epsilon, work2, out=epsilon, where=gradient_active)
    work3 *= work0
    work3 /= 3.0
    work2 += work3
    np.square(sigma, out=work3)
    work3 /= density
    work3 *= coefficient
    work3 *= 4.0 / 3.0
    work2 -= work3  # v1 exchange
    np.add(
        local_potential,
        work2,
        out=local_potential,
        where=gradient_active,
    )

    # PBE correlation correction (QE XClib ``pbec``, iflag=1).  The arrays
    # holding ec and vc become A and A*B only after those local quantities have
    # been consumed, keeping the formula visible without retaining its many
    # intermediate fields simultaneously.
    np.cbrt(density, out=work0)
    np.divide(0.6203504908994, work0, out=work0)  # r_s
    np.divide(1.919158292677513, work0, out=work1)
    np.sqrt(work1, out=work1)
    work1 *= 1.128379167095513  # k_s
    np.multiply(work1, density, out=work2)
    work2 *= 2.0
    np.divide(sigma, work2, out=work2)  # t
    gamma = 0.0310906908696548950
    np.divide(correlation, -gamma, out=work3)
    np.exp(work3, out=work3)  # exp(-ec/gamma)
    correlation_potential -= correlation
    correlation_potential *= work3  # B
    work3 -= 1.0
    np.divide(beta / gamma, work3, out=correlation)  # A
    correlation_potential *= correlation  # A*B
    np.square(work2, out=work0)
    work0 *= correlation  # y=A*t^2
    np.square(work0, out=work3)
    work3 += work0
    work3 += 1.0  # 1+y+y^2
    np.add(work0, 1.0, out=correlation)
    correlation /= work3  # x(y)
    np.add(work0, 2.0, out=work1)
    work1 *= work0
    work1 *= work0
    np.square(work3, out=work3)
    work1 /= work3  # q(y)
    np.square(work2, out=work3)
    work3 *= beta / gamma
    work3 *= correlation
    work3 += 1.0  # logarithm argument

    # dh0 is accumulated in correlation_potential before x and q are reused
    # for v2c.  k_s is reconstructed below from r_s, which costs one sqrt but
    # avoids another persistent grid-sized array.
    correlation_potential /= beta
    correlation_potential -= 7.0 / 3.0
    correlation_potential *= work1
    work0 = np.multiply(correlation, 7.0 / 3.0, out=work0)
    correlation_potential += work0
    correlation_potential *= -1.0
    np.square(work2, out=work0)
    work0 *= beta
    work0 /= work3
    correlation_potential *= work0  # dh0

    correlation -= work1  # x(y)-q(y)
    np.cbrt(density, out=work0)
    np.divide(0.6203504908994, work0, out=work0)
    np.divide(1.919158292677513, work0, out=work0)
    work0 *= 1.128379167095513**2  # k_s^2
    work0 *= 2.0
    work0 *= density
    work0 *= work3
    correlation *= beta
    correlation /= work0  # v2 correlation
    coefficient += correlation

    np.log(work3, out=work0)
    work0 *= gamma  # H0
    np.add(epsilon, work0, out=epsilon, where=gradient_active)
    work0 += correlation_potential
    np.add(
        local_potential,
        work0,
        out=local_potential,
        where=gradient_active,
    )
    coefficient[lda_active] = 0.0
    return epsilon, local_potential, coefficient


def pw92_lda_unpolarized(
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Slater exchange plus PW92 LDA correlation in Hartree."""
    density = np.abs(np.asarray(rho, dtype=np.float64))
    epsilon, potential = pw92_unpolarized(density)
    active = density > 1.0e-10
    np.maximum(density, 1.0e-10, out=density)
    exchange = np.cbrt(density)
    exchange *= (-0.687247939924714 * (2.0 / 3.0)) / 0.6203504908994
    np.add(epsilon, exchange, out=epsilon, where=active)
    exchange *= 4.0 / 3.0
    np.add(potential, exchange, out=potential, where=active)
    return epsilon, potential


def _pw92_component(
    rs: np.ndarray,
    a: float,
    a1: float,
    b1: float,
    b2: float,
    b3: float,
    b4: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized PW92 rational component used by ``pw_spin``."""
    root = np.sqrt(rs)
    rs32 = rs * root
    rs2 = rs * rs
    omega = 2.0 * a * (
        b1 * root + b2 * rs + b3 * rs32 + b4 * rs2
    )
    derivative = 2.0 * a * (
        0.5 * b1 * root
        + b2 * rs
        + 1.5 * b3 * rs32
        + 2.0 * b4 * rs2
    )
    logarithm = np.log1p(1.0 / omega)
    energy = -2.0 * a * (1.0 + a1 * rs) * logarithm
    potential = (
        -2.0 * a * (1.0 + (2.0 / 3.0) * a1 * rs) * logarithm
        - (2.0 / 3.0)
        * a
        * (1.0 + a1 * rs)
        * derivative
        / (omega * (omega + 1.0))
    )
    return energy, potential


def _pz81_correlation_unpolarized(
    rs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    low_rs = rs < 1.0
    logarithm = np.log(rs)
    energy_high = (
        0.0311 * logarithm
        - 0.048
        + 0.0020 * rs * logarithm
        - 0.0116 * rs
    )
    potential_high = (
        0.0311 * logarithm
        - (0.048 + 0.0311 / 3.0)
        + (2.0 / 3.0) * 0.0020 * rs * logarithm
        + (2.0 * -0.0116 - 0.0020) * rs / 3.0
    )
    denominator = 1.0 + 1.0529 * np.sqrt(rs) + 0.3334 * rs
    energy_low = -0.1423 / denominator
    potential_low = energy_low * (
        1.0 + (7.0 / 6.0) * 1.0529 * np.sqrt(rs)
        + (4.0 / 3.0) * 0.3334 * rs
    ) / denominator
    return (
        np.where(low_rs, energy_high, energy_low),
        np.where(low_rs, potential_high, potential_low),
    )


def _pz81_correlation_polarized(
    rs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    low_rs = rs < 1.0
    logarithm = np.log(rs)
    energy_high = (
        0.01555 * logarithm
        - 0.0269
        + 0.0007 * rs * logarithm
        - 0.0048 * rs
    )
    potential_high = (
        0.01555 * logarithm
        + (-0.0269 - 0.01555 / 3.0)
        + (2.0 / 3.0) * 0.0007 * rs * logarithm
        + (2.0 * -0.0048 - 0.0007) * rs / 3.0
    )
    denominator = 1.0 + 1.3981 * np.sqrt(rs) + 0.2611 * rs
    energy_low = -0.0843 / denominator
    potential_low = energy_low * (
        1.0 + (7.0 / 6.0) * 1.3981 * np.sqrt(rs)
        + (4.0 / 3.0) * 0.2611 * rs
    ) / denominator
    return (
        np.where(low_rs, energy_high, energy_low),
        np.where(low_rs, potential_high, potential_low),
    )


def _pw92_spin_correlation(
    rs: np.ndarray, zeta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return QE ``pw_spin`` correlation energy and spin potentials."""
    p43 = 4.0 / 3.0
    denominator = 2.0**p43 - 2.0
    fz = (
        (1.0 + zeta) ** p43
        + (1.0 - zeta) ** p43
        - 2.0
    ) / denominator
    dfz = p43 * (
        np.cbrt(1.0 + zeta) - np.cbrt(1.0 - zeta)
    ) / denominator
    ecu, vcu = _pw92_component(
        rs, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294
    )
    ecp, vcp = _pw92_component(
        rs, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517
    )
    stiffness_energy, stiffness_potential = _pw92_component(
        rs, 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671
    )
    alpha = -stiffness_energy
    alpha_potential = -stiffness_potential
    zeta3 = zeta**3
    zeta4 = zeta3 * zeta
    fz0 = 1.709921
    correlation = (
        ecu
        + alpha * fz * (1.0 - zeta4) / fz0
        + (ecp - ecu) * fz * zeta4
    )
    spin_derivative = (
        alpha
        / fz0
        * (dfz * (1.0 - zeta4) - 4.0 * fz * zeta3)
        + (ecp - ecu) * (dfz * zeta4 + 4.0 * fz * zeta3)
    )
    common = (
        vcu
        + alpha_potential * fz * (1.0 - zeta4) / fz0
        + (vcp - vcu) * fz * zeta4
    )
    return (
        correlation,
        common + spin_derivative * (1.0 - zeta),
        common - spin_derivative * (1.0 + zeta),
    )


def lsda_lda(
    spin_density: np.ndarray,
    functional: str = "pz",
) -> tuple[np.ndarray, np.ndarray]:
    """Return collinear LSDA ``epsilon_xc`` and ``(v_up, v_down)``.

    The implementation follows QE 7.5 XClib's ``xc_lsda``, including the
    Perdew--Zunger 1981 and Perdew--Wang 1992 spin interpolation formulas.
    ``spin_density`` is ordered as majority/up then minority/down.
    """
    spins = np.asarray(spin_density, dtype=np.float64)
    if spins.ndim < 2 or spins.shape[0] != 2:
        raise ValueError("spin_density must have shape (2, ...)")
    if functional not in LDA_FUNCTIONALS:
        raise ValueError(f"unsupported LSDA functional {functional!r}")
    point_count = spins[0].size
    if point_count <= GGA_POINT_BLOCK_SIZE:
        return _lsda_lda_block(spins, functional)

    # LSDA spin interpolation carries substantially more pointwise work arrays
    # than scalar LDA.  Tile it through the same cache-sized blocks used by GGA
    # so large FFT grids do not stream dozens of full-grid temporaries through
    # memory for every v_xc call.
    epsilon = np.empty(spins.shape[1:], dtype=np.float64, order="C")
    potentials = np.empty(spins.shape, dtype=np.float64, order="C")
    spin_flat = spins.reshape(2, -1)
    epsilon_flat = epsilon.reshape(-1)
    potential_flat = potentials.reshape(2, -1)
    for first in range(0, point_count, GGA_POINT_BLOCK_SIZE):
        last = min(first + GGA_POINT_BLOCK_SIZE, point_count)
        block_epsilon, block_potentials = _lsda_lda_block(
            spin_flat[:, first:last], functional
        )
        epsilon_flat[first:last] = block_epsilon
        potential_flat[:, first:last] = block_potentials
    return epsilon, potentials


def _lsda_lda_block(
    spins: np.ndarray,
    functional: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate PZ/PW LSDA on one cache-sized contiguous density block."""
    total = np.abs(spins[0] + spins[1])
    active = total > 1.0e-10
    safe_total = np.maximum(total, 1.0e-10)
    zeta = np.clip((spins[0] - spins[1]) / safe_total, -1.0, 1.0)
    rs = 0.6203504908994 / np.cbrt(safe_total)

    exchange_constant = -1.10783814957303361 * (2.0 / 3.0)
    up_root = np.cbrt(np.maximum(0.0, (1.0 + zeta) * safe_total))
    down_root = np.cbrt(np.maximum(0.0, (1.0 - zeta) * safe_total))
    exchange_up = exchange_constant * up_root
    exchange_down = exchange_constant * down_root
    exchange = 0.5 * (
        (1.0 + zeta) * exchange_up
        + (1.0 - zeta) * exchange_down
    )
    potential_up = (4.0 / 3.0) * exchange_up
    potential_down = (4.0 / 3.0) * exchange_down

    p43 = 4.0 / 3.0
    denominator = 2.0**p43 - 2.0
    fz = (
        (1.0 + zeta) ** p43
        + (1.0 - zeta) ** p43
        - 2.0
    ) / denominator
    dfz = p43 * (
        np.cbrt(1.0 + zeta) - np.cbrt(1.0 - zeta)
    ) / denominator
    if functional == "pz":
        ecu, vcu = _pz81_correlation_unpolarized(rs)
        ecp, vcp = _pz81_correlation_polarized(rs)
        difference = ecp - ecu
        correlation = ecu + fz * difference
        common = vcu + fz * (vcp - vcu)
        correlation_up = common + difference * dfz * (1.0 - zeta)
        correlation_down = common + difference * dfz * (-1.0 - zeta)
    else:
        correlation, correlation_up, correlation_down = (
            _pw92_spin_correlation(rs, zeta)
        )

    epsilon = exchange + correlation
    potentials = np.stack(
        (potential_up + correlation_up, potential_down + correlation_down)
    )
    epsilon[~active] = 0.0
    potentials[:, ~active] = 0.0
    return epsilon, potentials


def pbe_spin_components(
    spin_density: np.ndarray,
    spin_gradient: np.ndarray,
    functional: str = "pbe",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return collinear PBE-family XC pointwise data.

    Results are ``epsilon_xc``, the density-local spin potentials, the two
    exchange flux coefficients, and the common correlation flux coefficient.
    The full variational potential is obtained by subtracting the divergence
    of ``c_x[s] grad(n_s) + c_c grad(n_up+n_down)`` from each local channel.
    This is QE 7.5's ``gcx_spin`` plus ``pbec_spin`` representation.
    """
    spins = np.asarray(spin_density, dtype=np.float64)
    gradients = np.asarray(spin_gradient, dtype=np.float64)
    if spins.ndim < 2 or spins.shape[0] != 2:
        raise ValueError("spin_density must have shape (2, ...)")
    if gradients.shape != (2, 3, *spins.shape[1:]):
        raise ValueError("spin_gradient must have shape (2, 3, ...)")
    try:
        kappa, mu, beta, exchange_form = _PBE_FAMILY_PARAMETERS[functional]
    except KeyError as exc:
        raise ValueError(
            f"unsupported PBE-family functional {functional!r}"
        ) from exc

    total = spins[0] + spins[1]
    total_active = total > 1.0e-10
    safe_total = np.maximum(total, 1.0e-10)
    zeta = np.clip(
        (spins[0] - spins[1]) / safe_total,
        -1.0 + 1.0e-6,
        1.0 - 1.0e-6,
    )
    epsilon_lda, local_potential = lsda_lda(spins, "pw")
    energy_density = total * epsilon_lda
    exchange_coefficients = np.zeros_like(spins)

    for spin in range(2):
        density = 2.0 * spins[spin]
        sigma = 4.0 * np.einsum(
            "i...,i...->...", gradients[spin], gradients[spin]
        )
        active = total_active & (density > 1.0e-10) & (sigma > 1.0e-20)
        safe_density = np.maximum(density, 1.0e-10)
        safe_sigma = np.maximum(sigma, 1.0e-20)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gradient_norm = np.sqrt(safe_sigma)
            kf = 3.093667726280136 * np.cbrt(safe_density)
            reduced = gradient_norm / (2.0 * kf * safe_density)
            argument = mu * reduced * reduced / kappa
            if exchange_form == "rational":
                denominator = 1.0 + argument
                enhancement = kappa * (1.0 - 1.0 / denominator)
                derivative = 2.0 * mu * reduced / denominator**2
            else:
                exponential = np.exp(-argument)
                enhancement = kappa * (1.0 - exponential)
                derivative = 2.0 * mu * reduced * exponential
            exchange_uniform = -(0.75 / np.pi) * kf
            correction_energy = (
                0.5 * safe_density * exchange_uniform * enhancement
            )
            correction_local = exchange_uniform * (
                (4.0 / 3.0) * enhancement
                - (4.0 / 3.0) * reduced * derivative
            )
            coefficient = (
                exchange_uniform
                * derivative
                / (kf * gradient_norm)
            )
        energy_density += np.where(active, correction_energy, 0.0)
        local_potential[spin] += np.where(active, correction_local, 0.0)
        exchange_coefficients[spin] = np.where(active, coefficient, 0.0)

    total_gradient = gradients[0] + gradients[1]
    total_sigma = np.einsum(
        "i...,i...->...", total_gradient, total_gradient
    )
    correlation_active = total_active & (total_sigma > 1.0e-20)
    safe_sigma = np.maximum(total_sigma, 1.0e-20)
    rs = 0.6203504908994 / np.cbrt(safe_total)
    correlation, correlation_up, correlation_down = (
        _pw92_spin_correlation(rs, zeta)
    )
    phi = 0.5 * (
        (1.0 + zeta) ** (2.0 / 3.0)
        + (1.0 - zeta) ** (2.0 / 3.0)
    )
    phi2 = phi * phi
    phi3 = phi2 * phi
    dphi = (
        (1.0 + zeta) ** (-1.0 / 3.0)
        - (1.0 - zeta) ** (-1.0 / 3.0)
    ) / 3.0
    gamma = 0.031091
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        kf = 1.919158292677513 / rs
        ks = 1.128379167095513 * np.sqrt(kf)
        reduced = np.sqrt(safe_sigma) / (
            2.0 * phi * ks * safe_total
        )
        exponential = np.exp(np.clip(
            -correlation / (phi3 * gamma), -700.0, 700.0
        ))
        a_parameter = beta / gamma / (exponential - 1.0)
        b_up = exponential * (correlation_up - correlation) / phi3
        b_down = exponential * (correlation_down - correlation) / phi3
        y = a_parameter * reduced * reduced
        y_denominator = 1.0 + y + y * y
        x_y = (1.0 + y) / y_denominator
        q_y = y * y * (2.0 + y) / y_denominator**2
        logarithm_argument = (
            1.0 + beta / gamma * reduced * reduced * x_y
        )
        h0 = phi3 * gamma * np.log(logarithm_argument)
        common_prefactor = beta * reduced * reduced * phi3 / logarithm_argument
        density_up = common_prefactor * (
            -(7.0 / 3.0) * x_y
            - q_y * (a_parameter * b_up / beta - 7.0 / 3.0)
        )
        density_down = common_prefactor * (
            -(7.0 / 3.0) * x_y
            - q_y * (a_parameter * b_down / beta - 7.0 / 3.0)
        )
        zeta_bracket = (
            3.0 * h0 / phi
            - beta * reduced * reduced * phi2 / logarithm_argument
            * (
                2.0 * x_y
                - q_y * (
                    3.0 * a_parameter * exponential * correlation
                    / (phi3 * beta)
                    + 2.0
                )
            )
        )
        density_up += zeta_bracket * dphi * (1.0 - zeta)
        density_down -= zeta_bracket * dphi * (1.0 + zeta)
        correlation_coefficient = (
            beta * phi / (2.0 * ks * ks * safe_total)
            * (x_y - q_y) / logarithm_argument
        )
    correction_energy = safe_total * h0
    energy_density += np.where(correlation_active, correction_energy, 0.0)
    local_potential[0] += np.where(
        correlation_active, h0 + density_up, 0.0
    )
    local_potential[1] += np.where(
        correlation_active, h0 + density_down, 0.0
    )
    correlation_coefficient = np.where(
        correlation_active, correlation_coefficient, 0.0
    )
    epsilon = np.zeros_like(total)
    np.divide(
        energy_density,
        safe_total,
        out=epsilon,
        where=total_active,
    )
    local_potential[:, ~total_active] = 0.0
    return (
        epsilon,
        local_potential,
        exchange_coefficients,
        correlation_coefficient,
    )


def pz81_unpolarized(
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (epsilon_xc, v_xc) in Hartree for an unpolarized density."""
    # Match QE's XClib LDA driver: it evaluates the functional at |rho|,
    # and returns exactly zero below rho_threshold_lda.
    density = np.abs(np.asarray(rho, dtype=np.float64))
    active = density > 1.0e-10
    np.maximum(density, 1.0e-10, out=density)
    rs = np.cbrt(density)
    np.divide(0.6203504908994, rs, out=rs)
    epsilon_xc = np.divide(
        -0.687247939924714 * (2.0 / 3.0), rs
    )
    potential_xc = (4.0 / 3.0) * epsilon_xc
    low_density_rs = rs < 1.0
    work = np.empty_like(density)
    auxiliary = np.empty_like(density)

    # High-density branch (rs < 1): Perdew-Zunger logarithmic form.
    np.log(rs, out=density)
    np.multiply(density, 0.0311, out=work)
    work -= 0.048
    np.multiply(rs, density, out=auxiliary)
    auxiliary *= 0.0020
    work += auxiliary
    np.multiply(rs, -0.0116, out=auxiliary)
    work += auxiliary
    np.add(epsilon_xc, work, out=epsilon_xc, where=low_density_rs)
    np.add(potential_xc, work, out=potential_xc, where=low_density_rs)
    np.add(density, 1.0, out=auxiliary)
    auxiliary *= 0.0020
    auxiliary -= 0.0116
    auxiliary *= rs
    auxiliary += 0.0311
    auxiliary /= 3.0
    np.subtract(
        potential_xc,
        auxiliary,
        out=potential_xc,
        where=low_density_rs,
    )

    # Low-density branch (rs >= 1): rational Perdew-Zunger form.
    np.sqrt(rs, out=density)
    np.multiply(density, 1.0529, out=work)
    work += 1.0
    np.multiply(rs, 0.3334, out=auxiliary)
    work += auxiliary  # denominator
    np.divide(-0.1423, work, out=auxiliary)
    np.add(epsilon_xc, auxiliary, out=epsilon_xc, where=~low_density_rs)
    np.add(potential_xc, auxiliary, out=potential_xc, where=~low_density_rs)
    np.reciprocal(density, out=auxiliary)
    auxiliary *= 0.5 * 1.0529
    auxiliary += 0.3334
    auxiliary *= rs
    auxiliary *= -0.1423 / 3.0
    np.square(work, out=work)
    auxiliary /= work
    np.add(
        potential_xc,
        auxiliary,
        out=potential_xc,
        where=~low_density_rs,
    )
    epsilon_xc[~active] = 0.0
    potential_xc[~active] = 0.0
    return epsilon_xc, potential_xc
