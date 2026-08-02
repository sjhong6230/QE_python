from pathlib import Path

import numpy as np

from qepy_pw.upf import _qe_real_spherical_harmonics, read_upf


def test_coulomb_upf_transform():
    pseudo = read_upf(Path(__file__).parents[1] / "examples" / "H.coulomb.UPF")
    q = np.array([0.0, 2.0])
    values, derivative = pseudo.fourier_with_derivative(q, volume=10.0)
    assert values[0] == derivative[0] == 0.0
    assert np.isclose(values[1], -np.pi / 10.0)
    assert np.isclose(derivative[1], np.pi / 10.0)


def test_norm_conserving_projector_channels_and_hermiticity():
    pseudo = read_upf(Path(__file__).parent / "data" / "He.local-nc.UPF")
    assert len(pseudo.projectors) == 3
    assert pseudo.number_of_projector_channels == 5
    assert pseudo.has_nlcc
    assert pseudo.has_atomic_density
    assert len(pseudo.atomic_wavefunctions) == 1
    assert pseudo.number_of_atomic_orbitals == 1
    core_charge = pseudo.core_density_fourier(np.array([0.0]), 20.0)[0] * 20.0
    atomic_charge = (
        pseudo.atomic_density_fourier(np.array([0.0]), 20.0)[0] * 20.0
    )
    assert core_charge > 0.0
    assert np.isclose(atomic_charge, pseudo.z_valence)
    gk = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [-1.0, 0.5, 0.0]]
    )
    beta, coupling = pseudo.atomic_projectors(gk, np.array([0.2, 0.1, 0.3]), 20.0)
    nonlocal_h = beta @ coupling @ beta.conj().T
    assert beta.shape == (4, 5)
    assert coupling.shape == (5, 5)
    assert coupling[0, 1] == coupling[1, 0] == 0.25  # PP_DIJ Ry -> Ha
    assert np.allclose(beta[0, 2:], 0.0, atol=1e-14)
    assert np.allclose(nonlocal_h, nonlocal_h.conj().T, atol=1e-13)
    orbitals = pseudo.atomic_orbitals(
        gk, np.array([0.2, 0.1, 0.3]), 20.0
    )
    centered = pseudo.atomic_orbital_basis(gk, 20.0)
    phase = np.exp(-1j * (gk @ np.array([0.2, 0.1, 0.3])))
    assert np.array_equal(orbitals, centered * phase[:, None])
    assert orbitals.shape == (4, 1)
    assert np.all(np.isfinite(orbitals))
    assert np.linalg.norm(orbitals) > 0.0


def test_frozen_core_fourier_derivative_matches_finite_difference():
    pseudo = read_upf(Path(__file__).parent / "data" / "He.local-nc.UPF")
    q = np.array([0.0, 0.4, 1.3, 2.1])
    derivative = pseudo.core_density_fourier_derivative(q, 20.0)
    step = 1.0e-5
    finite = np.zeros_like(q)
    finite[1:] = (
        pseudo.core_density_fourier(q[1:] + step, 20.0)
        - pseudo.core_density_fourier(q[1:] - step, 20.0)
    ) / (2.0 * step)
    # QE differentiates its one-sided four-point cubic table even at q=0;
    # the exact analytic derivative is zero but the interpolant leaves a
    # tiny O(dq^3) endpoint value.
    assert abs(derivative[0]) < 2.0e-8
    assert np.allclose(derivative[1:], finite[1:], atol=2.0e-11)


def test_frozen_core_uses_qe_dq_table_and_reuses_it():
    pseudo = read_upf(Path(__file__).parent / "data" / "He.local-nc.UPF")
    q = np.array([0.37, 0.91, 1.43])
    first = pseudo.core_density_fourier(q, 20.0)
    table = pseudo._core_density_table
    assert table is not None
    second = pseudo.core_density_fourier(q[::-1], 20.0)
    assert pseudo._core_density_table is table
    assert np.array_equal(second, first[::-1])


def test_analytic_projector_gradient_including_polar_axis():
    pseudo = read_upf(Path(__file__).parent / "data" / "He.local-nc.UPF")
    gk = np.array(
        [[0.4, 0.7, 0.9], [0.0, 0.0, 1.2], [0.0, 0.0, -1.4]]
    )
    _, _, gradient = pseudo.projector_basis_with_gradient(gk, 20.0)
    step = 1.0e-4
    for axis in range(3):
        plus = gk.copy()
        minus = gk.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        beta_plus, _ = pseudo.projector_basis(plus, 20.0)
        beta_minus, _ = pseudo.projector_basis(minus, 20.0)
        finite_gradient = (beta_plus - beta_minus) / (2.0 * step)
        assert np.allclose(
            gradient[:, :, axis], finite_gradient, atol=2.0e-8
        )


def test_qe_real_harmonic_l1_order_and_signs():
    vectors = np.eye(3)
    harmonics = _qe_real_spherical_harmonics(1, vectors)
    scale = np.sqrt(3.0 / (4.0 * np.pi))
    # QE order is z, -x, -y for l=1.
    assert np.allclose(harmonics, [[0, -scale, 0], [0, 0, -scale], [scale, 0, 0]])


def test_real_qe_fe_upf_with_fortran_info_and_nlcc():
    root = Path(__file__).parents[1]
    path = root / "quantum-espresso" / "pseudo" / "Fe.pz-n-nc.UPF"
    if not path.exists():
        return
    pseudo = read_upf(path)
    transform = pseudo.core_density_fourier(np.array([0.0, 1.0, 3.0]), 100.0)
    assert pseudo.has_nlcc
    assert len(pseudo.projectors) == 2
    assert pseudo.number_of_projector_channels == 8
    assert transform[0] * 100.0 > 0.0
    assert np.all(np.isfinite(transform))
