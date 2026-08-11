from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qepy_pw.upf import (
    AtomicWavefunction,
    LocalPotential,
    RadialProjector,
    _qe_cubic_interpolate_with_derivative,
    _qe_real_spherical_harmonics_with_gradient,
    _qe_simpson,
    read_upf,
)


PSEUDO_DIR = Path(__file__).resolve().parents[2] / "qe_reference" / "upstream" / "pseudo"


@pytest.mark.parametrize(
    ("filename", "element", "valence"),
    [("H.pz-vbc.UPF", "H", 1.0), ("Si.pz-vbc.UPF", "Si", 4.0)],
)
def test_upf_reader_recovers_normalized_nc_metadata(
    filename: str, element: str, valence: float
) -> None:
    pseudo = read_upf(PSEUDO_DIR / filename)
    assert pseudo.element.strip() == element
    assert pseudo.z_valence == pytest.approx(valence)
    assert pseudo.pseudo_type == "NC"
    assert pseudo.r is not None and len(pseudo.r) == pseudo.mesh_size
    assert pseudo.rab is not None and len(pseudo.rab) == pseudo.mesh_size
    assert np.all(np.diff(pseudo.r) > 0.0)


def test_upf_reader_preserves_fully_relativistic_quantum_numbers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "X.rel.UPF"
    path.write_text(
        """<UPF version="2.0.1">
<PP_HEADER element="X" pseudo_type="NC" relativistic="full"
 is_ultrasoft="F" is_paw="F" is_coulomb="F" has_so="T"
 core_correction="F" functional="PBE" z_valence="1"
 mesh_size="3" number_of_wfc="1" number_of_proj="1"/>
<PP_MESH><PP_R>0.1 0.2 0.3</PP_R><PP_RAB>0.1 0.1 0.1</PP_RAB></PP_MESH>
<PP_LOCAL>0.0 0.0 0.0</PP_LOCAL>
<PP_NONLOCAL>
 <PP_BETA.1 index="1" label="2P" angular_momentum="1"
  cutoff_radius_index="3">0.0 0.1 0.0</PP_BETA.1>
 <PP_DIJ>2.0</PP_DIJ>
</PP_NONLOCAL>
<PP_PSWFC><PP_CHI.1 index="1" label="2P" l="1" occupation="1">
 0.0 0.2 0.0</PP_CHI.1></PP_PSWFC>
<PP_RHOATOM>0.0 0.1 0.0</PP_RHOATOM>
<PP_SPIN_ORB>
 <PP_RELWFC.1 index="1" nn="2" lchi="1" jchi="1.5"/>
 <PP_RELBETA.1 index="1" lll="1" jjj="1.5"/>
</PP_SPIN_ORB>
</UPF>""",
        encoding="utf-8",
    )
    pseudo = read_upf(path)
    assert pseudo.fully_relativistic
    assert pseudo.relativistic == "full"
    assert pseudo.has_spin_orbit
    assert pseudo.projectors[0].total_angular_momentum == pytest.approx(1.5)
    assert pseudo.atomic_wavefunctions[0].total_angular_momentum == pytest.approx(1.5)
    assert pseudo.atomic_wavefunctions[0].principal_quantum_number == 2
    beta, coupling = pseudo.spinor_projector_basis(
        np.asarray([[0.2, -0.1, 0.4], [0.0, 0.3, -0.2]]), 80.0
    )
    assert beta.shape == (4, 4)
    assert coupling.shape == (4, 4)
    np.testing.assert_allclose(coupling, coupling.T)
    atomic = pseudo.spinor_atomic_orbital_basis(
        np.asarray([[0.2, -0.1, 0.4], [0.0, 0.3, -0.2]]), 80.0
    )
    assert pseudo.number_of_spinor_atomic_orbitals == 4
    assert pseudo.number_of_averaged_projector_channels == 3
    assert atomic.shape == (4, 4)


def test_disabling_lspinorb_uses_qe_j_averaged_spin_diagonal_projectors() -> None:
    radial_grid = np.linspace(0.1, 0.5, 5)
    pseudo = LocalPotential(
        element="X",
        z_valence=1.0,
        functional="PBE",
        r=radial_grid,
        rab=np.full(5, 0.1),
        vloc_ry=np.zeros(5),
        coulomb=False,
        relativistic="full",
        has_spin_orbit=True,
        projectors=(
            RadialProjector(1, "2P-", 1, np.asarray([0, 1, 2, 1, 0.0]), 5, 0.5),
            RadialProjector(2, "2P+", 1, np.asarray([0, 2, 3, 2, 0.0]), 5, 1.5),
        ),
        dij_ry=np.diag((2.0, 4.0)),
        atomic_wavefunctions=(
            AtomicWavefunction(1, "2P-", 1, 1.0, np.asarray([0, 1, 2, 1, 0.0]), 0.5, 2),
            AtomicWavefunction(2, "2P+", 1, 1.0, np.asarray([0, 2, 3, 2, 0.0]), 1.5, 2),
        ),
    )
    vectors = np.asarray([[0.2, -0.1, 0.4], [0.0, 0.3, -0.2]])
    averaged, coupling = pseudo.averaged_spinor_projector_basis(vectors, 80.0)
    assert averaged.shape == (4, 6)
    assert coupling.shape == (6, 6)
    np.testing.assert_allclose(np.diag(coupling), np.full(6, 5.0 / 3.0))
    np.testing.assert_allclose(averaged[:2, 3:], 0.0)
    np.testing.assert_allclose(averaged[2:, :3], 0.0)

    resolved, _ = pseudo.spinor_projector_basis(vectors, 80.0)
    assert not np.allclose(np.abs(resolved), np.abs(averaged))

    # At +z only Y_10 survives. For j=l-1/2 the two mj columns therefore
    # place it in down and up respectively: (Y_{1,-1} up, Y_10 down), then
    # (Y_10 up, Y_11 down). This guards QE sph_ind's m-1/m convention.
    axial, _ = pseudo.spinor_projector_basis(
        np.asarray([[0.0, 0.0, 0.4]]), 80.0
    )
    assert abs(axial[0, 0]) < 1.0e-14
    assert abs(axial[1, 0]) > 1.0e-10
    assert abs(axial[0, 1]) > 1.0e-10
    assert abs(axial[1, 1]) < 1.0e-14
    averaged_atomic = pseudo.averaged_atomic_orbital_basis(vectors, 80.0)
    assert pseudo.number_of_spinor_atomic_orbitals == 6
    assert pseudo.number_of_averaged_atomic_orbitals == 3
    assert averaged_atomic.shape == (2, 3)


def test_qe_simpson_integrates_quadratic_on_uniform_grid() -> None:
    x = np.arange(5, dtype=float)
    integral = _qe_simpson(x * x, np.ones(5))
    assert integral == pytest.approx(64.0 / 3.0, abs=2.0e-15)


def test_qe_cubic_interpolation_is_exact_for_cubic_polynomial() -> None:
    dq = 0.2
    grid = np.arange(20) * dq
    table = grid**3 - 2.0 * grid**2 + 0.5 * grid - 1.0
    q = np.asarray([0.27, 0.93, 2.41])
    values, derivatives = _qe_cubic_interpolate_with_derivative(table, q, dq)
    np.testing.assert_allclose(values, q**3 - 2.0 * q**2 + 0.5 * q - 1.0, atol=3e-15)
    np.testing.assert_allclose(derivatives, 3.0 * q**2 - 4.0 * q + 0.5, atol=2e-14)


@pytest.mark.parametrize("angular_momentum", [0, 1, 2, 3])
def test_real_spherical_harmonic_gradients_match_finite_difference(
    angular_momentum: int,
) -> None:
    vectors = np.asarray([[0.7, -0.4, 1.1], [-0.8, 0.9, 0.5]])
    values, gradients = _qe_real_spherical_harmonics_with_gradient(
        angular_momentum, vectors
    )
    step = 1.0e-6
    numerical = np.empty_like(gradients)
    for axis in range(3):
        plus = vectors.copy()
        minus = vectors.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        plus_values, _ = _qe_real_spherical_harmonics_with_gradient(
            angular_momentum, plus
        )
        minus_values, _ = _qe_real_spherical_harmonics_with_gradient(
            angular_momentum, minus
        )
        numerical[:, :, axis] = (plus_values - minus_values) / (2.0 * step)
    np.testing.assert_allclose(values.shape, (2, 2 * angular_momentum + 1))
    np.testing.assert_allclose(gradients, numerical, atol=2.0e-9)


def test_local_pseudopotential_radial_derivative_matches_finite_difference() -> None:
    pseudo = read_upf(PSEUDO_DIR / "Si.pz-vbc.UPF")
    q = np.asarray([0.31, 0.84, 1.27])
    volume = 125.0
    values, derivative = pseudo.fourier_with_derivative(q, volume)
    step = 1.0e-6
    numerical = (
        pseudo.fourier(q + step, volume) - pseudo.fourier(q - step, volume)
    ) / (2.0 * step)
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(derivative, numerical, rtol=3.0e-6, atol=2.0e-10)

