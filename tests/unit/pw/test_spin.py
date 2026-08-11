from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qepy_pw.errors import QEInputError
from qepy_pw.basis import PlaneWaveBasis
from qepy_pw.input import read_pw_input
from qepy_pw.mpi import MPIContext
from qepy_pw.occupations import (
    default_number_of_bands,
    fixed_occupations,
    smeared_occupations,
    spin_electron_counts,
)
from qepy_pw.xc import (
    lsda_lda,
    pbe_spin_components,
    pbe_unpolarized_components,
    pw92_lda_unpolarized,
    pz81_unpolarized,
)
from qepy_pw.scf import SCFResult, run_scf
from qepy_pw.pw.output import format_footer
from qepy_pw.pw.scf import _spinor_nonlocal_derivatives
from qepy_pw.pw.save import (
    read_saved_density,
    read_saved_wavefunction,
    resolve_save_directory,
    write_qe_save,
)


def _lsda_input(system: str, kpoints: str = "K_POINTS gamma") -> str:
    return f"""\
&CONTROL
  calculation = 'scf'
/
&SYSTEM
  ibrav = 1, celldm(1) = 10.0,
  nat = 1, ntyp = 1, ecutwfc = 20.0,
  {system}
/
&ELECTRONS
/
ATOMIC_SPECIES
Fe 55.845 Fe.UPF
ATOMIC_POSITIONS crystal
Fe 0.0 0.0 0.0
{kpoints}
"""


def _read(system: str, kpoints: str = "K_POINTS gamma"):
    return read_pw_input(io.StringIO(_lsda_input(system, kpoints)))


def test_lsda_defaults_and_qe_kpoint_spin_order() -> None:
    pw = _read(
        "nspin=2, occupations='fixed', tot_magnetization=2",
        """K_POINTS crystal
2
0.0 0.0 0.0 0.25
0.5 0.0 0.0 0.75""",
    )
    assert pw.system["starting_magnetization(1)"] == 0.0
    assert pw.system["tot_magnetization"] == 2.0
    assert pw.spatial_kpoint_count == 2
    assert pw.kpoint_spins == (1, 1, 2, 2)
    assert len(pw.kpoints) == 4
    assert np.allclose(pw.kpoints[0].crystal, pw.kpoints[2].crystal)
    assert np.allclose(pw.kpoints[1].crystal, pw.kpoints[3].crystal)
    assert [point.weight for point in pw.kpoints] == [0.25, 0.75, 0.25, 0.75]


def test_scalar_magnetization_defaults_are_explicit() -> None:
    pw = _read("nspin=1, occupations='fixed'")
    assert pw.system["nspin"] == 1
    assert pw.system["starting_magnetization(1)"] == 0.0
    assert pw.system["tot_magnetization"] == -10000.0
    assert pw.kpoint_spins == (1,)


def test_noncollinear_input_promotes_density_components_without_k_duplication() -> None:
    pw = _read(
        "noncolin=.true., lspinorb=.true., starting_spin_angle=.true., no_t_rev=.true., "
        "starting_magnetization(1)=0.4, angle1(1)=90, angle2(1)=35",
        """K_POINTS crystal
2
0.0 0.0 0.0 0.25
0.5 0.0 0.0 0.75""",
    )
    assert pw.system["nspin"] == 4
    assert pw.system["noncolin"] is True
    assert pw.system["lspinorb"] is True
    assert pw.system["starting_spin_angle"] is True
    assert pw.system["no_t_rev"] is True
    assert pw.system["_domag"] is True
    assert pw.system["angle1(1)"] == pytest.approx(90.0)
    assert pw.system["angle2(1)"] == pytest.approx(35.0)
    assert pw.spatial_kpoint_count == 2
    assert pw.kpoint_spins == (1, 1)
    assert len(pw.kpoints) == 2


def test_zero_noncollinear_starting_magnetization_disables_domag() -> None:
    pw = _read(
        "nspin=4, angle1(1)=72, angle2(1)=-15",
        """K_POINTS crystal
1
0.0 0.0 0.0 1.0""",
    )
    assert pw.system["noncolin"] is True
    assert pw.system["_domag"] is False
    assert pw.system["starting_magnetization(1)"] == 0.0


def test_noncollinear_input_rejects_gamma_only_representation() -> None:
    with pytest.raises(QEInputError, match="gamma_only and noncolin"):
        _read("noncolin=.true.")


def test_noncollinear_input_rejects_conflicting_spin_flags() -> None:
    with pytest.raises(QEInputError, match="conflicting flags"):
        _read("nspin=2, noncolin=.true.")
    with pytest.raises(QEInputError, match="lspinorb requires noncolin"):
        _read("lspinorb=.true.")


def test_fixed_lsda_requires_integer_total_magnetization() -> None:
    with pytest.raises(
        QEInputError, match="fixed occupations and lsda need tot_magnetization"
    ):
        _read("nspin=2, occupations='fixed'")
    with pytest.raises(
        QEInputError, match="requires integer tot_magnetization"
    ):
        _read("nspin=2, occupations='fixed', tot_magnetization=0.5")


@pytest.mark.parametrize(
    "system",
    [
        "nspin=1, occupations='fixed', tot_magnetization=0",
        "nspin=2, occupations='smearing', degauss=0.01, tot_magnetization=0",
        "nspin=2, occupations='tetrahedra', tot_magnetization=0",
    ],
)
def test_total_magnetization_is_rejected_outside_fixed_lsda(system: str) -> None:
    with pytest.raises(
        QEInputError,
        match="allowed only for nspin=2 with fixed occupations",
    ):
        _read(system)


def test_lsda_fixed_and_smeared_occupations_have_unit_spin_capacity() -> None:
    assert spin_electron_counts(8.0, 2.0) == (5.0, 3.0)
    fixed = fixed_occupations(6, 5.0, 3.0, 2)
    assert len(fixed) == 4
    assert [float(np.sum(row)) for row in fixed] == [5.0, 5.0, 3.0, 3.0]
    assert all(float(np.max(row)) == 1.0 for row in fixed)

    eigenvalues = [
        np.asarray([-0.8, -0.2, 0.3]),
        np.asarray([-0.7, -0.1, 0.4]),
        np.asarray([-0.6, 0.0, 0.5]),
        np.asarray([-0.5, 0.1, 0.6]),
    ]
    weights = np.asarray([0.4, 0.6, 0.4, 0.6])
    _fermi, occupations, _demet = smeared_occupations(
        eigenvalues,
        weights,
        3.0,
        0.1,
        0,
        spin_degeneracy=1.0,
    )
    assert sum(
        weight * float(np.sum(row))
        for weight, row in zip(weights, occupations)
    ) == pytest.approx(3.0, abs=2.0e-12)
    assert max(float(np.max(row)) for row in occupations) <= 1.0


def test_lsda_default_band_count_uses_larger_spin_population() -> None:
    assert default_number_of_bands(
        8.0,
        "fixed",
        nspin=2,
        tot_magnetization=2.0,
    ) == 5
    assert default_number_of_bands(
        8.0,
        "smearing",
        nspin=2,
        tot_magnetization=-10000.0,
    ) == 8


@pytest.mark.parametrize(
    ("functional", "unpolarized"),
    [("pz", pz81_unpolarized), ("pw", pw92_lda_unpolarized)],
)
def test_lsda_reduces_to_lda_at_zero_spin_polarization(
    functional, unpolarized
) -> None:
    total = np.logspace(-9, 1, 41)
    epsilon, potential = lsda_lda(
        np.stack((0.5 * total, 0.5 * total)), functional
    )
    expected_epsilon, expected_potential = unpolarized(total)
    assert epsilon == pytest.approx(expected_epsilon, abs=2.0e-14)
    assert potential[0] == pytest.approx(expected_potential, abs=2.0e-14)
    assert potential[1] == pytest.approx(expected_potential, abs=2.0e-14)


@pytest.mark.parametrize("functional", ["pz", "pw"])
def test_lsda_potentials_are_energy_density_derivatives(functional: str) -> None:
    spin_density = np.asarray([[0.11, 0.43], [0.07, 0.19]])
    epsilon, potential = lsda_lda(spin_density, functional)
    energy_density = np.sum(spin_density, axis=0) * epsilon
    step = 1.0e-7
    for spin in range(2):
        displaced = spin_density.copy()
        displaced[spin] += step
        plus_epsilon, _ = lsda_lda(displaced, functional)
        plus = np.sum(displaced, axis=0) * plus_epsilon
        displaced[spin] -= 2.0 * step
        minus_epsilon, _ = lsda_lda(displaced, functional)
        minus = np.sum(displaced, axis=0) * minus_epsilon
        derivative = (plus - minus) / (2.0 * step)
        assert derivative == pytest.approx(potential[spin], abs=2.0e-8)
    assert np.all(np.isfinite(energy_density))


def test_lsda_spin_swap_swaps_only_the_potentials() -> None:
    density = np.asarray([[0.31, 0.04], [0.12, 0.27]])
    epsilon, potential = lsda_lda(density, "pz")
    swapped_epsilon, swapped_potential = lsda_lda(density[::-1], "pz")
    assert swapped_epsilon == pytest.approx(epsilon, abs=2.0e-15)
    assert swapped_potential == pytest.approx(potential[::-1], abs=2.0e-15)


@pytest.mark.parametrize("functional", ["pz", "pw"])
def test_large_lsda_grid_tiling_preserves_pointwise_results(functional: str) -> None:
    points = 40_000
    total = np.linspace(1.0e-8, 0.8, points)
    polarization = 0.7 * np.sin(np.linspace(-2.0, 3.0, points))
    density = np.stack((
        0.5 * total * (1.0 + polarization),
        0.5 * total * (1.0 - polarization),
    ))
    epsilon, potential = lsda_lda(density, functional)
    split = points // 2
    left_epsilon, left_potential = lsda_lda(
        density[:, :split], functional
    )
    right_epsilon, right_potential = lsda_lda(
        density[:, split:], functional
    )
    np.testing.assert_array_equal(
        epsilon, np.concatenate((left_epsilon, right_epsilon))
    )
    np.testing.assert_array_equal(
        potential,
        np.concatenate((left_potential, right_potential), axis=1),
    )


@pytest.mark.parametrize("functional", ["pz", "pw"])
def test_lsda_xc_depends_explicitly_on_magnetization(functional: str) -> None:
    total = np.asarray([0.08, 0.3, 1.2])
    unpolarized = np.stack((0.5 * total, 0.5 * total))
    zeta = 0.6
    polarized = np.stack((
        0.5 * total * (1.0 + zeta),
        0.5 * total * (1.0 - zeta),
    ))
    epsilon_zero, potential_zero = lsda_lda(unpolarized, functional)
    epsilon_m, potential_m = lsda_lda(polarized, functional)
    assert not np.allclose(epsilon_m, epsilon_zero, atol=1e-12)
    assert np.allclose(potential_zero[0], potential_zero[1], atol=1e-14)
    assert np.all(potential_m[0] < potential_m[1])

    # At fixed n, d[n*epsilon_xc(n,m)]/dm is the LSDA magnetic field,
    # one half of the up/down potential splitting.
    step = 1.0e-7
    plus = polarized.copy()
    minus = polarized.copy()
    plus[0] += 0.5 * step
    plus[1] -= 0.5 * step
    minus[0] -= 0.5 * step
    minus[1] += 0.5 * step
    epsilon_plus, _ = lsda_lda(plus, functional)
    epsilon_minus, _ = lsda_lda(minus, functional)
    derivative = total * (epsilon_plus - epsilon_minus) / (2.0 * step)
    np.testing.assert_allclose(
        derivative,
        0.5 * (potential_m[0] - potential_m[1]),
        atol=2e-8,
    )


@pytest.mark.parametrize("functional", ["pbe", "pbesol", "revpbe", "rpbe"])
def test_spin_gga_reduces_to_unpolarized_gga(functional: str) -> None:
    total = np.asarray([0.08, 0.3, 1.2])
    gradient = np.asarray([
        [0.013, -0.027, 0.041],
        [-0.008, 0.019, 0.033],
        [0.022, 0.011, -0.017],
    ])
    epsilon, local, exchange_coefficient, correlation_coefficient = (
        pbe_spin_components(
            np.stack((0.5 * total, 0.5 * total)),
            np.stack((0.5 * gradient, 0.5 * gradient)),
            functional,
        )
    )
    expected_epsilon, expected_local, expected_coefficient = (
        pbe_unpolarized_components(total, gradient, functional)
    )
    np.testing.assert_allclose(epsilon, expected_epsilon, atol=2e-12)
    np.testing.assert_allclose(local[0], expected_local, atol=2e-12)
    np.testing.assert_allclose(local[1], expected_local, atol=2e-12)
    for spin in range(2):
        flux = (
            exchange_coefficient[spin, None, ...]
            * 0.5
            * gradient
            + correlation_coefficient[None, ...] * gradient
        )
        np.testing.assert_allclose(
            flux,
            expected_coefficient[None, ...] * gradient,
            rtol=2e-5,
            atol=2e-9,
        )


@pytest.mark.parametrize("functional", ["pbe", "pbesol", "revpbe", "rpbe"])
def test_spin_gga_components_are_functional_derivatives(functional: str) -> None:
    density = np.asarray([[0.31], [0.13]])
    gradient = np.asarray([
        [[0.041], [-0.023], [0.017]],
        [[-0.012], [0.031], [0.026]],
    ])
    epsilon, local, exchange_coefficient, correlation_coefficient = (
        pbe_spin_components(density, gradient, functional)
    )
    total = np.sum(density, axis=0)
    step = 1.0e-7
    for spin in range(2):
        plus = density.copy()
        minus = density.copy()
        plus[spin] += step
        minus[spin] -= step
        epsilon_plus, *_ = pbe_spin_components(plus, gradient, functional)
        epsilon_minus, *_ = pbe_spin_components(minus, gradient, functional)
        derivative = (
            np.sum(plus, axis=0) * epsilon_plus
            - np.sum(minus, axis=0) * epsilon_minus
        ) / (2.0 * step)
        np.testing.assert_allclose(derivative, local[spin], atol=2e-8)

        for direction in range(3):
            plus_gradient = gradient.copy()
            minus_gradient = gradient.copy()
            plus_gradient[spin, direction] += step
            minus_gradient[spin, direction] -= step
            epsilon_plus, *_ = pbe_spin_components(
                density, plus_gradient, functional
            )
            epsilon_minus, *_ = pbe_spin_components(
                density, minus_gradient, functional
            )
            derivative = total * (epsilon_plus - epsilon_minus) / (2.0 * step)
            expected_flux = (
                exchange_coefficient[spin]
                * gradient[spin, direction]
                + correlation_coefficient
                * (gradient[0, direction] + gradient[1, direction])
            )
            np.testing.assert_allclose(
                derivative, expected_flux, atol=2e-8
            )


def test_lsda_scf_runs_two_spin_blocks_and_preserves_spatial_symmetry(
    tmp_path: Path,
) -> None:
    pw = read_pw_input(io.StringIO("""\
&CONTROL
  calculation='scf',
  pseudo_dir='./tests/qe_reference/upstream/pseudo',
  disk_io='medium', prefix='lsda-h2', tstress=.true., tprnfor=.true.
/
&SYSTEM
  ibrav=1, celldm(1)=10.0, nat=2, ntyp=1,
  ecutwfc=12.0, nbnd=1, nspin=2,
  occupations='fixed', tot_magnetization=0
/
&ELECTRONS
  conv_thr=1.d-5, electron_maxstep=30
/
ATOMIC_SPECIES
H 1.0008 H.pz-vbc.UPF
ATOMIC_POSITIONS angstrom
H 0.00 0.00 -0.35
H 0.00 0.00  0.35
K_POINTS gamma
"""))
    pw.control["outdir"] = str(tmp_path)
    result = run_scf(pw)
    assert result.converged
    assert result.density.shape[0] == 2
    assert pw.kpoint_spins == (1, 2)
    assert len(result.eigenvalues_ha) == 2
    assert result.forces_ha_per_bohr is not None
    assert result.stress_ha_per_bohr3 is not None
    assert [row[0] for row in result.occupations] == [1.0, 1.0]
    # A nonmagnetic starting state must remain invariant under spin exchange;
    # the crystal density symmetrizer is applied independently to each block.
    np.testing.assert_allclose(result.density[0], result.density[1], atol=1e-6)

    output = format_footer(pw, result)
    assert "SPIN UP" in output and "SPIN DOWN" in output
    assert "total magnetization" in output
    assert "    1.0000" in output

    write_qe_save(pw, result)
    save = resolve_save_directory(pw)
    import h5py

    with h5py.File(save / "charge-density.hdf5", "r") as h5:
        assert h5.attrs["nspin"] == 2
        assert "rhotot_g" in h5 and "rhodiff_g" in h5
    with h5py.File(save / "wfc1.hdf5", "r") as up:
        assert up.attrs["ispin"] == 1
    with h5py.File(save / "wfc2.hdf5", "r") as down:
        assert down.attrs["ispin"] == 2

    pw.control["calculation"] = "nscf"
    pw.control["tstress"] = False
    pw.control["tprnfor"] = False
    pw.electrons["startingpot"] = "file"
    pw.electrons["startingwfc"] = "file"
    nscf = run_scf(pw)
    assert nscf.converged
    np.testing.assert_allclose(nscf.density, result.density, atol=2e-12)


def test_noncollinear_save_round_trip_uses_qe_pauli_and_spinor_layouts(
    tmp_path: Path,
) -> None:
    pw = read_pw_input(io.StringIO("""\
&CONTROL
  calculation='scf', pseudo_dir='./tests/qe_reference/upstream/pseudo',
  disk_io='medium', prefix='noncolin-save'
/
&SYSTEM
  ibrav=1, celldm(1)=10.0, nat=1, ntyp=1,
  ecutwfc=12.0, nbnd=1, noncolin=.true., lspinorb=.true.,
  starting_magnetization(1)=0.4, angle1(1)=70.0, angle2(1)=-25.0
/
&ELECTRONS /
ATOMIC_SPECIES
H 1.0008 H.pz-vbc.UPF
ATOMIC_POSITIONS angstrom
H 0.0 0.0 0.0
K_POINTS crystal
1
0.0 0.0 0.0 1.0
"""))
    pw.control["outdir"] = str(tmp_path)
    shape = (4, 4, 4)
    charge = np.full(shape, 1.0 / pw.volume)
    density = np.stack((charge, 0.2 * charge, -0.1 * charge, 0.3 * charge))
    spinor = np.asarray([[1.0], [1.0j]]) / np.sqrt(2.0)
    result = SCFResult(
        converged=True,
        total_energy_ha=-0.5,
        eigenvalues_ha=[np.asarray([-0.5])],
        density=density,
        plane_waves_per_k=[1],
        occupations=[np.asarray([1.0])],
        wavefunctions=[spinor],
        wavefunction_miller_indices=[np.zeros((1, 3), dtype=np.int32)],
    )

    write_qe_save(pw, result)
    save = resolve_save_directory(pw)
    import h5py

    with h5py.File(save / "charge-density.hdf5", "r") as h5:
        assert h5.attrs["nspin"] == 4
        assert set(("rhotot_g", "m_x", "m_y", "m_z")) <= set(h5)
    with h5py.File(save / "wfc1.hdf5", "r") as h5:
        assert h5.attrs["npol"] == 2
        assert h5["evc"].shape == (1, 2)

    restored_density = read_saved_density(pw, shape, expected_electrons=1.0)
    restored_spinor = read_saved_wavefunction(
        pw, 0, np.zeros((1, 3), dtype=np.int32), 1
    )
    np.testing.assert_allclose(restored_density, density, atol=1e-14)
    np.testing.assert_allclose(restored_spinor, spinor, atol=1e-14)

    footer = format_footer(pw, result)
    assert "    1.0000" in footer
    assert "total magnetization" in footer


def test_starting_magnetization_splits_initial_spin_hamiltonians(
    monkeypatch,
) -> None:
    import qepy_pw.pw.scf as scf_module

    pw = read_pw_input(io.StringIO("""\
&CONTROL
  calculation='scf',
  pseudo_dir='./tests/qe_reference/upstream/pseudo',
  disk_io='none', tstress=.false.
/
&SYSTEM
  ibrav=1, celldm(1)=10.0, nat=2, ntyp=1,
  ecutwfc=12.0, nbnd=1, nspin=2,
  occupations='fixed', tot_magnetization=0,
  starting_magnetization(1)=0.5
/
&ELECTRONS
  conv_thr=1.d-14, electron_maxstep=1, startingwfc='atomic'
/
ATOMIC_SPECIES
H 1.0008 H.pz-vbc.UPF
ATOMIC_POSITIONS angstrom
H 0.00 0.00 -0.35
H 0.00 0.00  0.35
K_POINTS gamma
"""))
    projected_eigenvalues: list[np.ndarray] = []
    original_rotate = scf_module._rotate_starting_subspace

    def tracked_rotate(*args, **kwargs):
        vectors, values, applied = original_rotate(*args, **kwargs)
        projected_eigenvalues.append(values.copy())
        return vectors, values, applied

    monkeypatch.setattr(scf_module, "_rotate_starting_subspace", tracked_rotate)
    run_scf(pw)

    # QE uses the same scalar atomic orbitals for both collinear channels,
    # then rotates each trial subspace with its own spin-dependent initial
    # Hamiltonian. A nonzero starting moment must therefore split these two
    # projected spectra before the first Davidson solve.
    assert len(projected_eigenvalues) == 2
    assert not np.allclose(
        projected_eigenvalues[0], projected_eigenvalues[1], atol=1e-10
    )


def test_spin_gga_scf_force_and_stress_paths() -> None:
    # The analytic kernel tests above cover every supported PBE variant.
    # One representative end-to-end calculation is sufficient because all
    # variants share the same SCF, force, and stress plumbing.
    pw = read_pw_input(io.StringIO("""\
&CONTROL
  calculation='scf',
  pseudo_dir='./tests/qe_reference/upstream/pseudo',
  disk_io='none', tstress=.true., tprnfor=.true.
/
&SYSTEM
  ibrav=1, celldm(1)=10.0, nat=2, ntyp=1,
  ecutwfc=10.0, nbnd=1, nspin=2, input_dft='PBE',
  occupations='fixed', tot_magnetization=0,
  starting_magnetization(1)=0.25
/
&ELECTRONS
  conv_thr=2.d-5, electron_maxstep=30
/
ATOMIC_SPECIES
H 1.0008 H.pz-vbc.UPF
ATOMIC_POSITIONS angstrom
H 0.00 0.00 -0.35
H 0.00 0.00  0.35
K_POINTS gamma
"""))
    result = run_scf(pw)
    assert result.converged
    assert np.isfinite(result.total_energy_ha)
    assert np.all(np.isfinite(result.density))
    assert result.forces_ha_per_bohr is not None
    assert np.all(np.isfinite(result.forces_ha_per_bohr))
    assert result.stress_ha_per_bohr3 is not None
    assert np.all(np.isfinite(result.stress_ha_per_bohr3))


def test_spinor_nonlocal_force_is_negative_atomic_energy_derivative() -> None:
    class AnalyticSpinorPseudo:
        @staticmethod
        def spinor_projector_basis(
            gk: np.ndarray, volume: float
        ) -> tuple[np.ndarray, np.ndarray]:
            radial = np.exp(-0.17 * np.einsum("gi,gi->g", gk, gk))
            angular = 1.0 + 0.13j * gk[:, 0] - 0.07 * gk[:, 1]
            up = radial * angular / np.sqrt(volume)
            down = radial * (0.31 - 0.19j) / np.sqrt(volume)
            return np.concatenate((up, down))[:, None], np.asarray([[0.83]])

    basis = PlaneWaveBasis(
        np.asarray(((0, 0, 0), (1, 0, 0), (0, -1, 1)), dtype=np.int32),
        np.asarray(((0.11, -0.23, 0.07), (0.72, 0.08, -0.12),
                    (-0.18, -0.61, 0.49))),
        np.asarray((0.0, 0.0, 0.0)),
    )
    rng = np.random.default_rng(9713)
    coefficients = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    coefficients /= np.linalg.norm(coefficients, axis=0, keepdims=True)
    position = np.asarray((0.29, -0.17, 0.41))
    pseudo = AnalyticSpinorPseudo()
    workspace = SimpleNamespace(local_plane_wave_indices=np.arange(3))

    def evaluate(atom_position: np.ndarray, *, forces: bool):
        pw = SimpleNamespace(
            system={"noncolin": True, "lspinorb": True},
            atoms=[SimpleNamespace(label="X", position=atom_position)],
            volume=17.0,
        )
        return _spinor_nonlocal_derivatives(
            pw,
            {"X": pseudo},
            [basis],
            [coefficients],
            np.asarray((0.75,)),
            [np.asarray((0.9, 0.35))],
            MPIContext(),
            [workspace],
            forces=forces,
            stress=False,
        )

    _, force, _ = evaluate(position, forces=True)
    assert force is not None
    step = 2.0e-6
    numerical = np.empty(3)
    for direction in range(3):
        displacement = np.zeros(3)
        displacement[direction] = step
        plus = evaluate(position + displacement, forces=False)[0]
        minus = evaluate(position - displacement, forces=False)[0]
        numerical[direction] = -(plus - minus) / (2.0 * step)
    np.testing.assert_allclose(force[0], numerical, rtol=2.0e-8, atol=2.0e-10)
