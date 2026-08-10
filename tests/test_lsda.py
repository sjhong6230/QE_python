from __future__ import annotations

import io

import numpy as np
import pytest

from qepy_pw.errors import QEInputError
from qepy_pw.input import read_pw_input
from qepy_pw.occupations import (
    default_number_of_bands,
    fixed_occupations,
    smeared_occupations,
    spin_electron_counts,
)
from qepy_pw.xc import lsda_lda, pw92_lda_unpolarized, pz81_unpolarized
from qepy_pw.scf import run_scf


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


def test_lsda_scf_runs_two_spin_blocks_and_preserves_spatial_symmetry() -> None:
    pw = read_pw_input(io.StringIO("""\
&CONTROL
  calculation='scf',
  pseudo_dir='./tests/qe_reference/upstream/pseudo',
  disk_io='none', tstress=.true., tprnfor=.true.
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
