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
