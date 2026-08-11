from __future__ import annotations

import io

import numpy as np

from qepy_pw.input import read_pw_input
from qepy_pw.pp.namelist import parse_namelist


def test_postprocessing_namelist_accepts_whole_array_assignment() -> None:
    options = parse_namelist(
        """&PLOT
 e1 = 1.0, 0.0, -2.5,
 e2(1) = 0.0, 1.0, 0.0,
 x0 = 0.25,
 nx = 51
/
""",
        "plot",
    )

    assert options == {
        "e1(1)": 1.0,
        "e1(2)": 0.0,
        "e1(3)": -2.5,
        "e2(1)": 0.0,
        "e2(2)": 1.0,
        "e2(3)": 0.0,
        "x0(1)": 0.25,
        "nx": 51,
    }


def test_postprocessing_scalar_assignments_are_not_absorbed_into_array() -> None:
    options = parse_namelist(
        "&PLOT e1=1,0,0, nx=20, fileout='a,b.xsf' /",
        "plot",
    )
    assert [options[f"e1({index})"] for index in range(1, 4)] == [1, 0, 0]
    assert options["nx"] == 20
    assert options["fileout"] == "a,b.xsf"


def test_pw_namelist_accepts_whole_celldm_array_assignment() -> None:
    pw = read_pw_input(io.StringIO("""&CONTROL
 calculation='scf'
/
&SYSTEM
 ibrav=1, celldm=10.2, 0.0, 0.0, 0.0, 0.0, 0.0,
 nat=1, ntyp=1, ecutwfc=12.0
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
ATOMIC_POSITIONS crystal
Si 0.0 0.0 0.0
K_POINTS gamma
"""))

    assert [pw.system[f"celldm({index})"] for index in range(1, 7)] == [
        10.2, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    np.testing.assert_allclose(pw.lattice, np.eye(3) * 10.2)


def test_pw_indexed_array_assignment_can_fill_following_elements() -> None:
    pw = read_pw_input(io.StringIO("""&SYSTEM
 ibrav=1, celldm(1)=9.5, 0, 0, 0, 0, 0,
 nat=1, ntyp=1, ecutwfc=10
/
ATOMIC_SPECIES
H 1.0 H.UPF
ATOMIC_POSITIONS crystal
H 0 0 0
K_POINTS gamma
"""))
    assert pw.system["celldm(1)"] == 9.5
    assert pw.system["celldm(6)"] == 0


def test_single_value_whole_array_assignment_targets_first_element() -> None:
    plot = parse_namelist("&PLOT e1=2.5, nx=10 /", "plot")
    assert plot["e1(1)"] == 2.5
    assert "e1" not in plot

    pw = read_pw_input(io.StringIO("""&SYSTEM
 ibrav=1, celldm=8.0, nat=1, ntyp=1, ecutwfc=10
/
ATOMIC_SPECIES
H 1.0 H.UPF
ATOMIC_POSITIONS crystal
H 0 0 0
K_POINTS gamma
"""))
    assert pw.system["celldm(1)"] == 8.0
