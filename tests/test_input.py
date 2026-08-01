from io import StringIO
import numpy as np
import pytest

from qepy_pw.errors import QEInputError
from qepy_pw.constants import BOHR_PER_ANGSTROM
from qepy_pw.input import read_pw_input


def test_qe_namelists_cards_and_d_exponent():
    data = StringIO(
        """&CONTROL calculation='scf', pseudo_dir='./p/seudos' /
&SYSTEM ibrav=0, nat=1, ntyp=1, ecutwfc=1.2d1 /
&ELECTRONS conv_thr=1d-8 /
CELL_PARAMETERS angstrom
2 0 0
0 2 0
0 0 2
ATOMIC_SPECIES
H 1 H.UPF
ATOMIC_POSITIONS crystal
H .25 .5 .75
K_POINTS automatic
2 1 1 1 0 0
"""
    )
    pw = read_pw_input(data)
    assert pw.system["ecutwfc"] == 12.0
    assert pw.control["pseudo_dir"] == "./p/seudos"
    assert pw.full_kpoint_count == 2
    assert len(pw.kpoints) == 1
    assert np.isclose(sum(k.weight for k in pw.kpoints), 1.0)
    assert np.allclose(pw.atoms[0].position / pw.lattice.diagonal(), [.25, .5, .75])


def test_ibrav_two_matches_qe_convention():
    data = StringIO(
        """&CONTROL /
&SYSTEM ibrav=2, celldm(1)=10.2, nat=1, ntyp=1, ecutwfc=12 /
&ELECTRONS /
ATOMIC_SPECIES
Si 28 Si.UPF
ATOMIC_POSITIONS alat
Si 0 0 0
K_POINTS gamma
"""
    )
    pw = read_pw_input(data)
    assert np.allclose(pw.lattice[0], [-5.1, 0, 5.1])


def test_qe_ibrav_uses_modern_a_b_c_and_cosine_parameters():
    data = StringIO(
        """&CONTROL /
&SYSTEM ibrav=14, A=4.0, B=5.0, C=6.0, cosAB=0.2, cosAC=0.3, cosBC=0.4, nat=1, ntyp=1, ecutwfc=12 /
&ELECTRONS /
ATOMIC_SPECIES
Si 28 Si.UPF
ATOMIC_POSITIONS crystal
Si 0 0 0
K_POINTS gamma
"""
    )
    pw = read_pw_input(data)
    lengths = np.linalg.norm(pw.lattice, axis=1) / BOHR_PER_ANGSTROM
    assert np.allclose(lengths, [4.0, 5.0, 6.0])
    metric = pw.lattice @ pw.lattice.T
    assert np.isclose(metric[0, 1] / np.sqrt(metric[0, 0] * metric[1, 1]), 0.2)
    assert np.isclose(metric[0, 2] / np.sqrt(metric[0, 0] * metric[2, 2]), 0.3)
    assert np.isclose(metric[1, 2] / np.sqrt(metric[1, 1] * metric[2, 2]), 0.4)


def test_qe_ibrav_uses_celldm_ratios_and_monoclinic_cosine():
    data = StringIO(
        """&CONTROL /
&SYSTEM ibrav=-12, celldm(1)=10, celldm(2)=1.2, celldm(3)=1.4, celldm(5)=0.25, nat=1, ntyp=1, ecutwfc=12 /
&ELECTRONS /
ATOMIC_SPECIES
Si 28 Si.UPF
ATOMIC_POSITIONS crystal
Si 0 0 0
K_POINTS gamma
"""
    )
    pw = read_pw_input(data)
    assert np.allclose(np.linalg.norm(pw.lattice, axis=1), [10.0, 12.0, 14.0])
    assert np.isclose(
        np.dot(pw.lattice[0], pw.lattice[2]) / (10.0 * 14.0), 0.25
    )


@pytest.mark.parametrize("ibrav", [3, -3, 4, 5, -5, 6, 7, 8, 9, -9, 91, 10, 11, 12, -12, 13, -13, 14])
def test_standard_qe_bravais_lattices_have_positive_volume(ibrav: int):
    system = "ibrav=%s, celldm(1)=10, celldm(2)=1.1, celldm(3)=1.2, celldm(4)=0.2, celldm(5)=0.25, celldm(6)=0.3" % ibrav
    data = StringIO(
        "&CONTROL /\n&SYSTEM " + system + ", nat=1, ntyp=1, ecutwfc=12 /\n&ELECTRONS /\n"
        "ATOMIC_SPECIES\nSi 28 Si.UPF\nATOMIC_POSITIONS crystal\nSi 0 0 0\nK_POINTS gamma\n"
    )
    assert read_pw_input(data).volume > 0.0


def test_tprfor_typo_is_rejected_with_tprnfor_guidance():
    data = StringIO(
        """&CONTROL tprfor=.true. /
&SYSTEM ibrav=1, celldm(1)=8, nat=1, ntyp=1, ecutwfc=4 /
&ELECTRONS /
ATOMIC_SPECIES
H 1 H.UPF
ATOMIC_POSITIONS crystal
H 0 0 0
K_POINTS automatic
3 3 3 0 0 0
"""
    )
    with pytest.raises(QEInputError, match="use 'tprnfor'"):
        read_pw_input(data)


def test_tprnfor_keeps_normal_symmetry_reduction():
    data = StringIO(
        """&CONTROL tprnfor=.true. /
&SYSTEM ibrav=1, celldm(1)=8, nat=1, ntyp=1, ecutwfc=4 /
&ELECTRONS /
ATOMIC_SPECIES
H 1 H.UPF
ATOMIC_POSITIONS crystal
H 0 0 0
K_POINTS automatic
3 3 3 0 0 0
"""
    )
    pw = read_pw_input(data)
    assert pw.control["tprnfor"] is True
    assert pw.full_kpoint_count == 27
    assert len(pw.kpoints) < 27
