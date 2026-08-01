from io import StringIO

import numpy as np

from qepy_pw.input import read_pw_input
from qepy_pw.mpi import MPIContext
from qepy_pw.symmetry import (
    symmetrize_density,
    symmetrize_forces,
    symmetrize_stress,
)


def _diamond_si(mesh: int = 8):
    return read_pw_input(
        StringIO(
            f"""&CONTROL /
&SYSTEM ibrav=2, celldm(1)=10.18, nat=2, ntyp=1, ecutwfc=20 /
&ELECTRONS /
ATOMIC_SPECIES
Si 28 Si.upf
ATOMIC_POSITIONS crystal
Si 0 0 0
Si .25 .25 .25
K_POINTS automatic
{mesh} {mesh} {mesh} 0 0 0
"""
        )
    )


def _triclinic_p1(*, noinv: bool):
    flag = "noinv=.true.," if noinv else ""
    return read_pw_input(
        StringIO(
            f"""&CONTROL /
&SYSTEM ibrav=0, nat=3, ntyp=3, ecutwfc=4, {flag} /
&ELECTRONS /
CELL_PARAMETERS bohr
3.1 0.0 0.0
0.4 3.7 0.0
0.2 0.6 4.3
ATOMIC_SPECIES
H  1 H.upf
He 4 He.upf
Li 7 Li.upf
ATOMIC_POSITIONS crystal
H  .13 .27 .39
He .22 .44 .71
Li .51 .18 .83
K_POINTS automatic
3 3 3 0 0 0
"""
        )
    )


def test_diamond_space_group_and_8x8x8_irreducible_mesh():
    pw = _diamond_si()
    assert len(pw.symmetry_operations) == 48
    assert pw.full_kpoint_count == 512
    assert len(pw.kpoints) == 29
    assert np.isclose(sum(point.weight for point in pw.kpoints), 1.0)


def test_noinv_disables_only_additional_time_reversal_reduction():
    with_time_reversal = _triclinic_p1(noinv=False)
    without_time_reversal = _triclinic_p1(noinv=True)
    assert len(with_time_reversal.symmetry_operations) == 1
    assert len(without_time_reversal.symmetry_operations) == 1
    assert with_time_reversal.full_kpoint_count == 27
    assert len(with_time_reversal.kpoints) == 14
    assert len(without_time_reversal.kpoints) == 27
    assert np.isclose(
        sum(point.weight for point in without_time_reversal.kpoints),
        1.0,
    )


def test_force_symmorphic_removes_fractional_translation_operations():
    pw = read_pw_input(
        StringIO(
            """&CONTROL /
&SYSTEM ibrav=2, celldm(1)=10.18, nat=2, ntyp=1, ecutwfc=20, force_symmorphic=.true. /
&ELECTRONS /
ATOMIC_SPECIES
Si 28 Si.upf
ATOMIC_POSITIONS crystal
Si 0 0 0
Si .25 .25 .25
K_POINTS automatic
4 4 4 0 0 0
"""
        )
    )
    assert all(np.allclose(operation.translation, 0.0) for operation in pw.symmetry_operations)


def test_nosym_evc_disables_crystal_symmetry_and_completes_bravais_star():
    pw = read_pw_input(
        StringIO(
            """&CONTROL /
&SYSTEM ibrav=1, celldm(1)=8, nat=1, ntyp=1, ecutwfc=4, nosym_evc=.true. /
&ELECTRONS /
ATOMIC_SPECIES
H 1 H.upf
ATOMIC_POSITIONS crystal
H 0 0 0
K_POINTS automatic
2 3 1 0 0 0
"""
        )
    )
    assert pw.system["nosym_evc"] is True
    assert len(pw.symmetry_operations) == 1
    assert pw.full_kpoint_count > 6
    assert len(pw.kpoints) == pw.full_kpoint_count
    assert np.isclose(sum(point.weight for point in pw.kpoints), 1.0)


def test_qe_fractional_translation_rule_rejects_arbitrary_inversion_center():
    pw = read_pw_input(
        StringIO(
            """&CONTROL /
&SYSTEM ibrav=2, celldm(1)=10.18, nat=2, ntyp=1, ecutwfc=20 /
&ELECTRONS /
ATOMIC_SPECIES
Si 28 Si.upf
ATOMIC_POSITIONS crystal
Si 0 0 0
Si .26 .25 .25
K_POINTS automatic
2 2 2 0 0 0
"""
        )
    )
    assert len(pw.symmetry_operations) == 2
    assert all(np.allclose(operation.translation, 0.0) for operation in pw.symmetry_operations)
    assert len(pw.kpoints) == 6


def test_diamond_force_and_stress_projection_has_cubic_symmetry():
    pw = _diamond_si()
    fractional = np.array([atom.position for atom in pw.atoms]) @ np.linalg.inv(
        pw.lattice
    )
    forces = np.array([[1.0, -2.0, 0.5], [-0.3, 0.7, 1.1]])
    projected_forces = symmetrize_forces(
        forces,
        pw.lattice,
        fractional,
        [atom.label for atom in pw.atoms],
        pw.symmetry_operations,
    )
    assert np.allclose(projected_forces, 0.0, atol=2.0e-15)

    stress = np.array(
        [[3.0, 0.4, -0.2], [0.4, 1.0, 0.7], [-0.2, 0.7, 2.0]]
    )
    projected_stress = symmetrize_stress(
        stress, pw.lattice, pw.symmetry_operations
    )
    assert np.allclose(projected_stress, 2.0 * np.eye(3), atol=2.0e-15)


def test_density_symmetrizer_is_charge_preserving_projection():
    pw = _diamond_si()
    density = np.random.default_rng(7).random((8, 8, 8))
    once = symmetrize_density(density, pw.symmetry_operations)
    twice = symmetrize_density(once, pw.symmetry_operations)
    assert np.isclose(np.mean(once), np.mean(density))
    assert np.allclose(twice, once, atol=1e-13)


def test_density_symmetrizer_uses_rounded_indices_on_36_grid():
    pw = _diamond_si()
    shape = (36, 36, 36)
    axes = [
        np.rint(np.fft.fftfreq(size) * size).astype(int)
        for size in shape
    ]
    indices = np.stack(
        np.meshgrid(*axes, indexing="ij"), axis=-1
    )
    # A diamond-compatible structure factor multiplied by a radial function
    # is invariant under the full space group.
    vectors = indices @ pw.reciprocal
    g2 = np.einsum("...j,...j->...", vectors, vectors)
    structure = (
        1.0
        + np.exp(
            -0.5j * np.pi * np.sum(indices, axis=-1)
        )
    )
    density_g = np.exp(-0.02 * g2) * structure
    density_g[g2 > 50.0] = 0.0
    density = np.fft.ifftn(density_g * np.prod(shape)).real
    projected = symmetrize_density(
        density, pw.symmetry_operations
    )
    assert np.allclose(projected, density, atol=2.0e-13)


def test_serial_mpi_symmetry_path_matches_default():
    pw = _diamond_si()
    density = np.random.default_rng(11).random((8, 8, 8))
    expected = symmetrize_density(density, pw.symmetry_operations)
    actual = symmetrize_density(
        density, pw.symmetry_operations, MPIContext()
    )
    assert np.array_equal(actual, expected)


def test_grid_projection_matches_reciprocal_space_group_formula():
    pw = _diamond_si()
    density = np.random.default_rng(19).random((8, 8, 8))
    shape = density.shape
    rho_g = np.fft.fftn(density) / density.size
    axes = [
        np.rint(np.fft.fftfreq(size) * size).astype(int)
        for size in shape
    ]
    g_indices = np.stack(
        np.meshgrid(*axes, indexing="ij"), axis=-1
    )
    expected_g = np.zeros_like(rho_g)
    for operation in pw.symmetry_operations:
        source = g_indices @ operation.matrix.T
        slots = tuple(
            source[..., axis] % shape[axis] for axis in range(3)
        )
        phase = np.exp(
            -2j
            * np.pi
            * np.einsum(
                "...j,j->...", g_indices, operation.translation
            )
        )
        expected_g += rho_g[slots] * phase
    expected = np.fft.ifftn(
        expected_g * density.size / len(pw.symmetry_operations)
    ).real
    actual = symmetrize_density(
        density, pw.symmetry_operations
    )
    assert np.allclose(actual, expected, atol=3.0e-15)
