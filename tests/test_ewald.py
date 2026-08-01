import numpy as np

from qepy_pw.ewald import ewald_energy, ewald_forces


def test_ewald_matches_qe_full_precision_for_diamond_si():
    alat = 10.18
    lattice = alat * np.array(
        [
            [-0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [-0.5, 0.5, 0.0],
        ]
    )
    fractional = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    positions = fractional @ lattice
    energy_ha = ewald_energy(
        lattice, positions, np.array([4.0, 4.0]), 160.0
    )
    # QE 7.5 data-file-schema.xml stores energies in Hartree.
    assert np.isclose(
        energy_ha, -8.466480241488920, rtol=0.0, atol=1.0e-12
    )


def test_analytic_ewald_force_matches_energy_derivative():
    lattice = np.array(
        [[6.1, 0.0, 0.0], [0.4, 5.7, 0.0], [0.2, 0.3, 6.4]]
    )
    positions = np.array([[0.7, 1.1, 1.4], [3.2, 2.4, 4.1]])
    charges = np.array([1.5, 2.0])
    forces = ewald_forces(lattice, positions, charges, 80.0)
    step = 1.0e-5
    displaced_plus = positions.copy()
    displaced_minus = positions.copy()
    displaced_plus[0, 1] += step
    displaced_minus[0, 1] -= step
    derivative = (
        ewald_energy(lattice, displaced_plus, charges, 80.0)
        - ewald_energy(lattice, displaced_minus, charges, 80.0)
    ) / (2.0 * step)
    assert np.isclose(forces[0, 1], -derivative, atol=2.0e-9)
    assert np.allclose(np.sum(forces, axis=0), 0.0, atol=2.0e-12)
