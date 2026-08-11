from __future__ import annotations

import numpy as np
import pytest

from qepy_pw.symmetry import (
    DensitySymmetrizer,
    PauliDensitySymmetrizer,
    SymmetryOperation,
    fft_factors,
    find_space_group,
    magnetic_symmetry_operations,
    mesh_compatible_operations,
    reduce_kpoints,
    symmetrize_forces,
    symmetrize_stress,
)


def _identity() -> SymmetryOperation:
    return SymmetryOperation(np.eye(3, dtype=int), np.zeros(3))


def test_fractional_translations_determine_fft_divisibility() -> None:
    operations = (
        _identity(),
        SymmetryOperation(np.eye(3, dtype=int), np.asarray([0.5, 1.0 / 3.0, 0.25])),
    )
    assert fft_factors(operations) == (2, 3, 4)
    assert fft_factors(operations, use_all_frac=True) == (1, 1, 1)


def test_density_projection_is_charge_preserving_and_idempotent() -> None:
    inversion = SymmetryOperation(-np.eye(3, dtype=int), np.zeros(3))
    rng = np.random.default_rng(812)
    density = rng.random((6, 6, 6))
    symmetrizer = DensitySymmetrizer(density.shape, (_identity(), inversion))
    projected = symmetrizer.apply(density)
    second = symmetrizer.apply(projected)
    assert np.sum(projected) == pytest.approx(np.sum(density), abs=3.0e-14)
    np.testing.assert_allclose(second, projected, atol=2.0e-16)


def test_kpoint_reduction_returns_orbit_mapping_and_combined_weights() -> None:
    inversion = SymmetryOperation(-np.eye(3, dtype=int), np.zeros(3))
    coordinates = np.asarray([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0], [0.0, 0.0, 0.0]])
    points, weights, mapping = reduce_kpoints(
        coordinates,
        np.asarray([1.0, 2.0, 1.0]),
        (_identity(), inversion),
        time_reversal=False,
        return_mapping=True,
    )
    np.testing.assert_allclose(points, coordinates[[0, 2]])
    np.testing.assert_allclose(weights, [0.75, 0.25])
    np.testing.assert_array_equal(mapping, [0, 0, 1])


def test_mesh_filter_removes_operations_that_leave_sampled_set() -> None:
    swap_xy = SymmetryOperation(
        np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int), np.zeros(3)
    )
    line = np.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
    compatible = mesh_compatible_operations(line, (_identity(), swap_xy))
    assert len(compatible) == 1
    np.testing.assert_array_equal(compatible[0].matrix, np.eye(3, dtype=int))


def test_cubic_space_group_projects_force_and_stress_irreducibly() -> None:
    lattice = 5.0 * np.eye(3)
    operations = find_space_group(lattice, np.zeros((1, 3)), ["X"])
    assert len(operations) == 48
    force = symmetrize_forces(
        np.asarray([[1.0, -2.0, 0.5]]), lattice, np.zeros((1, 3)), ["X"], operations
    )
    np.testing.assert_allclose(force, 0.0, atol=2.0e-15)
    tensor = np.asarray([[2.0, 0.7, -0.3], [0.1, -1.0, 0.4], [0.2, -0.8, 5.0]])
    projected = symmetrize_stress(tensor, lattice, operations)
    np.testing.assert_allclose(
        projected, np.trace(tensor) / 3.0 * np.eye(3), atol=2.0e-15
    )


def test_magnetic_group_marks_spin_reversing_operations_antiunitary() -> None:
    lattice = 5.0 * np.eye(3)
    spatial = find_space_group(lattice, np.zeros((1, 3)), ["X"])
    magnetic = magnetic_symmetry_operations(
        lattice, spatial, np.asarray([[0.0, 0.0, 1.0]])
    )
    unitary_only = magnetic_symmetry_operations(
        lattice,
        spatial,
        np.asarray([[0.0, 0.0, 1.0]]),
        allow_time_reversal=False,
    )
    assert len(magnetic) > len(unitary_only)
    assert any(operation.time_reversal for operation in magnetic)
    for operation in magnetic:
        rotation = np.linalg.inv(lattice) @ operation.matrix @ lattice
        transformed = np.asarray([0.0, 0.0, 1.0]) @ (
            np.linalg.det(rotation) * rotation
        )
        if operation.time_reversal:
            transformed = -transformed
        np.testing.assert_allclose(transformed, [0.0, 0.0, 1.0], atol=2e-15)


def test_pauli_density_symmetry_rotates_axial_magnetization() -> None:
    lattice = np.eye(3)
    rotation = SymmetryOperation(
        np.asarray([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), np.zeros(3)
    )
    density = np.zeros((4, 2, 2, 2))
    density[0].fill(1.0)
    density[3].fill(0.25)
    symmetrizer = PauliDensitySymmetrizer(
        (2, 2, 2), (_identity(), rotation), lattice
    )
    projected = symmetrizer.apply_pauli(density)
    np.testing.assert_allclose(projected, density)


def test_antiunitary_operation_maps_k_to_minus_rotated_k() -> None:
    anti_identity = SymmetryOperation(
        np.eye(3, dtype=int), np.zeros(3), time_reversal=True
    )
    coordinates = np.asarray([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]])
    points, weights = reduce_kpoints(
        coordinates,
        np.ones(2),
        (anti_identity,),
        time_reversal=False,
    )
    assert len(points) == 1
    np.testing.assert_allclose(weights, [1.0])
