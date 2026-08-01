"""Small crystallographic symmetry engine for scalar periodic calculations.

Fractional coordinates are row vectors. A space-group operation is represented
as ``x' = x @ matrix + translation`` modulo lattice vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .mpi import MPIContext


@dataclass(frozen=True)
class SymmetryOperation:
    matrix: np.ndarray
    translation: np.ndarray


class DensitySymmetrizer:
    """Reusable scalar-density projection plan for one FFT grid."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        operations: tuple[SymmetryOperation, ...],
        mpi: "MPIContext | None" = None,
    ) -> None:
        self.shape = tuple(int(size) for size in shape)
        self.operations = operations
        self.mpi = mpi
        rank = 0 if mpi is None else mpi.rank
        processes = 1 if mpi is None else mpi.size
        distributed_grid = mpi is not None and processes > 1
        local_operations = () if distributed_grid else operations
        coordinates = [
            np.arange(size, dtype=float) / size
            for size in self.shape
        ]
        x = coordinates[0][:, None, None]
        y = coordinates[1][None, :, None]
        z = coordinates[2][None, None, :]
        mappings = []
        for operation in local_operations:
            slots = []
            for axis, size in enumerate(self.shape):
                transformed = (
                    x * operation.matrix[0, axis]
                    + y * operation.matrix[1, axis]
                    + z * operation.matrix[2, axis]
                    + operation.translation[axis]
                )
                scaled = transformed * size
                rounded = np.rint(scaled)
                if not np.allclose(
                    scaled, rounded, rtol=0.0, atol=1.0e-7
                ):
                    raise ValueError(
                        "symmetry operation is incompatible with FFT grid"
                    )
                slots.append(rounded.astype(np.int32) % size)
            linear = (
                (slots[0] * self.shape[1] + slots[1])
                * self.shape[2]
                + slots[2]
            )
            mappings.append(np.asarray(linear, dtype=np.int32).ravel())
        self.mappings = (
            np.stack(mappings)
            if mappings
            else np.empty((0, int(np.prod(self.shape))), dtype=np.int32)
        )
        self._permuted = np.empty(int(np.prod(self.shape)), dtype=float)
        self.local_plans: list[
            tuple[list[np.ndarray], list[np.ndarray]]
        ] = []
        if distributed_grid and len(operations) > 1:
            z_slab = mpi.slab(self.shape[2])
            local_coordinates = [
                np.arange(self.shape[0], dtype=float) / self.shape[0],
                np.arange(self.shape[1], dtype=float) / self.shape[1],
                np.arange(z_slab.start, z_slab.stop, dtype=float)
                / self.shape[2],
            ]
            grids = np.meshgrid(*local_coordinates, indexing="ij")
            stops = np.asarray(
                [
                    mpi.slab_for_rank(self.shape[2], owner).stop
                    for owner in range(processes)
                ]
            )
            for operation in operations:
                slots = []
                for axis, size in enumerate(self.shape):
                    transformed = operation.translation[axis]
                    for source_axis in range(3):
                        transformed = (
                            transformed
                            + grids[source_axis]
                            * operation.matrix[source_axis, axis]
                        )
                    slots.append(
                        np.rint(transformed * size).astype(np.int32) % size
                    )
                owners = np.searchsorted(
                    stops, slots[2].ravel(), side="right"
                )
                requests: list[np.ndarray] = []
                positions: list[np.ndarray] = []
                for owner in range(processes):
                    selected = np.flatnonzero(owners == owner)
                    owner_slab = mpi.slab_for_rank(self.shape[2], owner)
                    source_index = (
                        (
                            slots[0].ravel()[selected] * self.shape[1]
                            + slots[1].ravel()[selected]
                        )
                        * (owner_slab.stop - owner_slab.start)
                        + slots[2].ravel()[selected]
                        - owner_slab.start
                    )
                    requests.append(
                        np.asarray(source_index, dtype=np.int64)
                    )
                    positions.append(selected)
                self.local_plans.append((requests, positions))

    @property
    def mapping_bytes(self) -> int:
        return int(
            self.mappings.nbytes
            + sum(
                array.nbytes
                for requests, positions in self.local_plans
                for array in (*requests, *positions)
            )
        )

    def apply(self, density: np.ndarray) -> np.ndarray:
        array = np.asarray(density, dtype=float)
        if (
            self.mpi is not None
            and self.mpi.size > 1
            and array.shape
            == (
                self.shape[0],
                self.shape[1],
                self.mpi.slab(self.shape[2]).stop
                - self.mpi.slab(self.shape[2]).start,
            )
        ):
            if len(self.operations) <= 1:
                return array
            source = array.ravel()
            averaged = np.zeros_like(source)
            for requests, positions in self.local_plans:
                incoming_requests = self.mpi.comm.alltoall(requests)
                responses = [
                    np.ascontiguousarray(source[request])
                    for request in incoming_requests
                ]
                incoming_values = self.mpi.comm.alltoall(responses)
                for target_positions, values in zip(
                    positions, incoming_values
                ):
                    averaged[target_positions] += values
            return (averaged / len(self.operations)).reshape(array.shape)
        if array.shape != self.shape:
            raise ValueError("density does not match symmetry FFT grid")
        if len(self.operations) <= 1:
            return array
        source = array.ravel()
        averaged = np.zeros(source.size, dtype=float)
        for mapping in self.mappings:
            np.take(source, mapping, out=self._permuted)
            averaged += self._permuted
        result = averaged.reshape(self.shape)
        if self.mpi is not None and self.mpi.size > 1:
            result = self.mpi.sum_array(result)
        result /= len(self.operations)
        return result


def fft_factors(
    operations: tuple[SymmetryOperation, ...],
    tolerance: float = 1.0e-7,
) -> tuple[int, int, int]:
    """Return QE-style FFT divisibility factors for fractional translations."""
    factors = [1, 1, 1]
    for operation in operations:
        for axis, value in enumerate(operation.translation):
            reduced = float(value) % 1.0
            fraction = Fraction(reduced).limit_denominator(48)
            if abs(float(fraction) - reduced) <= tolerance:
                factors[axis] = math.lcm(factors[axis], fraction.denominator)
    return tuple(factors)


def _periodic_match(
    transformed: np.ndarray,
    labels: list[str],
    reference: np.ndarray,
    tolerance: float,
) -> bool:
    for label in set(labels):
        left = transformed[np.array([item == label for item in labels])]
        right = reference[np.array([item == label for item in labels])]
        unused = list(range(len(right)))
        for position in left:
            match = None
            for index in unused:
                delta = position - right[index]
                delta -= np.rint(delta)
                if np.linalg.norm(delta) < tolerance:
                    match = index
                    break
            if match is None:
                return False
            unused.remove(match)
    return True


def find_space_group(
    lattice: np.ndarray,
    fractional_positions: np.ndarray,
    labels: list[str],
    tolerance: float = 1.0e-7,
) -> tuple[SymmetryOperation, ...]:
    """Find integer lattice automorphisms that map the labeled atomic basis."""
    metric = lattice @ lattice.T
    scale = max(1.0, float(np.max(np.abs(metric))))
    operations: list[SymmetryOperation] = []
    first_label = labels[0]
    candidate_targets = fractional_positions[
        np.array([label == first_label for label in labels])
    ]
    for entries in itertools.product((-1, 0, 1), repeat=9):
        matrix = np.asarray(entries, dtype=int).reshape(3, 3)
        determinant = round(float(np.linalg.det(matrix)))
        if abs(determinant) != 1:
            continue
        if not np.allclose(
            matrix @ metric @ matrix.T,
            metric,
            rtol=0.0,
            atol=tolerance * scale,
        ):
            continue
        rotated = fractional_positions @ matrix
        for target in candidate_targets:
            translation = target - rotated[0]
            # QE accepts a nonsymmorphic fractional translation only when
            # each nonzero component is 1/n (up to sign and a lattice
            # vector), with n = 2, 3, 4, or 6.  This crystallographic
            # restriction is important for slightly displaced structures:
            # an arbitrary inversion centre may be a mathematical symmetry
            # of a two-identical-atom basis, but find_sym does not retain it.
            reduced_translation = translation - np.rint(translation)
            valid_translation = True
            for component in reduced_translation:
                magnitude = abs(float(component))
                if magnitude <= tolerance:
                    continue
                inverse = 1.0 / magnitude
                denominator = round(inverse)
                if (
                    denominator not in {2, 3, 4, 6}
                    or abs(inverse - denominator) > tolerance
                ):
                    valid_translation = False
                    break
            if not valid_translation:
                continue
            transformed = rotated + translation
            if _periodic_match(
                transformed, labels, fractional_positions, tolerance
            ):
                translation -= np.floor(translation)
                operations.append(
                    SymmetryOperation(matrix.copy(), translation.copy())
                )
                break
    if not operations:
        operations.append(
            SymmetryOperation(np.eye(3, dtype=int), np.zeros(3))
        )
    return tuple(operations)


def _kpoint_key(point: np.ndarray, decimals: int = 10) -> tuple[float, ...]:
    reduced = np.asarray(point) - np.floor(np.asarray(point))
    reduced[np.isclose(reduced, 1.0, atol=10.0 ** (-decimals))] = 0.0
    return tuple(np.round(reduced, decimals))


def reduce_kpoints(
    coordinates: np.ndarray,
    weights: np.ndarray,
    operations: tuple[SymmetryOperation, ...],
    time_reversal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a complete fractional reciprocal mesh into symmetry orbits."""
    coordinates = np.asarray(coordinates, dtype=float)
    weights = np.asarray(weights, dtype=float)
    lookup = {_kpoint_key(point): index for index, point in enumerate(coordinates)}
    operations = mesh_compatible_operations(coordinates, operations)
    time_reversal = time_reversal and all(
        _kpoint_key(-point) in lookup for point in coordinates
    )
    assigned: set[int] = set()
    representatives: list[np.ndarray] = []
    reduced_weights: list[float] = []
    for index, point in enumerate(coordinates):
        if index in assigned:
            continue
        orbit: set[int] = set()
        for operation in operations:
            reciprocal = point @ np.linalg.inv(operation.matrix).T
            signs = (1.0, -1.0) if time_reversal else (1.0,)
            for sign in signs:
                mapped = lookup.get(_kpoint_key(sign * reciprocal))
                if mapped is not None:
                    orbit.add(mapped)
        if not orbit:
            orbit.add(index)
        assigned.update(orbit)
        representatives.append(point.copy())
        reduced_weights.append(float(np.sum(weights[list(orbit)])))
    normalized = np.asarray(reduced_weights)
    normalized /= normalized.sum()
    return np.asarray(representatives), normalized


def mesh_compatible_operations(
    coordinates: np.ndarray,
    operations: tuple[SymmetryOperation, ...],
) -> tuple[SymmetryOperation, ...]:
    """Keep only crystal operations that map the complete sampled mesh to itself."""
    coordinates = np.asarray(coordinates, dtype=float)
    keys = {_kpoint_key(point) for point in coordinates}
    compatible = []
    for operation in operations:
        inverse_transpose = np.linalg.inv(operation.matrix).T
        if all(
            _kpoint_key(point @ inverse_transpose) in keys
            for point in coordinates
        ):
            compatible.append(operation)
    return tuple(compatible)


def symmetrize_density(
    density: np.ndarray,
    operations: tuple[SymmetryOperation, ...],
    mpi: "MPIContext | None" = None,
) -> np.ndarray:
    """Average a real-space density over FFT-grid-compatible operations."""
    return DensitySymmetrizer(
        density.shape, operations, mpi
    ).apply(density)


def symmetrize_forces(
    forces: np.ndarray,
    lattice: np.ndarray,
    fractional_positions: np.ndarray,
    labels: list[str],
    operations: tuple[SymmetryOperation, ...],
    tolerance: float = 1.0e-7,
) -> np.ndarray:
    """Project Cartesian atomic forces onto the crystal space group."""
    values = np.asarray(forces, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    fractional = np.asarray(fractional_positions, dtype=float)
    if values.shape != (len(fractional), 3):
        raise ValueError("forces must have shape (number of atoms, 3)")
    if len(operations) <= 1:
        return values.copy()
    inverse_lattice = np.linalg.inv(lattice)
    projected = np.zeros_like(values)
    for operation in operations:
        cartesian_rotation = (
            inverse_lattice @ operation.matrix @ lattice
        )
        transformed = (
            fractional @ operation.matrix + operation.translation
        )
        transformed -= np.floor(transformed)
        for source, (position, label) in enumerate(
            zip(transformed, labels)
        ):
            target = None
            for candidate, (reference, reference_label) in enumerate(
                zip(fractional, labels)
            ):
                if reference_label != label:
                    continue
                delta = position - reference
                delta -= np.rint(delta)
                if np.linalg.norm(delta) < tolerance:
                    target = candidate
                    break
            if target is None:
                raise ValueError("symmetry operation does not map the atomic basis")
            projected[target] += values[source] @ cartesian_rotation
    projected /= len(operations)
    return projected


def symmetrize_stress(
    stress: np.ndarray,
    lattice: np.ndarray,
    operations: tuple[SymmetryOperation, ...],
) -> np.ndarray:
    """Project a Cartesian symmetric stress tensor onto the point group."""
    tensor = np.asarray(stress, dtype=float)
    if tensor.shape != (3, 3):
        raise ValueError("stress must have shape (3, 3)")
    if len(operations) <= 1:
        return 0.5 * (tensor + tensor.T)
    inverse_lattice = np.linalg.inv(np.asarray(lattice, dtype=float))
    projected = np.zeros((3, 3), dtype=float)
    for operation in operations:
        cartesian_rotation = (
            inverse_lattice @ operation.matrix @ lattice
        )
        projected += (
            cartesian_rotation.T @ tensor @ cartesian_rotation
        )
    projected /= len(operations)
    return 0.5 * (projected + projected.T)
