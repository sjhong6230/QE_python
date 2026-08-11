"""Numerical irreducible characters of crystallographic double groups.

The implementation follows the same group-theoretical construction used by
QE's ``rap_point_group_so`` tables, but constructs the binary (spin) cover
from the actual little group.  This also avoids hard-coding 32 separate
character tables and keeps non-standard cell orientations harmless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from .errors import QEInputError
from .symmetry import SymmetryOperation


_QE_DOUBLE_NAMES = {
    "C_1": ("G_2",), "C_i": ("G_2+", "G_2-"),
    "C_s": ("G_3", "G_4"), "C_2": ("G_3", "G_4"),
    "C_3": ("G_4", "G_5", "G_6"),
    "C_4": ("G_5", "G_6", "G_7", "G_8"),
    "C_6": ("G_7", "G_8", "G_9", "G_10", "G_11", "G_12"),
    "D_2": ("G_5",), "D_3": ("G_4", "G_5", "G_6"),
    "D_4": ("G_6", "G_7"), "D_6": ("G_7", "G_8", "G_9"),
    "C_2v": ("G_5",), "C_3v": ("G_4", "G_5", "G_6"),
    "C_4v": ("G_6", "G_7"), "C_6v": ("G_7", "G_8", "G_9"),
    "C_2h": ("G_3+", "G_4+", "G_3-", "G_4-"),
    "C_3h": ("G_7", "G_8", "G_9", "G_10", "G_11", "G_12"),
    "C_4h": ("G_5+", "G_6+", "G_7+", "G_8+", "G_5-", "G_6-", "G_7-", "G_8-"),
    "C_6h": tuple(f"G_{number}{parity}" for parity in ("+", "-") for number in range(7, 13)),
    "D_2h": ("G_5+", "G_5-"),
    "D_3h": ("G_7", "G_8", "G_9"),
    "D_4h": ("G_6+", "G_7+", "G_6-", "G_7-"),
    "D_6h": ("G_7+", "G_8+", "G_9+", "G_7-", "G_8-", "G_9-"),
    "D_2d": ("G_6", "G_7"),
    "D_3d": ("G_4+", "G_5+", "G_6+", "G_4-", "G_5-", "G_6-"),
    "S_4": ("G_5", "G_6", "G_7", "G_8"),
    "S_6": ("G_4+", "G_5+", "G_6+", "G_4-", "G_5-", "G_6-"),
    "T": ("G_5", "G_6", "G_7"),
    "T_h": ("G_5+", "G_6+", "G_7+", "G_5-", "G_6-", "G_7-"),
    "T_d": ("G_6", "G_7", "G_8"),
    "O": ("G_6", "G_7", "G_8"),
    "O_h": ("G_6+", "G_7+", "G_8+", "G_6-", "G_7-", "G_8-"),
}


@dataclass(frozen=True)
class DoubleGroupTable:
    names: tuple[str, ...]
    dimensions: np.ndarray
    # Characters of the principal SU(2) lift of each spatial operation.
    characters: np.ndarray


def spin_half_rotation(
    lattice: np.ndarray, operation: SymmetryOperation
) -> np.ndarray:
    """Return the principal SU(2) lift of an axial crystal rotation."""
    inverse_lattice = np.linalg.inv(np.asarray(lattice, dtype=float))
    cartesian = inverse_lattice @ operation.matrix @ np.asarray(lattice)
    axial = np.linalg.det(cartesian) * cartesian
    vector = Rotation.from_matrix(axial.T).as_rotvec()
    pauli = (
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        np.asarray([[0.0, -1j], [1j, 0.0]], dtype=complex),
        np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    )
    generator = sum(component * matrix for component, matrix in zip(vector, pauli))
    return expm(-0.5j * generator)


def _spatial_product(
    first: SymmetryOperation,
    second: SymmetryOperation,
    operations: tuple[SymmetryOperation, ...],
) -> int:
    matrix = first.matrix @ second.matrix
    translation = first.translation @ second.matrix + second.translation
    for index, candidate in enumerate(operations):
        delta = translation - candidate.translation
        delta -= np.rint(delta)
        if np.array_equal(matrix, candidate.matrix) and np.linalg.norm(delta) < 1.0e-7:
            return index
    raise QEInputError("little-group operations are not closed")


def _multiplication_table(
    operations: tuple[SymmetryOperation, ...], lifts: tuple[np.ndarray, ...]
) -> np.ndarray:
    count = len(operations)
    table = np.empty((2 * count, 2 * count), dtype=np.int32)
    for first in range(count):
        for second in range(count):
            product = _spatial_product(operations[first], operations[second], operations)
            # Row-coordinate spatial operations compose as S_first S_second,
            # while their active column rotations (and SU(2) lifts) compose
            # in the reverse matrix order.
            lifted = lifts[second] @ lifts[first]
            sign = 0 if np.linalg.norm(lifted - lifts[product]) < np.linalg.norm(lifted + lifts[product]) else 1
            for first_sign in range(2):
                for second_sign in range(2):
                    result_sign = sign ^ first_sign ^ second_sign
                    table[2 * first + first_sign, 2 * second + second_sign] = (
                        2 * product + result_sign
                    )
    return table


def _conjugacy_classes(table: np.ndarray, identity: int) -> list[tuple[int, ...]]:
    order = len(table)
    inverse = np.empty(order, dtype=np.int32)
    for element in range(order):
        matches = np.flatnonzero(table[element] == identity)
        if len(matches) != 1:
            raise QEInputError("invalid double-group inverse")
        inverse[element] = matches[0]
    unassigned = set(range(order))
    classes = []
    while unassigned:
        element = min(unassigned)
        members = {
            int(table[table[group, element], inverse[group]])
            for group in range(order)
        }
        classes.append(tuple(sorted(members)))
        unassigned.difference_update(members)
    return classes


def _regular_class_sum(table: np.ndarray, members: tuple[int, ...]) -> np.ndarray:
    order = len(table)
    result = np.zeros((order, order), dtype=np.complex128)
    columns = np.arange(order)
    for member in members:
        result[table[member], columns] += 1.0
    return result


def _cluster_eigenvalues(values: np.ndarray, tolerance: float = 2.0e-7) -> list[np.ndarray]:
    remaining = set(range(len(values)))
    groups = []
    while remaining:
        first = min(remaining)
        scale = max(1.0, abs(values[first]))
        group = np.asarray([
            index for index in sorted(remaining)
            if abs(values[index] - values[first]) <= tolerance * scale
        ], dtype=int)
        groups.append(group)
        remaining.difference_update(group.tolist())
    return groups


def _isotypic_subspaces(
    class_sums: tuple[np.ndarray, ...]
) -> list[np.ndarray]:
    """Simultaneously diagonalize the center of the regular representation."""
    order = len(class_sums[0])
    spaces = [np.eye(order, dtype=np.complex128)]
    hermitian_centers = []
    for matrix in class_sums:
        hermitian_centers.extend((
            0.5 * (matrix + matrix.conj().T),
            (matrix - matrix.conj().T) / (2j),
        ))
    for center in hermitian_centers:
        refined = []
        for basis in spaces:
            projected = basis.conj().T @ center @ basis
            projected = 0.5 * (projected + projected.conj().T)
            eigenvalues, eigenvectors = np.linalg.eigh(projected)
            for indices in _cluster_eigenvalues(eigenvalues):
                candidate = basis @ eigenvectors[:, indices]
                candidate, _ = np.linalg.qr(candidate)
                refined.append(candidate)
        spaces = refined
    return spaces


def _qe_like_names(
    group_name: str, dimensions: np.ndarray, characters: np.ndarray,
    operations: tuple[SymmetryOperation, ...],
) -> tuple[str, ...]:
    """Assign Koster/QE labels where their invariant distinction is simple."""
    inversion = next(
        (index for index, op in enumerate(operations) if np.array_equal(op.matrix, -np.eye(3, dtype=int))),
        None,
    )
    order_four = next(
        (
            index for index, op in enumerate(operations)
            if round(np.linalg.det(op.matrix)) == 1
            and abs(round(np.trace(op.matrix)) - 1) == 0
            and not np.array_equal(op.matrix, np.eye(3, dtype=int))
        ),
        None,
    )
    names = []
    for index, dimension in enumerate(dimensions):
        parity = ""
        if inversion is not None:
            parity = "+" if np.real(characters[index, inversion]) > 0 else "-"
        if group_name in {"O", "O_h", "T_d", "D_4h", "C_4v"}:
            if dimension == 4:
                number = 8
            elif order_four is not None and np.real(characters[index, order_four]) < 0:
                number = 7
            else:
                number = 6
            names.append(f"G_{number}{parity}")
        else:
            names.append(f"DG_{index + 1}{parity}")
    requested = _QE_DOUBLE_NAMES.get(group_name)
    if requested is None or len(requested) != len(names):
        return tuple(names)
    if group_name in {"O", "O_h", "T_d", "D_4h", "C_4v"}:
        # These labels are fixed above by dimension, C4 character, and parity.
        return tuple(names)
    return requested


def double_group_character_table(
    lattice: np.ndarray,
    operations: tuple[SymmetryOperation, ...],
    group_name: str = "",
) -> DoubleGroupTable:
    """Construct all double-valued irreducible characters of a little group."""
    unitary = tuple(operation for operation in operations if not operation.time_reversal)
    if not unitary:
        raise QEInputError("double-group analysis requires a unitary subgroup")
    identity_spatial = next(
        (
            index for index, operation in enumerate(unitary)
            if np.array_equal(operation.matrix, np.eye(3, dtype=int))
            and np.linalg.norm(operation.translation - np.rint(operation.translation)) < 1.0e-7
        ),
        None,
    )
    if identity_spatial is None:
        raise QEInputError("little group has no identity")
    lifts = tuple(spin_half_rotation(lattice, operation) for operation in unitary)
    multiplication = _multiplication_table(unitary, lifts)
    identity = 2 * identity_spatial
    classes = _conjugacy_classes(multiplication, identity)
    sums = tuple(_regular_class_sum(multiplication, members) for members in classes)
    irreps = []
    for basis in _isotypic_subspaces(sums):
        multiplicity_space = basis.shape[1]
        dimension = int(round(np.sqrt(multiplicity_space)))
        if dimension * dimension != multiplicity_space:
            raise QEInputError("could not resolve double-group regular representation")
        class_characters = []
        for members, class_sum in zip(classes, sums):
            scalar = (
                np.trace(basis.conj().T @ class_sum @ basis)
                / multiplicity_space
            )
            class_characters.append(scalar * dimension / len(members))
        minus_identity_class = next(
            index for index, members in enumerate(classes)
            if 2 * identity_spatial + 1 in members
        )
        if abs(class_characters[minus_identity_class] + dimension) < 2.0e-5:
            irreps.append((dimension, np.asarray(class_characters)))
    if not irreps:
        raise QEInputError("no double-valued irreducible representations found")
    spatial_characters = []
    for _dimension, class_characters in irreps:
        values = np.empty(len(unitary), dtype=np.complex128)
        for operation in range(len(unitary)):
            element = 2 * operation
            class_index = next(i for i, members in enumerate(classes) if element in members)
            values[operation] = class_characters[class_index]
        spatial_characters.append(values)
    dimensions = np.asarray([item[0] for item in irreps], dtype=int)
    characters = np.asarray(spatial_characters)
    ordering = sorted(
        range(len(irreps)),
        key=lambda index: (
            dimensions[index],
            tuple(np.round(characters[index].real, 8)),
            tuple(np.round(characters[index].imag, 8)),
        ),
    )
    dimensions = dimensions[ordering]
    characters = characters[ordering]
    names = _qe_like_names(group_name, dimensions, characters, unitary)
    qe_order = _QE_DOUBLE_NAMES.get(group_name)
    if (
        qe_order is not None
        and len(qe_order) == len(names)
        and len(set(names)) == len(names)
        and set(qe_order) == set(names)
    ):
        label_order = [names.index(name) for name in qe_order]
        dimensions = dimensions[label_order]
        characters = characters[label_order]
        names = qe_order
    return DoubleGroupTable(names, dimensions, characters)
