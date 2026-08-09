"""Lightweight character tables for all 32 crystallographic point groups.

The implementation follows the scalar (single-valued) group information
printed by Quantum ESPRESSO.  It deliberately has no IrRep or spglib import:
conjugacy classes and their characters are obtained from the at-most 48
already-discovered rotation matrices by diagonalizing the at-most 12 by 12
class algebra.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np


@dataclass(frozen=True)
class PointGroupClass:
    label: str
    operation_indices: tuple[int, ...]
    description: str


@dataclass(frozen=True)
class PointGroupTable:
    schoenflies: str
    international: str
    classes: tuple[PointGroupClass, ...]
    irreps: tuple[tuple[str, tuple[complex, ...]], ...]


# Schoenflies symbol, Hermann--Mauguin symbol, Mulliken labels and dimensions.
# Complex one-dimensional partners used by QE retain the E/E* convention.
_GROUPS = {
    "C_1": ("1", ("A",), (1,)),
    "C_i": ("-1", ("A_g", "A_u"), (1, 1)),
    "C_s": ("m", ("A'", "A''"), (1, 1)),
    "C_2": ("2", ("A", "B"), (1, 1)),
    "C_3": ("3", ("A", "E", "E*"), (1, 1, 1)),
    "C_4": ("4", ("A", "B", "E", "E*"), (1, 1, 1, 1)),
    "C_6": ("6", ("A", "B", "E_1", "E_1*", "E_2", "E_2*"), (1,) * 6),
    "D_2": ("222", ("A", "B_1", "B_2", "B_3"), (1,) * 4),
    "D_3": ("32", ("A_1", "A_2", "E"), (1, 1, 2)),
    "D_4": ("422", ("A_1", "A_2", "B_1", "B_2", "E"), (1, 1, 1, 1, 2)),
    "D_6": ("622", ("A_1", "A_2", "B_1", "B_2", "E_1", "E_2"), (1, 1, 1, 1, 2, 2)),
    "C_2v": ("mm2", ("A_1", "A_2", "B_1", "B_2"), (1,) * 4),
    "C_3v": ("3m", ("A_1", "A_2", "E"), (1, 1, 2)),
    "C_4v": ("4mm", ("A_1", "A_2", "B_1", "B_2", "E"), (1, 1, 1, 1, 2)),
    "C_6v": ("6mm", ("A_1", "A_2", "B_1", "B_2", "E_1", "E_2"), (1, 1, 1, 1, 2, 2)),
    "C_2h": ("2/m", ("A_g", "B_g", "A_u", "B_u"), (1,) * 4),
    "C_3h": ("-6", ("A'", "E'", "E'*", "A''", "E''", "E''*"), (1,) * 6),
    "C_4h": ("4/m", ("A_g", "B_g", "E_g", "E_g*", "A_u", "B_u", "E_u", "E_u*"), (1,) * 8),
    "C_6h": ("6/m", ("A_g", "B_g", "E_1g", "E_1g*", "E_2g", "E_2g*", "A_u", "B_u", "E_1u", "E_1u*", "E_2u", "E_2u*"), (1,) * 12),
    "D_2h": ("mmm", ("A_g", "B_1g", "B_2g", "B_3g", "A_u", "B_1u", "B_2u", "B_3u"), (1,) * 8),
    "D_3h": ("-62m", ("A'_1", "A'_2", "E'", "A''_1", "A''_2", "E''"), (1, 1, 2, 1, 1, 2)),
    "D_4h": ("4/mmm", ("A_1g", "A_2g", "B_1g", "B_2g", "E_g", "A_1u", "A_2u", "B_1u", "B_2u", "E_u"), (1, 1, 1, 1, 2, 1, 1, 1, 1, 2)),
    "D_6h": ("6/mmm", ("A_1g", "A_2g", "B_1g", "B_2g", "E_1g", "E_2g", "A_1u", "A_2u", "B_1u", "B_2u", "E_1u", "E_2u"), (1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2)),
    "D_2d": ("-42m", ("A_1", "A_2", "B_1", "B_2", "E"), (1, 1, 1, 1, 2)),
    "D_3d": ("-3m", ("A_1g", "A_2g", "E_g", "A_1u", "A_2u", "E_u"), (1, 1, 2, 1, 1, 2)),
    "S_4": ("-4", ("A", "B", "E", "E*"), (1,) * 4),
    "S_6": ("-3", ("A_g", "E_g", "E_g*", "A_u", "E_u", "E_u*"), (1,) * 6),
    "T": ("23", ("A", "E", "E*", "T"), (1, 1, 1, 3)),
    "T_h": ("m-3", ("A_g", "E_g", "E_g*", "T_g", "A_u", "E_u", "E_u*", "T_u"), (1, 1, 1, 3, 1, 1, 1, 3)),
    "T_d": ("-43m", ("A_1", "A_2", "E", "T_1", "T_2"), (1, 1, 2, 3, 3)),
    "O": ("432", ("A_1", "A_2", "E", "T_1", "T_2"), (1, 1, 2, 3, 3)),
    "O_h": ("m-3m", ("A_1g", "A_2g", "E_g", "T_1g", "T_2g", "A_1u", "A_2u", "E_u", "T_1u", "T_2u"), (1, 1, 2, 3, 3, 1, 1, 2, 3, 3)),
}


_CLASS_LABELS = {
    "C_1": ("E",),
    "C_i": ("E", "i"),
    "C_s": ("E", "s"),
    "C_2": ("E", "C2"),
    "C_3": ("E", "C3", "C3^2"),
    "C_4": ("E", "C4", "C2", "C4^3"),
    "C_6": ("E", "C6", "C3", "C2", "C3^2", "C6^5"),
    "D_2": ("E", "C2", "C2'", "C2''"),
    "D_3": ("E", "2C3", "3C2'"),
    "D_4": ("E", "2C4", "C2", "2C2'", "2C2''"),
    "D_6": ("E", "2C6", "2C3", "C2", "3C2'", "3C2''"),
    "C_2v": ("E", "C2", "s_v", "s_v'"),
    "C_3v": ("E", "2C3", "3s_v"),
    "C_4v": ("E", "2C4", "C2", "2s_v", "2s_d"),
    "C_6v": ("E", "2C6", "2C3", "C2", "3s_v", "3s_d"),
    "C_2h": ("E", "C2", "i", "s_h"),
    "C_3h": ("E", "C3", "C3^2", "s_h", "S3", "S3^5"),
    "C_4h": ("E", "C4", "C2", "C4^3", "i", "S4^3", "s_h", "S4"),
    "C_6h": ("E", "C6", "C3", "C2", "C3^2", "C6^5", "i", "S3^5", "S6^5", "s_h", "S6", "S3"),
    "D_2h": ("E", "C2", "C2'", "C2''", "i", "s_v", "s_v'", "s_v''"),
    "D_3h": ("E", "2C3", "3C2", "s_h", "2S3", "3s_v"),
    "D_4h": ("E", "2C4", "C2", "2C2'", "2C2''", "i", "2S4", "s_h", "2s_v", "2s_d"),
    "D_6h": ("E", "2C6", "2C3", "C2", "3C2'", "3C2''", "i", "2S3", "2S6", "s_h", "3s_d", "3s_v"),
    "D_2d": ("E", "2S4", "C2", "2C2'", "2s_d"),
    "D_3d": ("E", "2C3", "3C2'", "i", "2S6", "3s_d"),
    "S_4": ("E", "S4^3", "C2", "S4"),
    "S_6": ("E", "C3", "C3^2", "i", "S6^5", "S6"),
    "T": ("E", "3C2", "4C3", "4C3'"),
    "T_h": ("E", "3C2", "4C3", "4C3'", "i", "3s_h", "4S6^5", "4S6"),
    "T_d": ("E", "8C3", "3C2", "6S4", "6s_d"),
    "O": ("E", "8C3", "3C2", "6C4", "6C2'"),
    "O_h": ("E", "8C3", "3C2", "6C4", "6C2'", "i", "8S6", "3s_h", "6S4", "6s_d"),
}


# (principal operation, first secondary C2/mirror) in the class order above.
_NAMING_SLOTS = {
    "C_2": (1, None), "C_3": (1, None), "C_4": (1, None),
    "C_6": (1, None), "C_2h": (1, None), "C_3h": (1, None),
    "C_4h": (1, None), "C_6h": (1, None), "S_4": (1, None),
    "S_6": (1, None), "D_3": (1, 2), "D_4": (1, 3),
    "D_6": (1, 4), "C_2v": (1, 2), "C_3v": (1, 2),
    "C_4v": (1, 3), "C_6v": (1, 4), "D_3h": (1, 2),
    "D_4h": (1, 3), "D_6h": (1, 4), "D_2d": (1, 3),
    "D_3d": (1, 2), "T": (2, None), "T_h": (2, None),
    "T_d": (None, 3), "O": (None, 3), "O_h": (None, 3),
}


def _matrix_key(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.rint(matrix).astype(int).ravel())


def _matrix_order(matrix: np.ndarray) -> int:
    product = np.eye(3, dtype=int)
    integer = np.rint(matrix).astype(int)
    for order in range(1, 13):
        product = product @ integer
        if np.array_equal(product, np.eye(3, dtype=int)):
            return order
    raise ValueError("operation is not crystallographic")


def _operation_kind(matrix: np.ndarray) -> str:
    integer = np.rint(matrix).astype(int)
    if np.array_equal(integer, np.eye(3, dtype=int)):
        return "identity"
    if np.array_equal(integer, -np.eye(3, dtype=int)):
        return "inversion"
    determinant = int(round(np.linalg.det(integer)))
    if determinant > 0:
        return "twofold" if _matrix_order(integer) == 2 else "rotation"
    return "mirror" if _matrix_order(integer) == 2 else "improper"


def _identify_group(matrices: list[np.ndarray]) -> str:
    """Port the invariant decision tree used by QE's ``find_group``."""
    kinds = [_operation_kind(matrix) for matrix in matrices]
    count = {kind: kinds.count(kind) for kind in set(kinds)}
    nrot = len(matrices)
    inversion = count.get("inversion", 0)
    if inversion == 0:
        if nrot == 1:
            return "C_1"
        if nrot == 2:
            return "C_2" if count.get("twofold", 0) else "C_s"
        if nrot == 3:
            return "C_3"
        if nrot == 4:
            if count.get("improper", 0):
                return "S_4"
            if count.get("mirror", 0):
                return "C_2v"
            if count.get("rotation", 0):
                return "C_4"
            return "D_2"
        if nrot == 6:
            if count.get("mirror", 0) == 3:
                return "C_3v"
            if count.get("mirror", 0) == 1:
                return "C_3h"
            if count.get("twofold", 0) == 3:
                return "D_3"
            return "C_6"
        if nrot == 8:
            if count.get("mirror", 0) == 4:
                return "C_4v"
            if count.get("mirror", 0) == 2:
                return "D_2d"
            return "D_4"
        if nrot == 12:
            if count.get("mirror", 0) == 6:
                return "C_6v"
            if count.get("mirror", 0) == 4:
                return "D_3h"
            if count.get("twofold", 0) > 6:
                return "D_6"
            return "T"
        if nrot == 24:
            return "T_d" if count.get("mirror", 0) else "O"
    elif inversion == 1:
        if nrot == 2:
            return "C_i"
        if nrot == 4:
            return "C_2h"
        if nrot == 6:
            return "S_6"
        if nrot == 8:
            return "D_2h" if count.get("mirror", 0) == 3 else "C_4h"
        if nrot == 12:
            return "D_3d" if count.get("mirror", 0) == 3 else "C_6h"
        if nrot == 16:
            return "D_4h"
        if nrot == 24:
            return "D_6h" if count.get("mirror", 0) > 6 else "T_h"
        if nrot == 48:
            return "O_h"
    raise ValueError("rotation matrices do not form a crystallographic point group")


def _conjugacy_classes(matrices: list[np.ndarray]) -> list[list[int]]:
    lookup = {_matrix_key(matrix): index for index, matrix in enumerate(matrices)}
    remaining = set(range(len(matrices)))
    classes: list[list[int]] = []
    while remaining:
        representative = min(remaining)
        members = {
            lookup[_matrix_key(a @ matrices[representative] @ np.linalg.inv(a))]
            for a in matrices
        }
        ordered = sorted(members)
        classes.append(ordered)
        remaining.difference_update(members)
    return classes


def _class_characters(
    matrices: list[np.ndarray], classes: list[list[int]]
) -> list[tuple[int, tuple[complex, ...]]]:
    """Recover all irreducible characters from the finite class algebra."""
    element_class = {
        _matrix_key(matrices[element]): class_index
        for class_index, members in enumerate(classes)
        for element in members
    }
    number = len(classes)
    multiplication = []
    for left in classes:
        operator = np.zeros((number, number), dtype=float)
        for column, right in enumerate(classes):
            counts = np.zeros(number, dtype=int)
            for first in left:
                for second in right:
                    product = matrices[first] @ matrices[second]
                    counts[element_class[_matrix_key(product)]] += 1
            for row, members in enumerate(classes):
                operator[row, column] = counts[row] / len(members)
        multiplication.append(operator)

    # A generic complex combination separates the one-dimensional joint
    # eigenspaces of these mutually commuting central operators.
    for attempt in range(5):
        weights = np.exp(
            1j * (attempt + 1) * np.arange(1, number + 1) * math.sqrt(2.0)
        )
        combined = sum(weight * op for weight, op in zip(weights, multiplication))
        eigenvalues, eigenvectors = np.linalg.eig(combined)
        if min(abs(a - b) for i, a in enumerate(eigenvalues) for b in eigenvalues[i + 1:]) > 1.0e-7 if number > 1 else True:
            break
    result = []
    sizes = np.asarray([len(members) for members in classes], dtype=float)
    for vector in eigenvectors.T:
        norm = np.vdot(vector, vector)
        central_eigenvalues = np.asarray(
            [np.vdot(vector, operator @ vector) / norm for operator in multiplication]
        )
        ratios = central_eigenvalues / sizes
        dimension = int(round(math.sqrt(len(matrices) / float(np.sum(sizes * abs(ratios) ** 2)))))
        characters = dimension * ratios
        characters.real[np.abs(characters.real) < 1.0e-9] = 0.0
        characters.imag[np.abs(characters.imag) < 1.0e-9] = 0.0
        characters = np.round(characters, 10)
        result.append((dimension, tuple(complex(value) for value in characters)))
    return result


def _improper_fold(matrix: np.ndarray) -> int:
    proper = -matrix
    angle = math.degrees(
        math.acos(np.clip((np.trace(proper) - 1.0) / 2.0, -1.0, 1.0))
    )
    effective = (angle + 180.0) % 360.0
    effective = min(effective, 360.0 - effective)
    return max(2, int(round(360.0 / effective))) if effective > 1.0e-8 else 2


def _expected_class_descriptor(label: str) -> tuple[int, str, int]:
    match = re.match(r"(\d+)?(.*)", label)
    assert match is not None
    size = int(match.group(1) or 1)
    core = match.group(2)
    if core == "E":
        return size, "identity", 1
    if core == "i":
        return size, "inversion", 1
    if core.startswith("s"):
        return size, "mirror", 2
    fold = int(re.search(r"\d+", core).group())
    return size, "rotation" if core.startswith("C") else "improper", fold


def _actual_class_descriptor(
    matrix: np.ndarray, members: list[int]
) -> tuple[int, str, int]:
    kind = _operation_kind(matrix)
    if kind == "twofold":
        kind = "rotation"
    fold = (
        _matrix_order(matrix)
        if kind == "rotation"
        else _improper_fold(matrix)
        if kind == "improper"
        else 2
        if kind == "mirror"
        else 1
    )
    return len(members), kind, fold


def _order_classes(
    group_name: str,
    matrices: list[np.ndarray],
    classes: list[list[int]],
) -> list[list[int]]:
    """Match conjugacy classes to QE labels by size and operation type."""
    remaining = list(classes)
    ordered = []
    for label in _CLASS_LABELS[group_name]:
        expected = _expected_class_descriptor(label)
        match = next(
            (
                index
                for index, members in enumerate(remaining)
                if _actual_class_descriptor(matrices[members[0]], members)
                == expected
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"cannot match point-group class {label} in {group_name}"
            )
        ordered.append(remaining.pop(match))
    if remaining:
        raise ValueError(f"unmatched conjugacy classes in {group_name}")
    # Couple singleton power classes to the same chosen generator.  Without
    # this step C3 and C3^2 are individually valid but an independently
    # ordered S3/S3^5 pair can acquire the opposite power label.
    power_slots = {
        "C_3": {2: 2},
        "C_4": {2: 2, 3: 3},
        "C_6": {2: 2, 3: 3, 4: 4, 5: 5},
        "C_3h": {2: 2},
        "C_4h": {2: 2, 3: 3},
        "C_6h": {2: 2, 3: 3, 4: 4, 5: 5},
        "S_6": {2: 2},
        "S_4": {2: 2, 3: 3},
    }.get(group_name, {})
    if power_slots:
        generator = matrices[ordered[1][0]]
        by_matrix = {
            _matrix_key(matrices[members[0]]): members for members in ordered
        }
        for slot, power in power_slots.items():
            ordered[slot] = by_matrix[_matrix_key(np.linalg.matrix_power(generator, power))]

    partner = None
    partner_slots: dict[int, int] = {}
    if group_name == "C_3h":
        partner, partner_slots = matrices[ordered[3][0]], {4: 1, 5: 2}
    elif group_name == "C_4h":
        partner, partner_slots = matrices[ordered[4][0]], {5: 1, 6: 2, 7: 3}
    elif group_name == "C_6h":
        partner = matrices[ordered[6][0]]
        partner_slots = {7: 1, 8: 2, 9: 3, 10: 4, 11: 5}
    elif group_name == "S_6":
        partner, partner_slots = matrices[ordered[3][0]], {4: 1, 5: 2}
    if partner is not None:
        generator = matrices[ordered[1][0]]
        by_matrix = {
            _matrix_key(matrices[members[0]]): members for members in ordered
        }
        for slot, power in partner_slots.items():
            product = np.linalg.matrix_power(generator, power) @ partner
            ordered[slot] = by_matrix[_matrix_key(product)]
    return ordered


def _which_c2_axis(matrix: np.ndarray, eigenvalue: float) -> int:
    """Return QE ``which_c2``'s Cartesian direction code (1 through 13)."""
    _u, _s, vh = np.linalg.svd(
        np.asarray(matrix, dtype=float) - eigenvalue * np.eye(3)
    )
    axis = vh[-1]
    axis /= max(float(np.max(np.abs(axis))), 1.0e-15)
    x, y, z = axis
    tolerance = 1.0e-7
    if abs(y) < tolerance and abs(z) < tolerance:
        return 1
    if abs(x) < tolerance and abs(z) < tolerance:
        return 2
    if abs(x) < tolerance and abs(y) < tolerance:
        return 3
    if abs(x) < tolerance:
        if abs(y - z) < tolerance:
            return 4
        if abs(y + z) < tolerance:
            return 5
    elif abs(y) < tolerance:
        if abs(x - z) < tolerance:
            return 6
        if abs(x + z) < tolerance:
            return 7
    elif abs(z) < tolerance:
        sqrt_three = math.sqrt(3.0)
        tests = (
            (8, x - y),
            (9, x + y),
            (10, x - y / sqrt_three),
            (11, x + y / sqrt_three),
            (12, x - y * sqrt_three),
            (13, x + y * sqrt_three),
        )
        for code, residual in tests:
            if abs(residual) < tolerance:
                return code
    raise ValueError("C2 or mirror axis is not in QE's crystallographic list")


_QE_C2V_AXIS_ORDERS = {
    (1, 2, 3), (1, 4, 5), (2, 3, 1), (2, 7, 6),
    (3, 1, 2), (3, 8, 9), (4, 5, 1), (5, 1, 4),
    (6, 2, 7), (7, 6, 2), (8, 3, 9), (9, 8, 3),
    (3, 11, 12), (3, 13, 10), (12, 3, 11), (13, 10, 3),
    (10, 3, 13), (11, 12, 3),
}


def _order_c2v_mirrors(
    classes: list[list[int]], cartesian: list[np.ndarray]
) -> list[list[int]]:
    """Order C2v mirrors as QE ``is_c2v`` does for B1/B2 labels."""
    principal = _which_c2_axis(cartesian[classes[1][0]], 1.0)
    first = _which_c2_axis(cartesian[classes[2][0]], -1.0)
    second = _which_c2_axis(cartesian[classes[3][0]], -1.0)
    if (principal, first, second) in _QE_C2V_AXIS_ORDERS:
        return classes
    if (principal, second, first) in _QE_C2V_AXIS_ORDERS:
        classes[2], classes[3] = classes[3], classes[2]
        return classes
    raise ValueError("cannot order C2v mirror classes using QE convention")


def _axis(matrix: np.ndarray, eigenvalue: float) -> tuple[int, int, int]:
    _u, _s, vh = np.linalg.svd(matrix - eigenvalue * np.eye(3))
    axis = vh[-1]
    axis /= max(np.max(np.abs(axis)), 1.0e-15)
    rounded = np.rint(axis).astype(int)
    if not np.any(rounded):
        rounded[np.argmax(np.abs(axis))] = 1
    if rounded[np.flatnonzero(rounded)[0]] > 0:
        rounded *= -1
    return tuple(int(value) for value in rounded)


def _description(matrix: np.ndarray) -> str:
    kind = _operation_kind(matrix)
    if kind == "identity":
        return "identity"
    if kind == "inversion":
        return "inversion"
    improper = np.linalg.det(matrix) < 0
    proper = -matrix if improper else matrix
    order = _matrix_order(proper)
    angle = int(round(360.0 / order))
    axis = _axis(proper, 1.0)
    prefix = "inv. " if improper else ""
    return f"{prefix}{angle:3d} deg rotation - cart. axis [{axis[0]},{axis[1]},{axis[2]}]"


def operation_description(lattice: np.ndarray, operation) -> str:
    """Return QE's human-readable name for a crystal symmetry operation."""
    cartesian = (
        np.linalg.inv(np.asarray(lattice, dtype=float))
        @ np.asarray(operation.matrix, dtype=float)
        @ np.asarray(lattice, dtype=float)
    )
    return _description(cartesian)


def _irrep_sort_key(item, inversion_slot: int | None, mirror_slot: int | None):
    dimension, characters = item
    trivial = all(abs(value - 1.0) < 1.0e-7 for value in characters)
    parity = 0
    if inversion_slot is not None:
        parity = 0 if characters[inversion_slot].real > 0 else 1
    elif mirror_slot is not None:
        parity = 0 if characters[mirror_slot].real > 0 else 1
    signature = tuple((round(-value.real, 8), round(-value.imag, 8)) for value in characters)
    return (0 if trivial else 1, parity, dimension, signature)


def _assign_irrep_names(
    group_name: str,
    characters: list[tuple[int, tuple[complex, ...]]],
    inversion_slot: int | None,
    mirror_slot: int | None,
) -> list[tuple[str, tuple[complex, ...]]]:
    """Attach QE's Mulliken names using their defining generator signs."""
    _international, labels, dimensions = _GROUPS[group_name]
    principal_slot, secondary_slot = _NAMING_SLOTS.get(
        group_name, (None, None)
    )
    remaining = list(characters)
    result = []
    for name, expected_dimension in zip(labels, dimensions):
        best_index = None
        best_score = math.inf
        for index, (dimension, row) in enumerate(remaining):
            if dimension != expected_dimension:
                continue
            score = 0.0
            if name.endswith("g") or "g*" in name:
                if inversion_slot is not None:
                    score += abs(row[inversion_slot] - dimension) * 100.0
            if name.endswith("u") or "u*" in name:
                if inversion_slot is not None:
                    score += abs(row[inversion_slot] + dimension) * 100.0
            if "''" in name and mirror_slot is not None:
                score += abs(row[mirror_slot] + dimension) * 100.0
            elif "'" in name and mirror_slot is not None:
                score += abs(row[mirror_slot] - dimension) * 100.0

            family = name[0]
            if principal_slot is not None and dimension == 1:
                principal = row[principal_slot]
                if family == "A":
                    score += abs(principal - 1.0) * 20.0
                elif family == "B":
                    score += abs(principal + 1.0) * 20.0
                elif family == "E":
                    # QE assigns the unstarred member positive imaginary
                    # character for its chosen principal operation.
                    expected_sign = -1.0 if "*" in name else 1.0
                    score += max(0.0, -expected_sign * principal.imag) * 20.0

            subscript = re.search(r"[_']([123])", name)
            if secondary_slot is not None and subscript and dimension == 1:
                number = int(subscript.group(1))
                if number in {1, 2}:
                    expected = 1.0 if number == 1 else -1.0
                    score += abs(row[secondary_slot] - expected) * 10.0
            if secondary_slot is not None and family == "T":
                number = 1 if "_1" in name else 2
                expected = 1.0 if number == 1 else -1.0
                score += abs(row[secondary_slot] - expected) * 10.0

            # D2 has three equivalent twofold generators; the conventional
            # B labels are fixed by the first two columns in QE's table.
            if group_name in {"D_2", "D_2h"} and family == "B":
                number_match = re.search(r"_([123])", name)
                if number_match:
                    targets = {1: (1.0, -1.0), 2: (-1.0, 1.0), 3: (-1.0, -1.0)}
                    first, second = targets[int(number_match.group(1))]
                    score += abs(row[1] - first) * 10.0
                    score += abs(row[2] - second) * 10.0

            trivial = all(abs(value - 1.0) < 1.0e-7 for value in row)
            if name.startswith("A_1") or name in {"A", "A_g", "A'"}:
                score += 0.0 if trivial else 1.0
            if score < best_score:
                best_index, best_score = index, float(score)
        if best_index is None:
            raise ValueError(f"cannot assign irrep {name} in {group_name}")
        _dimension, row = remaining.pop(best_index)
        result.append((name, row))
    return result


def point_group_character_table(pw, operations) -> PointGroupTable:
    """Return a scalar character table for any crystallographic point group."""
    fractional_matrices = [np.asarray(operation.matrix, dtype=int) for operation in operations]
    group_name = _identify_group(fractional_matrices)
    international, _labels, _dimensions = _GROUPS[group_name]
    inverse_lattice = np.linalg.inv(pw.lattice)
    cartesian = [inverse_lattice @ matrix @ pw.lattice for matrix in fractional_matrices]
    raw_classes = _order_classes(
        group_name,
        fractional_matrices,
        _conjugacy_classes(fractional_matrices),
    )
    if group_name == "C_2v":
        raw_classes = _order_c2v_mirrors(raw_classes, cartesian)
    characters = _class_characters(fractional_matrices, raw_classes)

    inversion_slot = next((i for i, members in enumerate(raw_classes) if _operation_kind(fractional_matrices[members[0]]) == "inversion"), None)
    mirror_slot = next((i for i, members in enumerate(raw_classes) if len(members) == 1 and _operation_kind(fractional_matrices[members[0]]) == "mirror"), None)
    characters.sort(key=lambda item: _irrep_sort_key(item, inversion_slot, mirror_slot))
    ordered_irreps = _assign_irrep_names(
        group_name, characters, inversion_slot, mirror_slot
    )

    class_names = _CLASS_LABELS[group_name]
    classes = tuple(
        PointGroupClass(
            class_names[index],
            tuple(member + 1 for member in members),
            _description(cartesian[members[0]]),
        )
        for index, members in enumerate(raw_classes)
    )
    return PointGroupTable(group_name, international, classes, tuple(ordered_irreps))


# Compatibility for callers introduced with the initial O_h-only version.
oh_character_table = point_group_character_table


__all__ = [
    "PointGroupClass", "PointGroupTable", "point_group_character_table",
    "oh_character_table", "operation_description",
]
