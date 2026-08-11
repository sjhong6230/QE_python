"""Scalar implementation of the Quantum ESPRESSO ``bands.x`` workflow."""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..cli_options import add_input_file_argument
from ..errors import QEInputError, UnsupportedFeatureError, emit_qe_error
from ..qe_format import format_qe_closing, format_qe_opening, format_qe_timing
from ..point_group import point_group_character_table
from ..symmetry import SymmetryOperation, find_space_group
from ..version import __version__
from .band_data import (
    BandData,
    read_saved_bands,
    resolve_save_directory,
    write_band_file,
    write_gnuplot,
)
from .namelist import parse_namelist
from .p_matrix import compute_and_write_p_avg
from .xml_data import find, findall, findtext, upstream_qe_xml


_SUPPORTED_KEYS = {
    "prefix", "outdir", "filband", "spin_component", "lsym",
    "no_overlap", "plot_2d", "firstk", "lastk", "lp", "filp",
    "lsigma(1)", "lsigma(2)", "lsigma(3)", "lsigma(4)",
}


# QE's ``name_rap`` strings include legacy Koster/Bradley-Cracknell labels
# for the little groups where more than one convention is commonly used.
# The integer .rap value remains the one-based row of the character table.
_QE_IRREP_DISPLAY_NAMES = {
    "C_2v": ("A_1  D_1  S_1", "A_2  D_2  S_2", "B_1  D_3  S_3", "B_2  D_4  S_4"),
    "C_3v": ("A_1  L_1", "A_2  L_2", "E    L_3"),
    "C_4v": ("A_1  G_1 D_1", "A_2  G_2 D_1'", "B_1  G_3 D_2", "B_2  G_4 D_2'", "E    G_5 D_5"),
    "D_4h": (
        "A_1g X_1  M_1", "A_2g X_4  M_4", "B_1g X_2  M_2",
        "B_2g X_3  M_3", "E_g  X_5  M_5", "A_1u X_1' M_1'",
        "A_2u X_4' M_4'", "B_1u X_2' M_2'", "B_2u X_3' M_3'",
        "E_u  X_5' M_5'",
    ),
    "D_2d": ("A_1  X_1  W_1", "A_2  X_4  W_2'", "B_1  X_2  W_1'", "B_2  X_3  W_2", "E    X_5  W_3"),
    "D_3d": ("A_1g L_1", "A_2g L_2", "E_g  L_3", "A_1u L_1'", "A_2u L_2'", "E_u  L_3'"),
    "S_4": ("A    W_1", "B    W_3", "E    W_4", "E*   W_2"),
    "T_d": ("A_1  G_1  P_1", "A_2  G_2  P_2", "E    G_12 P_3", "T_1  G_25 P_5", "T_2  G_15 P_4"),
    "O_h": (
        "A_1g G_1   G_1+", "A_2g G_2   G_2+", "E_g  G_12  G_3+",
        "T_1g G_15' G_4+", "T_2g G_25' G_5+", "A_1u G_1'  G_1-",
        "A_2u G_2'  G_2-", "E_u  G_12' G_3-", "T_1u G_15  G_4-",
        "T_2u G_25  G_5-",
    ),
}


def _qe_irrep_display_name(table, irrep_index: int, name: str) -> str:
    aliases = _QE_IRREP_DISPLAY_NAMES.get(table.schoenflies)
    return aliases[irrep_index - 1] if aliases is not None else name


def _read_wavefunctions(
    directory: Path,
    nks: int,
    indices: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    import h5py

    result = []
    file_indices = range(1, nks + 1) if indices is None else (indices + 1)
    for index in file_indices:
        path = directory / f"wfc{index}.hdf5"
        try:
            with h5py.File(path, "r") as h5:
                miller = np.asarray(h5["MillerIndices"][:], dtype=np.int32)
                raw = np.asarray(h5["evc"][:])
                if np.issubdtype(raw.dtype, np.floating):
                    if raw.ndim != 2 or raw.shape[1] != 2 * len(miller):
                        raise ValueError("invalid interleaved QE evc dataset")
                    coefficients = (
                        raw[:, 0::2] + 1j * raw[:, 1::2]
                    ).astype(np.complex128, copy=False).T
                else:
                    coefficients = np.asarray(raw, dtype=np.complex128).T
        except (OSError, KeyError, ValueError) as exc:
            raise QEInputError(f"cannot read wavefunctions {path}: {exc}") from exc
        result.append((miller, coefficients))
    return result


def reorder_by_overlap(
    data: BandData,
    wavefunctions: list[tuple[np.ndarray, np.ndarray]],
) -> BandData:
    """Follow bands by maximum one-to-one wavefunction overlap."""
    energies = data.energies_ev.copy()
    ordered = [wavefunctions[0]]
    for kpoint in range(1, data.nks):
        previous_miller, previous = ordered[-1]
        current_miller, current = wavefunctions[kpoint]
        previous_lookup = {tuple(row): i for i, row in enumerate(previous_miller)}
        pairs = [
            (previous_lookup[tuple(row)], current_index)
            for current_index, row in enumerate(current_miller)
            if tuple(row) in previous_lookup
        ]
        if not pairs:
            raise QEInputError(f"adjacent k points {kpoint} and {kpoint + 1} share no plane waves")
        left, right = np.asarray(pairs, dtype=int).T
        overlap = previous[left].conj().T @ current[right]
        rows, columns = linear_sum_assignment(-np.abs(overlap) ** 2)
        permutation = np.empty(data.nbnd, dtype=int)
        permutation[rows] = columns
        energies[kpoint] = energies[kpoint, permutation]
        ordered.append((current_miller, current[:, permutation]))
    wavefunctions[:] = ordered
    return BandData(data.kpoints, energies, data.spins)


def _saved_structure(
    directory: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[SymmetryOperation, ...]]:
    root = ET.parse(directory / "data-file-schema.xml").getroot()
    lattice = np.vstack([
        np.fromstring(findtext(
            root, f"output/atomic_structure/cell/a{i}"
        ) or "", sep=" ")
        for i in range(1, 4)
    ])
    atoms = findall(root, "output/atomic_structure/atomic_positions/atom")
    positions = np.vstack([np.fromstring(atom.text or "", sep=" ") for atom in atoms])
    labels = [atom.attrib.get("name", "") for atom in atoms]
    saved_operations = []
    symmetry_path = (
        "output/symmetries/symmetry"
        if upstream_qe_xml(root)
        else "output/band_structure/symmetry_operations/symmetry"
    )
    for entry in findall(root, symmetry_path):
        rotation = np.fromstring(
            findtext(entry, "rotation", "") or "", sep=" "
        )
        translation = np.fromstring(
            findtext(entry, "fractional_translation", "") or "",
            sep=" ",
        )
        if rotation.size == 9 and translation.size == 3:
            matrix = np.rint(rotation).astype(int).reshape(3, 3)
            if upstream_qe_xml(root):
                matrix = matrix.T
            saved_operations.append(
                SymmetryOperation(
                    matrix, translation
                )
            )
    fractional = positions @ np.linalg.inv(lattice)
    operations = tuple(saved_operations) or find_space_group(
        lattice, fractional, labels
    )
    return lattice, fractional, labels, operations


def _qe_plot_data(directory: Path, data: BandData) -> BandData:
    """Convert saved crystal k coordinates to QE's 2pi/alat convention."""
    root = ET.parse(directory / "data-file-schema.xml").getroot()
    structure = find(root, "output/atomic_structure")
    lattice = np.vstack([
        np.fromstring(
            findtext(root, f"output/atomic_structure/cell/a{i}", "") or "",
            sep=" ",
        )
        for i in range(1, 4)
    ])
    if lattice.shape != (3, 3):
        return data
    alat = (
        float(structure.attrib["alat"])
        if structure is not None and "alat" in structure.attrib
        else float(np.linalg.norm(lattice[0]))
    )
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    plot_points = data.kpoints @ reciprocal * (alat / (2.0 * np.pi))
    return BandData(plot_points, data.energies_ev, data.spins)


def _little_group(kpoint: np.ndarray, operations) -> list:
    little = []
    for operation in operations:
        rotated = kpoint @ np.linalg.inv(operation.matrix).T
        if np.allclose(rotated - kpoint, np.rint(rotated - kpoint), atol=1.0e-7):
            little.append(operation)
    return little


def _symmetry_matrix(kpoint, miller, coefficients, operation) -> np.ndarray:
    transformed = (kpoint + miller) @ np.linalg.inv(operation.matrix).T
    target = np.rint(transformed - kpoint).astype(np.int32)
    if not np.allclose(transformed - kpoint, target, atol=1.0e-7):
        raise QEInputError("little-group operation does not preserve the plane-wave basis")
    lookup = {tuple(row): index for index, row in enumerate(miller)}
    try:
        destination = np.asarray([lookup[tuple(row)] for row in target], dtype=int)
    except KeyError as exc:
        raise QEInputError("symmetry-transformed plane wave is absent from saved basis") from exc
    # QE's rotate_all_psi acts on the lattice-periodic part u_k, not on the
    # complete Bloch function.  After q=k+G is rotated back into the same
    # little-group k point, ``target`` is the destination G index.  Its
    # translation phase is exp(-i 2pi G'.t); using q' instead adds an
    # operation-dependent Bloch phase and corrupts general-k irreps whenever
    # the little-group operation carries a reciprocal shift.
    phase = np.exp(-2j * np.pi * (target @ operation.translation))
    transformed_coefficients = np.empty_like(coefficients)
    transformed_coefficients[destination] = phase[:, None] * coefficients
    return coefficients.conj().T @ transformed_coefficients


def _format_group_info(table) -> str:
    out = io.StringIO()
    group = f"{table.schoenflies} ({table.international})"
    print(f"     point group {group:<11s}", file=out)
    print(f"     there are{len(table.classes):3d} classes", file=out)
    print("     the character table:\n", file=out)
    print("       " + "".join(f"{item.label:<6s}" for item in table.classes), file=out)
    for name, characters in table.irreps:
        print(
            f"{name:<5s}" + "".join(f"{value.real:6.2f}" for value in characters),
            file=out,
        )
    if any(
        abs(value.imag) > 1.0e-8
        for _name, characters in table.irreps
        for value in characters
    ):
        print("     imaginary part", file=out)
        for name, characters in table.irreps:
            print(
                f"{name:<5s}"
                + "".join(f"{value.imag:6.2f}" for value in characters),
                file=out,
            )
    print(
        "\n     the symmetry operations in each class and the name "
        "of the first element:\n",
        file=out,
    )
    for item in table.classes:
        print(
            f"     {item.label:<5s}"
            + "".join(f"{index:5d}" for index in item.operation_indices),
            file=out,
        )
        print(f"          {item.description}", file=out)
    return out.getvalue()


def classify_irreps(
    data: BandData,
    wavefunctions: list[tuple[np.ndarray, np.ndarray]],
    directory: Path,
    firstk: int = 0,
    lastk: int = 10_000_000,
    degeneracy_tolerance_ev: float = 1.0e-4,
    reports: list[str] | None = None,
    report_points: np.ndarray | None = None,
    group_signatures: list[object | None] | None = None,
) -> np.ndarray:
    """Decompose degenerate band subspaces into QE little-group irreps."""
    lattice, _fractional, _labels, operations = _saved_structure(directory)
    classes = np.full((data.nks, data.nbnd), -1, dtype=int)
    dummy_pw = type("SavedCell", (), {"lattice": lattice})()
    last_group_signature = None
    for ik, (point, energies, (miller, coefficients)) in enumerate(
        zip(data.kpoints, data.energies_ev, wavefunctions), start=1
    ):
        if ik < max(1, firstk) or ik > lastk:
            continue
        little = _little_group(point, operations)
        if not little:
            continue
        table = point_group_character_table(dummy_pw, little)
        group_signature = (
            table.schoenflies,
            tuple(item.label for item in table.classes),
        )
        if group_signatures is not None:
            group_signatures[ik - 1] = group_signature
        shown_point = point if report_points is None else report_points[ik - 1]
        if reports is not None:
            reports.append(
                "\n " + "*" * 74 + "\n\n"
                f"                    xk=({shown_point[0]:10.5f},"
                f"{shown_point[1]:10.5f},{shown_point[2]:10.5f}  )\n"
            )
        if any(
            not np.allclose(operation.translation, 0.0)
            and np.any(
                np.rint(
                    point @ np.linalg.inv(operation.matrix).T - point
                ).astype(int)
            )
            for operation in little
        ):
            if reports is not None:
                reports.append(
                    "\n     zone border point and non-symmorphic group\n"
                    "     symmetry decomposition not available\n"
                    "\n " + "*" * 74 + "\n"
                )
            continue
        matrices = [
            _symmetry_matrix(point, miller, coefficients, operation)
            for operation in little
        ]
        class_members = [
            tuple(index - 1 for index in item.operation_indices)
            for item in table.classes
        ]
        class_sizes = np.asarray([len(items) for items in class_members])
        if reports is not None:
            if group_signature != last_group_signature:
                reports.append("\n" + _format_group_info(table))
                last_group_signature = group_signature
            group = f"{table.schoenflies} ({table.international})"
            reports.append(
                f"\n     Band symmetry, {group:11s} point group:\n\n"
            )
        start = 0
        while start < data.nbnd:
            stop = start + 1
            while stop < data.nbnd and abs(energies[stop] - energies[start]) <= degeneracy_tolerance_ev:
                stop += 1
            subspace_characters = np.asarray([
                np.mean([
                    np.trace(matrices[index][start:stop, start:stop])
                    for index in members
                ])
                for members in class_members
            ])
            decomposition: list[tuple[int, str, int, int]] = []
            irrep_slots: list[int] = []
            for irrep_index, (name, characters) in enumerate(
                table.irreps, start=1
            ):
                multiplicity_value = np.sum(
                    class_sizes
                    * subspace_characters
                    * np.conjugate(np.asarray(characters))
                ) / len(little)
                multiplicity = int(np.rint(multiplicity_value.real))
                if (
                    multiplicity <= 0
                    or abs(multiplicity_value.imag) > 2.0e-4
                    or abs(multiplicity_value.real - multiplicity) > 2.0e-3
                ):
                    continue
                dimension = int(np.rint(characters[0].real))
                count = multiplicity * dimension
                irrep_slots.extend([irrep_index] * count)
                decomposition.append((
                    irrep_index,
                    _qe_irrep_display_name(table, irrep_index, name),
                    multiplicity,
                    dimension,
                ))
            if len(irrep_slots) != stop - start:
                classes[ik - 1, start:stop] = 0
            else:
                classes[ik - 1, start:stop] = irrep_slots
            if reports is not None:
                prefix = (
                    f"     e({start + 1:3d} -{stop:3d}) = "
                    f"{energies[start]:12.5f}  eV   "
                    f"{stop - start:3d}   --> "
                )
                if len(irrep_slots) != stop - start:
                    reports.append(prefix + "  ?\n")
                else:
                    for _index, name, multiplicity, _dimension in decomposition:
                        representation = (
                            f"{multiplicity:3d} {name:<15s}"
                            if multiplicity > 1 else f"{name:<15s}"
                        )
                        reports.append(prefix + representation + "\n")
            start = stop
        if reports is not None:
            reports.append("\n " + "*" * 74 + "\n")
    return classes


def write_irrep_file(
    path: str | Path,
    data: BandData,
    irreps: np.ndarray,
    firstk: int = 0,
    lastk: int = 10_000_000,
    group_signatures: list[object | None] | None = None,
) -> Path:
    """Write QE's integer ``&plot_rap`` companion format."""
    output = Path(path)
    first = max(1, int(firstk))
    last = min(data.nks, int(lastk))
    if last < first:
        raise QEInputError("empty k-point range for symmetry analysis")
    points = data.kpoints[first - 1:last]
    labels_by_point = irreps[first - 1:last]
    signatures = (
        list(group_signatures[first - 1:last])
        if group_signatures is not None
        else [None] * len(points)
    )
    high_symmetry = np.zeros(len(points), dtype=bool)
    high_symmetry[[0, len(points) - 1]] = True
    if len(points) > 2:
        steps = np.diff(points, axis=0)
        for index in range(1, len(points) - 1):
            left, right = steps[index - 1], steps[index]
            left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
            if left_norm < 1.0e-6 or right_norm < 1.0e-6:
                continue
            cosine = float(np.dot(left, right) / left_norm / right_norm)
            high_symmetry[index] = abs(cosine - 1.0) > 1.0e-4
            if np.dot(points[index], points[index]) < 1.0e-9:
                high_symmetry[index] = True

    if len(points) > 1:
        saved_step = float(np.linalg.norm(points[1] - points[0]))
        for index in range(1, len(points)):
            distance = float(np.linalg.norm(points[index] - points[index - 1]))
            if distance < 1.0e-6:
                high_symmetry[index] = high_symmetry[index - 1]
            elif distance < 5.0 * saved_step:
                if not high_symmetry[index - 1]:
                    high_symmetry[index] = (
                        signatures[index] != signatures[index - 1]
                        or high_symmetry[index]
                    )
                if distance > 1.0e-3:
                    saved_step = distance
            else:
                high_symmetry[index] = True
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            f" &plot_rap nbnd_rap={data.nbnd:4d}, nks_rap={len(points):4d} /\n"
        )
        for index, (point, labels) in enumerate(zip(points, labels_by_point)):
            point = np.where(np.abs(point) < 5.0e-7, 0.0, point)
            stream.write(
                "          "
                + "".join(f"{value:10.6f}" for value in point)
                + ("    T\n" if high_symmetry[index] else "    F\n")
            )
            for start in range(0, data.nbnd, 10):
                stream.write(
                    "".join(f"{int(value):8d}" for value in labels[start:start + 10])
                    + "\n"
                )
    return output


def write_band_grid_2d(
    filband: str | Path,
    data: BandData,
    tolerance: float = 1.0e-8,
) -> list[Path]:
    """Write QE ``plot_2d`` files, one ``kx ky energy`` grid per band."""
    if data.nks < 4:
        raise QEInputError("plot_2d requires at least a 2 by 2 k-point grid")
    origin = data.kpoints[0]
    dky = data.kpoints[1] - origin
    if np.linalg.norm(dky) <= tolerance:
        raise QEInputError("plot_2d has coincident first two k points")
    n2 = None
    dkx = None
    for index in range(2, data.nks):
        expected = origin + index * dky
        if not np.allclose(data.kpoints[index], expected, rtol=0.0, atol=tolerance):
            n2 = index
            dkx = data.kpoints[index] - origin
            break
    if n2 is None or dkx is None or np.linalg.norm(dkx) <= tolerance:
        raise QEInputError("plot_2d k points span only one direction")
    if data.nks % n2:
        raise QEInputError("plot_2d k-point count is not a rectangular grid")
    n1 = data.nks // n2
    for i1 in range(n1):
        for i2 in range(n2):
            index = i1 * n2 + i2
            expected = origin + i1 * dkx + i2 * dky
            if not np.allclose(
                data.kpoints[index], expected, rtol=0.0, atol=tolerance
            ):
                raise QEInputError(
                    f"plot_2d k point {index + 1} is inconsistent with "
                    "the inferred rectangular grid"
                )
    length_x = float(np.linalg.norm(dkx))
    length_y = float(np.linalg.norm(dky))
    base = Path(filband)
    outputs = []
    for band in range(data.nbnd):
        output = Path(f"{base}.{band + 1}")
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for i1 in range(n1):
                for i2 in range(n2):
                    index = i1 * n2 + i2
                    stream.write(
                        f"{length_x * i1:16.6f}"
                        f"{length_y * i2:16.6f}"
                        f"{data.energies_ev[index, band]:16.6f}\n"
                    )
        outputs.append(output)
    return outputs


def _report_plottable_bands(
    stdout: io.TextIOBase, filband: Path, data: BandData
) -> None:
    coordinate = data.path_coordinate()
    if data.nks <= 2:
        high_symmetry = range(data.nks)
    else:
        selected = {0, data.nks - 1}
        steps = np.diff(data.kpoints, axis=0)
        for index in range(1, data.nks - 1):
            left, right = steps[index - 1], steps[index]
            left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
            if left_norm <= 1.0e-4 or right_norm <= 1.0e-4:
                continue
            cosine = float(np.dot(left, right) / left_norm / right_norm)
            if abs(cosine - 1.0) > 1.0e-4 or np.dot(
                data.kpoints[index], data.kpoints[index]
            ) < 1.0e-8:
                selected.add(index)
        high_symmetry = sorted(selected)
    for index in high_symmetry:
        point = data.kpoints[index]
        print(
            f"     high-symmetry point: {point[0]:7.4f}"
            f"{point[1]:7.4f}{point[2]:7.4f}   x coordinate"
            f"{coordinate[index]:9.4f}",
            file=stdout,
        )
    print(
        f"\n     Plottable bands (eV) written to file {filband}.gnu",
        file=stdout,
    )


def run_bands(
    options: dict[str, object], stdout: io.TextIOBase | None = None
) -> tuple[Path, Path | None]:
    unknown = set(options) - _SUPPORTED_KEYS
    if unknown:
        raise QEInputError(f"unknown &BANDS variable {sorted(unknown)[0]!r}")
    spin_component = int(options.get("spin_component", 1))
    if spin_component not in {1, 2}:
        raise QEInputError("spin_component must be 1 (up) or 2 (down)")
    if any(bool(options.get(f"lsigma({i})", False)) for i in range(1, 5)):
        raise UnsupportedFeatureError("spin-matrix post-processing is not implemented")
    prefix = str(options.get("prefix", "pwscf"))
    outdir = str(options["outdir"]) if "outdir" in options else None
    directory = resolve_save_directory(prefix, outdir)
    if stdout is not None:
        print(
            f"\n     Reading xml data from directory:\n\n"
            f"     {directory}{os.sep}\n",
            file=stdout,
        )
    saved_data = read_saved_bands(prefix, outdir)
    if saved_data.nspin == 1 and spin_component != 1:
        raise QEInputError("spin_component requires an LSDA calculation")
    selected_indices = np.flatnonzero(saved_data.spins == spin_component)
    data = saved_data.select_spin(spin_component)
    plot_data = _qe_plot_data(directory, data)
    filband = Path(str(options.get("filband", "bands.out")))
    if bool(options.get("plot_2d", False)):
        write_band_grid_2d(filband, plot_data)
        return filband, None
    wavefunctions = None
    lsym = bool(options.get("lsym", True))
    lp = bool(options.get("lp", False))
    if lsym or lp or not bool(options.get("no_overlap", True)):
        wavefunctions = _read_wavefunctions(
            directory, saved_data.nks, selected_indices
        )
    if not lsym and not bool(options.get("no_overlap", True)):
        assert wavefunctions is not None
        data = reorder_by_overlap(data, wavefunctions)
        plot_data = BandData(plot_data.kpoints, data.energies_ev, data.spins)
    else:
        plot_data = BandData(plot_data.kpoints, data.energies_ev, data.spins)
    write_band_file(filband, plot_data)
    write_gnuplot(f"{filband}.gnu", plot_data)
    if stdout is not None:
        _report_plottable_bands(stdout, filband, plot_data)
        print(f"     Bands written to file {filband}", file=stdout)
    if lsym:
        assert wavefunctions is not None
        firstk = int(options.get("firstk", 0))
        lastk = int(options.get("lastk", 10_000_000))
        reports: list[str] = []
        group_signatures: list[object | None] = [None] * data.nks
        irreps = classify_irreps(
            data, wavefunctions, directory,
            firstk, lastk,
            reports=reports,
            report_points=plot_data.kpoints,
            group_signatures=group_signatures,
        )
        write_irrep_file(
            f"{filband}.rap", plot_data, irreps, firstk, lastk,
            group_signatures,
        )
        if stdout is not None:
            print("".join(reports), end="", file=stdout)
    if lp:
        assert wavefunctions is not None
        compute_and_write_p_avg(
            str(options.get("filp", "p_avg.dat")),
            data,
            wavefunctions,
            directory,
            int(options.get("firstk", 0)),
            int(options.get("lastk", 10_000_000)),
            output_data=plot_data,
        )
    return filband, Path(f"{filband}.gnu")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bands.py")
    add_input_file_argument(parser)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        print(format_qe_opening("BANDS-PY", __version__), end="")
        text = Path(args.input_file).read_text(encoding="utf-8") if args.input_file else sys.stdin.read()
        run_bands(parse_namelist(text, "bands"), stdout=sys.stdout)
        elapsed = time.perf_counter() - started
        cpu_elapsed = time.process_time() - cpu_started
        print(format_qe_timing("BANDS", cpu_elapsed, elapsed))
        print(format_qe_closing(), end="")
        return 0
    except (QEInputError, UnsupportedFeatureError, OSError, ValueError) as exc:
        emit_qe_error(exc, routine="bands.py")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
