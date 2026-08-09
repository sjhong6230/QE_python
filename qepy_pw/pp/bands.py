"""Scalar implementation of the Quantum ESPRESSO ``bands.x`` workflow."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..errors import QEInputError, UnsupportedFeatureError, format_qe_error
from ..qe_format import format_qe_closing, format_qe_duration, qe_date_and_time
from ..pw.save import QES_NAMESPACE
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


_SUPPORTED_KEYS = {
    "prefix", "outdir", "filband", "spin_component", "lsym",
    "no_overlap", "plot_2d", "firstk", "lastk", "lp", "filp",
    "lsigma(1)", "lsigma(2)", "lsigma(3)", "lsigma(4)",
}


def _read_wavefunctions(directory: Path, nks: int) -> list[tuple[np.ndarray, np.ndarray]]:
    import h5py

    result = []
    for index in range(1, nks + 1):
        path = directory / f"wfc{index}.hdf5"
        try:
            with h5py.File(path, "r") as h5:
                miller = np.asarray(h5["MillerIndices"][:], dtype=np.int32)
                coefficients = np.asarray(h5["evc"][:], dtype=np.complex128).T
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
    return BandData(data.kpoints, energies)


def _saved_structure(
    directory: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[SymmetryOperation, ...]]:
    root = ET.parse(directory / "data-file-schema.xml").getroot()
    ns = {"qes": QES_NAMESPACE}
    lattice = np.vstack([
        np.fromstring(root.findtext(
            f"qes:output/qes:atomic_structure/qes:cell/qes:a{i}",
            namespaces=ns,
        ) or "", sep=" ")
        for i in range(1, 4)
    ])
    atoms = root.findall(
        "qes:output/qes:atomic_structure/qes:atomic_positions/qes:atom", ns
    )
    positions = np.vstack([np.fromstring(atom.text or "", sep=" ") for atom in atoms])
    labels = [atom.attrib.get("name", "") for atom in atoms]
    saved_operations = []
    for entry in root.findall(
        "qes:output/qes:band_structure/qes:symmetry_operations/qes:symmetry",
        ns,
    ):
        rotation = np.fromstring(
            entry.findtext("qes:rotation", default="", namespaces=ns), sep=" "
        )
        translation = np.fromstring(
            entry.findtext(
                "qes:fractional_translation", default="", namespaces=ns
            ),
            sep=" ",
        )
        if rotation.size == 9 and translation.size == 3:
            saved_operations.append(
                SymmetryOperation(
                    np.rint(rotation).astype(int).reshape(3, 3), translation
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
    ns = {"qes": QES_NAMESPACE}
    structure = root.find("qes:output/qes:atomic_structure", ns)
    lattice = np.vstack([
        np.fromstring(
            root.findtext(
                f"qes:output/qes:atomic_structure/qes:cell/qes:a{i}",
                default="", namespaces=ns,
            ),
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
    return BandData(plot_points, data.energies_ev)


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
    phase = np.exp(-2j * np.pi * (transformed @ operation.translation))
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
        table = point_group_character_table(dummy_pw, little)
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
            group_signature = (
                table.schoenflies,
                tuple(item.label for item in table.classes),
            )
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
                decomposition.append(
                    (irrep_index, name, multiplicity, dimension)
                )
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
) -> Path:
    """Write QE's integer ``&plot_rap`` companion format."""
    output = Path(path)
    high_symmetry = np.zeros(data.nks, dtype=bool)
    if data.nks:
        high_symmetry[[0, data.nks - 1]] = True
    if data.nks > 2:
        steps = np.diff(data.kpoints, axis=0)
        for index in range(1, data.nks - 1):
            left, right = steps[index - 1], steps[index]
            left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
            if left_norm > 1.0e-8 and right_norm > 1.0e-8:
                cosine = float(np.dot(left, right) / left_norm / right_norm)
                high_symmetry[index] = abs(cosine - 1.0) > 1.0e-4
            if np.dot(data.kpoints[index], data.kpoints[index]) < 1.0e-9:
                high_symmetry[index] = True
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            f" &plot_rap nbnd_rap={data.nbnd:4d}, nks_rap={data.nks:4d} /\n"
        )
        for index, (point, labels) in enumerate(
            zip(data.kpoints, irreps), start=1
        ):
            analyzed = max(1, firstk) <= index <= lastk
            stream.write(
                "          "
                + "".join(f"{value:10.6f}" for value in point)
                + ("    T\n" if analyzed and high_symmetry[index - 1] else "    F\n")
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
    if int(options.get("spin_component", 1)) != 1:
        raise UnsupportedFeatureError("spin_component requires an LSDA calculation")
    if any(bool(options.get(f"lsigma({i})", False)) for i in range(1, 5)):
        raise UnsupportedFeatureError("spin-matrix post-processing is not implemented")
    prefix = str(options.get("prefix", "pwscf"))
    outdir = str(options["outdir"]) if "outdir" in options else None
    directory = resolve_save_directory(prefix, outdir)
    if stdout is not None:
        print(f"\n     Reading data from directory:\n     {directory}\n", file=stdout)
    data = read_saved_bands(prefix, outdir)
    plot_data = _qe_plot_data(directory, data)
    filband = Path(str(options.get("filband", "bands.out")))
    if bool(options.get("plot_2d", False)):
        write_band_grid_2d(filband, plot_data)
        return filband, None
    wavefunctions = None
    lsym = bool(options.get("lsym", True))
    lp = bool(options.get("lp", False))
    if lsym or lp or not bool(options.get("no_overlap", True)):
        wavefunctions = _read_wavefunctions(directory, data.nks)
    if lp:
        assert wavefunctions is not None
        compute_and_write_p_avg(
            str(options.get("filp", "p_avg.dat")),
            data,
            wavefunctions,
            directory,
            int(options.get("firstk", 0)),
            int(options.get("lastk", 10_000_000)),
        )
    if not lsym and not bool(options.get("no_overlap", True)):
        assert wavefunctions is not None
        data = reorder_by_overlap(data, wavefunctions)
        plot_data = BandData(plot_data.kpoints, data.energies_ev)
    else:
        plot_data = BandData(plot_data.kpoints, data.energies_ev)
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
        irreps = classify_irreps(
            data, wavefunctions, directory,
            firstk, lastk,
            reports=reports,
            report_points=plot_data.kpoints,
        )
        write_irrep_file(
            f"{filband}.rap", plot_data, irreps, firstk, lastk
        )
        if stdout is not None:
            print("".join(reports), end="", file=stdout)
    return filband, Path(f"{filband}.gnu")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bands.py")
    parser.add_argument("-i", "-in", "-inp", "--input", dest="input_file")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        cdate, ctime = qe_date_and_time()
        print(
            f"\n     Program BANDS-PY v.{__version__} starts on "
            f"{cdate} at {ctime}\n"
        )
        text = Path(args.input_file).read_text(encoding="utf-8") if args.input_file else sys.stdin.read()
        run_bands(parse_namelist(text, "bands"), stdout=sys.stdout)
        elapsed = time.perf_counter() - started
        cpu_elapsed = time.process_time() - cpu_started
        print(
            "\n     BANDS        : "
            f"{format_qe_duration(cpu_elapsed, 'CPU')} "
            f"{format_qe_duration(elapsed, 'WALL')}\n"
        )
        print(format_qe_closing(), end="")
        return 0
    except (QEInputError, UnsupportedFeatureError, OSError, ValueError) as exc:
        print(format_qe_error(exc), end="", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
