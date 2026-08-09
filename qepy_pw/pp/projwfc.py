"""Scalar atomic-orbital projections corresponding to QE ``projwfc.x``."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import TextIO
import xml.etree.ElementTree as ET

import numpy as np

from ..cli_options import add_input_file_argument
from ..constants import BOHR_PER_ANGSTROM, EV_PER_HARTREE
from ..errors import QEInputError, UnsupportedFeatureError, emit_qe_error
from ..occupations import smearing_density, smearing_order
from ..pw.save import QES_NAMESPACE
from ..qe_format import format_qe_closing, format_qe_opening, format_qe_timing
from ..symmetry import SymmetryOperation
from ..upf import _qe_real_spherical_harmonics, read_upf
from ..version import __version__
from .band_data import resolve_save_directory
from .dos import DOSData, read_saved_dos
from .namelist import parse_namelist


_L_NAMES = "spdfgh"
_ALLOWED = {
    "prefix", "outdir", "ngauss", "degauss", "emin", "emax", "deltae",
    "lsym", "diag_basis", "pawproj", "filpdos", "filproj",
    "lwrite_overlaps", "lbinary_data", "kresolveddos", "tdosinboxes",
    "io_choice", "smoothing",
    "n_proj_boxes", "plotboxes",
}


@dataclass(frozen=True)
class Orbital:
    atom: int
    symbol: str
    wfc: int
    l: int
    m: int
    label: str


@dataclass(frozen=True)
class ProjectionData:
    energies_ev: np.ndarray
    weights: np.ndarray
    projections: np.ndarray  # (nks, nbnd, nwfc), squared amplitudes
    amplitudes: np.ndarray  # (nks, nbnd, nwfc), complex overlaps
    occupations: np.ndarray
    orbitals: tuple[Orbital, ...]
    overlaps: tuple[np.ndarray, ...]
    fermi_ev: float | None


@dataclass(frozen=True)
class BoxData:
    energies_ev: np.ndarray
    weights: np.ndarray
    occupations: np.ndarray
    box_weights: np.ndarray
    shape: tuple[int, int, int]
    masks: tuple[np.ndarray, ...]


def _xml_vector(element: ET.Element | None, name: str, ns: dict[str, str]) -> np.ndarray:
    if element is None:
        raise QEInputError(f"saved data is missing {name}")
    values = np.fromstring(element.text or "", sep=" ")
    if values.shape != (3,):
        raise QEInputError(f"saved {name} is not a three-vector")
    return values


def _saved_geometry(directory: Path):
    path = directory / "data-file-schema.xml"
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise QEInputError(f"cannot read saved projection metadata {path}: {exc}") from exc
    ns = {"qes": QES_NAMESPACE}
    output = root.find("qes:output", ns)
    if output is None:
        raise QEInputError("saved data contains no output section")
    structure = output.find("qes:atomic_structure", ns)
    cell = structure.find("qes:cell", ns) if structure is not None else None
    lattice = np.vstack([_xml_vector(cell.find(f"qes:a{i}", ns) if cell is not None else None, f"a{i}", ns) for i in range(1, 4)])
    reciprocal_element = output.find("qes:basis_set/qes:reciprocal_lattice", ns)
    reciprocal = np.vstack([_xml_vector(reciprocal_element.find(f"qes:b{i}", ns) if reciprocal_element is not None else None, f"b{i}", ns) for i in range(1, 4)])
    species_files = {
        entry.attrib.get("name", ""): entry.findtext("qes:pseudo_file", namespaces=ns)
        for entry in output.findall("qes:atomic_species/qes:species", ns)
    }
    atoms = []
    if structure is not None:
        for element in structure.findall("qes:atomic_positions/qes:atom", ns):
            atoms.append((element.attrib.get("name", ""), _xml_vector(element, "atom position", ns)))
    if not atoms or any(symbol not in species_files for symbol, _ in atoms):
        raise QEInputError("saved atomic species or positions are incomplete")
    operations = []
    for entry in output.findall("qes:band_structure/qes:symmetry_operations/qes:symmetry", ns):
        rotation = np.fromstring(entry.findtext("qes:rotation", default="", namespaces=ns), sep=" ", dtype=int)
        translation = np.fromstring(entry.findtext("qes:fractional_translation", default="", namespaces=ns), sep=" ")
        if rotation.size != 9 or translation.shape != (3,):
            raise QEInputError("saved symmetry operation is malformed")
        operations.append(SymmetryOperation(rotation.reshape(3, 3), translation))
    if not operations:
        operations.append(SymmetryOperation(np.eye(3, dtype=int), np.zeros(3)))
    return lattice, reciprocal, atoms, species_files, tuple(operations)


def _orbital_basis(
    directory: Path,
    miller: np.ndarray,
    kpoint: np.ndarray,
    geometry=None,
    pseudo_cache: dict | None = None,
):
    lattice, reciprocal, atoms, species_files, _operations = geometry or _saved_geometry(directory)
    volume = abs(float(np.linalg.det(lattice)))
    gk = np.asarray(miller, dtype=float) @ reciprocal + np.asarray(kpoint)
    blocks, descriptors = [], []
    cache = {} if pseudo_cache is None else pseudo_cache
    for atom_index, (symbol, position) in enumerate(atoms, start=1):
        pseudo = cache.get(symbol)
        if pseudo is None:
            pseudo_name = species_files[symbol]
            if not pseudo_name:
                raise QEInputError(f"saved species {symbol} has no pseudopotential")
            pseudo = read_upf(directory / Path(pseudo_name).name)
            if pseudo.pseudo_type.upper() != "NC":
                raise UnsupportedFeatureError("projwfc.py currently supports norm-conserving UPFs")
            cache[symbol] = pseudo
        block = pseudo.atomic_orbitals(gk, position, volume)
        blocks.append(block)
        column = 0
        for wfc_index, wavefunction in enumerate(
            (item for item in pseudo.atomic_wavefunctions if item.occupation >= 0.0), start=1
        ):
            for m in range(2 * wavefunction.angular_momentum + 1):
                descriptors.append(Orbital(atom_index, symbol, wfc_index, wavefunction.angular_momentum, m, wavefunction.label))
                column += 1
        if column != block.shape[1]:
            raise QEInputError("UPF atomic-wavefunction metadata are inconsistent")
    if not blocks or sum(block.shape[1] for block in blocks) == 0:
        raise QEInputError("Cannot project on zero atomic wavefunctions")
    return np.column_stack(blocks), tuple(descriptors)


def _lowdin_basis(atomic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    overlap = atomic.conj().T @ atomic
    overlap = 0.5 * (overlap + overlap.conj().T)
    eigenvalues, vectors = np.linalg.eigh(overlap)
    cutoff = max(1.0e-12, 1.0e-10 * float(np.max(eigenvalues)))
    if np.any(eigenvalues <= cutoff):
        raise QEInputError("atomic orbital overlap matrix is singular")
    inverse_sqrt = (vectors / np.sqrt(eigenvalues)[None, :]) @ vectors.conj().T
    return atomic @ inverse_sqrt, overlap


def _real_harmonic_rotation(l: int, cartesian_rotation: np.ndarray) -> np.ndarray:
    """Return the orthogonal QE-real-harmonic representation of a rotation."""
    size = 2 * l + 1
    if l == 0:
        return np.ones((1, 1))
    # An overdetermined deterministic spherical sample avoids special-axis
    # rank loss. Polar decomposition removes least-squares roundoff.
    index = np.arange(max(24, 4 * size), dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / len(index)
    phi = index * (np.pi * (3.0 - np.sqrt(5.0)))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    vectors = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    source = _qe_real_spherical_harmonics(l, vectors)
    target = _qe_real_spherical_harmonics(l, vectors @ cartesian_rotation)
    representation = np.linalg.lstsq(source, target, rcond=None)[0]
    left, _singular, right = np.linalg.svd(representation)
    return left @ right


def _atom_mapping(atoms, operation: SymmetryOperation) -> np.ndarray:
    labels = [symbol for symbol, _position in atoms]
    fractional = np.vstack([position for _symbol, position in atoms])
    transformed = fractional @ operation.matrix + operation.translation
    mapping = np.empty(len(atoms), dtype=np.int32)
    for source, (position, label) in enumerate(zip(transformed, labels)):
        matches = []
        for target, (reference, reference_label) in enumerate(zip(fractional, labels)):
            delta = position - reference
            delta -= np.rint(delta)
            if reference_label == label and np.linalg.norm(delta) < 1.0e-7:
                matches.append(target)
        if len(matches) != 1:
            raise QEInputError("symmetry operation does not uniquely map atomic orbitals")
        mapping[source] = matches[0]
    return mapping


def symmetrize_projection_weights(
    amplitudes: np.ndarray,
    orbitals: tuple[Orbital, ...],
    lattice: np.ndarray,
    atoms,
    operations: tuple[SymmetryOperation, ...],
    *,
    diag_basis: bool = False,
    kpoint_weights: np.ndarray | None = None,
    occupations: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate and average the full orbital projection density matrices."""
    values = np.asarray(amplitudes, dtype=np.complex128)
    if len(operations) <= 1 and not diag_basis:
        return np.abs(values) ** 2
    inverse_lattice = np.linalg.inv(lattice)
    fractional_atoms = tuple(
        (symbol, np.asarray(position) @ inverse_lattice)
        for symbol, position in atoms
    )
    averaged = np.zeros(values.shape, dtype=float)
    lookup = {
        (orbital.atom, orbital.wfc, orbital.l, orbital.m): index
        for index, orbital in enumerate(orbitals)
    }
    transforms = []
    for operation in operations:
        cartesian = inverse_lattice @ operation.matrix @ lattice
        atom_map = _atom_mapping(fractional_atoms, operation)
        transform = np.zeros((len(orbitals), len(orbitals)), dtype=float)
        representations = {}
        for source, orbital in enumerate(orbitals):
            representation = representations.get(orbital.l)
            if representation is None:
                representation = _real_harmonic_rotation(orbital.l, cartesian)
                representations[orbital.l] = representation
            target_atom = int(atom_map[orbital.atom - 1]) + 1
            for target_m in range(2 * orbital.l + 1):
                target = lookup[(target_atom, orbital.wfc, orbital.l, target_m)]
                transform[target, source] = representation[orbital.m, target_m]
        transforms.append(transform)
    rotations = None
    if diag_basis:
        if kpoint_weights is None or occupations is None:
            raise ValueError("diag_basis requires k-point weights and occupations")
        density = np.zeros((len(orbitals), len(orbitals)), dtype=np.complex128)
        for transform in transforms:
            rotated = np.einsum("...o,po->...p", values, transform)
            density += np.einsum(
                "k,kb,kbi,kbj->ij",
                kpoint_weights,
                occupations,
                rotated.conj(),
                rotated,
            ) / len(transforms)
        rotations = {}
        for _orbital, columns in _groups(orbitals):
            block = density[columns, columns]
            _eigenvalues, eigenvectors = np.linalg.eigh(
                0.5 * (block + block.conj().T)
            )
            rotations[(columns.start, columns.stop)] = eigenvectors
    for transform in transforms:
        rotated = np.einsum("...o,po->...p", values, transform)
        if rotations is not None:
            for key, eigenvectors in rotations.items():
                columns = slice(*key)
                rotated[..., columns] = rotated[..., columns] @ eigenvectors
        averaged += np.abs(rotated) ** 2
    return averaged / len(operations)


def compute_projections(
    prefix: str = "pwscf", outdir: str | None = None, *, symmetrize: bool = True,
    diag_basis: bool = False,
) -> ProjectionData:
    import h5py

    directory = resolve_save_directory(prefix, outdir)
    saved = read_saved_dos(prefix, outdir)
    try:
        root = ET.parse(directory / "data-file-schema.xml").getroot()
    except (OSError, ET.ParseError) as exc:
        raise QEInputError(f"cannot read saved occupations: {exc}") from exc
    ns = {"qes": QES_NAMESPACE}
    records = root.findall("qes:output/qes:band_structure/qes:ks_energies", ns)
    occupations = np.vstack([
        np.fromstring(record.findtext("qes:occupations", default="", namespaces=ns), sep=" ")
        for record in records
    ])
    if occupations.shape != saved.eigenvalues_ev.shape:
        raise QEInputError("saved occupations do not match the eigenvalues")
    all_amplitudes, overlaps = [], []
    orbitals = None
    geometry = _saved_geometry(directory)
    pseudo_cache = {}
    for index in range(1, len(saved.weights) + 1):
        path = directory / f"wfc{index}.hdf5"
        try:
            with h5py.File(path, "r") as h5:
                miller = np.asarray(h5["MillerIndices"][:], dtype=np.int32)
                wavefunctions = np.asarray(h5["evc"][:], dtype=np.complex128).T
                kpoint = np.asarray(h5.attrs["xk"], dtype=float)
        except (OSError, KeyError, ValueError) as exc:
            raise QEInputError(f"cannot read projection wavefunctions {path}: {exc}") from exc
        atomic, current_orbitals = _orbital_basis(
            directory, miller, kpoint, geometry, pseudo_cache
        )
        if orbitals is None:
            orbitals = current_orbitals
        elif orbitals != current_orbitals:
            raise QEInputError("atomic orbital ordering changes between k points")
        orthogonal, overlap = _lowdin_basis(atomic)
        amplitudes = orthogonal.conj().T @ wavefunctions
        all_amplitudes.append(amplitudes.T)
        overlaps.append(overlap)
    assert orbitals is not None
    amplitudes_array = np.asarray(all_amplitudes)
    lattice, _reciprocal, atoms, _species, operations = geometry
    projection_operations = (
        operations
        if symmetrize
        else (SymmetryOperation(np.eye(3, dtype=int), np.zeros(3)),)
    )
    projections = (
        symmetrize_projection_weights(
            amplitudes_array, orbitals, lattice, atoms, projection_operations,
            diag_basis=diag_basis,
            kpoint_weights=saved.weights,
            occupations=occupations,
        )
        if symmetrize or diag_basis
        else np.abs(amplitudes_array) ** 2
    )
    return ProjectionData(saved.eigenvalues_ev, saved.weights, projections, amplitudes_array, occupations, orbitals, tuple(overlaps), saved.fermi_ev)


def _dos_kernel(energies: np.ndarray, grid: np.ndarray, width: float, ngauss: int) -> np.ndarray:
    if ngauss not in {0, 1, -1, -99}:
        raise QEInputError("ngauss must be 0, 1, -1, or -99")
    if width <= 0:
        raise QEInputError("projected DOS requires positive degauss")
    x = (grid[:, None, None] - energies[None, :, :]) / width
    return smearing_density(x, ngauss) / width


def _indexed_box(options: dict[str, object], stem: str, box: int, shape) -> tuple[np.ndarray, ...]:
    indices = []
    for axis, size in enumerate(shape, start=1):
        low = int(options.get(f"irmin({axis},{box})", 1))
        high = int(options.get(f"irmax({axis},{box})", 0))
        low = (low - 1) % size
        high = size - 1 if high == 0 else (high - 1) % size
        if high >= low:
            values = np.arange(low, high + 1)
        else:
            values = np.concatenate((np.arange(0, high + 1), np.arange(low, size)))
        indices.append(values)
    return tuple(indices)


def _box_projections(prefix: str, outdir: str | None, options: dict[str, object]) -> BoxData:
    import h5py

    directory = resolve_save_directory(prefix, outdir)
    saved = read_saved_dos(prefix, outdir)
    root = ET.parse(directory / "data-file-schema.xml").getroot()
    ns = {"qes": QES_NAMESPACE}
    fft = root.find("qes:output/qes:basis_set/qes:fft_grid", ns)
    if fft is None:
        raise QEInputError("box LDOS requires a saved FFT grid")
    shape = tuple(int(fft.attrib[f"nr{i}"]) for i in range(1, 4))
    records = root.findall("qes:output/qes:band_structure/qes:ks_energies", ns)
    occupations = np.vstack([
        np.fromstring(record.findtext("qes:occupations", default="", namespaces=ns), sep=" ")
        for record in records
    ])
    count = int(options.get("n_proj_boxes", 1))
    if count < 1:
        raise QEInputError("n_proj_boxes must be positive")
    boxes = [_indexed_box(options, "ir", box, shape) for box in range(1, count + 1)]
    masks = []
    for axes in boxes:
        mask = np.zeros(shape, dtype=bool)
        mask[np.ix_(*axes)] = True
        masks.append(mask)
    projections = []
    points = int(np.prod(shape))
    for ik in range(1, len(saved.weights) + 1):
        path = directory / f"wfc{ik}.hdf5"
        with h5py.File(path, "r") as h5:
            miller = np.asarray(h5["MillerIndices"][:], dtype=np.int32)
            coefficients = np.asarray(h5["evc"][:], dtype=np.complex128)
        reciprocal_grid = np.zeros(shape, dtype=np.complex128)
        band_weights = np.empty((coefficients.shape[0], count), dtype=float)
        slots = tuple((miller[:, axis] % shape[axis]) for axis in range(3))
        for band, values in enumerate(coefficients):
            reciprocal_grid.fill(0.0)
            reciprocal_grid[slots] = values
            probability = np.abs(np.fft.ifftn(reciprocal_grid) * np.sqrt(points)) ** 2
            for box, mask in enumerate(masks):
                band_weights[band, box] = float(np.sum(probability[mask]))
        projections.append(band_weights)
    result = np.asarray(projections)
    if np.any(result < -1.0e-12) or np.any(result > 1.0 + 1.0e-8):
        raise QEInputError("real-space box projections violate wavefunction normalization")
    return BoxData(saved.eigenvalues_ev, saved.weights, occupations, result, shape, tuple(masks))


def _write_box_xsf(path: Path, mask: np.ndarray, lattice: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("CRYSTAL\nPRIMVEC\n")
        for vector in lattice:
            stream.write(" ".join(f"{value:16.9f}" for value in vector) + "\n")
        stream.write("PRIMCOORD\n0 1\nBEGIN_BLOCK_DATAGRID_3D\nbox\nBEGIN_DATAGRID_3D_box\n")
        stream.write(f"{mask.shape[0]} {mask.shape[1]} {mask.shape[2]}\n0 0 0\n")
        for vector in lattice:
            stream.write(" ".join(f"{value:16.9f}" for value in vector) + "\n")
        flat = mask.astype(int).ravel(order="F")
        for start in range(0, len(flat), 12):
            stream.write(" ".join(map(str, flat[start:start + 12])) + "\n")
        stream.write("END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n")


def _run_boxes(options: dict[str, object], prefix: str, outdir: str | None) -> tuple[BoxData, list[Path]]:
    data = _box_projections(prefix, outdir, options)
    saved = read_saved_dos(prefix, outdir)
    delta = float(options.get("deltae", 0.01))
    if delta <= 0:
        raise QEInputError("DeltaE must be positive")
    degauss_ry, ngauss, _source = _smearing_parameters(saved, options, delta)
    width = 0.5 * degauss_ry * EV_PER_HARTREE
    emin = float(options.get("emin", np.min(data.energies_ev) - 3.0 * width))
    emax = float(options.get("emax", np.max(data.energies_ev) + 3.0 * width))
    grid = emin + np.arange(int(np.floor((emax - emin) / delta + 1.000001))) * delta
    kernel = _dos_kernel(data.energies_ev, grid, width, ngauss)
    kresolved = bool(options.get("kresolveddos", False))
    weights = np.ones_like(data.weights) if kresolved else data.weights
    box_dos = 2.0 * np.einsum("ekb,k,kbx->ex", kernel, weights, data.box_weights)
    output = Path(f"{options.get('filpdos', prefix)}.ldos_boxes")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(("# ik E (eV)" if kresolved else "# E (eV)") + "".join(f" LDOS_box{box}" for box in range(1, box_dos.shape[1] + 1)) + "\n")
        if kresolved:
            for ik in range(len(data.weights)):
                values_k = 2.0 * np.einsum("eb,bx->ex", kernel[:, ik], data.box_weights[ik])
                for energy, values in zip(grid, values_k):
                    stream.write(f"{ik + 1:6d} {energy:12.6f}" + "".join(f" {value:14.6e}" for value in values) + "\n")
        else:
            for energy, values in zip(grid, box_dos):
                stream.write(f"{energy:12.6f}" + "".join(f" {value:14.6e}" for value in values) + "\n")
    paths = [output]
    if bool(options.get("plotboxes", False)):
        lattice, _reciprocal, _atoms, _species, _operations = _saved_geometry(resolve_save_directory(prefix, outdir))
        for box, mask in enumerate(data.masks, start=1):
            path = Path(f"box#{box}.xsf")
            _write_box_xsf(path, mask, lattice / BOHR_PER_ANGSTROM)
            paths.append(path)
    if "filproj" in options and str(options["filproj"]).strip():
        path = Path(str(options["filproj"]))
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for ik, bands in enumerate(data.box_weights, start=1):
                for band, values in enumerate(bands, start=1):
                    stream.write(f"k = {ik:5d} band = {band:5d}" + "".join(f" box#{box + 1} = {value:.10e}" for box, value in enumerate(values)) + "\n")
        paths.append(path)
    return data, paths


def _write_atomic_proj(path: Path, data: ProjectionData, include_overlaps: bool) -> None:
    root = ET.Element("ATOMIC_PROJECTIONS")
    states = ET.SubElement(root, "ATOMIC_WFC", number_of_wfc=str(len(data.orbitals)))
    for index, orbital in enumerate(data.orbitals, start=1):
        ET.SubElement(states, "STATE", index=str(index), atom=str(orbital.atom), species=orbital.symbol, wfc=str(orbital.wfc), l=str(orbital.l), m=str(orbital.m + 1), label=orbital.label)
    for ik, (energies, projections) in enumerate(zip(data.energies_ev, data.projections), start=1):
        point = ET.SubElement(root, "K_POINT", index=str(ik))
        for band, (energy, values) in enumerate(zip(energies, projections), start=1):
            state = ET.SubElement(point, "EIGENSTATE", band=str(band), energy_ev=f"{energy:.16e}")
            state.text = " ".join(f"{value:.16e}" for value in values)
        if include_overlaps:
            overlap = ET.SubElement(point, "OVERLAP", size=str(len(data.orbitals)))
            overlap.text = " ".join(f"{value.real:.16e},{value.imag:.16e}" for value in data.overlaps[ik - 1].ravel())
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _groups(orbitals: tuple[Orbital, ...]):
    result = []
    start = 0
    while start < len(orbitals):
        first = orbitals[start]
        end = start + 1
        while end < len(orbitals) and (orbitals[end].atom, orbitals[end].wfc) == (first.atom, first.wfc):
            end += 1
        result.append((first, slice(start, end)))
        start = end
    return result


def _smearing_parameters(
    saved: DOSData, options: dict[str, object], delta: float
) -> tuple[float, int, str]:
    input_degauss = float(options.get("degauss", 0.0))
    if input_degauss != 0.0:
        return input_degauss, int(options.get("ngauss", 0)), "read from input"
    if saved.degauss_ry > 0.0:
        return saved.degauss_ry, smearing_order(saved.smearing), "read from file"
    return delta / (0.5 * EV_PER_HARTREE), 0, "default values"


def run_projwfc(
    options: dict[str, object], stdout: TextIO | None = None
) -> tuple[ProjectionData | BoxData, list[Path]]:
    unknown = {
        key for key in options
        if key not in _ALLOWED and not key.startswith("irmin(") and not key.startswith("irmax(")
    }
    if unknown:
        raise QEInputError(f"unknown &PROJWFC variable {sorted(unknown)[0]!r}")
    for key in ("pawproj", "lbinary_data"):
        if bool(options.get(key, False)):
            raise UnsupportedFeatureError(f"projwfc {key}=.true. is not ported")
    if not bool(options.get("lsym", True)) and not bool(options.get("kresolveddos", False)):
        raise QEInputError("lsym=.false. requires kresolveddos=.true.")
    prefix = str(options.get("prefix", "pwscf"))
    outdir = str(options["outdir"]) if "outdir" in options else None
    if stdout is not None:
        directory = resolve_save_directory(prefix, outdir)
        print(
            f"\n     Reading xml data from directory:\n\n"
            f"     {directory}{os.sep}",
            file=stdout,
        )
    saved = read_saved_dos(prefix, outdir)
    input_degauss = float(options.get("degauss", 0.0))
    if saved.occupations_kind.lower().startswith("tetra") and input_degauss == 0.0:
        raise UnsupportedFeatureError("projected tetrahedron DOS is not yet ported; specify degauss for smearing")
    delta = float(options.get("deltae", 0.01))
    if delta <= 0:
        raise QEInputError("DeltaE must be positive")
    degauss_ry, ngauss, source = _smearing_parameters(saved, options, delta)
    if stdout is not None:
        print(
            f"\n     Gaussian broadening ({source}): "
            f"ngauss,degauss={ngauss:4d}{degauss_ry:12.6f}\n",
            file=stdout,
        )
    if bool(options.get("tdosinboxes", False)):
        return _run_boxes(options, prefix, outdir)
    data = compute_projections(
        prefix, outdir, symmetrize=bool(options.get("lsym", True)),
        diag_basis=bool(options.get("diag_basis", False)),
    )
    width = 0.5 * degauss_ry * EV_PER_HARTREE
    emin = float(options.get("emin", np.min(data.energies_ev) - 3.0 * width))
    emax = float(options.get("emax", np.max(data.energies_ev) + 3.0 * width))
    if emax < emin:
        raise QEInputError("Emax must be greater than or equal to Emin")
    grid = emin + np.arange(int(np.floor((emax - emin) / delta + 1.000001))) * delta
    kernel = _dos_kernel(data.energies_ev, grid, width, ngauss)
    kresolved = bool(options.get("kresolveddos", False))
    weights = np.ones_like(data.weights) if kresolved else data.weights
    total = 2.0 * np.einsum("ekb,k->e", kernel, weights)
    projected = 2.0 * np.einsum("ekb,k,kbo->eo", kernel, weights, data.projections)
    base = str(options.get("filpdos", prefix))
    paths = []
    total_path = Path(f"{base}.pdos_tot")
    with total_path.open("w", encoding="utf-8", newline="\n") as stream:
        if kresolved:
            stream.write("# ik E (eV) DOS(E) PDOS(E)\n")
            for ik in range(len(data.weights)):
                dos_k = 2.0 * np.sum(kernel[:, ik, :], axis=1)
                pdos_k = 2.0 * np.einsum("eb,bo->e", kernel[:, ik, :], data.projections[ik]).sum(axis=1)
                for energy, dos, pdos in zip(grid, dos_k, pdos_k):
                    stream.write(f"{ik + 1:6d} {energy:12.6f} {dos:14.6e} {pdos:14.6e}\n")
        else:
            stream.write("# E (eV) DOS(E) PDOS(E)\n")
            for energy, dos, pdos in zip(grid, total, np.sum(projected, axis=1)):
                stream.write(f"{energy:12.6f} {dos:14.6e} {pdos:14.6e}\n")
    paths.append(total_path)
    lnames = _L_NAMES
    for orbital, columns in _groups(data.orbitals):
        label = lnames[orbital.l] if orbital.l < len(lnames) else f"l{orbital.l}"
        path = Path(f"{base}.pdos_atm#{orbital.atom}({orbital.symbol})_wfc#{orbital.wfc}({label})")
        components = projected[:, columns]
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            prefix_header = "# ik E (eV) LDOS(E)" if kresolved else "# E (eV) LDOS(E)"
            stream.write(prefix_header + "".join(f" PDOS_{m + 1}(E)" for m in range(components.shape[1])) + "\n")
            if kresolved:
                for ik in range(len(data.weights)):
                    components_k = 2.0 * np.einsum("eb,bo->eo", kernel[:, ik, :], data.projections[ik][:, columns])
                    for energy, values in zip(grid, components_k):
                        stream.write(f"{ik + 1:6d} {energy:12.6f} {np.sum(values):14.6e}" + "".join(f" {value:14.6e}" for value in values) + "\n")
            else:
                for energy, values in zip(grid, components):
                    stream.write(f"{energy:12.6f} {np.sum(values):14.6e}" + "".join(f" {value:14.6e}" for value in values) + "\n")
        paths.append(path)
    directory = resolve_save_directory(prefix, outdir)
    atomic_proj = directory / "atomic_proj.xml"
    _write_atomic_proj(atomic_proj, data, bool(options.get("lwrite_overlaps", False)))
    paths.append(atomic_proj)
    if "filproj" in options and str(options["filproj"]).strip():
        filproj = Path(str(options["filproj"]))
        with filproj.open("w", encoding="utf-8", newline="\n") as stream:
            for index, orbital in enumerate(data.orbitals, start=1):
                stream.write(f"state # {index}: atom {orbital.atom} ({orbital.symbol}), wfc {orbital.wfc} (l={orbital.l} m={orbital.m + 1})\n")
            for ik, (energies, projections) in enumerate(zip(data.energies_ev, data.projections), start=1):
                for band, (energy, values) in enumerate(zip(energies, projections), start=1):
                    stream.write(f"k = {ik:5d} band = {band:5d} e = {energy:12.6f} |psi|^2 = {np.sum(values):8.5f}\n")
        paths.append(filproj)
    return data, paths


_M_LABELS = {
    1: ("z", "x", "y"),
    2: ("z2", "xz", "yz", "x2-y2", "xy"),
    3: ("z3", "xz2", "yz2", "zx2-zy2", "xyz", "x3-3xy2", "3yx2-y3"),
}


def _format_projection_summary(data: ProjectionData, *, diag_basis: bool = False) -> str:
    """Render the atomic-state and Lowdin-charge blocks printed by ``projwfc.x``."""
    lines = [
        "",
        "     Atomic states used for projection",
        "     (read from pseudopotential files):",
        "",
    ]
    for index, orbital in enumerate(data.orbitals, start=1):
        lines.append(
            f"     state #{index:4d}: atom {orbital.atom:3d} ({orbital.symbol:>3}), "
            f"wfc {orbital.wfc:2d} (l={orbital.l:d} m={orbital.m + 1:2d})"
        )

    charges = np.einsum(
        "k,kb,kbo->o", data.weights, data.occupations, data.projections
    )
    lines.extend(("", "Lowdin Charges: ", ""))
    atom_numbers = sorted({orbital.atom for orbital in data.orbitals})
    maximum_l = max((orbital.l for orbital in data.orbitals), default=0)
    for atom in atom_numbers:
        atom_indices = [
            index for index, orbital in enumerate(data.orbitals)
            if orbital.atom == atom
        ]
        total = float(np.sum(charges[atom_indices]))
        for angular_momentum in range(maximum_l + 1):
            component_indices = [
                index for index in atom_indices
                if data.orbitals[index].l == angular_momentum
            ]
            label = (
                _L_NAMES[angular_momentum]
                if angular_momentum < len(_L_NAMES)
                else f"l{angular_momentum}"
            )
            angular_charge = float(np.sum(charges[component_indices]))
            line = (
                f"     Atom # {atom:3d}: total charge = {total:8.4f}, "
                f"{label} ={angular_charge:8.4f}"
            )
            if angular_momentum:
                labels = (
                    tuple(str(index + 1) for index in range(2 * angular_momentum + 1))
                    if diag_basis
                    else _M_LABELS.get(
                        angular_momentum,
                        tuple(str(index + 1) for index in range(2 * angular_momentum + 1)),
                    )
                )
                component_charges = {
                    magnetic: float(np.sum([
                        charges[index] for index in component_indices
                        if data.orbitals[index].m == magnetic
                    ]))
                    for magnetic in range(2 * angular_momentum + 1)
                }
                line += "".join(
                    f"{label}{component_label}={component_charges[magnetic]:8.4f}, "
                    for magnetic, component_label in enumerate(labels)
                )
            lines.append(line)

    electrons = float(np.einsum("k,kb->", data.weights, data.occupations))
    spilling = 1.0 - float(np.sum(charges)) / max(1.0e-14, electrons)
    lines.extend(("", f"     Spilling Parameter: {spilling:8.4f}", ""))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="projwfc.py")
    add_input_file_argument(parser)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        print(format_qe_opening("PROJWFC-PY", __version__), end="")
        text = Path(args.input_file).read_text(encoding="utf-8") if args.input_file else sys.stdin.read()
        options = parse_namelist(text, "projwfc")
        data, _paths = run_projwfc(options, stdout=sys.stdout)
        if isinstance(data, BoxData):
            charges = np.einsum(
                "k,kb,kbx->x", data.weights, data.occupations, data.box_weights
            )
            for box, charge in enumerate(charges, start=1):
                print(f"     box # {box:4d}: integrated occupied weight = {charge:11.6f}")
        else:
            print(
                _format_projection_summary(
                    data, diag_basis=bool(options.get("diag_basis", False))
                )
            )
        print(
            format_qe_timing(
                "PROJWFC",
                time.process_time() - cpu_started,
                time.perf_counter() - started,
            )
        )
        print(format_qe_closing(), end="")
        return 0
    except (QEInputError, UnsupportedFeatureError, OSError, ValueError, ET.ParseError) as exc:
        emit_qe_error(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
