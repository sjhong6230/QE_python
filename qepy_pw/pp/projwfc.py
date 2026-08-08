"""Scalar atomic-orbital projections corresponding to QE ``projwfc.x``."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from ..constants import EV_PER_HARTREE
from ..errors import QEInputError, UnsupportedFeatureError, format_qe_error
from ..occupations import smearing_order, wgauss
from ..pw.save import QES_NAMESPACE
from ..upf import read_upf
from .band_data import resolve_save_directory
from .dos import read_saved_dos
from .namelist import parse_namelist


_L_NAMES = "spdfgh"
_ALLOWED = {
    "prefix", "outdir", "ngauss", "degauss", "emin", "emax", "deltae",
    "lsym", "diag_basis", "pawproj", "filpdos", "filproj",
    "lwrite_overlaps", "lbinary_data", "kresolveddos", "tdosinboxes",
    "io_choice", "smoothing",
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
    occupations: np.ndarray
    orbitals: tuple[Orbital, ...]
    overlaps: tuple[np.ndarray, ...]
    fermi_ev: float | None


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
    return lattice, reciprocal, atoms, species_files


def _orbital_basis(
    directory: Path,
    miller: np.ndarray,
    kpoint: np.ndarray,
    geometry=None,
    pseudo_cache: dict | None = None,
):
    lattice, reciprocal, atoms, species_files = geometry or _saved_geometry(directory)
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


def compute_projections(prefix: str = "pwscf", outdir: str | None = None) -> ProjectionData:
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
    all_projection, overlaps = [], []
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
        all_projection.append(np.abs(amplitudes.T) ** 2)
        overlaps.append(overlap)
    assert orbitals is not None
    return ProjectionData(saved.eigenvalues_ev, saved.weights, np.asarray(all_projection), occupations, orbitals, tuple(overlaps), saved.fermi_ev)


def _dos_kernel(energies: np.ndarray, grid: np.ndarray, width: float, ngauss: int) -> np.ndarray:
    if ngauss not in {0, 1, -1, -99}:
        raise QEInputError("ngauss must be 0, 1, -1, or -99")
    if width <= 0:
        raise QEInputError("projected DOS requires positive degauss")
    x = (grid[:, None, None] - energies[None, :, :]) / width
    step = 1.0e-5
    return (wgauss(x + step, ngauss) - wgauss(x - step, ngauss)) / (2.0 * step * width)


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


def run_projwfc(options: dict[str, object]) -> tuple[ProjectionData, list[Path]]:
    unknown = set(options) - _ALLOWED
    if unknown:
        raise QEInputError(f"unknown &PROJWFC variable {sorted(unknown)[0]!r}")
    for key in ("diag_basis", "pawproj", "lbinary_data", "tdosinboxes"):
        if bool(options.get(key, False)):
            raise UnsupportedFeatureError(f"projwfc {key}=.true. is not ported")
    if not bool(options.get("lsym", True)) and not bool(options.get("kresolveddos", False)):
        raise QEInputError("lsym=.false. requires kresolveddos=.true.")
    prefix = str(options.get("prefix", "pwscf"))
    outdir = str(options["outdir"]) if "outdir" in options else None
    saved = read_saved_dos(prefix, outdir)
    if saved.occupations_kind.lower().startswith("tetra") and "degauss" not in options:
        raise UnsupportedFeatureError("projected tetrahedron DOS is not yet ported; specify degauss for smearing")
    data = compute_projections(prefix, outdir)
    delta = float(options.get("deltae", 0.01))
    if delta <= 0:
        raise QEInputError("DeltaE must be positive")
    degauss_ry = float(options.get("degauss", saved.degauss_ry))
    if degauss_ry <= 0:
        degauss_ry = delta / (0.5 * EV_PER_HARTREE)
    width = 0.5 * degauss_ry * EV_PER_HARTREE
    ngauss = int(options.get("ngauss", smearing_order(saved.smearing) if saved.degauss_ry > 0 else 0))
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="projwfc.py")
    parser.add_argument("-i", "-in", "--input", dest="input_file")
    args = parser.parse_args(argv)
    try:
        text = Path(args.input_file).read_text(encoding="utf-8") if args.input_file else sys.stdin.read()
        data, paths = run_projwfc(parse_namelist(text, "projwfc"))
        charges = np.einsum("k,kb,kbo->o", data.weights, data.occupations, data.projections)
        for index, (orbital, charge) in enumerate(zip(data.orbitals, charges), start=1):
            print(f"     state # {index:4d}: atom {orbital.atom:4d} ({orbital.symbol}), wfc {orbital.wfc} (l={orbital.l} m={orbital.m + 1}) charge = {charge:9.5f}")
        electrons = float(np.einsum("k,kb->", data.weights, data.occupations))
        print(f"     Spilling Parameter: {1.0 - float(np.sum(charges)) / max(1.0e-14, electrons):9.5f}")
        print("     PROJWFC files written: " + ", ".join(str(path) for path in paths))
        return 0
    except (QEInputError, UnsupportedFeatureError, OSError, ValueError, ET.ParseError) as exc:
        print(format_qe_error(exc), end="", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
