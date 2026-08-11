"""Scalar norm-conserving implementation of Quantum ESPRESSO ``pp.x``."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import TextIO
import xml.etree.ElementTree as ET

import numpy as np
import pyfftw
from scipy.special import spherical_jn

from ..cli_options import add_input_file_argument
from ..constants import EV_PER_HARTREE
from ..errors import QEInputError, UnsupportedFeatureError, emit_qe_error
from ..occupations import smearing_density, smearing_order, w1gauss
from ..qe_format import format_qe_closing, format_qe_opening, format_qe_timing
from ..symmetry import DensitySymmetrizer, SymmetryOperation
from ..spinor import (
    eigenchannel_densities,
    potential_components as noncollinear_potential_components,
)
from ..threads import hybrid_thread_count
from ..upf import read_upf
from ..version import __version__
from ..xc import (
    lsda_lda,
    pbe_spin_components,
    pbe_unpolarized_components,
    pw92_lda_unpolarized,
    pz81_unpolarized,
)
from .band_data import resolve_save_directory
from .namelist import parse_namelist
from .spin import normalize_spin_weights, spin_labels, validate_spin_blocks
from .xml_data import find, findall, findtext, upstream_qe_xml


# QE 7.5 ``Modules/constants.f90`` value used by XSF writers.  Keep this
# output conversion distinct from the input-side legacy conversion constant.
_ANGSTROM_PER_BOHR = 0.529177210903
_INPUT_KEYS = {
    "title", "prefix", "outdir", "filplot", "plot_num", "sample_bias",
    "z", "dz", "emin", "emax", "delta_e", "degauss_ldos",
    "use_gauss_ldos", "lsign", "kpoint", "kpoint(1)", "kpoint(2)",
    "kband", "kband(1)", "kband(2)", "spin_component",
    "spin_component(1)", "spin_component(2)", "spin_component(3)",
    "nc(1)", "nc(2)", "nc(3)", "n0(1)", "n0(2)", "n0(3)",
}
_PLOT_KEYS = {
    "nfile", "iflag", "output_format", "fileout", "interpolation",
    "nx", "ny", "nz", "radius", "isostm_flag", "isovalue",
    "heightmin", "heightmax", "direction",
    *(f"filepp({index})" for index in range(1, 101)),
    *(f"weight({index})" for index in range(1, 101)),
    *(f"{name}({index})" for name in ("e1", "e2", "e3", "x0") for index in range(1, 4)),
}
_XC_FUNCTIONALS = {
    "PZ": "pz",
    "PW": "pw",
    "PBE": "pbe",
    "PBESOL": "pbesol",
    "REVPBE": "revpbe",
    "RPBE": "rpbe",
    "SLA PZ NOGX NOGC": "pz",
    "SLA PW NOGX NOGC": "pw",
    "SLA PW PBX PBC": "pbe",
    "SLA PW PSX PSC": "pbesol",
    "SLA PW REVX PBC": "revpbe",
    "SLA PW HHNX PBC": "rpbe",
}
_PLOT_NAMES = {
    0: "1D spherical average", 1: "1D along a line", 2: "2D contour",
    3: "3D", 4: "2D polar on a sphere",
}
_FORMAT_NAMES = {
    0: "gnuplot", 2: "plotrho.x", 3: "XCrySDen",
    5: "XCrySDen", 6: "Gaussian cube", 7: "gnuplot x,y,f",
}


@dataclass(frozen=True)
class PPAtom:
    symbol: str
    position: np.ndarray  # Cartesian bohr
    species: int
    valence: float


@dataclass(frozen=True)
class SavedPPState:
    directory: Path
    lattice: np.ndarray  # lattice vectors as rows, Cartesian bohr
    alat: float
    shape: tuple[int, int, int]
    atoms: tuple[PPAtom, ...]
    species: tuple[tuple[str, float], ...]
    starting_magnetizations: tuple[float, ...]
    pseudos: dict[str, object]
    density: np.ndarray
    spin_densities: np.ndarray
    energies_ha: np.ndarray
    occupations: np.ndarray
    weights: np.ndarray
    fermi_ha: float | None
    smearing: str
    degauss_ha: float
    functional: str
    ecutwfc_ry: float
    ecutrho_ry: float
    symmetry_operations: tuple[SymmetryOperation, ...]
    gamma_only: bool
    spins: np.ndarray
    noncolin: bool = False
    spinorbit: bool = False
    magnetization_density: np.ndarray | None = None
    noncollinear_gga_axis: np.ndarray | None = None
    ibrav: int = 0
    celldm: tuple[float, float, float, float, float, float] = (0.0,) * 6

    @property
    def reciprocal(self) -> np.ndarray:
        return 2.0 * np.pi * np.linalg.inv(self.lattice).T

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.lattice)))

    @property
    def nspin(self) -> int:
        if self.noncolin:
            return 4
        return int(np.max(self.spins, initial=1))

    @property
    def magnetization(self) -> np.ndarray:
        if self.noncolin:
            if self.magnetization_density is None:
                raise QEInputError(
                    "saved noncollinear magnetization density is missing",
                    routine="postproc",
                )
            return np.linalg.norm(self.magnetization_density, axis=0)
        if self.nspin == 1:
            return np.zeros_like(self.density)
        return self.spin_densities[0] - self.spin_densities[1]


@dataclass(frozen=True)
class PlotGrid:
    title: str
    plot_num: int
    values: np.ndarray
    lattice: np.ndarray
    alat: float
    atoms: tuple[PPAtom, ...]
    species: tuple[tuple[str, float], ...]
    ecutwfc_ry: float = 0.0
    ecutrho_ry: float = 0.0
    ibrav: int = 0
    celldm: tuple[float, float, float, float, float, float] = (0.0,) * 6
    gcutm: float | None = None


def _cell_parameters_from_lattice(
    ibrav: int, lattice: np.ndarray, alat: float
) -> tuple[float, float, float, float, float, float]:
    """Recover QE ``celldm`` metadata from a saved Bravais lattice.

    QE's XML schema retains ``bravais_index`` and the Cartesian primitive
    vectors, but not the original six ``celldm`` values.  ``plot_io`` needs
    those values because a nonzero Bravais index suppresses the three
    explicit unit-cell-vector records in a formatted ``filplot`` file.
    """
    cell = np.asarray(lattice, dtype=float)
    if ibrav == 0:
        return (float(alat), 0.0, 0.0, 0.0, 0.0, 0.0)

    a_vector: np.ndarray
    b_vector: np.ndarray
    c_vector: np.ndarray
    cosine_index = 0
    if ibrav in {1, 2, 3, -3}:
        return (float(alat), 0.0, 0.0, 0.0, 0.0, 0.0)
    if ibrav == 4:
        a_vector, b_vector, c_vector = cell[0], cell[1], cell[2]
    elif ibrav in {5, -5}:
        cosine = float(np.dot(cell[0], cell[1]) / alat**2)
        return (float(alat), 0.0, 0.0, cosine, 0.0, 0.0)
    elif ibrav == 6:
        a_vector, b_vector, c_vector = cell[0], cell[1], cell[2]
    elif ibrav == 7:
        a_vector = cell[0] - cell[1]
        b_vector = cell[0] - cell[1]
        c_vector = cell[1] + cell[2]
    elif ibrav == 8:
        a_vector, b_vector, c_vector = cell[0], cell[1], cell[2]
    elif ibrav == 9:
        a_vector, b_vector, c_vector = cell[0] - cell[1], cell[0] + cell[1], cell[2]
    elif ibrav == -9:
        a_vector, b_vector, c_vector = cell[0] + cell[1], cell[1] - cell[0], cell[2]
    elif ibrav == 91:
        a_vector, b_vector, c_vector = cell[0], cell[1] + cell[2], cell[2] - cell[1]
    elif ibrav == 10:
        a_vector = cell[1] + cell[2] - cell[0]
        b_vector = cell[0] + cell[2] - cell[1]
        c_vector = cell[0] + cell[1] - cell[2]
    elif ibrav == 11:
        a_vector, b_vector, c_vector = cell[0] - cell[1], cell[1] - cell[2], cell[0] + cell[2]
    elif ibrav == 12:
        a_vector, b_vector, c_vector = cell[0], cell[1], cell[2]
        cosine_index = 4
    elif ibrav == -12:
        a_vector, b_vector, c_vector = cell[0], cell[1], cell[2]
        cosine_index = 5
    elif ibrav == 13:
        a_vector, b_vector, c_vector = cell[0] + cell[2], cell[1], cell[2] - cell[0]
        cosine_index = 4
    elif ibrav == -13:
        a_vector, b_vector, c_vector = cell[0] + cell[1], cell[1] - cell[0], cell[2]
        cosine_index = 5
    elif ibrav == 14:
        a_vector, b_vector, c_vector = cell[0], cell[1], cell[2]
        cosine_index = -1
    else:
        # Unknown future QE Bravais indices remain fully representable as
        # ibrav=0, without fabricating a parameterization.
        return (float(alat), 0.0, 0.0, 0.0, 0.0, 0.0)

    a_length = float(np.linalg.norm(a_vector))
    b_length = float(np.linalg.norm(b_vector))
    c_length = float(np.linalg.norm(c_vector))
    values = [float(alat), b_length / a_length, c_length / a_length, 0.0, 0.0, 0.0]
    if cosine_index in {4, -1}:
        values[3] = float(np.dot(a_vector, b_vector) / (a_length * b_length))
    if cosine_index in {5, -1}:
        values[4] = float(np.dot(a_vector, c_vector) / (a_length * c_length))
    if cosine_index == -1:
        values[5] = float(np.dot(b_vector, c_vector) / (b_length * c_length))
    return tuple(values)  # type: ignore[return-value]


def _bool_text(value: object) -> bool:
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="ignore")
    return str(value or "").strip().lower() in {"true", ".true.", "t", "1"}


def _xml_vector(root: ET.Element, path: str) -> np.ndarray:
    values = np.fromstring(findtext(root, path, "") or "", sep=" ")
    if values.shape != (3,):
        raise QEInputError(f"saved data contains no valid {path}", routine="postproc")
    return values


def _saved_smearing_metadata(
    band_structure: ET.Element | None,
) -> tuple[str, float]:
    """Read QE's smearing name and Hartree width from saved XML.

    Both upstream QE and qepy-pw serialize the width as the ``degauss``
    attribute of ``<smearing>``.  Retain a child-element fallback for older
    qepy-pw saves written before that schema was followed consistently.
    """
    if band_structure is None:
        return "fixed", 0.0
    smearing = find(band_structure, "smearing")
    if smearing is None:
        # Legacy qepy-pw wrote the input-side Ry value as a child element.
        return "fixed", 0.5 * float(findtext(band_structure, "degauss", "0") or 0.0)
    name = (smearing.text or "fixed").strip() or "fixed"
    if "degauss" in smearing.attrib:
        # QE XML energy values, including this attribute, are in Hartree.
        return name, float(smearing.attrib["degauss"])
    # Backward compatibility with old qepy-pw saves: the child was in Ry.
    return name, 0.5 * float(findtext(band_structure, "degauss", "0") or 0.0)


def _read_density(
    directory: Path, shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    path = directory / "charge-density.hdf5"
    try:
        with h5py.File(path, "r") as h5:
            miller = np.asarray(h5["MillerIndices"][:], dtype=np.int64)
            raw = np.asarray(h5["rhotot_g"][:])
            gamma = _bool_text(h5.attrs.get("gamma_only", False))
            nspin = int(h5.attrs.get("nspin", 1))
            raw_difference = (
                np.asarray(h5["rhodiff_g"][:]) if nspin == 2 else None
            )
            raw_magnetization = (
                tuple(np.asarray(h5[name][:]) for name in ("m_x", "m_y", "m_z"))
                if nspin == 4
                else ()
            )
    except (OSError, KeyError, ValueError) as exc:
        raise QEInputError(f"cannot read charge density {path}: {exc}", routine="postproc") from exc

    def coefficients(values: np.ndarray) -> np.ndarray:
        if np.issubdtype(values.dtype, np.floating):
            if values.ndim != 1 or len(values) != 2 * len(miller):
                raise QEInputError("saved charge density has an unsupported layout", routine="postproc")
            return values[0::2] + 1j * values[1::2]
        result = np.asarray(values, dtype=np.complex128).reshape(-1)
        if len(result) != len(miller):
            raise QEInputError("saved charge density has an unsupported layout", routine="postproc")
        return result

    total_coefficients = coefficients(raw)
    if miller.shape != (len(total_coefficients), 3) or nspin not in {1, 2, 4}:
        raise QEInputError("saved charge density has an unsupported layout", routine="postproc")

    def real_field(values: np.ndarray) -> np.ndarray:
        grid = np.zeros(shape, dtype=np.complex128)
        slots = tuple((miller % np.asarray(shape)).T)
        grid[slots] = values
        if gamma:
            negative = tuple(((-miller) % np.asarray(shape)).T)
            empty = np.abs(grid[negative]) == 0.0
            grid[tuple(axis[empty] for axis in negative)] = np.conjugate(values[empty])
        return np.real(np.fft.ifftn(grid * np.prod(shape)))

    total = real_field(total_coefficients)
    if nspin == 1:
        return total, total[None, ...], np.zeros((3,) + shape)
    if nspin == 4:
        if len(raw_magnetization) != 3:
            raise QEInputError(
                "saved noncollinear charge density has no magnetization components",
                routine="postproc",
            )
        magnetization = np.asarray(
            [real_field(coefficients(component)) for component in raw_magnetization]
        )
        return total, total[None, ...], magnetization
    assert raw_difference is not None
    difference = real_field(coefficients(raw_difference))
    return (
        total,
        np.asarray((0.5 * (total + difference), 0.5 * (total - difference))),
        np.zeros((3,) + shape),
    )


def read_saved_pp(prefix: str = "pwscf", outdir: str | None = None) -> SavedPPState:
    """Read scalar NC metadata and density from a QE-compatible save directory."""
    directory = resolve_save_directory(prefix, outdir)
    path = directory / "data-file-schema.xml"
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise QEInputError(f"cannot read saved XML {path}: {exc}", routine="postproc") from exc

    if _bool_text(findtext(root, "output/algorithmic_info/paw")):
        raise UnsupportedFeatureError("PAW pp.x post-processing", routine="postproc")
    if _bool_text(findtext(root, "output/algorithmic_info/uspp")):
        raise UnsupportedFeatureError("ultrasoft pp.x post-processing", routine="postproc")
    lsda = _bool_text(findtext(root, "output/band_structure/lsda"))
    noncolin = _bool_text(findtext(root, "output/band_structure/noncolin"))
    spinorbit = _bool_text(findtext(root, "output/band_structure/spinorbit"))
    if lsda and noncolin:
        raise QEInputError(
            "saved data cannot be both LSDA and noncollinear", routine="postproc"
        )

    structure = find(root, "output/atomic_structure")
    if structure is None:
        raise QEInputError("saved data contains no atomic structure", routine="postproc")
    alat = float(structure.attrib.get("alat", "0"))
    ibrav = int(structure.attrib.get("bravais_index", "0"))
    lattice = np.vstack([_xml_vector(root, f"output/atomic_structure/cell/a{i}") for i in range(1, 4)])
    if alat <= 0.0:
        alat = float(np.linalg.norm(lattice[0]))
    celldm = _cell_parameters_from_lattice(ibrav, lattice, alat)
    fft = find(root, "output/basis_set/fft_grid")
    if fft is None:
        fft = find(root, "input/basis/fft_grid")
    if fft is None:
        raise QEInputError("saved data contains no FFT grid", routine="postproc")
    shape = tuple(int(fft.attrib[f"nr{i}"]) for i in range(1, 4))

    species_entries = findall(root, "output/atomic_species/species")
    pseudo_names = {
        entry.attrib.get("name", ""): findtext(entry, "pseudo_file", "") or ""
        for entry in species_entries
    }
    saved_pseudo_dir_text = (
        findtext(root, "input/control_variables/pseudo_dir", "") or ""
    ).strip()
    saved_pseudo_dir = Path(saved_pseudo_dir_text).expanduser()
    pseudos = {}
    species = []
    starting_magnetizations = []
    starting_vectors: dict[str, np.ndarray] = {}
    for symbol, pseudo_name in pseudo_names.items():
        if not pseudo_name:
            raise QEInputError(f"saved species {symbol} has no pseudopotential", routine="postproc")
        pseudo_path = directory / Path(pseudo_name).name
        if not pseudo_path.is_file() and saved_pseudo_dir_text:
            pseudo_path = saved_pseudo_dir / Path(pseudo_name).name
        pseudo = read_upf(pseudo_path)
        if pseudo.pseudo_type.upper() != "NC":
            raise UnsupportedFeatureError(
                f"{pseudo.pseudo_type} pp.x post-processing", routine="postproc"
            )
        pseudos[symbol] = pseudo
        species.append((symbol, float(pseudo.z_valence)))
        species_entry = next(
            entry
            for entry in species_entries
            if entry.attrib.get("name", "") == symbol
        )
        starting_magnetization = float(
            findtext(species_entry, "starting_magnetization", "0") or 0.0
        )
        starting_magnetizations.append(starting_magnetization)
        theta = float(findtext(species_entry, "spin_teta", "0") or 0.0)
        phi = float(findtext(species_entry, "spin_phi", "0") or 0.0)
        starting_vectors[symbol] = starting_magnetization * np.asarray(
            (
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            )
        )
    species_index = {symbol: index + 1 for index, (symbol, _z) in enumerate(species)}
    atoms = []
    for entry in findall(root, "output/atomic_structure/atomic_positions/atom"):
        symbol = entry.attrib.get("name", "")
        if symbol not in pseudos:
            raise QEInputError(f"saved atom has unknown species {symbol!r}", routine="postproc")
        position = np.fromstring(entry.text or "", sep=" ")
        atoms.append(PPAtom(symbol, position, species_index[symbol], float(pseudos[symbol].z_valence)))
    gga_axis = None
    atom_vectors = [starting_vectors[atom.symbol] for atom in atoms]
    for atom_index, vector in enumerate(atom_vectors):
        if float(np.dot(vector, vector)) <= 1.0e-12:
            continue
        if all(
            float(np.dot(np.cross(vector, other), np.cross(vector, other)))
            < 1.0e-6
            for other in atom_vectors[atom_index + 1 :]
        ):
            gga_axis = vector / np.linalg.norm(vector)
        break

    bands = find(root, "output/band_structure")
    smearing, degauss_ha = _saved_smearing_metadata(bands)
    records = [] if bands is None else findall(bands, "ks_energies")
    energies, occupations, weights, kpoints = [], [], [], []
    for record in records:
        energies.append(np.fromstring(findtext(record, "eigenvalues", "") or "", sep=" "))
        occupations.append(np.fromstring(findtext(record, "occupations", "") or "", sep=" "))
        point = find(record, "k_point")
        weights.append(float(point.attrib.get("weight", "nan")) if point is not None else np.nan)
        kpoints.append(
            np.fromstring(point.text or "", sep=" ")
            if point is not None
            else np.empty(0)
        )
    if not energies or len({len(row) for row in energies}) != 1:
        raise QEInputError("saved band energies are missing or inconsistent", routine="postproc")
    energy_array = np.vstack(energies)
    occupation_array = np.vstack(occupations)
    if occupation_array.shape != energy_array.shape:
        raise QEInputError("saved occupations are missing or inconsistent", routine="postproc")
    # Upstream QE's scalar XML stores per-spin occupations in [0, 1], while
    # qepy-pw stores the already spin-degenerate [0, 2] occupations used by
    # its SCF density builder.  Normalize both representations to [0, 2].
    if upstream_qe_xml(root) and not lsda and not noncolin:
        occupation_array *= 2.0
    spins = spin_labels(lsda, len(records))
    validate_spin_blocks(np.vstack(kpoints), spins)
    try:
        weight_array = normalize_spin_weights(np.asarray(weights, dtype=float), spins)
    except QEInputError as exc:
        raise QEInputError(str(exc), routine="postproc") from exc
    fermi_text = None if bands is None else findtext(bands, "fermi_energy")
    functional_text = findtext(root, "output/dft/functional", "") or ""
    functional = _XC_FUNCTIONALS.get(" ".join(functional_text.upper().split()))
    if functional is None:
        raise UnsupportedFeatureError(
            f"exchange-correlation functional {functional_text!r} in pp.x", routine="postproc"
        )
    ecutwfc_ha = float(findtext(root, "output/basis_set/ecutwfc", "0") or 0.0)
    ecutrho_ha = float(findtext(root, "output/basis_set/ecutrho", "0") or 0.0)
    operations = []
    symmetry_entries = findall(root, "output/band_structure/symmetry_operations/symmetry")
    if not symmetry_entries:
        symmetry_entries = findall(root, "output/symmetries/symmetry")
        symmetry_count = int(
            findtext(root, "output/symmetries/nsym", str(len(symmetry_entries)))
            or len(symmetry_entries)
        )
        symmetry_entries = symmetry_entries[:symmetry_count]
    for entry in symmetry_entries:
        rotation_values = np.fromstring(findtext(entry, "rotation", "") or "", sep=" ")
        translation = np.fromstring(findtext(entry, "fractional_translation", "") or "", sep=" ")
        if rotation_values.size != 9 or translation.shape != (3,):
            raise QEInputError("saved symmetry operation is malformed", routine="postproc")
        rotation = np.rint(rotation_values).astype(int).reshape(3, 3)
        if not np.allclose(rotation_values.reshape(3, 3), rotation, rtol=0.0, atol=1.0e-10):
            raise QEInputError("saved symmetry rotation is not integral", routine="postproc")
        if upstream_qe_xml(root):
            rotation = rotation.T
            translation = -translation
        operations.append(SymmetryOperation(
            rotation,
            translation,
            _bool_text(findtext(entry, "time_reversal", "false")),
        ))
    if not operations:
        operations.append(SymmetryOperation(np.eye(3, dtype=int), np.zeros(3)))
    density, spin_densities, magnetization_density = _read_density(directory, shape)
    if (len(spin_densities) == 2) != lsda or (
        np.any(magnetization_density) and not noncolin
    ):
        raise QEInputError(
            "XML spin metadata disagrees with the saved charge density",
            routine="postproc",
        )
    return SavedPPState(
        directory=directory,
        lattice=lattice,
        alat=alat,
        shape=shape,
        atoms=tuple(atoms),
        species=tuple(species),
        starting_magnetizations=tuple(starting_magnetizations),
        pseudos=pseudos,
        density=density,
        spin_densities=spin_densities,
        magnetization_density=magnetization_density,
        energies_ha=energy_array,
        occupations=occupation_array,
        weights=weight_array,
        fermi_ha=None if fermi_text is None else float(fermi_text),
        smearing=smearing,
        degauss_ha=degauss_ha,
        functional=functional,
        ecutwfc_ry=2.0 * ecutwfc_ha,
        ecutrho_ry=2.0 * ecutrho_ha,
        symmetry_operations=tuple(operations),
        gamma_only=_bool_text(findtext(root, "output/basis_set/gamma_only", "false")),
        spins=spins,
        noncolin=noncolin,
        spinorbit=spinorbit,
        noncollinear_gga_axis=gga_axis,
        ibrav=ibrav,
        celldm=celldm,
    )


def _miller_and_g(shape: tuple[int, int, int], reciprocal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = [np.rint(np.fft.fftfreq(size) * size).astype(np.int64) for size in shape]
    miller = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return miller, miller @ reciprocal


def _inverse_coefficients(coefficients: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifftn(coefficients * coefficients.size))


def _species_fourier_field(
    state: SavedPPState,
    method: str,
    species_scales: tuple[float, ...] | None = None,
) -> np.ndarray:
    _miller, g = _miller_and_g(state.shape, state.reciprocal)
    flat_g = g.reshape(-1, 3)
    q = np.linalg.norm(flat_g, axis=1)
    coefficients = np.zeros(len(flat_g), dtype=np.complex128)
    scales = (
        (1.0,) * len(state.species)
        if species_scales is None
        else species_scales
    )
    if len(scales) != len(state.species):
        raise QEInputError(
            "saved species magnetizations are inconsistent", routine="postproc"
        )
    for (symbol, _z), scale in zip(state.species, scales):
        pseudo = state.pseudos[symbol]
        positions = np.vstack([atom.position for atom in state.atoms if atom.symbol == symbol])
        structure = np.sum(np.exp(-1j * (flat_g @ positions.T)), axis=1)
        radial = getattr(pseudo, method)(q, state.volume)
        radial[q * q > state.ecutrho_ry * (1.0 + 1.0e-12)] = 0.0
        coefficients += float(scale) * radial * structure
    return _inverse_coefficients(coefficients.reshape(state.shape))


def _ionic_potential(state: SavedPPState) -> np.ndarray:
    return _species_fourier_field(state, "fourier")


def _atomic_density(state: SavedPPState) -> np.ndarray:
    return _species_fourier_field(state, "atomic_density_fourier")


def _atomic_magnetization(state: SavedPPState) -> np.ndarray:
    return _species_fourier_field(
        state,
        "atomic_density_fourier",
        state.starting_magnetizations,
    )


def _core_density(state: SavedPPState) -> np.ndarray:
    return _species_fourier_field(state, "core_density_fourier")


def _hartree_potential(state: SavedPPState) -> np.ndarray:
    coefficients = np.fft.fftn(state.density) / state.density.size
    _miller, g = _miller_and_g(state.shape, state.reciprocal)
    g2 = np.einsum("...i,...i->...", g, g)
    potential = np.zeros_like(coefficients)
    mask = g2 > 1.0e-14
    potential[mask] = 4.0 * np.pi * coefficients[mask] / g2[mask]
    return _inverse_coefficients(potential)


def _spectral_gradient_hessian(values: np.ndarray, lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.fft.fftn(values) / values.size
    _miller, g = _miller_and_g(values.shape, 2.0 * np.pi * np.linalg.inv(lattice).T)
    gradient = np.empty((3,) + values.shape, dtype=float)
    hessian = np.empty((3, 3) + values.shape, dtype=float)
    for first in range(3):
        gradient[first] = _inverse_coefficients(1j * g[..., first] * coefficients)
        for second in range(first, 3):
            component = _inverse_coefficients(-g[..., first] * g[..., second] * coefficients)
            hessian[first, second] = component
            hessian[second, first] = component
    return gradient, hessian


def _spectral_gradient(values: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    coefficients = np.fft.fftn(values) / values.size
    _miller, g = _miller_and_g(values.shape, 2.0 * np.pi * np.linalg.inv(lattice).T)
    return np.asarray(
        [
            _inverse_coefficients(1j * g[..., axis] * coefficients)
            for axis in range(3)
        ]
    )


def _spectral_hessian(values: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    coefficients = np.fft.fftn(values) / values.size
    _miller, g = _miller_and_g(values.shape, 2.0 * np.pi * np.linalg.inv(lattice).T)
    hessian = np.empty((3, 3) + values.shape, dtype=float)
    for first in range(3):
        for second in range(first, 3):
            component = _inverse_coefficients(
                -g[..., first] * g[..., second] * coefficients
            )
            hessian[first, second] = component
            hessian[second, first] = component
    return hessian


def _xc_potentials(state: SavedPPState) -> np.ndarray:
    core = _core_density(state)
    direction = None
    if state.noncolin:
        if state.magnetization_density is None:
            raise QEInputError(
                "saved noncollinear magnetization density is missing",
                routine="postproc",
            )
        pauli_density = np.concatenate(
            ((state.density + core)[None, ...], state.magnetization_density),
            axis=0,
        )
        densities, direction = eigenchannel_densities(
            pauli_density, state.noncollinear_gga_axis
        )
        if state.functional in {"pz", "pw"}:
            _epsilon, channel_potentials = lsda_lda(
                densities, state.functional
            )
            return noncollinear_potential_components(
                channel_potentials, direction
            )
        gradients = np.asarray([
            _spectral_gradient(density, state.lattice) for density in densities
        ])
        _epsilon, local_potentials, exchange, correlation = pbe_spin_components(
            densities, gradients, functional=state.functional
        )
        total_gradient = gradients[0] + gradients[1]
        fluxes = np.asarray([
            np.moveaxis(
                exchange[spin][None, ...] * gradients[spin]
                + correlation[None, ...] * total_gradient,
                0,
                -1,
            )
            for spin in range(2)
        ])
    elif state.nspin == 1:
        density = state.density + core
        if state.functional == "pz":
            return pz81_unpolarized(density)[1][None, ...]
        if state.functional == "pw":
            return pw92_lda_unpolarized(density)[1][None, ...]
        gradient = _spectral_gradient(density, state.lattice)
        _epsilon, local, coefficient = pbe_unpolarized_components(
            density, gradient, functional=state.functional
        )
        fluxes = (np.moveaxis(gradient, 0, -1) * coefficient[..., None])[None, ...]
        local_potentials = local[None, ...]
    else:
        densities = state.spin_densities + 0.5 * core[None, ...]
        if state.functional in {"pz", "pw"}:
            _epsilon, potentials = lsda_lda(densities, state.functional)
            return potentials
        gradients = np.asarray([
            _spectral_gradient(density, state.lattice) for density in densities
        ])
        _epsilon, local_potentials, exchange, correlation = pbe_spin_components(
            densities, gradients, functional=state.functional
        )
        total_gradient = gradients[0] + gradients[1]
        fluxes = np.asarray([
            np.moveaxis(
                exchange[spin][None, ...] * gradients[spin]
                + correlation[None, ...] * total_gradient,
                0,
                -1,
            )
            for spin in range(2)
        ])

    _miller, g = _miller_and_g(state.shape, state.reciprocal)
    potentials = np.empty_like(local_potentials)
    for spin in range(len(local_potentials)):
        divergence = np.zeros(state.shape, dtype=float)
        for axis in range(3):
            component = np.fft.fftn(fluxes[spin, ..., axis]) / state.density.size
            divergence += _inverse_coefficients(1j * g[..., axis] * component)
        potentials[spin] = local_potentials[spin] - divergence
    if direction is not None:
        return noncollinear_potential_components(potentials, direction)
    return potentials


def _xc_potential(state: SavedPPState, spin_component: int = 0) -> np.ndarray:
    potentials = _xc_potentials(state)
    if state.nspin in {1, 4}:
        return potentials[0]
    if spin_component == 0:
        return 0.5 * (potentials[0] + potentials[1])
    return potentials[spin_component - 1]


def _read_wavefunction(state: SavedPPState, kpoint: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    path = state.directory / f"wfc{kpoint + 1}.hdf5"
    try:
        with h5py.File(path, "r") as h5:
            miller = np.asarray(h5["MillerIndices"][:], dtype=np.int64)
            raw = np.asarray(h5["evc"][:])
            xk = np.asarray(h5.attrs.get("xk", np.zeros(3)), dtype=float)
            npol = int(h5.attrs.get("npol", 2 if state.noncolin else 1))
    except (OSError, KeyError, ValueError) as exc:
        raise QEInputError(f"cannot read wavefunctions {path}: {exc}", routine="punch_plot") from exc
    if np.issubdtype(raw.dtype, np.floating):
        if raw.ndim != 2 or raw.shape[1] != 2 * npol * len(miller):
            raise QEInputError(f"invalid interleaved wavefunctions in {path}", routine="punch_plot")
        coefficients = (raw[:, 0::2] + 1j * raw[:, 1::2]).T
    else:
        coefficients = np.asarray(raw, dtype=np.complex128).T
    if npol not in {1, 2} or coefficients.shape[0] != npol * len(miller):
        raise QEInputError(f"invalid spinor layout in {path}", routine="punch_plot")
    if state.noncolin != (npol == 2):
        raise QEInputError(
            f"wavefunction spinor metadata disagrees with {path}",
            routine="punch_plot",
        )
    return miller, coefficients, xk


def _wave_grid(miller: np.ndarray, coefficients: np.ndarray, shape: tuple[int, int, int], volume: float) -> np.ndarray:
    rows = len(miller)
    if len(coefficients) not in {rows, 2 * rows}:
        raise QEInputError(
            "wavefunction coefficients do not match Miller indices",
            routine="punch_plot",
        )
    components = len(coefficients) // rows
    grids = np.zeros((components,) + shape, dtype=np.complex128)
    slots = tuple((miller % np.asarray(shape)).T)
    for component in range(components):
        grids[(component,) + slots] = coefficients[
            component * rows : (component + 1) * rows
        ]
    waves = np.fft.ifftn(grids, axes=(-3, -2, -1))
    waves *= int(np.prod(shape)) / np.sqrt(volume)
    return waves[0] if components == 1 else waves


def _spinor_grid_quantity(wave: np.ndarray, spin_component: int) -> np.ndarray:
    """Return QE's charge or Pauli component for one real-space spinor."""
    if wave.ndim == 3:
        if spin_component:
            raise QEInputError(
                "spin_component requires a noncollinear calculation",
                routine="punch_plot",
            )
        return np.abs(wave) ** 2
    up, down = wave
    if spin_component == 0:
        return np.abs(up) ** 2 + np.abs(down) ** 2
    coherence = np.conjugate(up) * down
    if spin_component == 1:
        return 2.0 * np.real(coherence)
    if spin_component == 2:
        return 2.0 * np.imag(coherence)
    if spin_component == 3:
        return np.abs(up) ** 2 - np.abs(down) ** 2
    raise QEInputError("wrong spin_component", routine="postproc")


def _wave_density_sum(
    state: SavedPPState,
    factors: np.ndarray,
    *,
    kinetic: bool = False,
) -> np.ndarray:
    if kinetic:
        return _kinetic_density_sum(state, factors)
    result = np.zeros(state.shape, dtype=float)
    for ik in range(len(state.weights)):
        if not np.any(factors[ik]):
            continue
        miller, coefficients, xk = _read_wavefunction(state, ik)
        for band in np.flatnonzero(factors[ik]):
            wave = _wave_grid(miller, coefficients[:, band], state.shape, state.volume)
            result += factors[ik, band] * _spinor_grid_quantity(wave, 0)
    return result


def _kinetic_density_sum(state: SavedPPState, factors: np.ndarray) -> np.ndarray:
    """Accumulate QE's kinetic-energy density with batched reusable FFTs."""
    workers = hybrid_thread_count()
    shape_array = np.asarray(state.shape)
    point_count = int(np.prod(state.shape))
    spin_components = 2 if state.noncolin else 1
    wave_scale_squared = point_count**2 / state.volume
    bytes_per_band = spin_components * 3 * point_count * (16 + 16 + 8)
    # Bound all concurrent complex FFT buffers and modulus workspaces to
    # approximately 256 MiB, independently of the requested thread count.
    per_worker_budget = (256 * 1024**2) // workers
    bands_per_chunk = max(1, per_worker_budget // bytes_per_band)
    thread_data = threading.local()

    def one_kpoint(ik: int) -> np.ndarray:
        contribution = np.zeros(state.shape, dtype=float)
        bands = np.flatnonzero(factors[ik])
        if not len(bands):
            return contribution
        miller, coefficients, xk = _read_wavefunction(state, ik)
        gk = miller @ state.reciprocal + xk
        coefficient_components = coefficients.reshape(
            spin_components, len(miller), coefficients.shape[1]
        )
        indices = tuple((miller % shape_array).T)
        plans = getattr(thread_data, "plans", None)
        if plans is None:
            plans = {}
            thread_data.plans = plans
        for begin in range(0, len(bands), bands_per_chunk):
            selected = bands[begin : begin + bands_per_chunk]
            fft_batch = spin_components * len(selected)
            plan_data = plans.get(fft_batch)
            if plan_data is None:
                reciprocal_grids = pyfftw.empty_aligned(
                    (fft_batch, 3) + state.shape, dtype="complex128"
                )
                derivatives = pyfftw.empty_aligned(
                    reciprocal_grids.shape, dtype="complex128"
                )
                fft_plan = pyfftw.FFTW(
                    reciprocal_grids,
                    derivatives,
                    axes=(-3, -2, -1),
                    direction="FFTW_BACKWARD",
                    flags=("FFTW_MEASURE",),
                    threads=1,
                    normalise_idft=True,
                )
                plan_data = reciprocal_grids, derivatives, fft_plan
                plans[fft_batch] = plan_data
            reciprocal_grids, derivatives, fft_plan = plan_data
            reciprocal_grids.fill(0.0)
            selected_coefficients = coefficient_components[:, :, selected]
            selected_coefficients = selected_coefficients.transpose(0, 2, 1)
            selected_coefficients = selected_coefficients.reshape(
                fft_batch, len(miller)
            )
            for axis in range(3):
                reciprocal_grids[(slice(None), axis) + indices] = (
                    1j * gk[:, axis][None, :] * selected_coefficients
                )
            fft_plan()
            contribution += np.einsum(
                "b,baxyz->xyz",
                np.tile(factors[ik, selected], spin_components),
                np.abs(derivatives) ** 2 * wave_scale_squared,
                optimize=True,
            )
        return contribution

    result = np.zeros(state.shape, dtype=float)
    if workers == 1:
        for ik in range(len(state.weights)):
            result += one_kpoint(ik)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # executor.map yields in k-point order, preserving the serial
            # accumulation order and its final floating-point rounding.
            for contribution in executor.map(one_kpoint, range(len(state.weights))):
                result += contribution
    return result


def _qe_local_dos_weights(weights: np.ndarray) -> np.ndarray:
    """Apply QE ``local_dos``'s positive relative-weight cutoff.

    Methfessel-Paxton and Marzari-Vanderbilt (cold) delta approximants can
    have negative side lobes.  QE's ``local_dos.f90`` accumulates a state only when
    ``wg > epsilon(DP) * maxval(wg)``; consequently those negative lobes are
    deliberately omitted from plot_num 3 (and the same local_dos path used
    by plot_num 4).
    """
    values = np.asarray(weights, dtype=float)
    maximum = float(np.max(values)) if values.size else 0.0
    cutoff = np.finfo(float).eps * maximum
    return np.where(values > cutoff, values, 0.0)


def _stm_window_weights(
    energies_ha: np.ndarray,
    lower_ha: float,
    upper_ha: float,
    width_ha: float,
    ngauss: int,
) -> np.ndarray:
    """Reproduce the bias-window weights used by QE's ``stm.f90``."""
    energies = np.asarray(energies_ha, dtype=float)
    factors = np.empty_like(energies)
    inside = (energies > lower_ha) & (energies < upper_ha)
    below = energies <= lower_ha
    above = ~(inside | below)
    factors[inside] = 1.0
    # QE deliberately calls w0gauss, the dimensionless smearing kernel,
    # rather than the integrated occupation function wgauss at both edges.
    factors[below] = smearing_density(
        (lower_ha - energies[below]) / width_ha, ngauss
    )
    factors[above] = smearing_density(
        (upper_ha - energies[above]) / width_ha, ngauss
    )
    return factors


def _symmetrize_wave_field(state: SavedPPState, values: np.ndarray) -> np.ndarray:
    if state.gamma_only or len(state.symmetry_operations) <= 1:
        return values
    try:
        return DensitySymmetrizer(state.shape, state.symmetry_operations).apply(values)
    except ValueError as exc:
        raise QEInputError(f"cannot symmetrize post-processing field: {exc}", routine="sym_rho") from exc


def _range_option(options: dict[str, object], name: str, maximum: int) -> tuple[int, int]:
    first = int(options.get(f"{name}(1)", options.get(name, 0)))
    second = int(options.get(f"{name}(2)", first))
    if second == 0:
        second = first
    if first < 1 or second < first or second > maximum:
        raise QEInputError(f"wrong {name} specified", routine="local_dos")
    return first, second


def _selected_density(state: SavedPPState, spin_component: int) -> np.ndarray:
    if spin_component not in {0, 1, 2}:
        raise QEInputError("wrong spin_component", routine="postproc")
    if spin_component == 0:
        return state.density
    if state.nspin != 2:
        raise QEInputError(
            "spin_component requires an LSDA calculation", routine="postproc"
        )
    return state.spin_densities[spin_component - 1]


def _selected_spin_rows(state: SavedPPState, spin_component: int) -> np.ndarray:
    if spin_component == 0:
        return np.ones(len(state.weights), dtype=bool)
    if state.nspin != 2:
        raise QEInputError(
            "spin_component requires an LSDA calculation", routine="postproc"
        )
    return state.spins == spin_component


def _grid_from_state(state: SavedPPState, title: str, plot_num: int, values: np.ndarray) -> PlotGrid:
    return PlotGrid(
        title=title[:75], plot_num=plot_num, values=np.asarray(values, dtype=float),
        lattice=state.lattice, alat=state.alat, atoms=state.atoms,
        species=state.species, ecutwfc_ry=state.ecutwfc_ry,
        ecutrho_ry=state.ecutrho_ry, ibrav=state.ibrav, celldm=state.celldm,
    )


def extract_plot_grids(state: SavedPPState, options: dict[str, object]) -> list[tuple[Path, PlotGrid]]:
    """Compute the ``plot_num`` quantity and its QE intermediate filename(s)."""
    unknown = set(options) - _INPUT_KEYS
    if unknown:
        raise QEInputError(f"unknown &INPUTPP variable {sorted(unknown)[0]!r}", routine="postproc")
    plot_num = int(options.get("plot_num", -1))
    if plot_num == -1:
        return []
    # QE declares INPUTPP%spin_component as a length-three Fortran array.
    # Consequently an ordinary ``spin_component = 1`` namelist assignment is
    # parsed as its first element, just like ``spin_component(1) = 1``.
    # All collinear PP paths consume that first element as a scalar selector.
    spin_component = int(
        options.get("spin_component", options.get("spin_component(1)", 0))
    )
    allowed_spin_components = (
        {0, 1, 2, 3} if plot_num in {7, 13, 18} else {0, 1, 2}
    )
    if spin_component not in allowed_spin_components:
        raise QEInputError("wrong spin_component", routine="postproc")
    if any(int(options.get(f"nc({index})", 1)) != 1 for index in range(1, 4)):
        raise QEInputError("nc can be used only for plot_num=25", routine="postproc")
    if any(int(options.get(f"n0({index})", 0)) != 0 for index in range(1, 4)):
        raise QEInputError("n0 can be used only for plot_num=25", routine="postproc")
    unsupported = {
        12: "sawtooth electric-field potential",
        14: "polarization", 15: "polarization",
        16: "polarization", 17: "PAW all-electron valence density",
        21: "PAW all-electron density",
        24: "ultrasoft all-electron reconstruction", 25: "DFT+U Hubbard projectors",
    }
    if plot_num in unsupported:
        raise UnsupportedFeatureError(f"pp.x plot_num={plot_num} ({unsupported[plot_num]})", routine="punch_plot")
    if plot_num not in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 18, 19, 20, 22, 23, 123}:
        raise QEInputError("Wrong plot_num", routine="postproc")
    filplot = Path(str(options.get("filplot", "tmp.pp")))
    title = str(options.get("title", " "))

    if plot_num == 0:
        values = _selected_density(state, spin_component)
        return [(filplot, _grid_from_state(state, title, plot_num, values))]
    if plot_num == 1:
        _selected_density(state, spin_component)  # validate LSDA-only selection
        values = 2.0 * (
            _ionic_potential(state)
            + _hartree_potential(state)
            + _xc_potential(state, spin_component)
        )
        return [(filplot, _grid_from_state(state, title, plot_num, values))]
    if plot_num == 2:
        return [(filplot, _grid_from_state(state, title, plot_num, 2.0 * _ionic_potential(state)))]
    if plot_num == 9:
        density = _selected_density(state, spin_component)
        atomic = _atomic_density(state)
        if spin_component:
            atomic_magnetization = _atomic_magnetization(state)
            sign = 1.0 if spin_component == 1 else -1.0
            atomic = 0.5 * (atomic + sign * atomic_magnetization)
        return [(filplot, _grid_from_state(state, title, plot_num, density - atomic))]
    if plot_num == 6:
        if state.nspin != 2:
            raise QEInputError("plot_num=6 requires an LSDA calculation", routine="postproc")
        return [(
            filplot,
            _grid_from_state(state, title, plot_num, state.magnetization),
        )]
    if plot_num == 13:
        if not state.noncolin or state.magnetization_density is None:
            raise QEInputError(
                "plot_num=13 requires a noncollinear calculation",
                routine="postproc",
            )
        values = (
            np.linalg.norm(state.magnetization_density, axis=0)
            if spin_component == 0
            else state.magnetization_density[spin_component - 1]
        )
        return [(filplot, _grid_from_state(state, title, plot_num, values))]
    if plot_num == 18:
        if not state.noncolin:
            raise QEInputError(
                "plot_num=18 requires a noncollinear calculation",
                routine="postproc",
            )
        magnetic_potential = _xc_potentials(state)[1:]
        values = (
            np.linalg.norm(magnetic_potential, axis=0)
            if spin_component == 0
            else magnetic_potential[spin_component - 1]
        )
        # Internal XC potentials are Hartree; pp.x writes Rydberg fields.
        return [(filplot, _grid_from_state(state, title, plot_num, 2.0 * values))]
    if plot_num == 11:
        values = 2.0 * (_ionic_potential(state) + _hartree_potential(state))
        return [(filplot, _grid_from_state(state, title, plot_num, values))]

    if plot_num == 7:
        first_k, last_k = _range_option(options, "kpoint", len(state.weights))
        first_b, last_b = _range_option(options, "kband", state.energies_ha.shape[1])
        multiple = (last_k - first_k + 1) * (last_b - first_b + 1) > 1
        outputs = []
        for kpoint in range(first_k, last_k + 1):
            miller, coefficients, _xk = _read_wavefunction(state, kpoint - 1)
            for band in range(first_b, last_b + 1):
                wave = _wave_grid(miller, coefficients[:, band - 1], state.shape, state.volume)
                values = _spinor_grid_quantity(wave, spin_component)
                if bool(options.get("lsign", False)):
                    if wave.ndim != 3:
                        raise QEInputError(
                            "lsign is not defined for spinor orbitals",
                            routine="local_dos",
                        )
                    if np.linalg.norm(_xk) > 1.0e-9:
                        raise QEInputError("k must be zero", routine="local_dos")
                    largest = np.unravel_index(np.argmax(np.abs(wave)), wave.shape)
                    phase = wave[largest] / max(1.0e-30, abs(wave[largest]))
                    values *= np.sign(np.real(wave * np.conjugate(phase)))
                path = Path(f"{filplot}_K{kpoint:03d}_B{band:03d}") if multiple else filplot
                wave_title = f"k_point {kpoint:4d}, band {band:4d}"
                outputs.append((path, _grid_from_state(state, wave_title, plot_num, values)))
        return outputs

    if plot_num in {3, 4, 5, 10, 23}:
        energies_ev = state.energies_ha * EV_PER_HARTREE
        degeneracy = 2.0 if state.nspin == 1 else 1.0
        if plot_num == 3:
            if state.degauss_ha <= 0.0:
                raise QEInputError("gaussian broadening needed", routine="local_dos")
            fermi_ev = 0.0 if state.fermi_ha is None else state.fermi_ha * EV_PER_HARTREE
            emin = float(options.get("emin", fermi_ev))
            emax = float(options.get("emax", fermi_ev))
            delta = float(options.get("delta_e", 0.1))
            if emax < emin or delta <= 0.0:
                raise QEInputError("invalid LDOS energy grid", routine="postproc")
            width = float(options.get("degauss_ldos", state.degauss_ha * EV_PER_HARTREE))
            if width <= 0.0:
                raise QEInputError("LDOS requires positive broadening", routine="local_dos")
            ngauss = 0 if bool(options.get("use_gauss_ldos", False)) else smearing_order(state.smearing)
            energy_grid = emin + np.arange(int(np.floor((emax - emin) / delta + 1.000001))) * delta
            outputs = []
            for index, energy in enumerate(energy_grid, start=1):
                factors = degeneracy * state.weights[:, None] * smearing_density(
                    (energy - energies_ev) / width, ngauss
                ) / (2.0 * width / EV_PER_HARTREE)
                factors = _qe_local_dos_weights(factors)
                values = _symmetrize_wave_field(state, _wave_density_sum(state, factors))
                path = Path(f"{filplot}{index:03d}") if len(energy_grid) > 1 else filplot
                label = f" Energy = {energy:8.4f} eV, broadening = {width:8.4f} eV"
                outputs.append((path, _grid_from_state(state, label, plot_num, values)))
            return outputs
        if plot_num == 4:
            if state.fermi_ha is None or state.degauss_ha <= 0.0:
                raise QEInputError("gaussian broadening needed", routine="local_dos")
            ngauss = smearing_order(state.smearing)
            entropy = -degeneracy * state.weights[:, None] * w1gauss(
                (state.fermi_ha - state.energies_ha) / state.degauss_ha,
                ngauss,
            )
            entropy = _qe_local_dos_weights(entropy)
            values = _symmetrize_wave_field(state, _wave_density_sum(state, entropy))
            return [(filplot, _grid_from_state(state, title, plot_num, values))]
        if plot_num == 5:
            bias_ry = float(options.get("sample_bias", 0.01))
            if state.degauss_ha > 0.0 and state.fermi_ha is not None:
                fermi_ha = state.fermi_ha
                width_ha = state.degauss_ha
                ngauss = smearing_order(state.smearing)
            else:
                electron_count = float(np.sum(state.weights[:, None] * state.occupations))
                occupied_bands = int(np.floor(electron_count / 2.0 + 0.5))
                if state.energies_ha.shape[1] <= occupied_bands + 1:
                    raise QEInputError("not enough bands", routine="stm")
                valence = float(np.max(state.energies_ha[:, occupied_bands - 1]))
                conduction = float(np.min(state.energies_ha[:, occupied_bands]))
                fermi_ha = 0.5 * (valence + conduction)
                width_ha = 5.0e-6
                ngauss = 0
            lower_ha, upper_ha = sorted((fermi_ha, fermi_ha + 0.5 * bias_ry))
            factors = _stm_window_weights(
                state.energies_ha, lower_ha, upper_ha, width_ha, ngauss
            )
            factors *= degeneracy * state.weights[:, None]
            candidate = (
                (state.energies_ha >= lower_ha - 3.0 * width_ha)
                & (state.energies_ha <= upper_ha + 3.0 * width_ha)
            )
            factors *= candidate
            label = f" Bias in eV = {bias_ry * 0.5 * EV_PER_HARTREE:10.4f} # states{int(np.sum(candidate)):4d}"
            values = _symmetrize_wave_field(state, _wave_density_sum(state, factors))
            return [(filplot, _grid_from_state(state, label, plot_num, values))]
        else:
            if "emin" not in options:
                raise QEInputError("emin is required", routine="postproc")
            lower = float(options["emin"])
            upper = float(
                options.get(
                    "emax",
                    state.fermi_ha * EV_PER_HARTREE if plot_num == 10 and state.fermi_ha is not None else lower,
                )
            )
        if upper < lower:
            raise QEInputError("emin > emax", routine="postproc")
        selected_rows = _selected_spin_rows(state, spin_component)
        factors = (
            degeneracy
            * state.weights[:, None]
            * ((energies_ev >= lower) & (energies_ev <= upper))
            * selected_rows[:, None]
        )
        label = f"Density for spins between{lower:8.4f} eV and {upper:8.4f} eV"
        values = _wave_density_sum(state, factors)
        # QE local_dos iflag=3 (plot_num=10) symmetrizes the integrated
        # density.  iflag=4 (plot_num=23) deliberately returns beforehand,
        # retaining the unsymmetrized selected-state density.
        if plot_num == 10:
            values = _symmetrize_wave_field(state, values)
        return [(filplot, _grid_from_state(state, label, plot_num, values))]

    if plot_num == 19:
        gradient = _spectral_gradient(state.density, state.lattice)
        grad2 = np.einsum("i...,i...->...", gradient, gradient)
        factor = 0.5 / np.cbrt(3.0 * np.pi**2)
        denominator = np.maximum(np.abs(state.density), 1.0e-30) ** (4.0 / 3.0)
        values = factor * np.sqrt(grad2) / denominator
        values[state.density > 0.05] = factor * 100.0 / denominator[state.density > 0.05]
    elif plot_num == 20:
        hessian = _spectral_hessian(state.density, state.lattice)
        matrices = np.moveaxis(hessian, (0, 1), (-2, -1))
        second = np.linalg.eigvalsh(matrices)[..., 1]
        values = np.where(second < 0.0, -state.density, state.density)
    elif plot_num == 123:
        gradient, hessian = _spectral_gradient_hessian(state.density, state.lattice)
        grad2 = np.einsum("i...,i...->...", gradient, gradient)
        laplacian = np.trace(hessian, axis1=0, axis2=1)
        hessian_gradient = np.einsum("ij...,j...->i...", hessian, gradient)
        vector = state.density[None, ...] * hessian_gradient - gradient * grad2[None, ...]
        theta = 4.0 * np.einsum("i...,i...->...", vector, vector) / (grad2 + 1.0e-5) ** 3
        values = theta / (1.0 + theta)
    else:
        factors = state.weights[:, None] * state.occupations
        if plot_num == 22:
            factors = factors * _selected_spin_rows(
                state, spin_component
            )[:, None]
        kinetic = _symmetrize_wave_field(
            state,
            _wave_density_sum(state, factors, kinetic=True),
        )
        if plot_num == 22:
            values = kinetic
        else:
            gradient = _spectral_gradient(state.density, state.lattice)
            grad2 = np.einsum("i...,i...->...", gradient, gradient)
            factor = 5.0 / (3.0 * (6.0 * np.pi**2) ** (2.0 / 3.0))
            bosonic = grad2 / np.maximum(state.density, 1.0e-30)
            dvalue = factor * (kinetic - 0.25 * bosonic + 1.0e-5) / np.maximum(state.density, 1.0e-30) ** (5.0 / 3.0)
            values = np.where(state.density > 1.0e-30, 1.0 / (1.0 + dvalue**2), 0.0)
    return [(filplot, _grid_from_state(state, title, plot_num, values))]


def write_plot_file(path: str | Path, grid: PlotGrid) -> Path:
    """Write QE's formatted intermediate ``filplot`` representation."""
    destination = Path(path)
    shape = grid.values.shape
    if len(shape) != 3:
        raise QEInputError("plot data must be a three-dimensional FFT grid", routine="plot_io")
    species_lookup = {symbol: (index + 1, valence) for index, (symbol, valence) in enumerate(grid.species)}
    dual = grid.ecutrho_ry / grid.ecutwfc_ry if grid.ecutwfc_ry > 0.0 else 0.0
    tpiba2 = (2.0 * np.pi / grid.alat) ** 2
    gcutm = grid.ecutrho_ry / tpiba2 if tpiba2 > 0.0 else 0.0
    with destination.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(grid.title[:75].ljust(75) + "\n")
        stream.write("".join(f"{value:8d}" for value in (*shape, *shape, len(grid.atoms), len(grid.species))) + "\n")
        celldm = grid.celldm if grid.ibrav != 0 else (grid.alat, 0, 0, 0, 0, 0)
        stream.write(f"{grid.ibrav:6d}  " + "".join(f"{value:16.8f}" for value in celldm) + "\n")
        if grid.ibrav == 0:
            for vector in grid.lattice / grid.alat:
                stream.write(" ".join(f"{value:23.15E}" for value in vector) + "\n")
        stream.write(f"{gcutm:20.10f}{dual:20.10f}{grid.ecutwfc_ry:20.10f}{grid.plot_num:6d}\n")
        for index, (symbol, valence) in enumerate(grid.species, start=1):
            stream.write(f"{index:4d}   {symbol[:2]:2s}   {valence:5.2f}\n")
        for index, atom in enumerate(grid.atoms, start=1):
            species_index, _valence = species_lookup[atom.symbol]
            tau = atom.position / grid.alat
            stream.write(f"{index:4d}   {tau[0]:15.9f}{tau[1]:15.9f}{tau[2]:15.9f}   {species_index:2d}\n")
        _write_standard_e_values(
            stream, grid.values.ravel(order="F"), width=5,
            field_width=17, precision=9,
        )
    return destination


def read_plot_file(path: str | Path) -> PlotGrid:
    """Read a QE formatted intermediate ``filplot`` file."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise QEInputError(f"cannot read plot file {source}: {exc}", routine="plot_io") from exc
    if len(lines) < 5:
        raise QEInputError(f"invalid plot file {source}", routine="plot_io")
    cursor = 0
    title = lines[cursor].rstrip()
    cursor += 1
    dimensions = [int(value) for value in lines[cursor].split()]
    cursor += 1
    if len(dimensions) != 8:
        raise QEInputError("invalid plot-grid dimensions", routine="plot_io")
    shape = tuple(dimensions[3:6])
    nat, ntyp = dimensions[6:8]
    cell_data = lines[cursor].split()
    cursor += 1
    ibrav = int(cell_data[0])
    celldm = [float(value.replace("D", "E")) for value in cell_data[1:7]]
    alat = celldm[0]
    if ibrav == 0:
        at = np.vstack([
            np.fromstring(lines[cursor + index].replace("D", "E"), sep=" ")
            for index in range(3)
        ])
        cursor += 3
        lattice = at * alat
    else:
        from ..pw.input import _lattice

        system = {"ibrav": ibrav, **{f"celldm({index + 1})": value for index, value in enumerate(celldm)}}
        lattice, alat = _lattice(system, {})
    cutoff = lines[cursor].split()
    cursor += 1
    if len(cutoff) < 4:
        raise QEInputError("invalid plot-file cutoff header", routine="plot_io")
    gcutm, dual, ecutwfc_ry = map(float, cutoff[:3])
    plot_num = int(cutoff[3])
    species = []
    for _index in range(ntyp):
        fields = lines[cursor].split()
        cursor += 1
        species.append((fields[1], float(fields[2])))
    atoms = []
    for _index in range(nat):
        fields = lines[cursor].split()
        cursor += 1
        species_index = int(fields[4])
        symbol, valence = species[species_index - 1]
        atoms.append(PPAtom(symbol, np.asarray(fields[1:4], dtype=float) * alat, species_index, valence))
    raw = np.fromstring(" ".join(lines[cursor:]).replace("D", "E"), sep=" ")
    expected = int(np.prod(shape))
    if len(raw) != expected:
        raise QEInputError(
            f"plot file contains {len(raw)} values; expected {expected}", routine="plot_io"
        )
    values = raw.reshape(shape, order="F")
    return PlotGrid(
        title, plot_num, values, lattice, alat, tuple(atoms), tuple(species),
        ecutwfc_ry, dual * ecutwfc_ry, ibrav, tuple(celldm), gcutm,
    )


def _vector(options: dict[str, object], name: str, default: np.ndarray | None = None) -> np.ndarray:
    values = []
    for index in range(1, 4):
        key = f"{name}({index})"
        if key in options:
            values.append(float(options[key]))
        elif default is not None:
            values.append(float(default[index - 1]))
        else:
            values.append(0.0)
    return np.asarray(values)


def _plot_fourier_components(
    grid: PlotGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the G-vectors retained by QE's ``chdens`` interpolation.

    A formatted ``filplot`` contains the full real-space FFT box, but QE
    rebuilds ``ngm`` from the stored ``gcutm`` and discards every Fourier
    coefficient outside that sphere.  Retaining those coefficients would
    turn decimal I/O roundoff into visible high-frequency ringing.
    """
    coefficients = np.fft.fftn(grid.values).reshape(-1) / grid.values.size
    miller, g = _miller_and_g(
        grid.values.shape, 2.0 * np.pi * np.linalg.inv(grid.lattice).T
    )
    flat_miller = miller.reshape(-1, 3)
    flat_g = g.reshape(-1, 3)
    tpiba2 = (2.0 * np.pi / grid.alat) ** 2
    cutoff = (
        float(grid.gcutm) * tpiba2
        if grid.gcutm is not None
        else float(grid.ecutrho_ry)
    )
    active = np.einsum("gi,gi->g", flat_g, flat_g) <= cutoff
    return coefficients[active], flat_miller[active], flat_g[active]


def _fourier_values(grid: PlotGrid, cartesian_points: np.ndarray) -> np.ndarray:
    flat_coefficients, flat_miller, _flat_g = _plot_fourier_components(grid)
    fractional = np.asarray(cartesian_points) @ np.linalg.inv(grid.lattice)
    result = np.empty(len(fractional), dtype=np.complex128)
    for begin in range(0, len(fractional), 512):
        phase = np.exp(2j * np.pi * (fractional[begin:begin + 512] @ flat_miller.T))
        result[begin:begin + 512] = phase @ flat_coefficients
    return result


def _fourier_regular_values(
    grid: PlotGrid,
    origin_cartesian: np.ndarray,
    direction_cartesian: list[np.ndarray],
    coordinates: list[np.ndarray],
) -> np.ndarray:
    """Evaluate a one-, two-, or three-dimensional regular Fourier grid.

    For ``r = r0 + sum(t_i e_i)``, each plane-wave phase is a product of
    one-dimensional phases.  Using that separability avoids constructing the
    ``npoint x ng`` phase matrix used by the arbitrary-point evaluator and
    turns the accumulation into optimized matrix products.  G vectors are
    still chunked so large production FFT grids do not require excessive RAM.
    """
    ndim = len(coordinates)
    if ndim not in {1, 2, 3} or len(direction_cartesian) != ndim:
        raise ValueError("regular Fourier evaluation requires one to three axes")
    coefficients, flat_miller, _flat_g = _plot_fourier_components(grid)

    inverse_lattice = np.linalg.inv(grid.lattice)
    origin_fractional = np.asarray(origin_cartesian, dtype=float) @ inverse_lattice
    directions_fractional = [
        np.asarray(vector, dtype=float) @ inverse_lattice
        for vector in direction_cartesian
    ]
    coordinate_arrays = [np.asarray(axis, dtype=float) for axis in coordinates]
    result = np.zeros(tuple(len(axis) for axis in coordinate_arrays), dtype=np.complex128)

    if ndim == 1:
        # A line only depends on G through the scalar projected frequency.
        # Common crystallographic directions therefore collapse an entire 3D
        # FFT grid to O(nr) frequencies.  Keep exact floating-point keys here;
        # if a low-symmetry direction does not collapse, use the general
        # bounded-memory path below without changing any phase.
        projected = flat_miller @ directions_fractional[0]
        frequencies, inverse = np.unique(projected, return_inverse=True)
        if len(frequencies) <= 3 * len(projected) // 4:
            phased = coefficients * np.exp(
                2j * np.pi * (flat_miller @ origin_fractional)
            )
            collapsed = (
                np.bincount(inverse, weights=phased.real)
                + 1j * np.bincount(inverse, weights=phased.imag)
            )
            axis = coordinate_arrays[0]
            block_size = max(1, 4_000_000 // max(1, len(axis)))
            for begin in range(0, len(frequencies), block_size):
                stop = min(begin + block_size, len(frequencies))
                result += np.exp(
                    2j * np.pi * np.outer(axis, frequencies[begin:stop])
                ) @ collapsed[begin:stop]
            return np.real(result)

    # Roughly 64 MiB of complex phase arrays per G-vector block.
    phase_rows = max(1, sum(len(axis) for axis in coordinate_arrays))
    block_size = max(1, 4_000_000 // phase_rows)
    for begin in range(0, len(coefficients), block_size):
        stop = min(begin + block_size, len(coefficients))
        block_miller = flat_miller[begin:stop]
        block_coefficients = coefficients[begin:stop] * np.exp(
            2j * np.pi * (block_miller @ origin_fractional)
        )
        phases = [
            np.exp(
                2j * np.pi * np.outer(axis, block_miller @ direction)
            )
            for axis, direction in zip(coordinate_arrays, directions_fractional)
        ]
        if ndim == 1:
            result += phases[0] @ block_coefficients
        elif ndim == 2:
            result += (phases[0] * block_coefficients) @ phases[1].T
        else:
            first = phases[0] * block_coefficients
            for index in range(len(coordinate_arrays[2])):
                result[:, :, index] += (first * phases[2][index]) @ phases[1].T
    return np.real(result)


def _spherical_average_values(
    grid: PlotGrid,
    origin_cartesian: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Evaluate QE's Fourier-space spherical average efficiently."""
    coefficients, _flat_miller, flat_g = _plot_fourier_components(grid)
    phased = coefficients * np.exp(1j * (flat_g @ np.asarray(origin_cartesian)))
    norms = np.linalg.norm(flat_g, axis=1)

    # Symmetry-related G vectors should have the same norm.  Rounding only
    # the shell key absorbs last-bit differences from Cartesian matrix
    # multiplication; the representative norm remains an original value.
    shell_keys = np.round(norms, decimals=13)
    _keys, first, inverse = np.unique(
        shell_keys, return_index=True, return_inverse=True
    )
    shell_norms = norms[first]
    if len(shell_norms) <= 3 * len(norms) // 4:
        shell_coefficients = (
            np.bincount(inverse, weights=phased.real)
            + 1j * np.bincount(inverse, weights=phased.imag)
        )
    else:
        # A generic triclinic metric may have almost no shells to combine.
        # Avoid paying for a larger reordered copy in that case.
        shell_norms = norms
        shell_coefficients = phased

    radial = np.asarray(radii, dtype=float)
    result = np.zeros(len(radial), dtype=np.complex128)
    block_size = max(1, 4_000_000 // max(1, len(radial)))
    for begin in range(0, len(shell_norms), block_size):
        stop = min(begin + block_size, len(shell_norms))
        result += spherical_jn(
            0, np.outer(radial, shell_norms[begin:stop])
        ) @ shell_coefficients[begin:stop]
    return np.real(result)


def _atomic_number(symbol: str) -> int:
    try:
        from ase.data import atomic_numbers

        return int(atomic_numbers[symbol.strip().capitalize()])
    except (ImportError, KeyError):
        return 0


def _write_xsf_structure(stream: TextIO, grid: PlotGrid) -> None:
    stream.write(" CRYSTAL\n PRIMVEC\n")
    for vector in grid.lattice * _ANGSTROM_PER_BOHR:
        stream.write("".join(f"{value:15.9f}" for value in vector) + "\n")
    stream.write(f" PRIMCOORD\n{len(grid.atoms):12d}{1:12d}\n")
    for atom in grid.atoms:
        position = atom.position * _ANGSTROM_PER_BOHR
        stream.write(f"{atom.symbol[:3]:<3s}   " + "".join(f"{value:15.9f}" for value in position) + "\n")


def _fortran_e(value: float, field_width: int, precision: int) -> str:
    """Format an unscaled Fortran ``Ew.d`` field (mantissa below one)."""
    number = float(value)
    if number == 0.0:
        rendered = f"{number:.{precision}f}E+00"
    else:
        exponent = int(np.floor(np.log10(abs(number)))) + 1
        mantissa = number / 10.0**exponent
        rendered_mantissa = f"{mantissa:.{precision}f}"
        if abs(float(rendered_mantissa)) >= 1.0:
            exponent += 1
            mantissa /= 10.0
            rendered_mantissa = f"{mantissa:.{precision}f}"
        rendered = f"{rendered_mantissa}E{exponent:+03d}"
    return rendered.rjust(field_width)


def _fortran_e_fields(
    values: np.ndarray, field_width: int, precision: int
) -> list[str]:
    """Return flattened QE ``Ew.d`` fields with vectorized exponent math.

    NumPy performs the expensive floating-point conversion in compiled code.
    This matters for XSF and cube files, where a realistic FFT grid contains
    millions of values.  QE's unscaled ``Ew.d`` convention is retained: the
    mantissa lies below one and the exponent is one larger than conventional
    scientific notation.
    """
    numbers = np.asarray(values, dtype=float)
    exponents = np.zeros(numbers.shape, dtype=np.int32)
    nonzero = numbers != 0.0
    exponents[nonzero] = np.floor(np.log10(np.abs(numbers[nonzero]))).astype(np.int32) + 1
    mantissas = np.zeros(numbers.shape, dtype=float)
    mantissas[nonzero] = numbers[nonzero] / np.power(10.0, exponents[nonzero])

    # Match the scalar formatter when rounding promotes 0.999... to 1.000....
    promoted = np.abs(np.round(mantissas, precision)) >= 1.0
    exponents[promoted] += 1
    mantissas[promoted] /= 10.0
    # Once exponent arithmetic is vectorized, CPython's float formatter is
    # substantially faster here than numpy.char's per-element ufunc while
    # producing the same rounding as the scalar reference implementation.
    rendered = [
        f"{mantissa:.{precision}f}E{exponent:+03d}".rjust(field_width)
        for mantissa, exponent in zip(mantissas.reshape(-1), exponents.reshape(-1))
    ]
    return rendered


def _formatted_fortran_e(
    values: np.ndarray, field_width: int, precision: int
) -> np.ndarray:
    """Shaped-array equivalent of :func:`_fortran_e`."""
    numbers = np.asarray(values, dtype=float)
    return np.asarray(
        _fortran_e_fields(numbers, field_width, precision)
    ).reshape(numbers.shape)


def _write_string_rows(
    stream: TextIO,
    fields: np.ndarray,
    width: int,
    *,
    blank_lines: bool = False,
) -> None:
    """Write already formatted fields in large buffered chunks."""
    # Joining Python strings is markedly faster than iterating NumPy scalar
    # strings.  One conversion per chunk is cheaper than crossing the
    # Python/NumPy boundary for every field.
    flat = (
        fields
        if isinstance(fields, list)
        else np.asarray(fields).reshape(-1).tolist()
    )
    # Keep peak temporary string memory bounded while reducing Python calls
    # from one per output line to approximately one per 240k values while
    # keeping peak temporary string memory bounded.
    chunk_values = max(width, (240_000 // width) * width)
    separator = "\n\n" if blank_lines else "\n"
    for chunk_begin in range(0, len(flat), chunk_values):
        chunk = flat[chunk_begin:chunk_begin + chunk_values]
        rows = [
            "".join(chunk[begin:begin + width])
            for begin in range(0, len(chunk), width)
        ]
        stream.write(separator.join(rows) + separator)


def _write_standard_e_values(
    stream: TextIO,
    values: np.ndarray,
    width: int,
    *,
    field_width: int,
    precision: int,
) -> None:
    """Write QE ``1P Ew.d`` records using vectorized standard notation."""
    flat = np.asarray(values, dtype=float).reshape(-1)
    chunk_values = max(width, (240_000 // width) * width)
    for begin in range(0, len(flat), chunk_values):
        fields = [
            f"{value:{field_width}.{precision}E}"
            for value in flat[begin:begin + chunk_values]
        ]
        _write_string_rows(stream, fields, width)


def _write_values(
    stream: TextIO,
    values: np.ndarray,
    width: int = 6,
    *,
    field_width: int = 14,
    precision: int = 6,
) -> None:
    flat = np.asarray(values, dtype=float).reshape(-1)
    chunk_values = max(width, (240_000 // width) * width)
    for begin in range(0, len(flat), chunk_values):
        fields = _fortran_e_fields(
            flat[begin:begin + chunk_values], field_width, precision
        )
        _write_string_rows(stream, fields, width)


def _write_xsf_3d(
    path: Path,
    grid: PlotGrid,
    *,
    periodic_padding: bool = True,
    origin_bohr: np.ndarray | None = None,
) -> None:
    values = (
        np.pad(grid.values, ((0, 1), (0, 1), (0, 1)), mode="wrap")
        if periodic_padding
        else grid.values
    )
    origin = np.zeros(3) if origin_bohr is None else np.asarray(origin_bohr, dtype=float)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        _write_xsf_structure(stream, grid)
        marker = "BEGIN_DATAGRID_3D_UNKNOWN" if periodic_padding else "DATAGRID_3D_UNKNOWN"
        stream.write(f"BEGIN_BLOCK_DATAGRID_3D\n3D_PWSCF\n{marker}\n")
        stream.write("".join(f"{size:12d}" for size in values.shape) + "\n")
        stream.write("".join(f"{value:10.6f}" for value in origin * _ANGSTROM_PER_BOHR) + "\n")
        for vector in grid.lattice * _ANGSTROM_PER_BOHR:
            stream.write("".join(f"{value:12.6f}" for value in vector) + "\n")
        _write_values(stream, values.ravel(order="F"))
        stream.write("END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n")


def _write_cube(path: Path, grid: PlotGrid) -> None:
    shape = grid.values.shape
    inverse = np.linalg.inv(grid.lattice)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(" Cubefile created from PWScf calculation\n")
        title = grid.title.strip()
        stream.write((title if len(title) > 1 else "Contains the selected quantity on a FFT grid") + "\n")
        stream.write(f"{len(grid.atoms):5d}{0.0:12.6f}{0.0:12.6f}{0.0:12.6f}\n")
        for count, vector in zip(shape, grid.lattice):
            increment = vector / count
            stream.write(f"{count:5d}" + "".join(f"{value:12.6f}" for value in increment) + "\n")
        for atom in grid.atoms:
            number = _atomic_number(atom.symbol)
            fractional = atom.position @ inverse
            shifted = fractional - 0.5
            nearest = np.where(
                shifted >= 0.0,
                np.floor(shifted + 0.5),
                np.ceil(shifted - 0.5),
            )
            position = (fractional - nearest) @ grid.lattice
            stream.write(f"{number:5d}{float(number):12.6f}" + "".join(f"{value:12.6f}" for value in position) + "\n")
        _write_values(
            stream,
            grid.values.ravel(order="C"),
            field_width=13,
            precision=5,
        )


def combine_plot_files(paths: list[str | Path], weights: list[float]) -> PlotGrid:
    if len(paths) != len(weights) or not paths:
        raise QEInputError("filepp and weight counts disagree", routine="chdens")
    grids = [read_plot_file(path) for path in paths]
    reference = grids[0]
    values = np.zeros_like(reference.values)
    for grid, weight in zip(grids, weights):
        if grid.values.shape != reference.values.shape or not np.allclose(grid.lattice, reference.lattice):
            raise QEInputError("plot files have incompatible FFT grids or cells", routine="chdens")
        values += float(weight) * grid.values
    return PlotGrid(
        reference.title, reference.plot_num, values, reference.lattice,
        reference.alat, reference.atoms, reference.species,
        reference.ecutwfc_ry, reference.ecutrho_ry,
        reference.ibrav, reference.celldm, reference.gcutm,
    )


def write_output_plot(grid: PlotGrid, options: dict[str, object], stdout: TextIO | None = None) -> Path:
    """Render one combined intermediate grid using QE ``&PLOT`` controls."""
    unknown = set(options) - _PLOT_KEYS
    if unknown:
        raise QEInputError(f"unknown &PLOT variable {sorted(unknown)[0]!r}", routine="chdens")
    interpolation = str(options.get("interpolation", "fourier")).lower()
    if interpolation != "fourier":
        raise UnsupportedFeatureError("pp.x B-spline interpolation", routine="chdens")
    if bool(options.get("isostm_flag", False)):
        raise UnsupportedFeatureError("pp.x constant-current STM surface transform", routine="isostm")
    iflag = int(options.get("iflag", -1))
    output_format = int(options.get("output_format", -1))
    if iflag not in _PLOT_NAMES:
        raise QEInputError("wrong iflag", routine="chdens")
    if output_format in {-1, 1, 4} or not 0 <= output_format <= 7:
        raise QEInputError("output_format wrong or obsolete", routine="chdens")
    if "fileout" not in options or not str(options["fileout"]).strip():
        raise QEInputError("output filename missing", routine="chdens")
    destination = Path(str(options["fileout"]))

    if iflag in {0, 1}:
        nx = int(options.get("nx", 0))
        if nx < 2:
            raise QEInputError("nx required and must be at least 2", routine="chdens")
        x0 = _vector(options, "x0")
        direction = _vector(options, "e1")
        length = float(np.linalg.norm(direction))
        if length < 1.0e-12:
            raise QEInputError("e1 has zero length", routine="chdens")
        unit = direction / length
        distances = np.linspace(0.0, length, nx)
        if iflag == 1:
            values = _fourier_regular_values(
                grid, x0 * grid.alat, [unit * grid.alat], [distances]
            )
            columns = np.column_stack((distances, values))
        else:
            radii = distances * grid.alat
            values = _spherical_average_values(grid, x0 * grid.alat, radii)
            cumulative = np.cumsum(values * np.arange(nx) ** 2 * (length * grid.alat / (nx - 1)) ** 3)
            columns = np.column_stack((distances, values, cumulative))
        np.savetxt(destination, columns, fmt="%20.10f")
    elif iflag == 2:
        nx, ny = int(options.get("nx", 0)), int(options.get("ny", 0))
        if nx < 2 or ny < 2:
            raise QEInputError("nx,ny required and must be at least 2", routine="chdens")
        x0 = _vector(options, "x0")
        e1 = _vector(options, "e1")
        e2 = _vector(options, "e2")
        m1, m2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
        if m1 < 1.0e-12 or m2 < 1.0e-12:
            raise QEInputError("e1/e2 has zero length", routine="chdens")
        if abs(float(np.dot(e1, e2))) > 1.0e-6:
            raise QEInputError("e1 and e2 are not orthogonal", routine="chdens")
        u, v = np.linspace(0.0, m1, nx), np.linspace(0.0, m2, ny)
        values = _fourier_regular_values(
            grid,
            x0 * grid.alat,
            [(e1 / m1) * grid.alat, (e2 / m2) * grid.alat],
            [u, v],
        )
        if output_format == 7:
            with destination.open("w", encoding="ascii", newline="\n") as stream:
                formatted = _formatted_fortran_e(
                    np.column_stack(
                        (
                            np.repeat(grid.alat * u, ny),
                            np.tile(grid.alat * v, nx),
                            values.ravel(),
                        )
                    ),
                    20,
                    8,
                )
                for i in range(nx):
                    block = formatted[i * ny:(i + 1) * ny]
                    stream.write("\n".join("".join(row) for row in block) + "\n\n")
        elif output_format == 0:
            with destination.open("w", encoding="ascii", newline="\n") as stream:
                fields = _formatted_fortran_e(values, 25, 14)
                _write_string_rows(stream, fields, ny, blank_lines=True)
        elif output_format == 2:
            with destination.open("w", encoding="ascii", newline="\n") as stream:
                stream.write(f"{nx - 1:4d}{ny - 1:4d}\n")
                for coordinates in (u, v):
                    for begin in range(0, len(coordinates), 8):
                        stream.write(
                            "".join(f"{value:8.4f}" for value in coordinates[begin:begin + 8])
                            + "\n"
                        )
                _write_values(
                    stream,
                    values.T.ravel(),
                    width=6,
                    field_width=12,
                    precision=4,
                )
                for vector in (x0, e1, e2):
                    stream.write("".join(f"{value:8.4f}" for value in vector) + "\n")
                stream.write(f"{len(grid.atoms):4d}\n")
                for atom in grid.atoms:
                    tau = atom.position / grid.alat
                    stream.write("".join(f"{value:8.4f}" for value in tau) + f"{1:3d}\n")
                stream.write(f"{grid.alat:10.6f}\n")
                for vector in grid.lattice / grid.alat:
                    stream.write("".join(f"{value:12.6f}" for value in vector) + "\n")
        elif output_format == 3:
            with destination.open("w", encoding="ascii", newline="\n") as stream:
                _write_xsf_structure(stream, grid)
                stream.write("BEGIN_BLOCK_DATAGRID_2D\n2D_PWSCF\nDATAGRID_2D_UNKNOWN\n")
                stream.write(f" {nx} {ny}\n")
                stream.write("".join(f"{value:10.6f}" for value in x0 * grid.alat * _ANGSTROM_PER_BOHR) + "\n")
                for vector in (e1, e2):
                    stream.write("".join(f"{value:10.6f}" for value in vector * grid.alat * _ANGSTROM_PER_BOHR) + "\n")
                _write_values(stream, values.ravel(order="F"))
                stream.write("END_DATAGRID_2D\nEND_BLOCK_DATAGRID_2D\n")
        else:
            raise QEInputError("wrong output_format", routine="plot_2d")
    elif iflag == 3:
        if output_format == 5:
            _write_xsf_3d(destination, grid)
        elif output_format == 6:
            _write_cube(destination, grid)
        elif output_format == 3:
            nx, ny, nz = (int(options.get(name, 0)) for name in ("nx", "ny", "nz"))
            if min(nx, ny, nz) <= 0:
                raise QEInputError("nx,ny,nz required", routine="chdens")
            x0 = _vector(options, "x0")
            vectors = [_vector(options, name, grid.lattice[index] / grid.alat) for index, name in enumerate(("e1", "e2", "e3"))]
            if any(
                abs(float(np.dot(vectors[first], vectors[second]))) > 1.0e-6
                for first, second in ((0, 1), (0, 2), (1, 2))
            ):
                raise QEInputError("e1, e2, e3 are not orthogonal", routine="chdens")
            axes = [np.arange(size) / size for size in (nx, ny, nz)]
            values = _fourier_regular_values(
                grid,
                x0 * grid.alat,
                [vector * grid.alat for vector in vectors],
                axes,
            )
            custom = PlotGrid(grid.title, grid.plot_num, values, np.vstack(vectors) * grid.alat, grid.alat, grid.atoms, grid.species)
            _write_xsf_3d(
                destination,
                custom,
                periodic_padding=False,
                origin_bohr=x0 * grid.alat,
            )
        else:
            raise QEInputError("wrong output_format", routine="chdens")
    else:
        nx, ny = int(options.get("nx", 0)), int(options.get("ny", 0))
        radius = float(options.get("radius", 1.0))
        if nx < 2 or ny < 2 or radius <= 0.0:
            raise QEInputError("radius,nx,ny required", routine="chdens")
        phi = np.linspace(0.0, 2.0 * np.pi, nx)
        theta = np.linspace(0.0, np.pi, ny)
        phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")
        points = radius * np.stack(
            (
                np.sin(theta_grid) * np.cos(phi_grid),
                np.sin(theta_grid) * np.sin(phi_grid),
                np.cos(theta_grid),
            ),
            axis=-1,
        )
        if output_format != 0:
            raise QEInputError("plot not implemented", routine="plot_2ds")
        values = np.real(_fourier_values(grid, points.reshape(-1, 3))).reshape(points.shape[:-1])
        with destination.open("w", encoding="ascii", newline="\n") as stream:
            stream.write(f"{nx:8d}{ny:8d}\n")
            fields = _formatted_fortran_e(values, 25, 14)
            _write_string_rows(stream, fields, ny)
    if stdout is not None:
        print(
            f"     Plot Type: {_PLOT_NAMES[iflag]:20s}   "
            f"Output format: {_FORMAT_NAMES.get(output_format, str(output_format)):13s}",
            file=stdout,
        )
    return destination


def _has_namelist(text: str, name: str) -> bool:
    return re.search(rf"&{re.escape(name)}\b", text, flags=re.IGNORECASE) is not None


def run_pp(
    text: str,
    stdout: TextIO | None = None,
) -> tuple[list[Path], Path | list[Path] | None]:
    """Run extraction and plotting stages from a complete pp.x input string."""
    input_options = parse_namelist(text, "inputpp") if _has_namelist(text, "inputpp") else {}
    prefix = str(input_options.get("prefix", "pwscf"))
    outdir = str(input_options["outdir"]) if "outdir" in input_options else None
    extracted: list[tuple[Path, PlotGrid]] = []
    if int(input_options.get("plot_num", -1)) != -1:
        state = read_saved_pp(prefix, outdir)
        if stdout is not None:
            print(f"\n     Reading xml data from directory:\n\n     {state.directory}{os.sep}", file=stdout)
            if (
                int(input_options.get("plot_num", -1)) == 3
                and "degauss_ldos" not in input_options
            ):
                print(
                    "\n     degauss_ldos not set, defaults to degauss = "
                    f"{state.degauss_ha * EV_PER_HARTREE:6.4f} eV",
                    file=stdout,
                )
        extracted = extract_plot_grids(state, input_options)
        for path, grid in extracted:
            if stdout is not None:
                print(f"\n     Calling punch_plot, plot_num = {grid.plot_num:3d}", file=stdout)
                print(f"     Writing data to file  {path}", file=stdout)
            write_plot_file(path, grid)

    output: Path | list[Path] | None = None
    if _has_namelist(text, "plot"):
        plot_options = parse_namelist(text, "plot")
        nfile = int(plot_options.get("nfile", 1))
        if not 1 <= nfile <= 10:
            raise QEInputError("nfile < 1 or too large", routine="chdens")
        if len(extracted) > 1 and nfile > 1:
            raise QEInputError(
                "can't mix nfile > 1 with multiple output files", routine="chdens"
            )
        base_fileout = str(plot_options.get("fileout", ""))
        if not base_fileout.strip():
            raise QEInputError("output filename missing", routine="chdens")
        if len(extracted) > 1:
            tasks = [
                ([path], [float(plot_options.get("weight(1)", 1.0))], Path(f"{path}{base_fileout}"))
                for path, _grid in extracted
            ]
        else:
            paths, weights = [], []
            for index in range(1, nfile + 1):
                if extracted and index == 1:
                    path = extracted[0][0]
                else:
                    path = Path(str(plot_options.get(f"filepp({index})", "tmp.pp")))
                paths.append(path)
                weights.append(float(plot_options.get(f"weight({index})", 1.0)))
            tasks = [(paths, weights, Path(base_fileout))]

        rendered = []
        for paths, weights, destination in tasks:
            if stdout is not None:
                if not extracted:
                    print(f"     Reading header from file  {paths[0]}", file=stdout)
                for path in paths:
                    print(f"     Reading data from file  {path}", file=stdout)
            combined = combine_plot_files(paths, weights)
            current_options = dict(plot_options)
            current_options["fileout"] = str(destination)
            if stdout is not None and str(destination):
                print(f"\n     Writing data to be plotted to file {destination}", file=stdout)
            rendered.append(write_output_plot(combined, current_options, stdout=stdout))
        output = rendered[0] if len(rendered) == 1 else rendered
    return [path for path, _grid in extracted], output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pp.py")
    add_input_file_argument(parser)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        print(format_qe_opening("PP-PY", __version__), end="")
        text = Path(args.input_file).read_text(encoding="utf-8") if args.input_file else sys.stdin.read()
        run_pp(text, stdout=sys.stdout)
        print(format_qe_timing("POST-PROC", time.process_time() - cpu_started, time.perf_counter() - started), end="")
        print(format_qe_closing(), end="")
        return 0
    except (QEInputError, UnsupportedFeatureError, OSError, ValueError) as exc:
        emit_qe_error(exc, routine=getattr(exc, "routine", "pp.py"))
        return 1


__all__ = [
    "PPAtom", "PlotGrid", "SavedPPState", "combine_plot_files",
    "extract_plot_grids", "main", "read_plot_file", "read_saved_pp",
    "run_pp", "write_output_plot", "write_plot_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
