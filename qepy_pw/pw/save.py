"""Quantum ESPRESSO-style XML/HDF5 persistence for SCF calculations.

The output layout follows ``pw.x``: ``<outdir>/<prefix>.save`` contains
``data-file-schema.xml``, ``charge-density.hdf5``, per-k-point
``wfcN.hdf5`` files, and the pseudopotentials used by the run. No qepy-specific
result container is written.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from ..errors import QEInputError
from .buffers import resolve_disk_io, resolve_outdir, resolve_prefix
from .input import PWInput
from ..mpi import MPIContext
from .scf import SCFResult
from ..version import __version__
from ..xc import canonical_xc_name


QES_NAMESPACE = "http://www.quantum-espresso.org/ns/qes/qes-1.0"
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"
QES_SCHEMA = "http://www.quantum-espresso.org/ns/qes/qes_260226.xsd"

ET.register_namespace("qes", QES_NAMESPACE)
ET.register_namespace("xsi", XSI_NAMESPACE)


def _qes(name: str) -> str:
    return f"{{{QES_NAMESPACE}}}{name}"


def resolve_save_directory(pw: PWInput) -> Path:
    """Return QE's ``<outdir>/<prefix>.save`` path for an input."""

    return resolve_outdir(pw) / f"{resolve_prefix(pw)}.save"


def saving_enabled(pw: PWInput) -> bool:
    """Whether CONTROL/disk_io permits persistent output."""

    return resolve_disk_io(pw) != "none"


def _text(parent: ET.Element, tag: str, value: Any) -> ET.Element:
    element = ET.SubElement(parent, _qes(tag))
    if isinstance(value, (bool, np.bool_)):
        element.text = "true" if value else "false"
    elif isinstance(value, (float, np.floating)):
        element.text = f"{float(value):.16e}"
    else:
        element.text = str(value)
    return element


def _vector(vector: np.ndarray) -> str:
    return " ".join(f"{float(value):.16e}" for value in np.asarray(vector).ravel())


def _vector_element(parent: ET.Element, tag: str, values: np.ndarray) -> ET.Element:
    array = np.asarray(values).ravel()
    element = ET.SubElement(parent, _qes(tag), {"size": str(len(array))})
    element.text = _vector(array)
    return element


def _matrix_element(parent: ET.Element, tag: str, values: np.ndarray) -> ET.Element:
    matrix = np.asarray(values)
    element = ET.SubElement(
        parent,
        _qes(tag),
        {"rank": "2", "dims": f"{matrix.shape[1]} {matrix.shape[0]}"},
    )
    element.text = _vector(matrix)
    return element


def _atomic_species(parent: ET.Element, pw: PWInput) -> None:
    species_element = ET.SubElement(
        parent,
        _qes("atomic_species"),
        {"ntyp": str(len(pw.species)), "pseudo_dir": str(pw.pseudo_dir)},
    )
    for species in pw.species:
        entry = ET.SubElement(
            species_element, _qes("species"), {"name": species.label}
        )
        _text(entry, "mass", species.mass)
        _text(entry, "pseudo_file", Path(species.pseudo_file).name)


def _atomic_structure(parent: ET.Element, pw: PWInput) -> None:
    alat = float(pw.system.get("celldm(1)", np.linalg.norm(pw.lattice[0])))
    structure = ET.SubElement(
        parent,
        _qes("atomic_structure"),
        {
            "nat": str(len(pw.atoms)),
            "alat": f"{alat:.16e}",
            "bravais_index": str(int(pw.system.get("ibrav", 0))),
        },
    )
    positions = ET.SubElement(structure, _qes("atomic_positions"))
    for atom in pw.atoms:
        ET.SubElement(
            positions, _qes("atom"), {"name": atom.label}
        ).text = _vector(atom.position)
    cell = ET.SubElement(structure, _qes("cell"))
    for index, lattice_vector in enumerate(pw.lattice, start=1):
        _text(cell, f"a{index}", _vector(lattice_vector))


def _dft(parent: ET.Element, pw: PWInput) -> None:
    dft = ET.SubElement(parent, _qes("dft"))
    selected = canonical_xc_name(
        pw.system.get("_resolved_xc", pw.system.get("input_dft", "PZ"))
    ) or "pz"
    functional = {
        "pz": "SLA PZ NOGX NOGC",
        "pw": "SLA PW NOGX NOGC",
        "pbe": "SLA PW PBX PBC",
        "pbesol": "SLA PW PSX PSC",
        "revpbe": "SLA PW REVX PBC",
        "rpbe": "SLA PW HHNX PBC",
    }[selected]
    _text(dft, "functional", functional)


def _input_xml(root: ET.Element, pw: PWInput, result: SCFResult) -> None:
    input_element = ET.SubElement(root, _qes("input"))
    control = ET.SubElement(input_element, _qes("control_variables"))
    control_values = (
        ("title", pw.control.get("title", "")),
        ("calculation", pw.control.get("calculation", "scf")),
        ("restart_mode", pw.control.get("restart_mode", "from_scratch")),
        ("prefix", pw.control.get("prefix", "pwscf")),
        ("pseudo_dir", pw.pseudo_dir),
        ("outdir", resolve_save_directory(pw).parent),
        ("stress", pw.control.get("tstress", False)),
        ("forces", pw.control.get("tprnfor", False)),
        ("wf_collect", False),
        ("disk_io", resolve_disk_io(pw)),
        ("max_seconds", pw.control.get("max_seconds", 1.0e7)),
        ("etot_conv_thr", pw.control.get("etot_conv_thr", 1.0e-4)),
        ("forc_conv_thr", pw.control.get("forc_conv_thr", 1.0e-3)),
        ("press_conv_thr", pw.control.get("press_conv_thr", 0.5)),
        ("verbosity", pw.control.get("verbosity", "low")),
        ("print_every", pw.control.get("iprint", 100000)),
        ("fcp", False),
        ("rism", False),
    )
    for tag, value in control_values:
        _text(control, tag, value)

    _atomic_species(input_element, pw)
    _atomic_structure(input_element, pw)
    _dft(input_element, pw)
    spin = ET.SubElement(input_element, _qes("spin"))
    _text(spin, "lsda", False)
    _text(spin, "noncolin", False)
    _text(spin, "spinorbit", False)

    bands = ET.SubElement(input_element, _qes("bands"))
    _text(bands, "nbnd", max((len(v) for v in result.eigenvalues_ha), default=0))
    _text(bands, "tot_charge", float(pw.system.get("tot_charge", 0.0)))
    _text(bands, "occupations", pw.system.get("occupations", "fixed"))

    basis = ET.SubElement(input_element, _qes("basis"))
    # qepy-pw stores a complete complex G basis even for a Gamma-only input;
    # QE's gamma_only=.TRUE. representation instead stores only one member
    # of each +/-G pair and therefore must not be advertised here.
    _text(basis, "gamma_only", False)
    _text(basis, "ecutwfc", 0.5 * float(pw.system["ecutwfc"]))
    _text(
        basis,
        "ecutrho",
        0.5 * float(pw.system.get("ecutrho", 4.0 * pw.system["ecutwfc"])),
    )
    if np.ndim(result.density) == 3:
        ET.SubElement(
            basis,
            _qes("fft_grid"),
            dict(zip(("nr1", "nr2", "nr3"), map(str, result.density.shape))),
        )

    electrons = ET.SubElement(input_element, _qes("electron_control"))
    electron_values = (
        ("startingpot", pw.electrons.get("startingpot", "atomic")),
        ("startingwfc", pw.electrons.get("startingwfc", "atomic+random")),
        ("diagonalization", pw.electrons.get("diagonalization", "david")),
        ("mixing_mode", pw.electrons.get("mixing_mode", "plain")),
        ("mixing_beta", pw.electrons.get("mixing_beta", 0.7)),
        ("conv_thr", pw.electrons.get("conv_thr", 1.0e-6)),
        ("mixing_ndim", pw.electrons.get("mixing_ndim", 8)),
        (
            "mixing_pulay_frequency",
            pw.electrons.get("mixing_pulay_frequency", 1),
        ),
        ("max_nstep", pw.electrons.get("electron_maxstep", 100)),
        ("tq_smoothing", False),
        ("tbeta_smoothing", False),
        (
            "diago_thr_init",
            pw.electrons.get(
                "diago_thr_init",
                1.0e-5
                if str(pw.electrons.get("startingpot", "atomic")).lower()
                == "file"
                else 1.0e-2,
            ),
        ),
        ("diago_cg_maxiter", pw.electrons.get("diago_cg_maxiter", 20)),
        ("diago_david_ndim", pw.electrons.get("diago_david_ndim", 2)),
        ("diago_rmm_ndim", pw.electrons.get("diago_rmm_ndim", 4)),
        ("diago_rmm_conv", pw.electrons.get("diago_rmm_conv", False)),
        ("diago_gs_nblock", pw.electrons.get("diago_gs_nblock", 16)),
        ("diago_full_acc", pw.electrons.get("diago_full_acc", False)),
    )
    for tag, value in electron_values:
        _text(electrons, tag, value)

    kpoints = ET.SubElement(input_element, _qes("k_points_IBZ"))
    _text(kpoints, "nk", len(pw.kpoints))
    for point in pw.kpoints:
        ET.SubElement(
            kpoints, _qes("k_point"), {"weight": f"{point.weight:.16e}"}
        ).text = _vector(point.crystal)
    ions = ET.SubElement(input_element, _qes("ion_control"))
    _text(ions, "ion_dynamics", "none")
    cell_control = ET.SubElement(input_element, _qes("cell_control"))
    _text(cell_control, "cell_dynamics", "none")
    _text(cell_control, "pressure", 0.0)


def _electron_count(pw: PWInput, result: SCFResult) -> float:
    if result.occupations:
        return float(
            sum(
                point.weight * np.sum(occupations)
                for point, occupations in zip(pw.kpoints, result.occupations)
            )
        )
    return 0.0


def _charge_data(pw: PWInput, result: SCFResult) -> tuple[np.ndarray, np.ndarray]:
    density = np.asarray(result.density, dtype=np.float64)
    if density.ndim != 3:
        raise QEInputError("the gathered SCF density is not a three-dimensional grid")
    rho_g_grid = np.fft.fftn(density) / density.size
    axes = [np.rint(np.fft.fftfreq(n) * n).astype(np.int32) for n in density.shape]
    miller_grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    cartesian_g = miller_grid @ pw.reciprocal
    ecutrho_ry = float(pw.system.get("ecutrho", 4.0 * pw.system["ecutwfc"]))
    cutoff_mask = np.einsum("...i,...i->...", cartesian_g, cartesian_g) <= (
        ecutrho_ry + 1.0e-10
    )
    if pw.kpoint_mode == "gamma":
        gx = miller_grid[..., 0]
        gy = miller_grid[..., 1]
        gz = miller_grid[..., 2]
        gamma_half = (gz > 0) | ((gz == 0) & (gy > 0)) | (
            (gz == 0) & (gy == 0) & (gx >= 0)
        )
        cutoff_mask &= gamma_half
    return miller_grid[cutoff_mask], rho_g_grid[cutoff_mask]


def _output_xml(root: ET.Element, pw: PWInput, result: SCFResult) -> None:
    output = ET.SubElement(root, _qes("output"))
    convergence = ET.SubElement(output, _qes("convergence_info"))
    scf_convergence = ET.SubElement(convergence, _qes("scf_conv"))
    _text(scf_convergence, "convergence_achieved", result.converged)
    _text(scf_convergence, "n_scf_steps", len(result.iterations))
    _text(
        scf_convergence,
        "scf_error",
        result.iterations[-1].estimated_accuracy_ha if result.iterations else 0.0,
    )
    _text(convergence, "wf_collected", bool(result.wavefunctions))

    algorithm = ET.SubElement(output, _qes("algorithmic_info"))
    _text(algorithm, "real_space_q", False)
    _text(algorithm, "real_space_beta", False)
    _text(algorithm, "uspp", False)
    _text(algorithm, "paw", False)
    _atomic_species(output, pw)
    _atomic_structure(output, pw)

    basis = ET.SubElement(output, _qes("basis_set"))
    _text(basis, "gamma_only", False)
    _text(basis, "ecutwfc", 0.5 * float(pw.system["ecutwfc"]))
    _text(
        basis,
        "ecutrho",
        0.5 * float(pw.system.get("ecutrho", 4.0 * pw.system["ecutwfc"])),
    )
    shape = np.asarray(result.density).shape
    ET.SubElement(
        basis,
        _qes("fft_grid"),
        dict(zip(("nr1", "nr2", "nr3"), map(str, shape))),
    )
    miller, _ = _charge_data(pw, result)
    _text(basis, "ngm", len(miller))
    _text(basis, "npwx", max(result.plane_waves_per_k, default=0))
    reciprocal = ET.SubElement(basis, _qes("reciprocal_lattice"))
    for index, vector in enumerate(pw.reciprocal, start=1):
        _text(reciprocal, f"b{index}", _vector(vector))
    _dft(output, pw)

    energy = ET.SubElement(output, _qes("total_energy"))
    _text(energy, "etot", result.total_energy_ha)
    if result.energy_terms is not None:
        terms = result.energy_terms
        _text(energy, "eband", terms.one_electron_ha)
        _text(energy, "ehart", terms.hartree_ha)
        _text(energy, "etxc", terms.xc_ha)
        _text(energy, "ewald", terms.ewald_ha)
        _text(energy, "demet", terms.smearing_ha)

    bands = ET.SubElement(output, _qes("band_structure"))
    _text(bands, "lsda", False)
    _text(bands, "noncolin", False)
    _text(bands, "spinorbit", False)
    _text(bands, "nbnd", max((len(v) for v in result.eigenvalues_ha), default=0))
    _text(bands, "nelec", _electron_count(pw, result))
    if result.fermi_energy_ha is not None:
        _text(bands, "fermi_energy", result.fermi_energy_ha)
    _text(bands, "nks", len(pw.kpoints))
    _text(bands, "occupations_kind", pw.system.get("occupations", "fixed"))
    _text(bands, "smearing", pw.system.get("smearing", "gaussian"))
    _text(bands, "degauss", float(pw.system.get("degauss", 0.0)))
    if pw.kpoint_grid is not None:
        ET.SubElement(
            bands,
            _qes("monkhorst_pack"),
            {
                **{
                    f"nk{axis}": str(int(value))
                    for axis, value in enumerate(pw.kpoint_grid, start=1)
                },
                **{
                    f"k{axis}": str(int(value))
                    for axis, value in enumerate(
                        pw.kpoint_shift or (0, 0, 0), start=1
                    )
                },
            },
        )
        _text(
            bands,
            "full_to_irreducible",
            " ".join(str(int(value)) for value in pw.full_to_irreducible),
        )
    symmetries = ET.SubElement(bands, _qes("symmetry_operations"))
    for operation in pw.symmetry_operations:
        entry = ET.SubElement(symmetries, _qes("symmetry"))
        _text(
            entry,
            "rotation",
            " ".join(str(int(value)) for value in operation.matrix.ravel()),
        )
        _text(entry, "fractional_translation", _vector(operation.translation))
    _text(symmetries, "time_reversal", not bool(pw.system.get("noinv", False)))
    for index, (point, eigenvalues) in enumerate(
        zip(pw.kpoints, result.eigenvalues_ha)
    ):
        ks = ET.SubElement(bands, _qes("ks_energies"))
        ET.SubElement(
            ks, _qes("k_point"), {"weight": f"{point.weight:.16e}"}
        ).text = _vector(point.crystal)
        _text(ks, "npw", result.plane_waves_per_k[index])
        _vector_element(ks, "eigenvalues", eigenvalues)
        occupations = (
            result.occupations[index]
            if index < len(result.occupations)
            else np.zeros_like(eigenvalues)
        )
        _vector_element(ks, "occupations", occupations)

    if result.forces_ha_per_bohr is not None:
        _matrix_element(output, "forces", result.forces_ha_per_bohr)
    if result.stress_ha_per_bohr3 is not None:
        _matrix_element(output, "stress", result.stress_ha_per_bohr3)


def _write_xml(path: Path, pw: PWInput, result: SCFResult) -> None:
    now = datetime.now().astimezone()
    root = ET.Element(
        _qes("espresso"),
        {
            f"{{{XSI_NAMESPACE}}}schemaLocation": f"{QES_NAMESPACE} {QES_SCHEMA}",
            "Units": "Hartree atomic units",
        },
    )
    root.append(
        ET.Comment(
            "All quantities are in Hartree atomic units unless otherwise specified"
        )
    )
    general = ET.SubElement(root, _qes("general_info"))
    ET.SubElement(
        general, _qes("xml_format"), {"NAME": "QEXSD", "VERSION": "21.11.02"}
    ).text = "QEXSD_21.11.02"
    ET.SubElement(
        general, _qes("creator"), {"NAME": "PWSCF", "VERSION": __version__}
    ).text = "XML file generated by qepy-pw"
    ET.SubElement(
        general,
        _qes("created"),
        {"DATE": now.strftime("%d%b%Y"), "TIME": now.strftime("%H:%M:%S")},
    ).text = f"This run was terminated on: {now.isoformat()}"
    _text(general, "job", str(pw.control.get("title", "")))

    parallel = ET.SubElement(root, _qes("parallel_info"))
    for tag, value in (
        ("nprocs", result.mpi_processes),
        ("nthreads", 1),
        ("ntasks", 1),
        ("nbgrp", 1),
        ("npool", 1),
        ("ndiag", 1),
    ):
        _text(parallel, tag, value)
    _input_xml(root, pw, result)
    _output_xml(root, pw, result)
    _text(root, "exit_status", 0 if result.converged else 2)

    timing = ET.SubElement(root, _qes("timing_info"))
    total = ET.SubElement(timing, _qes("total"), {"label": "PWSCF"})
    _text(total, "cpu", max((step.cpu_seconds for step in result.iterations), default=0.0))
    _text(total, "wall", result.wall_seconds)
    for name, entry in sorted(result.timings.items()):
        partial = ET.SubElement(
            timing, _qes("partial"), {"label": name, "calls": str(entry.calls)}
        )
        _text(partial, "cpu", entry.cpu_seconds)
        _text(partial, "wall", entry.wall_seconds)
    ET.SubElement(
        root,
        _qes("closed"),
        {"DATE": now.strftime("%d%b%Y"), "TIME": now.strftime("%H:%M:%S")},
    ).text = "XML file closed normally"

    ET.indent(root, space="  ")
    with path.open("wb") as stream:
        ET.ElementTree(root).write(stream, encoding="utf-8", xml_declaration=True)


def _hdf5_module():
    try:
        import h5py
    except ImportError as exc:
        raise QEInputError(
            "HDF5 save output requires h5py; install qepy-pw with its declared dependencies"
        ) from exc
    return h5py


def _attribute_bool(value: Any) -> bool:
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).strip().upper() in {".TRUE.", "TRUE", "T", "1"}


def _saved_xml(pw: PWInput) -> ET.Element:
    path = resolve_save_directory(pw) / "data-file-schema.xml"
    if not path.is_file():
        raise QEInputError(f"restart metadata file not found: {path}")
    try:
        return ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise QEInputError(f"cannot read restart metadata {path}: {exc}") from exc


def _xml_vector(root: ET.Element, path: str) -> np.ndarray:
    element = root.find(path, {"qes": QES_NAMESPACE})
    if element is None or element.text is None:
        raise QEInputError(f"restart metadata is missing {path}")
    return np.fromstring(element.text, sep=" ")


def _miller_row_indices(
    saved_miller: np.ndarray, requested_miller: np.ndarray
) -> np.ndarray:
    """Map requested Miller rows without a Python tuple dictionary."""
    saved = np.ascontiguousarray(saved_miller, dtype=np.int32)
    requested = np.ascontiguousarray(requested_miller, dtype=np.int32)
    if saved.ndim != 2 or saved.shape[1] != 3:
        raise QEInputError("saved Miller indices have an invalid shape")
    if requested.ndim != 2 or requested.shape[1] != 3:
        raise QEInputError("requested Miller indices have an invalid shape")
    key_dtype = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4")])
    saved_keys = saved.view(key_dtype).reshape(-1)
    requested_keys = requested.view(key_dtype).reshape(-1)
    sorting = np.argsort(saved_keys, order=("x", "y", "z"))
    sorted_keys = saved_keys[sorting]
    positions = np.searchsorted(sorted_keys, requested_keys)
    if np.any(positions >= len(sorted_keys)):
        raise QEInputError("saved Miller indices do not cover the current basis")
    order = sorting[positions]
    if not np.array_equal(saved[order], requested):
        raise QEInputError("saved Miller indices do not match the current basis")
    return order


def validate_restart_metadata(
    pw: PWInput,
    shape: tuple[int, int, int],
    number_of_bands: int,
    *,
    electronic_states: bool = True,
) -> None:
    """Reject saved data incompatible with the requested calculation.

    A restart reuses both density and wavefunctions and therefore requires an
    identical FFT grid, band count, and k-point list.  NSCF/bands runs reuse
    only the density; their band count and k points are deliberately allowed
    to differ from the preceding SCF calculation.
    """
    root = _saved_xml(pw)
    ns = {"qes": QES_NAMESPACE}
    saved_lattice = np.vstack(
        [_xml_vector(root, f"qes:output/qes:atomic_structure/qes:cell/qes:a{i}")
         for i in range(1, 4)]
    )
    if not np.allclose(saved_lattice, pw.lattice, rtol=1.0e-10, atol=1.0e-10):
        raise QEInputError("restart lattice does not match the current input")
    saved_atoms = root.findall(
        "qes:output/qes:atomic_structure/qes:atomic_positions/qes:atom", ns
    )
    if len(saved_atoms) != len(pw.atoms):
        raise QEInputError("restart atom count does not match the current input")
    for index, (element, atom) in enumerate(zip(saved_atoms, pw.atoms), start=1):
        position = np.fromstring(element.text or "", sep=" ")
        if element.attrib.get("name") != atom.label or position.shape != (3,) or not np.allclose(
            position, atom.position, rtol=1.0e-10, atol=1.0e-10
        ):
            raise QEInputError(f"restart atom {index} does not match the current input")
    saved_ecut = root.findtext("qes:output/qes:basis_set/qes:ecutwfc", namespaces=ns)
    saved_ecutrho = root.findtext("qes:output/qes:basis_set/qes:ecutrho", namespaces=ns)
    current_ecut = 0.5 * float(pw.system["ecutwfc"])
    current_ecutrho = 0.5 * float(
        pw.system.get("ecutrho", 4.0 * pw.system["ecutwfc"])
    )
    if saved_ecut is None or not np.isclose(float(saved_ecut), current_ecut):
        raise QEInputError("restart wavefunction cutoff does not match the current input")
    if saved_ecutrho is None or not np.isclose(float(saved_ecutrho), current_ecutrho):
        raise QEInputError("restart charge-density cutoff does not match the current input")
    if not electronic_states:
        return
    grid = root.find("qes:output/qes:basis_set/qes:fft_grid", ns)
    saved_shape = (() if grid is None else tuple(
        int(grid.attrib[name]) for name in ("nr1", "nr2", "nr3")
    ))
    if saved_shape != tuple(shape):
        raise QEInputError(
            f"restart FFT grid {saved_shape} does not match current grid {shape}"
        )
    saved_nbnd = root.findtext("qes:output/qes:band_structure/qes:nbnd", namespaces=ns)
    if saved_nbnd is None or int(saved_nbnd) != number_of_bands:
        raise QEInputError("restart number of bands does not match the current input")
    saved_points = root.findall(
        "qes:output/qes:band_structure/qes:ks_energies/qes:k_point", ns
    )
    if len(saved_points) != len(pw.kpoints):
        raise QEInputError("restart k-point count does not match the current input")
    for index, (element, point) in enumerate(zip(saved_points, pw.kpoints), start=1):
        saved = np.fromstring(element.text or "", sep=" ")
        if saved.shape != (3,) or not np.allclose(
            saved, point.crystal, rtol=1.0e-10, atol=1.0e-10
        ):
            raise QEInputError(f"restart k point {index} does not match the current input")
        if not np.isclose(float(element.attrib.get("weight", "nan")), point.weight):
            raise QEInputError(f"restart k-point weight {index} does not match the current input")


def read_saved_density(
    pw: PWInput, shape: tuple[int, int, int], expected_electrons: float
) -> np.ndarray:
    """Reconstruct a real-space density from QE ``charge-density.hdf5``."""
    path = resolve_save_directory(pw) / "charge-density.hdf5"
    if not path.is_file():
        raise QEInputError(f"startingpot='file' requires {path}")
    h5py = _hdf5_module()
    try:
        with h5py.File(path, "r") as h5:
            miller = np.asarray(h5["MillerIndices"][:], dtype=np.int64)
            coefficients = np.asarray(h5["rhotot_g"][:], dtype=np.complex128)
            gamma_only = _attribute_bool(h5.attrs.get("gamma_only", False))
            nspin = int(h5.attrs.get("nspin", 1))
    except (OSError, KeyError, ValueError) as exc:
        raise QEInputError(f"cannot read saved charge density {path}: {exc}") from exc
    if nspin != 1 or miller.shape != (len(coefficients), 3):
        raise QEInputError("saved charge density has an unsupported layout")
    if len({tuple(row % np.asarray(shape)) for row in miller}) != len(miller):
        raise QEInputError("saved charge density contains aliased Miller indices")
    grid = np.zeros(shape, dtype=np.complex128)
    slots = tuple((miller % np.asarray(shape)).T)
    grid[slots] = coefficients
    if gamma_only:
        negative = tuple(((-miller) % np.asarray(shape)).T)
        empty = np.abs(grid[negative]) == 0.0
        negative_slots = tuple(axis[empty] for axis in negative)
        grid[negative_slots] = np.conjugate(coefficients[empty])
    density = np.real(np.fft.ifftn(grid * np.prod(shape)))
    charge = float(np.sum(density) * pw.volume / np.prod(shape))
    if not np.isfinite(charge) or abs(charge - expected_electrons) > 1.0e-7 * max(1.0, expected_electrons):
        raise QEInputError(
            f"saved density contains {charge:.12g} electrons; expected {expected_electrons:.12g}"
        )
    return density


def read_saved_density_coefficients(
    pw: PWInput,
    current_miller: np.ndarray,
) -> np.ndarray:
    """Read saved density directly onto a compact current G-vector basis."""
    path = resolve_save_directory(pw) / "charge-density.hdf5"
    if not path.is_file():
        raise QEInputError(f"startingpot='file' requires {path}")
    h5py = _hdf5_module()
    try:
        with h5py.File(path, "r") as h5:
            saved_miller = np.asarray(
                h5["MillerIndices"][:], dtype=np.int32
            )
            coefficients = np.asarray(
                h5["rhotot_g"][:], dtype=np.complex128
            )
            gamma_only = _attribute_bool(
                h5.attrs.get("gamma_only", False)
            )
            nspin = int(h5.attrs.get("nspin", 1))
    except (OSError, KeyError, ValueError) as exc:
        raise QEInputError(f"cannot read saved charge density {path}: {exc}") from exc
    if nspin != 1 or saved_miller.shape != (len(coefficients), 3):
        raise QEInputError("saved charge density has an unsupported layout")

    requested = np.ascontiguousarray(current_miller, dtype=np.int32).copy()
    conjugate = np.zeros(len(requested), dtype=bool)
    if gamma_only:
        gx, gy, gz = requested.T
        stored_half = (gz > 0) | ((gz == 0) & (gy > 0)) | (
            (gz == 0) & (gy == 0) & (gx >= 0)
        )
        conjugate = ~stored_half
        requested[conjugate] *= -1
    order = _miller_row_indices(saved_miller, requested)
    result = coefficients[order]
    result[conjugate] = np.conjugate(result[conjugate])
    if not np.all(np.isfinite(result)):
        raise QEInputError(f"saved charge density {path} contains non-finite values")
    return result


def read_saved_wavefunction(
    pw: PWInput,
    kpoint_index: int,
    current_miller: np.ndarray,
    number_of_bands: int,
    local_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Read and reorder one QE wavefunction file onto the current G basis."""
    path = resolve_save_directory(pw) / f"wfc{kpoint_index + 1}.hdf5"
    if not path.is_file():
        raise QEInputError(f"startingwfc='file' requires {path}")
    h5py = _hdf5_module()
    try:
        with h5py.File(path, "r") as h5:
            saved_miller = np.asarray(h5["MillerIndices"][:], dtype=np.int32)
            if int(h5.attrs.get("ik", -1)) != kpoint_index + 1:
                raise QEInputError(f"saved wavefunction k-point index is wrong in {path}")
            saved_xk = np.asarray(h5.attrs.get("xk", []), dtype=float)
            expected_xk = pw.kpoints[kpoint_index].crystal @ pw.reciprocal
            if saved_xk.shape != (3,) or not np.allclose(
                saved_xk, expected_xk, rtol=1.0e-10, atol=1.0e-10
            ):
                raise QEInputError(f"saved wavefunction k vector is wrong in {path}")
            if int(h5.attrs.get("ispin", -1)) != 1 or int(h5.attrs.get("npol", -1)) != 1:
                raise QEInputError("only scalar saved wavefunctions are supported")
            if int(h5.attrs.get("nbnd", -1)) != number_of_bands:
                raise QEInputError(
                    f"saved wavefunctions at k point {kpoint_index + 1} have a different band count"
                )
            if _attribute_bool(h5.attrs.get("gamma_only", False)):
                raise QEInputError("Gamma-packed saved wavefunctions are not supported")
            if h5["evc"].shape != (number_of_bands, len(saved_miller)):
                raise QEInputError("saved wavefunction coefficient dimensions are inconsistent")
            if len(saved_miller) != len(current_miller):
                raise QEInputError(
                    f"saved wavefunction basis size does not match k point {kpoint_index + 1}"
                )
            try:
                order = _miller_row_indices(saved_miller, current_miller)
            except QEInputError as exc:
                raise QEInputError(
                    f"saved wavefunction Miller indices do not match k point "
                    f"{kpoint_index + 1}"
                ) from exc
            del saved_miller
            selected = order if local_rows is None else order[np.asarray(local_rows)]
            del order
            # h5py requires increasing fancy indices; read bands one by one and
            # restore the requested current-basis ordering afterward.
            sorting = np.argsort(selected)
            inverse = np.argsort(sorting)
            vectors = np.empty((len(selected), number_of_bands), dtype=np.complex128)
            for band in range(number_of_bands):
                vectors[:, band] = h5["evc"][band, selected[sorting]][inverse]
    except QEInputError:
        raise
    except (OSError, KeyError, ValueError) as exc:
        raise QEInputError(f"cannot read saved wavefunctions {path}: {exc}") from exc
    if not np.all(np.isfinite(vectors)):
        raise QEInputError(f"saved wavefunctions {path} contain non-finite values")
    return vectors


def _write_charge_density(path: Path, pw: PWInput, result: SCFResult) -> None:
    h5py = _hdf5_module()
    miller, rho_g = _charge_data(pw, result)
    with h5py.File(path, "w") as h5:
        h5.attrs["gamma_only"] = (
            ".TRUE." if pw.kpoint_mode == "gamma" else ".FALSE."
        )
        h5.attrs["ngm_g"] = len(miller)
        h5.attrs["nspin"] = 1
        miller_dataset = h5.create_dataset("MillerIndices", data=miller)
        for index, vector in enumerate(pw.reciprocal, start=1):
            miller_dataset.attrs[f"bg{index}"] = np.asarray(vector, dtype=np.float64)
        h5.create_dataset("rhotot_g", data=rho_g)


def _write_wavefunction(
    path: Path,
    pw: PWInput,
    result: SCFResult,
    kpoint_index: int,
) -> None:
    """Write one scalar k point using QE's native ``wfcN.hdf5`` layout."""
    coefficients = np.asarray(
        result.wavefunctions[kpoint_index], dtype=np.complex128
    )
    miller = np.asarray(
        result.wavefunction_miller_indices[kpoint_index], dtype=np.int32
    )
    _write_wavefunction_data(
        path, pw, result, kpoint_index, coefficients, miller
    )


def _write_wavefunction_data(
    path: Path,
    pw: PWInput,
    result: SCFResult,
    kpoint_index: int,
    coefficients: np.ndarray,
    miller: np.ndarray,
) -> None:
    """Write one already-collected wavefunction matrix."""
    h5py = _hdf5_module()
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    miller = np.asarray(miller, dtype=np.int32)
    eigenvalues = np.asarray(result.eigenvalues_ha[kpoint_index])
    if miller.ndim != 2 or miller.shape[1] != 3:
        raise QEInputError("wavefunction Miller indices must have shape (npw, 3)")
    if coefficients.shape != (len(miller), len(eigenvalues)):
        raise QEInputError(
            "wavefunction coefficients do not match the plane-wave/band counts"
        )
    point = pw.kpoints[kpoint_index]
    with h5py.File(path, "w") as h5:
        h5.attrs["ik"] = kpoint_index + 1
        h5.attrs["xk"] = np.asarray(
            point.crystal @ pw.reciprocal, dtype=np.float64
        )
        h5.attrs["ispin"] = 1
        h5.attrs["gamma_only"] = ".FALSE."
        h5.attrs["scale_factor"] = 1.0
        h5.attrs["ngw"] = len(miller)
        h5.attrs["igwx"] = len(miller)
        h5.attrs["npol"] = 1
        h5.attrs["nbnd"] = coefficients.shape[1]
        miller_dataset = h5.create_dataset("MillerIndices", data=miller)
        for index, vector in enumerate(pw.reciprocal, start=1):
            miller_dataset.attrs[f"bg{index}"] = np.asarray(
                vector, dtype=np.float64
            )
        miller_dataset.attrs["doc"] = (
            "Miller Indices of the wave-vectors, same ordering as "
            "wave-function components"
        )
        evc = h5.create_dataset(
            "evc",
            shape=(coefficients.shape[1], coefficients.shape[0]),
            dtype=np.complex128,
        )
        for band in range(coefficients.shape[1]):
            evc[band, :] = coefficients[:, band]
        evc.attrs["doc:"] = (
            "Wave Functions, (npwx,nbnd), each contiguous line represents "
            "a wave function"
        )


def _temporary_path(directory: Path, final_name: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{final_name}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    return Path(name)


def _atomic_write(
    save_directory: Path,
    final_name: str,
    writer: Any,
) -> None:
    temporary = _temporary_path(save_directory, final_name)
    try:
        writer(temporary)
        os.replace(temporary, save_directory / final_name)
    finally:
        temporary.unlink(missing_ok=True)


def _finish_save_directory(
    save_directory: Path,
    pw: PWInput,
    wavefunction_count: int,
) -> None:
    stale_results = save_directory / "results.hdf5"
    if stale_results.exists():
        stale_results.unlink()
    expected_wavefunctions = {
        f"wfc{index + 1}.hdf5" for index in range(wavefunction_count)
    }
    for stale_wavefunction in save_directory.glob("wfc*.hdf5"):
        if stale_wavefunction.name not in expected_wavefunctions:
            stale_wavefunction.unlink()
    for species in pw.species:
        source = pw.pseudo_dir / species.pseudo_file
        destination = save_directory / Path(species.pseudo_file).name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)


def _write_qe_save_distributed(
    pw: PWInput,
    result: SCFResult,
    mpi: MPIContext,
) -> Path | None:
    """Collect, write, and release one distributed k point at a time."""
    kpoints = len(pw.kpoints)
    validation_error: str | None = None
    if len(result.wavefunctions) != kpoints:
        validation_error = (
            "distributed saved wavefunctions do not cover every k point"
        )
    elif len(result.wavefunction_row_indices) != kpoints:
        validation_error = (
            "distributed wavefunction row maps do not cover every k point"
        )
    elif mpi.is_root and len(result.wavefunction_miller_indices) != kpoints:
        validation_error = (
            "root Miller-index maps do not cover every k point"
        )
    validation_errors = mpi.comm.allgather(validation_error)
    first_error = next(
        (error for error in validation_errors if error is not None), None
    )
    if first_error is not None:
        raise QEInputError(first_error)

    save_directory = resolve_save_directory(pw)
    writes_density = (
        str(pw.control.get("calculation", "scf")).strip().lower() == "scf"
    )
    setup_error: str | None = None
    if mpi.is_root:
        try:
            save_directory.mkdir(parents=True, exist_ok=True)
            _atomic_write(
                save_directory,
                "data-file-schema.xml",
                lambda path: _write_xml(path, pw, result),
            )
            if writes_density:
                _atomic_write(
                    save_directory,
                    "charge-density.hdf5",
                    lambda path: _write_charge_density(path, pw, result),
                )
            _finish_save_directory(save_directory, pw, kpoints)
        except Exception as exc:  # propagate root I/O failure to every rank
            setup_error = str(exc)
    setup_error = mpi.broadcast(setup_error)
    if setup_error is not None:
        raise QEInputError(f"cannot initialize distributed save: {setup_error}")

    for kpoint_index, total_rows in enumerate(result.plane_waves_per_k):
        gathered = mpi.gather_indexed_rows_root(
            result.wavefunctions[kpoint_index],
            result.wavefunction_row_indices[kpoint_index],
            int(total_rows),
        )
        write_error: str | None = None
        if mpi.is_root:
            try:
                miller = result.wavefunction_miller_indices[kpoint_index]
                _atomic_write(
                    save_directory,
                    f"wfc{kpoint_index + 1}.hdf5",
                    lambda path, coefficients=gathered, indices=miller, index=kpoint_index: (
                        _write_wavefunction_data(
                            path,
                            pw,
                            result,
                            index,
                            coefficients,
                            indices,
                        )
                    ),
                )
            except Exception as exc:
                write_error = str(exc)
        write_error = mpi.broadcast(write_error)
        if write_error is not None:
            raise QEInputError(
                f"cannot write distributed wavefunction {kpoint_index + 1}: "
                f"{write_error}"
            )

    return save_directory if mpi.is_root else None


def write_qe_save(
    pw: PWInput,
    result: SCFResult,
    mpi: MPIContext | None = None,
) -> Path | None:
    """Write QE-style XML, density, wavefunctions, and UPFs."""

    if not saving_enabled(pw):
        return None
    if result.wavefunctions_distributed:
        if mpi is None or mpi.size <= 1:
            raise QEInputError(
                "distributed wavefunctions require their MPI context when saving"
            )
        return _write_qe_save_distributed(pw, result, mpi)
    save_directory = resolve_save_directory(pw)
    save_directory.mkdir(parents=True, exist_ok=True)
    if len(result.wavefunctions) != len(result.wavefunction_miller_indices):
        raise QEInputError("wavefunction coefficient and Miller-index lists differ")
    if result.wavefunctions and len(result.wavefunctions) != len(pw.kpoints):
        raise QEInputError("saved wavefunctions do not cover every k point")
    writes_density = (
        str(pw.control.get("calculation", "scf")).strip().lower() == "scf"
    )
    writers: list[tuple[str, Any]] = [
        ("data-file-schema.xml", _write_xml),
    ]
    if writes_density:
        writers.append(("charge-density.hdf5", _write_charge_density))
    temporary_files: list[Path] = []
    try:
        for final_name, writer in writers:
            temporary = _temporary_path(save_directory, final_name)
            temporary_files.append(temporary)
            writer(temporary, pw, result)
            os.replace(temporary, save_directory / final_name)
            temporary_files.remove(temporary)
        _finish_save_directory(
            save_directory, pw, len(result.wavefunctions)
        )
        for kpoint_index in range(len(result.wavefunctions)):
            final_name = f"wfc{kpoint_index + 1}.hdf5"
            temporary = _temporary_path(save_directory, final_name)
            temporary_files.append(temporary)
            _write_wavefunction(temporary, pw, result, kpoint_index)
            os.replace(temporary, save_directory / final_name)
            temporary_files.remove(temporary)
    finally:
        for temporary in temporary_files:
            temporary.unlink(missing_ok=True)
    return save_directory


__all__ = [
    "QES_NAMESPACE", "read_saved_density", "read_saved_density_coefficients",
    "read_saved_wavefunction",
    "resolve_save_directory", "saving_enabled", "validate_restart_metadata",
    "write_qe_save",
]
