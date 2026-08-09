"""Total electronic density of states corresponding to QE ``dos.x``."""

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
from ..constants import EV_PER_HARTREE
from ..errors import QEInputError, emit_qe_error
from ..occupations import _linear_tetra_moments_block, _tetrahedra, smearing_order, wgauss
from ..pw.save import QES_NAMESPACE
from ..qe_format import format_qe_closing, format_qe_opening, format_qe_timing
from ..version import __version__
from .band_data import resolve_save_directory
from .namelist import parse_namelist


@dataclass(frozen=True)
class DOSData:
    eigenvalues_ev: np.ndarray
    weights: np.ndarray
    fermi_ev: float | None
    occupations_kind: str
    smearing: str
    degauss_ry: float
    grid: tuple[int, int, int] | None
    mapping: np.ndarray | None
    reciprocal: np.ndarray | None


def read_saved_dos(prefix: str, outdir: str | None) -> DOSData:
    directory = resolve_save_directory(prefix, outdir)
    root = ET.parse(directory / "data-file-schema.xml").getroot()
    ns = {"qes": QES_NAMESPACE}
    bands = root.find("qes:output/qes:band_structure", ns)
    if bands is None:
        raise QEInputError("saved data contains no band structure")
    records = bands.findall("qes:ks_energies", ns)
    energies, weights = [], []
    for record in records:
        energies.append(np.fromstring(record.findtext("qes:eigenvalues", default="", namespaces=ns), sep=" "))
        point = record.find("qes:k_point", ns)
        weights.append(float(point.attrib.get("weight", "nan")) if point is not None else np.nan)
    if not energies or len({len(row) for row in energies}) != 1:
        raise QEInputError("saved DOS eigenvalues are missing or inconsistent")
    weights_array = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights_array)) or np.sum(weights_array) <= 0:
        raise QEInputError("saved DOS k-point weights are invalid")
    weights_array /= np.sum(weights_array)
    fermi = bands.findtext("qes:fermi_energy", namespaces=ns)
    pack = bands.find("qes:monkhorst_pack", ns)
    grid = None if pack is None else tuple(int(pack.attrib[f"nk{i}"]) for i in range(1, 4))
    mapping_text = bands.findtext("qes:full_to_irreducible", namespaces=ns)
    mapping = None if mapping_text is None else np.fromstring(mapping_text, sep=" ", dtype=np.int32)
    reciprocal_element = root.find("qes:output/qes:basis_set/qes:reciprocal_lattice", ns)
    reciprocal = None
    if reciprocal_element is not None:
        reciprocal = np.vstack([
            np.fromstring(reciprocal_element.findtext(f"qes:b{i}", default="", namespaces=ns), sep=" ")
            for i in range(1, 4)
        ])
    return DOSData(
        np.vstack(energies) * EV_PER_HARTREE,
        weights_array,
        None if fermi is None else float(fermi) * EV_PER_HARTREE,
        bands.findtext("qes:occupations_kind", default="fixed", namespaces=ns),
        bands.findtext("qes:smearing", default="gaussian", namespaces=ns),
        float(bands.findtext("qes:degauss", default="0", namespaces=ns)),
        grid, mapping, reciprocal,
    )


def smearing_dos(energies_ev, weights, grid_ev, width_ev, ngauss):
    if ngauss not in {0, 1, -1, -99}:
        raise QEInputError("ngauss must be 0, 1, -1, or -99")
    if width_ev <= 0:
        raise QEInputError("smearing DOS requires positive degauss")
    x = (grid_ev[:, None, None] - np.asarray(energies_ev)[None, :, :]) / width_ev
    step = 1.0e-5
    kernel = (wgauss(x + step, ngauss) - wgauss(x - step, ngauss)) / (2 * step * width_ev)
    return 2.0 * np.einsum("ekb,k->e", kernel, weights)


def tetrahedron_dos(data: DOSData, grid_ev: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    if data.grid is None or data.mapping is None or data.reciprocal is None:
        raise QEInputError("tetrahedron DOS requires saved automatic k-grid metadata")
    normalized = method.lower().replace("-", "_")
    optimized_connectivity = normalized in {"tetrahedra_lin", "tetrahedra_opt"}
    optimized = normalized == "tetrahedra_opt"
    tetra, interpolation = _tetrahedra(data.grid, data.mapping, data.reciprocal, optimized_connectivity, optimized)
    point_energies = data.eigenvalues_ev[tetra]
    effective = np.einsum("cp,tpb->tbc", interpolation, point_energies)
    sorted_e = np.sort(effective, axis=-1)
    dos, integrated = [], []
    ntetra = len(tetra)
    for energy in grid_ev:
        moments, density = _linear_tetra_moments_block(sorted_e, float(energy))
        dos.append(2.0 * float(np.sum(density)) / ntetra)
        integrated.append(2.0 * float(np.sum(moments)) / ntetra)
    return np.asarray(dos), np.asarray(integrated)


def run_dos(
    options: dict[str, object], stdout: TextIO | None = None
) -> Path:
    allowed = {"prefix", "outdir", "bz_sum", "ngauss", "degauss", "emin", "emax", "deltae", "fildos"}
    unknown = set(options) - allowed
    if unknown:
        raise QEInputError(f"unknown &DOS variable {sorted(unknown)[0]!r}")
    prefix = str(options.get("prefix", "pwscf"))
    outdir = str(options["outdir"]) if "outdir" in options else None
    if stdout is not None:
        directory = resolve_save_directory(prefix, outdir)
        print(
            f"\n     Reading xml data from directory:\n\n"
            f"     {directory}{os.sep}",
            file=stdout,
        )
    data = read_saved_dos(prefix, outdir)
    delta = float(options.get("deltae", 0.01))
    if delta <= 0:
        raise QEInputError("DeltaE must be positive")
    input_degauss = float(options.get("degauss", 0.0))
    explicit_degauss = input_degauss != 0.0
    method = str(options.get("bz_sum", "")).strip().lower()
    if not method:
        method = data.occupations_kind.lower()
    tetra = method in {"tetrahedra", "tetrahedra_lin", "tetrahedra_opt"} and not explicit_degauss
    if explicit_degauss:
        ngauss = int(options.get("ngauss", 0))
        degauss_ry = input_degauss
    elif data.degauss_ry > 0.0:
        ngauss = smearing_order(data.smearing)
        degauss_ry = data.degauss_ry
    else:
        ngauss = 0
        degauss_ry = delta / (0.5 * EV_PER_HARTREE)
    if stdout is not None:
        if tetra:
            messages = {
                "tetrahedra": "Tetrahedra used",
                "tetrahedra_lin": "Linear tetrahedron method is used",
                "tetrahedra_opt": "Optimized tetrahedron method used",
            }
            print(f"\n     {messages[method]}\n", file=stdout)
        else:
            if explicit_degauss:
                source = "read from input"
            elif data.degauss_ry > 0.0:
                source = "read from file"
            else:
                source = "default values"
            print(
                f"\n     Gaussian broadening ({source}): "
                f"ngauss,degauss={ngauss:4d}{degauss_ry:12.6f}\n",
                file=stdout,
            )
    width_ev = 0.5 * degauss_ry * EV_PER_HARTREE
    minimum, maximum = float(np.min(data.eigenvalues_ev)), float(np.max(data.eigenvalues_ev))
    emin = float(options.get("emin", minimum - (3 * width_ev if not tetra else 0)))
    emax = float(options.get("emax", maximum + (3 * width_ev if not tetra else 0)))
    if emax < emin:
        raise QEInputError("Emax must be greater than or equal to Emin")
    # QE uses NINT((Emax-Emin)/DeltaE + 0.500001).  Values here are
    # nonnegative, so floor(x + 1/2) reproduces Fortran NINT.
    count = int(np.floor((emax - emin) / delta + 1.000001))
    grid_ev = emin + np.arange(count) * delta
    if tetra:
        density, integrated = tetrahedron_dos(data, grid_ev, method)
    else:
        density = smearing_dos(data.eigenvalues_ev, data.weights, grid_ev, width_ev, ngauss)
        integrated = np.cumsum(density) * delta
    output = Path(str(options.get("fildos", f"{prefix}.dos")))
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        fermi = "" if data.fermi_ev is None else f" EFermi = {data.fermi_ev:8.3f} eV"
        stream.write(f"#  E (eV)   dos(E)     Int dos(E){fermi}\n")
        for energy, value, total in zip(grid_ev, density, integrated):
            stream.write(f"{energy:8.3f}{value:12.4e}{total:12.4e}\n")
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dos.py")
    add_input_file_argument(parser)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        print(format_qe_opening("DOS-PY", __version__), end="")
        text = Path(args.input_file).read_text(encoding="utf-8") if args.input_file else sys.stdin.read()
        run_dos(parse_namelist(text, "dos"), stdout=sys.stdout)
        print(
            format_qe_timing(
                "DOS",
                time.process_time() - cpu_started,
                time.perf_counter() - started,
            )
        )
        print(format_qe_closing(), end="")
        return 0
    except (QEInputError, OSError, ValueError, ET.ParseError) as exc:
        emit_qe_error(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
