"""Shared QE band-structure file and XML persistence formats."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from ..constants import EV_PER_HARTREE
from ..errors import QEInputError
from .xml_data import find, findall, findtext, upstream_qe_xml
from .spin import spin_labels, validate_spin_blocks


@dataclass(frozen=True)
class BandData:
    """K points and band energies, with energies stored in eV."""

    kpoints: np.ndarray
    energies_ev: np.ndarray
    spins: np.ndarray | None = None
    noncolin: bool = False

    def __post_init__(self) -> None:
        points = np.asarray(self.kpoints, dtype=float)
        energies = np.asarray(self.energies_ev, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("kpoints must have shape (nks, 3)")
        if energies.ndim != 2 or energies.shape[0] != points.shape[0]:
            raise ValueError("energies must have shape (nks, nbnd)")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(energies)):
            raise ValueError("band data contains non-finite values")
        spins = (
            np.ones(len(points), dtype=np.int8)
            if self.spins is None
            else np.asarray(self.spins, dtype=np.int8)
        )
        if spins.shape != (len(points),) or not np.all(np.isin(spins, (1, 2))):
            raise ValueError("spins must contain one collinear-spin label per k point")
        object.__setattr__(self, "kpoints", points)
        object.__setattr__(self, "energies_ev", energies)
        object.__setattr__(self, "spins", spins)

    @property
    def nks(self) -> int:
        return self.kpoints.shape[0]

    @property
    def nbnd(self) -> int:
        return self.energies_ev.shape[1]

    @property
    def nspin(self) -> int:
        if self.noncolin:
            return 4
        return int(np.max(self.spins, initial=1))

    def select_spin(self, component: int) -> "BandData":
        if component not in {1, 2}:
            raise QEInputError("spin_component must be 1 (up) or 2 (down)")
        selected = self.spins == component
        if not np.any(selected):
            raise QEInputError("spin_component requires an LSDA calculation")
        return BandData(
            self.kpoints[selected],
            self.energies_ev[selected],
            self.spins[selected],
            self.noncolin,
        )

    def path_coordinate(self) -> np.ndarray:
        """Return QE's plottable-band path coordinate."""
        if self.nks == 0:
            return np.empty(0)
        steps = np.linalg.norm(np.diff(self.kpoints, axis=0), axis=1)
        coordinate = np.zeros(self.nks)
        if not len(steps):
            return coordinate
        typical = float(steps[0])
        for index, distance in enumerate(steps, start=1):
            if typical > 0.0 and distance > 5.0 * typical:
                coordinate[index] = coordinate[index - 1]
            else:
                coordinate[index] = coordinate[index - 1] + distance
                if distance > 1.0e-4:
                    typical = float(distance)
        return coordinate


def resolve_save_directory(prefix: str, outdir: str | None) -> Path:
    prefix = prefix.strip() or "pwscf"
    if prefix in {".", ".."} or Path(prefix).name != prefix or any(
        separator in prefix for separator in ("/", "\\")
    ):
        raise QEInputError("BANDS prefix must be a file-name prefix, not a path")
    raw = outdir if outdir and outdir.strip() else os.environ.get(
        "ESPRESSO_TMPDIR", "."
    )
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve() / f"{prefix}.save"


def read_saved_bands(prefix: str = "pwscf", outdir: str | None = None) -> BandData:
    path = resolve_save_directory(prefix, outdir) / "data-file-schema.xml"
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise QEInputError(f"cannot read saved band data {path}: {exc}") from exc
    records = findall(root, "output/band_structure/ks_energies")
    if not records:
        raise QEInputError(f"saved data {path} contains no Kohn-Sham energies")
    points: list[np.ndarray] = []
    energies: list[np.ndarray] = []
    for index, record in enumerate(records, start=1):
        point = findtext(record, "k_point")
        values = findtext(record, "eigenvalues")
        kpoint = np.fromstring(point or "", sep=" ")
        eigenvalues = np.fromstring(values or "", sep=" ")
        if kpoint.shape != (3,) or eigenvalues.size == 0:
            raise QEInputError(f"invalid saved band record at k point {index}")
        points.append(kpoint)
        energies.append(eigenvalues * EV_PER_HARTREE)
    widths = {len(values) for values in energies}
    if len(widths) != 1:
        raise QEInputError("saved k points have inconsistent band counts")
    point_array = np.vstack(points)
    if upstream_qe_xml(root):
        reciprocal = find(root, "output/basis_set/reciprocal_lattice")
        if reciprocal is None:
            raise QEInputError("upstream QE save contains no reciprocal lattice")
        reciprocal_vectors = np.vstack([
            np.fromstring(findtext(reciprocal, f"b{i}") or "", sep=" ")
            for i in range(1, 4)
        ])
        if reciprocal_vectors.shape != (3, 3):
            raise QEInputError("upstream QE save contains an invalid reciprocal lattice")
        point_array = point_array @ np.linalg.inv(reciprocal_vectors)
    lsda = (findtext(root, "output/band_structure/lsda", "false") or "").strip().lower() in {
        "true", ".true.", "t", "1"
    }
    noncolin = (
        findtext(root, "output/band_structure/noncolin", "false") or ""
    ).strip().lower() in {"true", ".true.", "t", "1"}
    spins = spin_labels(lsda, len(point_array))
    validate_spin_blocks(point_array, spins)
    return BandData(point_array, np.vstack(energies), spins, noncolin)


_HEADER = re.compile(r"nbnd\s*=\s*(\d+).*nks\s*=\s*(\d+)", re.IGNORECASE)


def write_band_file(path: str | Path, data: BandData) -> Path:
    output = Path(path)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f" &plot nbnd={data.nbnd:4d}, nks={data.nks:6d} /\n")
        for point, values in zip(data.kpoints, data.energies_ev):
            stream.write("          " + "".join(f"{value:10.6f}" for value in point) + "\n")
            for start in range(0, data.nbnd, 10):
                stream.write("".join(f"{value:9.3f}" for value in values[start:start + 10]) + "\n")
    return output


def read_band_file(path: str | Path) -> BandData:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise QEInputError(f"cannot read band file {source}: {exc}") from exc
    if not lines:
        raise QEInputError(f"empty band file {source}")
    match = _HEADER.search(lines[0])
    if match is None:
        raise QEInputError(f"invalid &plot header in {source}")
    nbnd, nks = map(int, match.groups())
    tokens = [line.split() for line in lines[1:] if line.strip()]
    cursor = 0
    points: list[list[float]] = []
    energies: list[list[float]] = []
    try:
        for _ in range(nks):
            points.append([float(value) for value in tokens[cursor][:3]])
            if len(points[-1]) != 3:
                raise ValueError
            cursor += 1
            values: list[float] = []
            while len(values) < nbnd:
                values.extend(float(value) for value in tokens[cursor])
                cursor += 1
            if len(values) != nbnd:
                raise ValueError
            energies.append(values)
    except (IndexError, ValueError) as exc:
        raise QEInputError(f"truncated or invalid band data in {source}") from exc
    return BandData(np.asarray(points), np.asarray(energies))


def write_gnuplot(path: str | Path, data: BandData, reference_ev: float = 0.0) -> Path:
    output = Path(path)
    coordinate = data.path_coordinate()
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for band in range(data.nbnd):
            for x, energy in zip(coordinate, data.energies_ev[:, band]):
                stream.write(f"{x:10.4f}{energy - reference_ev:10.4f}\n")
            stream.write("\n")
    return output
