"""Velocity/momentum matrix elements for scalar norm-conserving bands."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..constants import EV_PER_HARTREE, TWO_PI
from ..errors import QEInputError
from ..pw.save import QES_NAMESPACE
from ..upf import LocalPotential, read_upf
from .band_data import BandData


def _saved_momentum_context(
    directory: Path,
) -> tuple[
    np.ndarray,
    list[tuple[str, np.ndarray]],
    dict[str, LocalPotential],
    np.ndarray,
    float | None,
]:
    """Read lattice, atoms, UPFs, and occupations from one QE save."""
    path = directory / "data-file-schema.xml"
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise QEInputError(f"cannot read saved momentum metadata {path}: {exc}") from exc
    ns = {"qes": QES_NAMESPACE}
    lattice = np.vstack([
        np.fromstring(
            root.findtext(
                f"qes:output/qes:atomic_structure/qes:cell/qes:a{i}",
                namespaces=ns,
            ) or "",
            sep=" ",
        )
        for i in range(1, 4)
    ])
    if lattice.shape != (3, 3) or abs(np.linalg.det(lattice)) < 1.0e-14:
        raise QEInputError("saved momentum metadata contains an invalid lattice")
    atom_elements = root.findall(
        "qes:output/qes:atomic_structure/qes:atomic_positions/qes:atom", ns
    )
    atoms = [
        (element.attrib.get("name", ""), np.fromstring(element.text or "", sep=" "))
        for element in atom_elements
    ]
    if any(not label or position.shape != (3,) for label, position in atoms):
        raise QEInputError("saved momentum metadata contains an invalid atom")
    species_elements = root.findall(
        "qes:output/qes:atomic_species/qes:species", ns
    )
    pseudo_files = {
        element.attrib.get("name", ""): element.findtext(
            "qes:pseudo_file", namespaces=ns
        )
        for element in species_elements
    }
    pseudos: dict[str, LocalPotential] = {}
    for label in {label for label, _position in atoms}:
        filename = pseudo_files.get(label)
        if not filename:
            raise QEInputError(f"saved momentum metadata has no UPF for species {label}")
        pseudos[label] = read_upf(directory / Path(filename).name)
    records = root.findall(
        "qes:output/qes:band_structure/qes:ks_energies", ns
    )
    occupations = []
    for index, record in enumerate(records, start=1):
        values = np.fromstring(
            record.findtext("qes:occupations", namespaces=ns) or "", sep=" "
        )
        if values.size == 0:
            raise QEInputError(f"saved occupations are missing at k point {index}")
        occupations.append(values)
    widths = {len(values) for values in occupations}
    if len(widths) != 1:
        raise QEInputError("saved k points have inconsistent occupation counts")
    fermi_text = root.findtext(
        "qes:output/qes:band_structure/qes:fermi_energy", namespaces=ns
    )
    fermi_ev = (
        float(fermi_text) * EV_PER_HARTREE
        if fermi_text is not None
        else None
    )
    return lattice, atoms, pseudos, np.vstack(occupations), fermi_ev


def momentum_matrices(
    data: BandData,
    wavefunctions: list[tuple[np.ndarray, np.ndarray]],
    lattice: np.ndarray,
    atoms: list[tuple[str, np.ndarray]],
    pseudos: dict[str, LocalPotential],
) -> np.ndarray:
    """Return ``<m,k|m*v|n,k>`` in atomic units.

    The result shape is ``(nks, 3, nbnd, nbnd)``. The local/kinetic term is
    diagonal in plane waves. The norm-conserving nonlocal commutator is
    evaluated as derivatives of the separable projectors, without forming a
    dense plane-wave operator.
    """
    lattice = np.asarray(lattice, dtype=float)
    reciprocal = TWO_PI * np.linalg.inv(lattice).T
    volume = abs(float(np.linalg.det(lattice)))
    if len(wavefunctions) != data.nks:
        raise QEInputError("wavefunctions do not cover every momentum k point")
    result = np.zeros((data.nks, 3, data.nbnd, data.nbnd), dtype=np.complex128)
    atoms_by_label: dict[str, list[np.ndarray]] = {}
    for label, position in atoms:
        atoms_by_label.setdefault(label, []).append(np.asarray(position, dtype=float))
    for ik, (point, (miller, coefficients)) in enumerate(
        zip(data.kpoints, wavefunctions)
    ):
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if coefficients.shape != (len(miller), data.nbnd):
            raise QEInputError(f"wavefunctions at k point {ik + 1} have invalid shape")
        gk = (np.asarray(miller, dtype=float) + point) @ reciprocal
        for axis in range(3):
            result[ik, axis] = coefficients.conj().T @ (
                gk[:, axis, None] * coefficients
            )
        for label, positions in atoms_by_label.items():
            pseudo = pseudos[label]
            beta, coupling, gradient = pseudo.projector_basis_with_gradient(
                gk, volume
            )
            if beta.shape[1] == 0:
                continue
            for position in positions:
                phase = np.exp(-1j * (gk @ position))
                atom_beta = beta * phase[:, None]
                beta_overlap = atom_beta.conj().T @ coefficients
                for axis in range(3):
                    # Atomic phase derivatives cancel between the two sides
                    # of [V_nl,r]; only the q derivative of the radial/angular
                    # projector remains.
                    atom_gradient = gradient[:, :, axis] * phase[:, None]
                    gradient_overlap = atom_gradient.conj().T @ coefficients
                    result[ik, axis] += (
                        gradient_overlap.conj().T @ coupling @ beta_overlap
                        + beta_overlap.conj().T @ coupling @ gradient_overlap
                    )
        # Suppress harmless anti-Hermitian roundoff from independent BLAS
        # contractions. The velocity operator is Hermitian.
        result[ik] = 0.5 * (
            result[ik] + result[ik].conj().transpose(0, 2, 1)
        )
    return result


def write_p_avg(
    path: str | Path,
    data: BandData,
    matrices: np.ndarray,
    occupations: np.ndarray,
    firstk: int = 0,
    lastk: int = 10_000_000,
    fermi_energy_ev: float | None = None,
) -> Path:
    """Write QE's ``&p_mat`` conduction--valence squared matrix elements."""
    output = Path(path)
    matrices = np.asarray(matrices)
    occupations = np.asarray(occupations, dtype=float)
    if matrices.shape != (data.nks, 3, data.nbnd, data.nbnd):
        raise QEInputError("momentum matrices have an invalid shape")
    if occupations.shape != (data.nks, data.nbnd):
        raise QEInputError("momentum occupations have an invalid shape")
    first = max(1, int(firstk))
    last = min(data.nks, int(lastk))
    if last < first:
        raise QEInputError("empty k-point range for momentum matrix elements")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f" &p_mat nbnd={data.nbnd:4d}, nks={last - first + 1:4d} /\n")
        for ik in range(first - 1, last):
            occupied_indices = np.flatnonzero(
                data.energies_ev[ik] <= fermi_energy_ev
                if fermi_energy_ev is not None
                else occupations[ik] > 1.0e-8
            )
            nbnd_occ = int(occupied_indices[-1]) + 1 if occupied_indices.size else 0
            point = data.kpoints[ik]
            stream.write(
                "          " + "".join(f"{value:10.6f}" for value in point)
                + f"{nbnd_occ:7d}\n"
            )
            for axis in range(3):
                stream.write(f"{axis + 1:3d}\n")
                for conduction in range(nbnd_occ, data.nbnd):
                    values = np.abs(matrices[ik, axis, conduction, :nbnd_occ]) ** 2
                    for start in range(0, nbnd_occ, 5):
                        stream.write(
                            "".join(f"{value:15.8f}" for value in values[start:start + 5])
                            + "\n"
                        )
    return output


def compute_and_write_p_avg(
    path: str | Path,
    data: BandData,
    wavefunctions: list[tuple[np.ndarray, np.ndarray]],
    directory: Path,
    firstk: int = 0,
    lastk: int = 10_000_000,
) -> Path:
    lattice, atoms, pseudos, occupations, fermi_ev = _saved_momentum_context(
        directory
    )
    if occupations.shape != (data.nks, data.nbnd):
        raise QEInputError("saved occupations do not match the band data")
    matrices = momentum_matrices(data, wavefunctions, lattice, atoms, pseudos)
    return write_p_avg(
        path,
        data,
        matrices,
        occupations,
        firstk,
        lastk,
        fermi_energy_ev=fermi_ev,
    )
