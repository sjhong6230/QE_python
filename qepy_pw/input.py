"""Parser for the pw.x namelists and cards used by the scalar SCF subset.

The parser intentionally accepts ordinary Fortran namelist spelling, indexed
keys such as ``celldm(1)``, comments, commas, and all standard QE card header
delimiters. It does not attempt to reinterpret unsupported physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shlex
from typing import Any, TextIO

import numpy as np

from .constants import BOHR_PER_ANGSTROM, TWO_PI
from .errors import QEInputError, UnsupportedFeatureError
from .symmetry import (
    SymmetryOperation,
    find_space_group,
    mesh_compatible_operations,
    reduce_kpoints,
)

_NAMELISTS = {"control", "system", "electrons", "ions", "cell"}
_CARDS = {"ATOMIC_SPECIES", "ATOMIC_POSITIONS", "K_POINTS", "CELL_PARAMETERS", "OCCUPATIONS"}


@dataclass(frozen=True)
class Species:
    label: str
    mass: float
    pseudo_file: str


@dataclass(frozen=True)
class Atom:
    label: str
    position: np.ndarray  # Cartesian, bohr


@dataclass(frozen=True)
class KPoint:
    crystal: np.ndarray
    weight: float


@dataclass
class PWInput:
    control: dict[str, Any]
    system: dict[str, Any]
    electrons: dict[str, Any]
    lattice: np.ndarray  # lattice vectors in rows, bohr
    species: list[Species]
    atoms: list[Atom]
    kpoints: list[KPoint]
    kpoint_mode: str = "gamma"
    full_kpoint_count: int = 1
    symmetry_operations: tuple[SymmetryOperation, ...] = field(
        default_factory=lambda: (
            SymmetryOperation(np.eye(3, dtype=int), np.zeros(3)),
        )
    )
    source: str = "<stdin>"
    warnings: list[str] = field(default_factory=list)

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.lattice)))

    @property
    def reciprocal(self) -> np.ndarray:
        """Reciprocal vectors in rows, including 2 pi."""
        return TWO_PI * np.linalg.inv(self.lattice).T

    @property
    def pseudo_dir(self) -> Path:
        value = str(self.control.get("pseudo_dir", "."))
        path = Path(value).expanduser()
        if not path.is_absolute() and self.source != "<stdin>":
            path = Path(self.source).resolve().parent / path
        return path.resolve()


def _strip_comment(line: str) -> str:
    quote = None
    out: list[str] = []
    for char in line:
        if char in {"'", '"'}:
            quote = None if quote == char else (char if quote is None else quote)
        if char in {"!", "#"} and quote is None:
            break
        out.append(char)
    return "".join(out)


def _split_assignments(body: str) -> list[str]:
    parts, token, quote, depth = [], [], None, 0
    for char in body:
        if char in {"'", '"'}:
            quote = None if quote == char else (char if quote is None else quote)
        if quote is None:
            depth += char == "("
            depth -= char == ")"
        if (char == "," or char == "\n") and quote is None and depth == 0:
            if "".join(token).strip():
                parts.append("".join(token).strip())
            token = []
        else:
            token.append(char)
    if "".join(token).strip():
        parts.append("".join(token).strip())
    return parts


def _namelist_terminator(text: str) -> int | None:
    """Locate a slash outside a quoted Fortran string."""
    quote = None
    for index, char in enumerate(text):
        if char in {"'", '"'}:
            quote = None if quote == char else (char if quote is None else quote)
        elif char == "/" and quote is None:
            return index
    return None


def _value(text: str) -> Any:
    value = text.strip().rstrip(",")
    low = value.lower()
    if low in {".true.", "true", "t"}:
        return True
    if low in {".false.", "false", "f"}:
        return False
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    numeric = re.sub(r"(?<=\d)[dD](?=[+-]?\d)", "e", value)
    try:
        return int(numeric)
    except ValueError:
        try:
            return float(numeric)
        except ValueError:
            return value


def _parse_namelists(lines: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    result: dict[str, dict[str, Any]] = {}
    card_lines: list[str] = []
    i = 0
    while i < len(lines):
        clean = _strip_comment(lines[i]).strip()
        if not clean:
            i += 1
            continue
        if clean.startswith("&"):
            name = clean[1:].split()[0].lower()
            body = clean[len(name) + 1 :]
            terminator = _namelist_terminator(body)
            while terminator is None:
                i += 1
                if i >= len(lines):
                    raise QEInputError(f"unterminated &{name} namelist")
                body += "\n" + _strip_comment(lines[i])
                terminator = _namelist_terminator(body)
            body = body[:terminator]
            values: dict[str, Any] = {}
            for assignment in _split_assignments(body):
                if "=" not in assignment:
                    continue
                key, raw = assignment.split("=", 1)
                values[key.strip().lower()] = _value(raw)
            result[name] = values
        else:
            card_lines.append(clean)
        i += 1
    return result, card_lines


def _header(line: str) -> tuple[str, str]:
    bits = line.strip().split(maxsplit=1)
    name = bits[0].upper()
    option = bits[1].strip().strip("(){}").lower() if len(bits) > 1 else ""
    return name, option


def _lattice(system: dict[str, Any], cards: dict[str, tuple[str, list[str]]]) -> tuple[np.ndarray, float]:
    ibrav = int(system.get("ibrav", 0))
    celldm_1 = float(system.get("celldm(1)", 0.0))
    if celldm_1 <= 0.0 and "a" in system:
        celldm_1 = float(system["a"]) * BOHR_PER_ANGSTROM
    alat = celldm_1
    if ibrav == 0:
        if "CELL_PARAMETERS" not in cards:
            raise QEInputError("ibrav=0 requires CELL_PARAMETERS")
        option, rows = cards["CELL_PARAMETERS"]
        if len(rows) < 3:
            raise QEInputError("CELL_PARAMETERS requires three vectors")
        cell = np.array([[float(x) for x in row.split()[:3]] for row in rows[:3]])
        if option in {"angstrom", "ang"}:
            cell *= BOHR_PER_ANGSTROM
        elif option == "bohr":
            pass
        elif option in {"alat", ""}:
            if not alat:
                raise QEInputError("CELL_PARAMETERS alat requires celldm(1) or A")
            cell *= alat
        else:
            raise QEInputError(f"unknown CELL_PARAMETERS unit {option!r}")
        return cell, alat or float(np.linalg.norm(cell[0]))
    if not alat:
        raise QEInputError("ibrav /= 0 requires celldm(1) or A")
    def _length(name: str, celldm_index: int, default: float) -> float:
        """Read QE's Angstrom length or its dimensionless celldm ratio."""
        if name in system:
            value = float(system[name]) * BOHR_PER_ANGSTROM
        else:
            value = alat * float(system.get(f"celldm({celldm_index})", default))
        if value <= 0.0:
            raise QEInputError(f"{name.upper()} must be positive")
        return value

    def _cosine(name: str, celldm_index: int) -> float:
        value = float(system.get(name, system.get(f"celldm({celldm_index})", 0.0)))
        if not -1.0 < value < 1.0:
            raise QEInputError(f"{name} must lie strictly between -1 and 1")
        return value

    def _positive_root(value: float, description: str) -> float:
        if value <= 1.0e-14:
            raise QEInputError(f"invalid lattice parameters: {description}")
        return float(np.sqrt(value))

    if ibrav == 1:
        return np.eye(3) * alat, alat
    if ibrav == 2:
        return alat * np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]]), alat
    if ibrav == 3:
        return 0.5 * alat * np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]), alat
    if ibrav == -3:
        return 0.5 * alat * np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]]), alat
    b = _length("b", 2, 1.0)
    c = _length("c", 3, 1.0)
    if ibrav == 4:
        return np.array([[alat, 0.0, 0.0], [-0.5 * alat, 0.5 * np.sqrt(3.0) * alat, 0.0], [0.0, 0.0, c]]), alat
    if ibrav == 5:
        cosab = _cosine("cosab", 4)
        xy = _positive_root((1.0 - cosab) / 2.0, "rhombohedral cosine")
        z = _positive_root((1.0 + 2.0 * cosab) / 3.0, "rhombohedral cosine")
        return alat * np.array([[xy, -xy / np.sqrt(3.0), z], [0.0, 2.0 * xy / np.sqrt(3.0), z], [-xy, -xy / np.sqrt(3.0), z]]), alat
    if ibrav == -5:
        cosab = _cosine("cosab", 4)
        # QE's alternate rhombohedral setting has its threefold axis along x.
        xy = _positive_root((1.0 - cosab) / 2.0, "rhombohedral cosine")
        z = _positive_root((1.0 + 2.0 * cosab) / 3.0, "rhombohedral cosine")
        return alat * np.array([[z, xy, -xy / np.sqrt(3.0)], [z, 0.0, 2.0 * xy / np.sqrt(3.0)], [z, -xy, -xy / np.sqrt(3.0)]]), alat
    if ibrav == 6:
        return np.diag([alat, alat, c]), alat
    if ibrav == 7:
        return 0.5 * np.array([[alat, -alat, c], [alat, alat, c], [-alat, -alat, c]]), alat
    if ibrav == 8:
        return np.diag([alat, b, c]), alat
    if ibrav == 9:
        return np.array([[0.5 * alat, 0.5 * b, 0.0], [-0.5 * alat, 0.5 * b, 0.0], [0.0, 0.0, c]]), alat
    if ibrav == -9:
        return np.array([[0.5 * alat, -0.5 * b, 0.0], [0.5 * alat, 0.5 * b, 0.0], [0.0, 0.0, c]]), alat
    if ibrav == 91:
        return np.array([[alat, 0.0, 0.0], [0.0, 0.5 * b, -0.5 * c], [0.0, 0.5 * b, 0.5 * c]]), alat
    if ibrav == 10:
        return 0.5 * np.array([[0.0, b, c], [alat, 0.0, c], [alat, b, 0.0]]), alat
    if ibrav == 11:
        return 0.5 * np.array([[alat, b, c], [-alat, b, c], [-alat, -b, c]]), alat
    if ibrav == 12:
        cosab = _cosine("cosab", 4)
        return np.array([[alat, 0.0, 0.0], [b * cosab, b * _positive_root(1.0 - cosab**2, "cosAB"), 0.0], [0.0, 0.0, c]]), alat
    if ibrav == -12:
        cosac = _cosine("cosac", 5)
        return np.array([[alat, 0.0, 0.0], [0.0, b, 0.0], [c * cosac, 0.0, c * _positive_root(1.0 - cosac**2, "cosAC")]]), alat
    if ibrav == 13:
        cosab = _cosine("cosab", 4)
        return np.array([[0.5 * alat, 0.0, -0.5 * c], [b * cosab, b * _positive_root(1.0 - cosab**2, "cosAB"), 0.0], [0.5 * alat, 0.0, 0.5 * c]]), alat
    if ibrav == -13:
        cosac = _cosine("cosac", 5)
        return np.array([[0.5 * alat, -0.5 * b, 0.0], [0.5 * alat, 0.5 * b, 0.0], [c * cosac, 0.0, c * _positive_root(1.0 - cosac**2, "cosAC")]]), alat
    if ibrav == 14:
        cosab = _cosine("cosab", 4)
        cosac = _cosine("cosac", 5)
        cosbc = _cosine("cosbc", 6)
        sinab = _positive_root(1.0 - cosab**2, "cosAB")
        third_y = c * (cosbc - cosab * cosac) / sinab
        third_z = c * _positive_root(
            1.0 - cosac**2 - ((cosbc - cosab * cosac) / sinab) ** 2,
            "triclinic cosines",
        )
        return np.array([[alat, 0.0, 0.0], [b * cosab, b * sinab, 0.0], [c * cosac, third_y, third_z]]), alat
    raise UnsupportedFeatureError(f"ibrav={ibrav} is not a supported QE Bravais lattice")


def _collect_cards(lines: list[str], system: dict[str, Any]) -> dict[str, tuple[str, list[str]]]:
    counts = {
        "ATOMIC_SPECIES": int(system.get("ntyp", 0)),
        "ATOMIC_POSITIONS": int(system.get("nat", 0)),
        "CELL_PARAMETERS": 3,
    }
    result: dict[str, tuple[str, list[str]]] = {}
    i = 0
    while i < len(lines):
        name, option = _header(lines[i])
        if name not in _CARDS:
            i += 1
            continue
        if name == "K_POINTS":
            if option in {"gamma", "automatic"}:
                count = 0 if option == "gamma" else 1
            else:
                if i + 1 >= len(lines):
                    raise QEInputError("K_POINTS is incomplete")
                count = int(lines[i + 1].split()[0]) + 1
        else:
            count = counts.get(name, 0)
        rows = lines[i + 1 : i + 1 + count]
        if len(rows) != count:
            raise QEInputError(f"{name} card is incomplete")
        result[name] = (option, rows)
        i += count + 1
    return result


def _parse_kpoints(card: tuple[str, list[str]], lattice: np.ndarray, alat: float) -> list[KPoint]:
    option, rows = card
    reciprocal = TWO_PI * np.linalg.inv(lattice).T
    if option == "gamma":
        return [KPoint(np.zeros(3), 1.0)]
    if option == "automatic":
        values = [int(x) for x in rows[0].split()]
        if len(values) != 6 or any(n <= 0 for n in values[:3]):
            raise QEInputError("K_POINTS automatic requires nk1 nk2 nk3 sk1 sk2 sk3")
        grid, shift = values[:3], values[3:]
        points = []
        for i in range(grid[0]):
            for j in range(grid[1]):
                for k in range(grid[2]):
                    frac = np.array(
                        [(i + 0.5 * shift[0]) / grid[0],
                         (j + 0.5 * shift[1]) / grid[1],
                         (k + 0.5 * shift[2]) / grid[2]]
                    )
                    # QE represents the boundary of the first reciprocal
                    # cell as -1/2, not +1/2.  np.rint uses ties-to-even and
                    # therefore left +0.5 unchanged, selecting a different
                    # (though equivalent) irreducible representative.  That
                    # also changes the deterministic atomic+random starting
                    # wavefunctions and the under-converged first SCF energy.
                    frac -= np.floor(frac + 0.5)
                    points.append(KPoint(frac, 1.0 / np.prod(grid)))
        return points
    n = int(rows[0].split()[0])
    raw = rows[1 : n + 1]
    weights = np.array([float(row.split()[3]) for row in raw])
    weights /= weights.sum()
    points = []
    for row, weight in zip(raw, weights):
        vector = np.array([float(x) for x in row.split()[:3]])
        if option in {"crystal", "crystal_b"}:
            crystal = vector
        else:
            cart = vector * (TWO_PI / alat)
            crystal = cart @ np.linalg.inv(reciprocal)
        points.append(KPoint(crystal, float(weight)))
    return points


def _bravais_kpoint_closure(
    kpoints: list[KPoint],
    operations: tuple[SymmetryOperation, ...],
    time_reversal: bool,
) -> list[KPoint]:
    """Expand a k-point list to its Bravais-lattice star for ``nosym_evc``."""
    accumulated: dict[tuple[float, ...], tuple[np.ndarray, float]] = {}
    signs = (1.0, -1.0) if time_reversal else (1.0,)
    for point in kpoints:
        images: dict[tuple[float, ...], np.ndarray] = {}
        for operation in operations:
            rotated = point.crystal @ np.linalg.inv(operation.matrix).T
            for sign in signs:
                coordinate = sign * rotated
                coordinate -= np.floor(coordinate)
                key = tuple(np.round(coordinate, 10))
                images[key] = coordinate
        share = point.weight / len(images)
        for key, coordinate in images.items():
            prior = accumulated.get(key)
            accumulated[key] = (
                coordinate,
                share if prior is None else prior[1] + share,
            )
    expanded = [KPoint(coordinate, weight) for coordinate, weight in accumulated.values()]
    total = sum(point.weight for point in expanded)
    return [KPoint(point.crystal, point.weight / total) for point in expanded]


def read_pw_input(source: str | Path | TextIO) -> PWInput:
    if hasattr(source, "read"):
        text, source_name = source.read(), getattr(source, "name", "<stdin>")
    else:
        path = Path(source)
        text, source_name = path.read_text(encoding="utf-8"), str(path)
    lines = text.splitlines()
    namelists, card_lines = _parse_namelists(lines)
    control = namelists.get("control", {})
    system = namelists.get("system", {})
    electrons = namelists.get("electrons", {})
    if "tprfor" in control:
        raise QEInputError("unknown CONTROL variable 'tprfor'; use 'tprnfor'")
    if str(control.get("calculation", "scf")).lower() != "scf":
        raise UnsupportedFeatureError("the first port supports calculation='scf' only")
    if int(system.get("nspin", 1)) != 1 or system.get("noncolin", False):
        raise UnsupportedFeatureError("spin-polarized and noncollinear calculations are not ported")
    forbidden = {
        "lda_plus_u": "DFT+U", "lspinorb": "spin-orbit coupling",
    }
    for key, feature in forbidden.items():
        value = system.get(key)
        normalized = value.lower() if isinstance(value, str) else value
        if key in system and normalized not in {False, "", "pz", "lda"}:
            raise UnsupportedFeatureError(f"{feature} is not ported")
    if "input_dft" in system:
        input_dft = str(system["input_dft"]).lower().replace("-", "")
        if input_dft not in {"pz", "pz81", "lda", "pbe"}:
            raise UnsupportedFeatureError(
                f"input_dft={system['input_dft']!r} is not ported; "
                "use 'PZ', 'LDA', or 'PBE'"
            )
    cards = _collect_cards(card_lines, system)
    lattice, alat = _lattice(system, cards)
    if "ATOMIC_SPECIES" not in cards or "ATOMIC_POSITIONS" not in cards:
        raise QEInputError("ATOMIC_SPECIES and ATOMIC_POSITIONS are required")
    species = []
    for row in cards["ATOMIC_SPECIES"][1]:
        bits = shlex.split(row)
        species.append(Species(bits[0], float(bits[1]), bits[2]))
    if len(species) != int(system.get("ntyp", len(species))):
        raise QEInputError("ntyp does not match ATOMIC_SPECIES")
    pos_unit, rows = cards["ATOMIC_POSITIONS"]
    atoms = []
    for row in rows:
        bits = row.split()
        vector = np.array([float(x) for x in bits[1:4]])
        if pos_unit == "crystal":
            vector = vector @ lattice
        elif pos_unit in {"angstrom", "ang"}:
            vector *= BOHR_PER_ANGSTROM
        elif pos_unit == "bohr":
            pass
        elif pos_unit in {"alat", ""}:
            vector *= alat
        else:
            raise QEInputError(f"unknown ATOMIC_POSITIONS unit {pos_unit!r}")
        atoms.append(Atom(bits[0], vector))
    if len(atoms) != int(system.get("nat", len(atoms))):
        raise QEInputError("nat does not match ATOMIC_POSITIONS")
    labels = {item.label for item in species}
    unknown = sorted({atom.label for atom in atoms} - labels)
    if unknown:
        raise QEInputError(f"ATOMIC_POSITIONS contains unknown species: {', '.join(unknown)}")
    kcard = cards.get("K_POINTS", ("gamma", []))
    kpoints = _parse_kpoints(kcard, lattice, alat)
    no_time_reversal = bool(system.get("noinv", False))
    nosym_evc = bool(system.get("nosym_evc", False))
    if nosym_evc:
        # QE uses only the identity in the electronic SCF path, but completes
        # the input k-point stars under the Bravais lattice before doing so.
        bravais_operations = find_space_group(
            lattice,
            np.zeros((1, 3)),
            ["_bravais_"],
        )
        kpoints = _bravais_kpoint_closure(
            kpoints, bravais_operations, time_reversal=not no_time_reversal
        )
    full_kpoint_count = len(kpoints)
    operations = (
        SymmetryOperation(np.eye(3, dtype=int), np.zeros(3)),
    )
    # QE's ``force_symmorphic`` retains only operations without a fractional
    # translation.  It is useful when a workflow cannot represent a
    # nonsymmorphic little group, while still allowing ordinary point-group
    # reduction and density/force symmetrization.
    if not system.get("nosym", False) and not nosym_evc:
        fractional = np.array([atom.position for atom in atoms]) @ np.linalg.inv(
            lattice
        )
        operations = find_space_group(
            lattice, fractional, [atom.label for atom in atoms]
        )
        if system.get("force_symmorphic", False):
            operations = tuple(
                operation
                for operation in operations
                if np.allclose(operation.translation, 0.0, atol=1.0e-10)
            )
            if not operations:
                operations = (
                    SymmetryOperation(np.eye(3, dtype=int), np.zeros(3)),
                )
    if kcard[0] == "automatic" and not system.get("nosym", False) and not nosym_evc:
        operations = mesh_compatible_operations(
            np.array([point.crystal for point in kpoints]), operations
        )
        coordinates, weights = reduce_kpoints(
            np.array([point.crystal for point in kpoints]),
            np.array([point.weight for point in kpoints]),
            operations,
            # ``noinv`` disables QE's ordinary inversion/time-reversal
            # reduction in this scalar, nonmagnetic implementation.
            time_reversal=not no_time_reversal,
        )
        kpoints = [
            KPoint(coordinate, float(weight))
            for coordinate, weight in zip(coordinates, weights)
        ]
    return PWInput(
        control=control,
        system=system,
        electrons=electrons,
        lattice=lattice,
        species=species,
        atoms=atoms,
        kpoints=kpoints,
        kpoint_mode=kcard[0] or "tpiba",
        full_kpoint_count=full_kpoint_count,
        symmetry_operations=operations,
        source=source_name,
    )
