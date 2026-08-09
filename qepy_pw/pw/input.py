"""Parser for the pw.x namelists and cards used by the scalar SCF subset.

The parser intentionally accepts ordinary Fortran namelist spelling, indexed
keys such as ``celldm(1)``, comments, commas, and all standard QE card header
delimiters. It does not attempt to reinterpret unsupported physics.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
import shlex
from typing import Any, TextIO
import warnings

import numpy as np

from ..constants import BOHR_PER_ANGSTROM, TWO_PI
from ..errors import QEInputError, QEWarning, UnsupportedFeatureError, not_implemented
from ..qe_input_schema import QE_NAMELIST_VARIABLES
from ..xc import canonical_xc_name
from ..symmetry import (
    SymmetryOperation,
    find_space_group,
    mesh_compatible_operations,
    reduce_kpoints,
)

_NAMELISTS = {"control", "system", "electrons", "ions", "cell"}
_CARDS = {"ATOMIC_SPECIES", "ATOMIC_POSITIONS", "K_POINTS", "CELL_PARAMETERS", "OCCUPATIONS"}

# Variables that affect the implemented scalar-SCF path.  Rejecting every
# other explicitly supplied variable is intentional: silently accepting a QE
# option whose physics is absent is substantially more dangerous than failing
# with a precise "not implemented" diagnostic.
_IMPLEMENTED_NAMELIST_VARIABLES = {
    "control": {
        "title", "calculation", "verbosity", "restart_mode", "tstress",
        "tprnfor", "outdir", "wfcdir", "prefix", "pseudo_dir", "disk_io", "iprint",
    },
    "system": {
        "ibrav", "celldm", "a", "b", "c", "cosab", "cosac", "cosbc",
        "nat", "ntyp", "nbnd", "ecutwfc", "ecutrho", "nosym",
        "nosym_evc", "noinv", "use_all_frac", "force_symmorphic",
        "starting_charge", "occupations", "degauss", "smearing", "input_dft",
        "nspin", "noncolin", "lda_plus_u", "lspinorb", "tot_charge",
        "space_group", "uniqueb", "origin_choice", "rhombohedral",
    },
    "electrons": {
        "electron_maxstep", "mixing_mode", "mixing_beta", "mixing_ndim",
        "mixing_pulay_frequency", "diagonalization", "startingpot",
        "startingwfc", "conv_thr", "diago_thr_init", "diago_cg_maxiter",
        "diago_david_ndim", "diago_rmm_ndim", "diago_rmm_conv",
        "diago_gs_nblock", "diago_full_acc",
    },
    "ions": set(),
    "cell": set(),
}


def _namelist_key_base(key: str) -> str:
    return key.split("(", 1)[0].strip().lower()


def _reject_unimplemented_variables(
    namelists: dict[str, dict[str, Any]],
) -> None:
    for name, values in namelists.items():
        implemented = _IMPLEMENTED_NAMELIST_VARIABLES[name]
        for key in values:
            base = _namelist_key_base(key)
            if base in implemented:
                continue
            if name in QE_NAMELIST_VARIABLES and base not in QE_NAMELIST_VARIABLES[name]:
                raise QEInputError(
                    f"bad line in namelist &{name.upper()}: unknown variable '{key}'",
                    routine="read_namelists",
                )
            else:
                raise UnsupportedFeatureError(
                    not_implemented(f"&{name.upper()} variable '{key}'"),
                    routine="iosys",
                )


def _input_number(
    values: dict[str, Any],
    key: str,
    default: Any,
    conversion: type[float] | type[int],
    namelist: str,
) -> float | int:
    try:
        return conversion(values.get(key, default))
    except (TypeError, ValueError, OverflowError) as exc:
        raise QEInputError(
            f"bad line in namelist &{namelist.upper()}: invalid value for '{key}'",
            routine="read_namelists",
        ) from exc


@dataclass(frozen=True)
class Species:
    label: str
    mass: float
    pseudo_file: str


@dataclass(frozen=True)
class Atom:
    label: str
    position: np.ndarray  # Cartesian, bohr
    if_pos: tuple[int, int, int] = (1, 1, 1)


@dataclass(frozen=True)
class KPoint:
    crystal: np.ndarray
    weight: float
    label: str | None = None


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
    full_kpoints: tuple[KPoint, ...] = ()
    full_to_irreducible: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.int32)
    )
    kpoint_grid: tuple[int, int, int] | None = None
    kpoint_shift: tuple[int, int, int] | None = None
    symmetry_operations: tuple[SymmetryOperation, ...] = field(
        default_factory=lambda: (
            SymmetryOperation(np.eye(3, dtype=int), np.zeros(3)),
        )
    )
    source: str = "<stdin>"
    warnings: list[QEWarning] = field(default_factory=list)

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
    # Fortran accepts both ``1d-8`` and ``1.d-8``.  The latter occurs
    # throughout QE's own test-suite, so do not require the character
    # immediately before the exponent marker to be a digit.
    numeric = re.sub(r"([0-9.])[dD]([+-]?\d)", r"\1e\2", value)
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
            if name not in _NAMELISTS:
                raise UnsupportedFeatureError(
                    not_implemented(f"namelist &{name.upper()}"),
                    routine="read_namelists",
                )
            if name in result:
                raise QEInputError(
                    f"two occurrences of namelist &{name.upper()}",
                    routine="read_namelists",
                )
            body = clean[len(name) + 1 :]
            terminator = _namelist_terminator(body)
            while terminator is None:
                i += 1
                if i >= len(lines):
                    raise QEInputError(
                        f"unterminated &{name} namelist",
                        routine="read_namelists",
                    )
                body += "\n" + _strip_comment(lines[i])
                terminator = _namelist_terminator(body)
            body = body[:terminator]
            values: dict[str, Any] = {}
            for assignment in _split_assignments(body):
                if "=" not in assignment:
                    raise QEInputError(
                        f"bad line in namelist &{name.upper()}: {assignment}",
                        routine="read_namelists",
                    )
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


def _hexagonal_to_rhombohedral(coordinates: np.ndarray) -> np.ndarray:
    """Convert ITA obverse hexagonal coordinates to QE ``ibrav=5`` axes.

    This is the trigonal ``ccord`` transformation used after QE expands
    ``crystal_sg`` positions for ``rhombohedral=.false.``.
    """
    x, y, z = np.asarray(coordinates, dtype=float)
    converted = np.asarray((x - y - z, y - z, -z - x))
    return converted - np.floor(converted)


def _coordinate_expression(token: str) -> float:
    """Evaluate the deliberately small expression language accepted by QE.

    QE permits only arithmetic operators in atomic coordinates.  Parsing an
    AST instead of calling ``eval`` keeps names, calls, attributes, and every
    other Python construct out of input files.
    """
    if token.startswith("+"):
        raise QEInputError(
            f"atomic coordinate expression {token!r} must not start with '+'"
        )
    normalized = re.sub(
        r"(?<=[0-9.])[dD](?=[+-]?\d)", "e", token
    ).replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise QEInputError(f"invalid atomic coordinate expression {token!r}") from exc

    binary = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.Pow: lambda left, right: left**right,
    }

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return float(binary[type(node.op)](evaluate(node.left), evaluate(node.right)))
        raise QEInputError(f"invalid atomic coordinate expression {token!r}")

    try:
        value = evaluate(tree)
    except (ArithmeticError, OverflowError, ValueError) as exc:
        raise QEInputError(f"invalid atomic coordinate expression {token!r}") from exc
    if not np.isfinite(value):
        raise QEInputError(f"atomic coordinate expression {token!r} is not finite")
    return value


def _position_constraints(tokens: list[str]) -> tuple[int, int, int]:
    if not tokens:
        return (1, 1, 1)
    if len(tokens) != 3:
        raise QEInputError("ATOMIC_POSITIONS constraints require exactly three if_pos values")
    try:
        values = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise QEInputError("ATOMIC_POSITIONS if_pos values must be 0 or 1") from exc
    if any(str(value) != token.strip() or value not in {0, 1} for token, value in zip(tokens, values)):
        raise QEInputError("ATOMIC_POSITIONS if_pos values must be 0 or 1")
    x, y, z = values
    return x, y, z


@lru_cache(maxsize=230 * 4)
def _space_group_hall_number(
    number: int,
    uniqueb: bool,
    origin_choice: int,
    rhombohedral: bool,
) -> int:
    """Choose the spglib Hall setting corresponding to QE input switches."""
    if not 1 <= number <= 230:
        raise QEInputError("space_group must be between 1 and 230")
    if origin_choice not in {1, 2}:
        raise QEInputError("origin_choice must be 1 or 2")
    try:
        import spglib
    except ImportError as exc:  # pragma: no cover - dependency installation failure
        raise UnsupportedFeatureError(
            "ATOMIC_POSITIONS crystal_sg requires the 'spglib' package"
        ) from exc

    candidates = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for hall_number in range(1, 531):
            kind = spglib.get_spacegroup_type(hall_number)
            if kind is not None and int(kind.number) == number:
                candidates.append((hall_number, str(kind.choice)))

    if 143 <= number <= 167 and any(choice in {"H", "R"} for _, choice in candidates):
        wanted = "R" if rhombohedral else "H"
        return next(hall for hall, choice in candidates if choice == wanted)

    if 3 <= number <= 15:
        axis = "b" if uniqueb else "c"
        primary = [
            hall for hall, choice in candidates
            if choice.startswith(axis) and choice.endswith("1")
        ]
        if primary:
            return primary[0]

    wanted_origin = str(origin_choice)
    exact = [hall for hall, choice in candidates if choice == wanted_origin]
    if exact:
        return exact[0]
    if origin_choice == 2:
        raise QEInputError(
            f"space_group={number} does not provide a second origin choice"
        )
    default = [hall for hall, choice in candidates if choice == ""]
    if default:
        return default[0]
    if candidates:
        return candidates[0][0]
    raise QEInputError(f"spglib has no Hall setting for space_group={number}")


def _space_group_ibrav(number: int, hall_number: int, uniqueb: bool) -> int:
    """Return QE's primitive Bravais convention used by ``wyckoff.ccord``."""
    try:
        import spglib
    except ImportError as exc:  # pragma: no cover - dependency installation failure
        raise UnsupportedFeatureError(
            "ATOMIC_POSITIONS crystal_sg requires the 'spglib' package"
        ) from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        group_type = spglib.get_spacegroup_type(hall_number)
    symbol = str(group_type.international_short)
    centering = symbol[0].upper()
    if number <= 2:
        return 14
    if number <= 15:
        base = 12 if centering == "P" else 13
        return -base if uniqueb else base
    if number <= 74:
        return {"P": 8, "C": 9, "A": 91, "B": 91, "F": 10, "I": 11}[centering]
    if number <= 142:
        return 7 if centering == "I" else 6
    if number <= 194:
        return 5 if centering == "R" else 4
    return {"P": 1, "F": 2, "I": 3}[centering]


def _qe_primitive_coordinates(coordinates: np.ndarray, ibrav: int) -> np.ndarray:
    """Apply QE ``Modules/wyckoff.f90:ccord`` to conventional coordinates."""
    values = np.asarray(coordinates, dtype=float)
    x, y, z = values.T
    if ibrav == 2:
        converted = np.column_stack((-x - y + z, x + y + z, -x - z + y))
    elif ibrav in {3, 11}:
        converted = np.column_stack((x + z, y - x, z - y))
    elif ibrav == 7:
        converted = np.column_stack((x - y, y + z, z - x))
    elif ibrav == 9:
        converted = np.column_stack((x + y, y - x, z))
    elif ibrav == 91:
        converted = np.column_stack((x, y + z, y - z))
    elif ibrav == 10:
        converted = np.column_stack((x - y + z, x + y - z, -x + y + z))
    elif ibrav == 13:
        converted = np.column_stack((x - z, y, z + x))
    elif ibrav == -13:
        converted = np.column_stack((x + y, y - x, z))
    else:
        converted = values.copy()
    return converted - np.floor(converted)


def _unique_fractional(coordinates: np.ndarray, tolerance: float = 1.0e-7) -> np.ndarray:
    unique: list[np.ndarray] = []
    for coordinate in np.asarray(coordinates, dtype=float):
        wrapped = coordinate - np.floor(coordinate)
        if not any(
            np.all(np.abs((wrapped - prior) - np.rint(wrapped - prior)) < tolerance)
            for prior in unique
        ):
            unique.append(wrapped)
    return np.asarray(unique)


def _wyckoff_coordinates(
    hall_number: int, label: str, free_coordinates: list[float]
) -> np.ndarray:
    try:
        from pyxtal.symmetry import Group
    except ImportError as exc:  # pragma: no cover - dependency installation failure
        raise UnsupportedFeatureError(
            "Wyckoff labels in ATOMIC_POSITIONS crystal_sg require the 'pyxtal' package"
        ) from exc
    group = Group(hall_number, use_hall=True)
    wyckoff = group.get_wyckoff_position(label.lower())
    if wyckoff is None:
        raise QEInputError(
            f"Wyckoff position {label!r} is not valid for space_group={group.number}"
        )
    required = int(wyckoff.get_dof())
    if len(free_coordinates) != required:
        raise QEInputError(
            f"Wyckoff position {label!r} requires {required} free coordinate(s), "
            f"not {len(free_coordinates)}"
        )
    seed = wyckoff.get_position_from_free_xyzs(free_coordinates)
    return np.asarray(wyckoff.get_all_positions(seed), dtype=float)


def _expand_crystal_sg(
    rows: list[str], system: dict[str, Any], lattice: np.ndarray
) -> list[Atom]:
    if "space_group" not in system or int(system["space_group"]) == 0:
        raise QEInputError("ATOMIC_POSITIONS crystal_sg requires space_group")
    number = int(system["space_group"])
    uniqueb = bool(system.get("uniqueb", False))
    origin_choice = int(system.get("origin_choice", 1))
    rhombohedral = bool(system.get("rhombohedral", True))
    hall_number = _space_group_hall_number(
        number, uniqueb, origin_choice, rhombohedral
    )
    qe_ibrav = _space_group_ibrav(number, hall_number, uniqueb)
    requested_ibrav = int(system.get("ibrav", 0))
    if requested_ibrav != 0 and requested_ibrav != qe_ibrav:
        raise QEInputError(
            f"space_group={number} requires QE ibrav={qe_ibrav}, "
            f"not ibrav={requested_ibrav}"
        )
    try:
        import spglib
    except ImportError as exc:  # pragma: no cover - dependency installation failure
        raise UnsupportedFeatureError(
            "ATOMIC_POSITIONS crystal_sg requires the 'spglib' package"
        ) from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        symmetry = spglib.get_symmetry_from_database(hall_number)
    rotations = np.asarray(symmetry["rotations"], dtype=float)
    translations = np.asarray(symmetry["translations"], dtype=float)

    atoms: list[Atom] = []
    for row in rows:
        bits = row.split()
        if len(bits) < 2:
            raise QEInputError(f"invalid ATOMIC_POSITIONS crystal_sg row: {row!r}")
        label = bits[0]
        wyckoff_match = re.fullmatch(r"(\d+)([A-Za-z])", bits[1])
        if wyckoff_match:
            try:
                from pyxtal.symmetry import Group
            except ImportError as exc:  # pragma: no cover
                raise UnsupportedFeatureError(
                    "Wyckoff labels in ATOMIC_POSITIONS crystal_sg require pyxtal"
                ) from exc
            wyckoff = Group(hall_number, use_hall=True).get_wyckoff_position(bits[1].lower())
            if wyckoff is None:
                raise QEInputError(
                    f"Wyckoff position {bits[1]!r} is invalid for space_group={number}"
                )
            required = int(wyckoff.get_dof())
            payload = bits[2:]
            if len(payload) not in {required, required + 3}:
                raise QEInputError(
                    f"Wyckoff position {bits[1]!r} requires {required} free coordinate(s) "
                    "followed optionally by three if_pos values"
                )
            free = [_coordinate_expression(token) for token in payload[:required]]
            if_pos = _position_constraints(payload[required:])
            conventional = _wyckoff_coordinates(hall_number, bits[1], free)
        else:
            if len(bits) not in {4, 7}:
                raise QEInputError(
                    "ATOMIC_POSITIONS crystal_sg coordinates require x y z "
                    "followed optionally by three if_pos values"
                )
            seed = np.asarray([_coordinate_expression(token) for token in bits[1:4]])
            if_pos = _position_constraints(bits[4:])
            conventional = (
                rotations @ seed[:, np.newaxis]
            ).squeeze(axis=2) + translations

        if qe_ibrav == 5 and not rhombohedral:
            primitive = np.asarray(
                [_hexagonal_to_rhombohedral(value) for value in conventional]
            )
        else:
            primitive = _qe_primitive_coordinates(conventional, qe_ibrav)
        for coordinate in _unique_fractional(primitive):
            atoms.append(Atom(label, coordinate @ lattice, if_pos))
    return atoms


def _lattice(system: dict[str, Any], cards: dict[str, tuple[str, list[str]]]) -> tuple[np.ndarray, float]:
    legacy_parameters = sorted(
        key for key in system if re.fullmatch(r"celldm\([1-6]\)", key)
    )
    modern_parameter_names = {"a", "b", "c", "cosab", "cosac", "cosbc"}
    modern_parameters = sorted(modern_parameter_names.intersection(system))
    if legacy_parameters and modern_parameters:
        raise QEInputError(
            "celldm(i) and A/B/C/cosAB/cosAC/cosBC cannot be used together"
        )
    ibrav = int(system.get("ibrav", 0))
    celldm_1 = float(system.get("celldm(1)", 0.0))
    if celldm_1 <= 0.0 and "a" in system:
        celldm_1 = float(system["a"]) * BOHR_PER_ANGSTROM
    alat = celldm_1
    if ibrav == 0:
        if "CELL_PARAMETERS" not in cards:
            raise QEInputError(
                "ibrav=0 requires CELL_PARAMETERS", routine="cell_base_init"
            )
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
        raise QEInputError(
            "invalid lattice parameters ( celldm or a )", routine="iosys"
        )
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
        if name in result:
            routine = {
                "ATOMIC_SPECIES": "card_atomic_species",
                "ATOMIC_POSITIONS": "card_atomic_positions",
                "K_POINTS": "card_kpoints",
                "CELL_PARAMETERS": "card_cell_parameters",
                "OCCUPATIONS": "card_occupations",
            }[name]
            raise QEInputError("two occurrences", routine=routine)
        if name in {"ATOMIC_SPECIES", "ATOMIC_POSITIONS"}:
            # Read the actual card through the next recognized card header.
            # Slicing by ntyp/nat would hide surplus rows and, when the
            # declaration is too large, consume the following card as data.
            end = i + 1
            while end < len(lines) and _header(lines[end])[0] not in _CARDS:
                end += 1
            result[name] = (option, lines[i + 1 : end])
            i = end
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


def _kpoint_real(token: str, description: str) -> float:
    normalized = re.sub(r"(?<=[0-9.])[dD](?=[+-]?\d)", "e", token)
    try:
        value = float(normalized)
    except ValueError as exc:
        raise QEInputError(f"invalid {description} value {token!r}") from exc
    if not np.isfinite(value):
        raise QEInputError(f"{description} values must be finite")
    return value


def _fortran_nint(value: float) -> int:
    """Fortran NINT, including its half-away-from-zero tie behavior."""
    return (
        int(np.floor(value + 0.5))
        if value >= 0.0
        else int(np.ceil(value - 0.5))
    )


def _kpoint_to_crystal(
    vector: np.ndarray,
    coordinate_type: str,
    reciprocal: np.ndarray,
    alat: float,
) -> np.ndarray:
    if coordinate_type == "crystal":
        return np.asarray(vector, dtype=float)
    cartesian = np.asarray(vector, dtype=float) * (TWO_PI / alat)
    return cartesian @ np.linalg.inv(reciprocal)


def _special_kpoint(
    label: str,
    lattice: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Resolve QE's default SC high-symmetry labels in the input cell basis."""
    try:
        from ase.cell import Cell
    except ImportError as exc:  # pragma: no cover - dependency installation failure
        raise UnsupportedFeatureError(
            "labeled K_POINTS paths require the 'ase' package"
        ) from exc
    requested = label.strip()
    key = requested
    if key.lower() in {"gamma", "gg"}:
        key = "G"
    elif len(key) > 1 and key[0] == "g":
        # QE prefixes Greek-letter labels with g: gG, gS, gS1, ...
        key = key[1:]
    key = key.upper()
    try:
        special_points = Cell(lattice).bandpath(npoints=0).special_points
    except (ValueError, RuntimeError) as exc:
        raise QEInputError(
            f"cannot determine the Brillouin-zone labels for {label!r}"
        ) from exc
    if key not in special_points:
        available = ", ".join(sorted(special_points))
        raise QEInputError(
            f"K_POINTS label {label!r} is not defined for this lattice; "
            f"available SC labels are: {available}"
        )
    return np.asarray(special_points[key], dtype=float), requested


def _band_path_kpoints(
    rows: list[str],
    coordinate_type: str,
    lattice: np.ndarray,
    reciprocal: np.ndarray,
    alat: float,
) -> list[KPoint]:
    anchors: list[tuple[np.ndarray, int, str | None]] = []
    for row in rows:
        bits = row.split()
        try:
            first_coordinate = _kpoint_real(bits[0], "K_POINTS coordinate")
            numeric = True
        except QEInputError:
            numeric = False
        if numeric:
            if len(bits) not in {4, 5}:
                raise QEInputError(
                    "K_POINTS *_b numeric rows require kx ky kz N "
                    "and an optional label"
                )
            vector = np.asarray(
                [
                    first_coordinate,
                    _kpoint_real(bits[1], "K_POINTS coordinate"),
                    _kpoint_real(bits[2], "K_POINTS coordinate"),
                ]
            )
            count_value = _kpoint_real(bits[3], "K_POINTS path count")
            coordinate = _kpoint_to_crystal(
                vector, coordinate_type, reciprocal, alat
            )
            label = bits[4] if len(bits) == 5 else None
        else:
            if len(bits) != 2:
                raise QEInputError(
                    "K_POINTS *_b label rows require a label followed by N"
                )
            coordinate, label = _special_kpoint(bits[0], lattice)
            count_value = _kpoint_real(bits[1], "K_POINTS path count")
        anchors.append((coordinate, _fortran_nint(count_value), label))

    generated: list[tuple[np.ndarray, str | None]] = [
        (anchors[0][0].copy(), anchors[0][2])
    ]
    for index in range(len(anchors) - 1):
        start, count, _start_label = anchors[index]
        stop, _unused_count, stop_label = anchors[index + 1]
        if count < 0:
            raise QEInputError("K_POINTS *_b path counts must be nonnegative")
        if count == 0:
            generated.append((stop.copy(), stop_label))
            continue
        for step in range(1, count + 1):
            fraction = step / count
            coordinate = start + fraction * (stop - start)
            generated.append(
                (coordinate, stop_label if step == count else None)
            )
    weight = 1.0 / len(generated)
    return [KPoint(point, weight, label) for point, label in generated]


def _contour_kpoints(
    rows: list[str],
    coordinate_type: str,
    reciprocal: np.ndarray,
    alat: float,
) -> list[KPoint]:
    if len(rows) != 3:
        raise QEInputError("K_POINTS *_c requires exactly 3 supplied k-points")
    anchors = []
    counts = []
    for row in rows:
        bits = row.split()
        if len(bits) != 4:
            raise QEInputError("K_POINTS *_c rows require kx ky kz N")
        vector = np.asarray(
            [_kpoint_real(token, "K_POINTS coordinate") for token in bits[:3]]
        )
        anchors.append(
            _kpoint_to_crystal(vector, coordinate_type, reciprocal, alat)
        )
        counts.append(_fortran_nint(_kpoint_real(bits[3], "K_POINTS mesh count")))
    n1, n2 = counts[1], counts[2]
    if n1 < 2 or n2 < 2:
        raise QEInputError("K_POINTS *_c direction counts must both be at least 2")
    first_step = (anchors[1] - anchors[0]) / (n1 - 1)
    second_step = (anchors[2] - anchors[0]) / (n2 - 1)
    weight = 1.0 / (n1 * n2)
    return [
        KPoint(anchors[0] + i * first_step + j * second_step, weight)
        for i in range(n1)
        for j in range(n2)
    ]


def _parse_kpoints(card: tuple[str, list[str]], lattice: np.ndarray, alat: float) -> list[KPoint]:
    option, rows = card
    reciprocal = TWO_PI * np.linalg.inv(lattice).T
    if option == "gamma":
        return [KPoint(np.zeros(3), 1.0)]
    if option == "automatic":
        try:
            values = [int(x) for x in rows[0].split()]
        except ValueError as exc:
            raise QEInputError(
                "K_POINTS automatic requires six integer values"
            ) from exc
        if len(values) != 6 or any(n <= 0 for n in values[:3]):
            raise QEInputError(
                "invalid values for nk1, nk2, nk3", routine="card_kpoints"
            )
        if any(shift not in {0, 1} for shift in values[3:]):
            raise QEInputError(
                "invalid offsets: must be 0 or 1", routine="card_kpoints"
            )
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
    allowed = {"", "tpiba", "crystal", "tpiba_b", "crystal_b", "tpiba_c", "crystal_c"}
    if option not in allowed:
        raise QEInputError(
            f"unknown K_POINTS option {option!r}", routine="card_kpoints"
        )
    try:
        n = int(rows[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise QEInputError(
            "K_POINTS requires an integer number of supplied points"
        ) from exc
    if n <= 0:
        raise QEInputError("K_POINTS requires a positive number of supplied points")
    raw = rows[1 : n + 1]
    coordinate_type = "crystal" if option.startswith("crystal") else "tpiba"
    if option.endswith("_b"):
        return _band_path_kpoints(
            raw, coordinate_type, lattice, reciprocal, alat
        )
    if option.endswith("_c"):
        return _contour_kpoints(raw, coordinate_type, reciprocal, alat)

    parsed_rows = [row.split() for row in raw]
    if any(len(bits) not in {4, 5} for bits in parsed_rows):
        raise QEInputError(
            "explicit K_POINTS rows require kx ky kz weight and an optional label"
        )
    weights = np.asarray(
        [_kpoint_real(bits[3], "K_POINTS weight") for bits in parsed_rows]
    )
    total_weight = float(weights.sum())
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise QEInputError("explicit K_POINTS weights must have a positive sum")
    weights /= total_weight
    points = []
    for bits, weight in zip(parsed_rows, weights):
        vector = np.asarray(
            [_kpoint_real(token, "K_POINTS coordinate") for token in bits[:3]]
        )
        crystal = _kpoint_to_crystal(
            vector, coordinate_type, reciprocal, alat
        )
        label = bits[4] if len(bits) == 5 else None
        points.append(KPoint(crystal, float(weight), label))
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
    input_warnings: list[QEWarning] = []
    if "tprfor" in control:
        raise QEInputError(
            "bad line in namelist &CONTROL: unknown variable 'tprfor'",
            routine="read_namelists",
        )
    _reject_unimplemented_variables(namelists)
    calculation = str(control.get("calculation", "scf")).strip().lower()
    if calculation not in {"scf", "nscf", "bands"}:
        raise UnsupportedFeatureError(
            not_implemented(f"calculation {calculation}"), routine="iosys"
        )
    control["calculation"] = calculation
    # QE defaults to memory-oriented I/O for SCF, but selects medium for
    # non-self-consistent calculations because their k-point sets can be
    # large. Explicit user input always takes precedence.
    default_disk_io = "low" if calculation == "scf" else "medium"
    disk_io = str(
        control.get("disk_io", default_disk_io)
    ).strip().lower()
    if disk_io not in {"none", "low", "medium", "high"}:
        raise QEInputError(
            f"unknown disk_io {disk_io}", routine="iosys"
        )
    control["disk_io"] = disk_io
    restart_mode = str(
        control.get("restart_mode", "from_scratch")
    ).strip().lower()
    if restart_mode not in {"from_scratch", "restart"}:
        raise QEInputError(
            f"unknown restart_mode {restart_mode}", routine="iosys"
        )

    starting_potential = str(
        electrons.get("startingpot", "atomic")
    ).strip().lower()
    starting_wavefunctions = str(
        electrons.get("startingwfc", "atomic+random")
    ).strip().lower()
    if calculation in {"nscf", "bands"}:
        # A non-self-consistent calculation always evaluates the Kohn--Sham
        # Hamiltonian built from the charge density saved by a preceding SCF
        # run.  Wavefunctions need not be reusable because the requested
        # k-point set commonly differs from the SCF mesh.
        starting_potential = "file"
        electrons["startingpot"] = "file"
    if starting_potential not in {"atomic", "file"}:
        input_warnings.append(
            QEWarning("iosys", "wrong startingpot: use default (1)")
        )
        starting_potential = "atomic"
        electrons["startingpot"] = starting_potential
    if starting_wavefunctions not in {"atomic", "atomic+random", "random", "file"}:
        input_warnings.append(
            QEWarning("iosys", "wrong startingwfc: use default (atomic+random)")
        )
        starting_wavefunctions = "atomic+random"
        electrons["startingwfc"] = starting_wavefunctions
    if restart_mode == "restart":
        if starting_wavefunctions != "file":
            input_warnings.append(
                QEWarning(
                    "input",
                    f'WARNING: "startingwfc" set to {starting_wavefunctions} may spoil restart',
                )
            )
        if starting_potential != "file":
            input_warnings.append(
                QEWarning(
                    "input",
                    f'WARNING: "startingpot" set to {starting_potential} may spoil restart',
                )
            )
            electrons["startingpot"] = "file"

    diagonalization = str(
        electrons.get("diagonalization", "david")
    ).strip().lower()
    diagonalization_options = {
        "david", "davidson", "cg", "paro", "rmm", "rmm-diis",
        "rmm-davidson", "rmm-paro", "direct", "dense",
    }
    if diagonalization == "ppcg":
        raise QEInputError(
            "PPCG diagonalization not supported anymore (Dec. 2024)",
            routine="iosys",
        )
    if diagonalization not in diagonalization_options:
        raise QEInputError(
            f"diagonalization {diagonalization} not implemented",
            routine="iosys",
        )

    mixing_mode = str(electrons.get("mixing_mode", "plain")).strip().lower()
    if mixing_mode == "potential":
        raise QEInputError(
            "potential mixing no longer implemented", routine="iosys"
        )
    if mixing_mode.replace("_", "-") not in {
        "plain", "default", "tf", "local-tf",
    }:
        raise QEInputError(
            f"unknown mixing {mixing_mode}", routine="iosys"
        )
    if "mixing_beta" in electrons and _input_number(
        electrons, "mixing_beta", 0.7, float, "electrons"
    ) < 0.0:
        electrons["mixing_beta"] = 0.7

    ecutwfc = float(_input_number(system, "ecutwfc", 0.0, float, "system"))
    if ecutwfc < 0.0:
        raise QEInputError("ecutwfc out of range", routine="iosys")
    if ecutwfc == 0.0:
        raise QEInputError("ecutwfc not set", routine="set_cutoff")
    if ecutwfc < 1.0 or ecutwfc > 10_000.0:
        raise QEInputError("meaningless value for ecutwfc", routine="setup")
    ecutrho = float(
        _input_number(system, "ecutrho", 4.0 * ecutwfc, float, "system")
    )
    system["ecutrho"] = ecutrho
    if ecutrho < 0.0:
        raise QEInputError("ecutrho out of range", routine="iosys")
    if ecutrho / ecutwfc <= 1.0:
        raise QEInputError("ecutrho <= ecutwfc?!?", routine="set_cutoff")
    if ecutrho < 4.0 * ecutwfc:
        input_warnings.append(
            QEWarning("set_cutoff", "ecutrho < 4*ecutwfc, are you sure?")
        )

    occupations = str(system.get("occupations", "fixed")).strip().lower()
    normalized_occupations = occupations.replace("-", "_")
    supported_occupations = {
        "fixed", "smearing", "tetrahedra", "tetrahedra_lin", "tetrahedra_opt",
    }
    if normalized_occupations == "from_input":
        raise UnsupportedFeatureError(
            not_implemented("occupations from_input"),
            routine="set_occupations",
        )
    if normalized_occupations not in supported_occupations:
        raise QEInputError(
            f"occupations {occupations} not implemented",
            routine="set_occupations",
        )
    degauss = float(_input_number(system, "degauss", 0.0, float, "system"))
    if normalized_occupations == "fixed" and degauss != 0.0:
        # QE 7.5 calls errore(..., -1) here; errore immediately returns for
        # ierr <= 0, so the broadening is reset without printed diagnostics.
        system["degauss"] = 0.0
    if normalized_occupations == "smearing":
        if degauss <= 0.0:
            raise QEInputError(
                "smearing requires a value for gaussian broadening (degauss)",
                routine="set_occupations",
            )
        smearing = str(system.get("smearing", "gaussian")).strip().lower()
        if smearing not in {
            "gaussian", "gauss", "0", "methfessel-paxton", "m-p", "mp",
            "marzari-vanderbilt", "cold", "m-v", "mv", "fermi-dirac",
            "f-d", "fd",
        }:
            raise QEInputError(
                f"smearing {smearing} unknown", routine="set_occupations"
            )
    nspin = int(_input_number(system, "nspin", 1, int, "system"))
    if nspin not in {1, 2, 4}:
        raise QEInputError("nspin out of range", routine="iosys")
    if nspin != 1 or system.get("noncolin", False):
        raise UnsupportedFeatureError(
            not_implemented("spin-polarized and noncollinear calculations"),
            routine="iosys",
        )
    if "nbnd" in system and _input_number(
        system, "nbnd", 0, int, "system"
    ) < 1:
        raise QEInputError("nbnd less than 1", routine="iosys")
    forbidden = {
        "lda_plus_u": "DFT+U", "lspinorb": "spin-orbit coupling",
    }
    for key, feature in forbidden.items():
        value = system.get(key)
        normalized = value.lower() if isinstance(value, str) else value
        if key in system and normalized not in {False, "", "pz", "lda"}:
            raise UnsupportedFeatureError(
                not_implemented(feature), routine="iosys"
            )
    if "input_dft" in system:
        if canonical_xc_name(system["input_dft"]) is None:
            raise UnsupportedFeatureError(
                f"input_dft={system['input_dft']!r} is not implemented in PWSCF-PY; "
                "supported LDA functionals are 'LDA'/'PZ'/'PZ81' and "
                "'PW'/'PW92'; supported GGA functionals are 'PBE', "
                "'PBEsol', 'revPBE', and 'RPBE'"
                ,
                routine="set_dft_from_name",
            )
    cards = _collect_cards(card_lines, system)
    if "ATOMIC_POSITIONS" in cards and cards["ATOMIC_POSITIONS"][0] == "":
        input_warnings.extend(
            (
                QEWarning(
                    "read_cards",
                    "DEPRECATED: no units specified in ATOMIC_POSITIONS card",
                ),
                QEWarning("read_cards", "ATOMIC_POSITIONS: units set to alat"),
            )
        )
    if "CELL_PARAMETERS" in cards and cards["CELL_PARAMETERS"][0] == "":
        input_warnings.append(
            QEWarning(
                "read_cards",
                "DEPRECATED: no units specified in CELL_PARAMETERS card",
            )
        )
    lattice, alat = _lattice(system, cards)
    if "ATOMIC_SPECIES" not in cards or "ATOMIC_POSITIONS" not in cards:
        raise QEInputError("ATOMIC_SPECIES and ATOMIC_POSITIONS are required")
    species = []
    for row in cards["ATOMIC_SPECIES"][1]:
        bits = shlex.split(row)
        if len(bits) != 3:
            raise QEInputError(
                f"cannot read atomic specie from: {row}",
                routine="card_atomic_species",
            )
        try:
            mass = float(bits[1])
        except ValueError as exc:
            raise QEInputError(
                f"cannot read atomic specie from: {row}",
                routine="card_atomic_species",
            ) from exc
        species.append(Species(bits[0], mass, bits[2]))
    if len({item.label for item in species}) != len(species):
        raise QEInputError(
            "two occurrences of the same atomic label",
            routine="card_atomic_species",
        )
    declared_ntyp = int(system.get("ntyp", len(species)))
    if len(species) != declared_ntyp:
        raise QEInputError(
            f"ntyp={declared_ntyp} does not match "
            f"{len(species)} ATOMIC_SPECIES entries"
        )
    pos_unit, rows = cards["ATOMIC_POSITIONS"]
    declared_nat = int(system.get("nat", len(rows)))
    if len(rows) != declared_nat:
        raise QEInputError(
            f"nat={declared_nat} does not match "
            f"{len(rows)} ATOMIC_POSITIONS entries"
        )
    if pos_unit == "crystal_sg":
        atoms = _expand_crystal_sg(rows, system, lattice)
        # QE replaces the input count of inequivalent sites with the expanded
        # number before entering the electronic-structure path.
        system["nat"] = len(atoms)
    else:
        use_hexagonal_rhombohedral_coordinates = (
            int(system.get("ibrav", 0)) == 5
            and not bool(system.get("rhombohedral", True))
            and pos_unit == "crystal"
        )
        atoms = []
        for row in rows:
            bits = row.split()
            if len(bits) not in {4, 7}:
                raise QEInputError(
                    "wrong number of columns in ATOMIC_POSITIONS",
                    routine="card_atomic_positions",
                )
            vector = np.array([_coordinate_expression(x) for x in bits[1:4]])
            if_pos = _position_constraints(bits[4:])
            if pos_unit == "crystal":
                if use_hexagonal_rhombohedral_coordinates:
                    vector = _hexagonal_to_rhombohedral(vector)
                vector = vector @ lattice
            elif pos_unit in {"angstrom", "ang"}:
                vector *= BOHR_PER_ANGSTROM
            elif pos_unit == "bohr":
                pass
            elif pos_unit in {"alat", ""}:
                vector *= alat
            else:
                raise QEInputError(f"unknown ATOMIC_POSITIONS unit {pos_unit!r}")
            atoms.append(Atom(bits[0], vector, if_pos))
    labels = {item.label for item in species}
    unknown = sorted({atom.label for atom in atoms} - labels)
    if unknown:
        raise QEInputError(
            f"species {unknown[0]} in ATOMIC_POSITIONS is nonexistent",
            routine="card_atomic_positions",
        )
    fractional_atoms = np.asarray([atom.position for atom in atoms]) @ np.linalg.inv(lattice)
    for first in range(len(atoms)):
        for second in range(first + 1, len(atoms)):
            difference = fractional_atoms[first] - fractional_atoms[second]
            nearest_lattice_vector = np.rint(difference).astype(int)
            wrapped = difference - nearest_lattice_vector
            if np.linalg.norm(wrapped @ lattice) >= 1.0e-5:
                continue
            if np.all(nearest_lattice_vector == 0):
                message = f"atoms # {first + 1:3d} and # {second + 1:3d} overlap!"
            else:
                i, j, k = nearest_lattice_vector
                message = (
                    f"atoms # {first + 1:3d} and # {second + 1:3d} differ by "
                    f"lattice vector ({i:2d},{j:2d},{k:2d}) in crystal axis"
                )
            raise QEInputError(message, routine="check_atoms")
    kcard = cards.get("K_POINTS", ("gamma", []))
    kpoints = _parse_kpoints(kcard, lattice, alat)
    if kcard[0] == "automatic":
        automatic_values = tuple(int(value) for value in kcard[1][0].split())
        kpoint_grid = automatic_values[:3]
        kpoint_shift = automatic_values[3:]
    else:
        kpoint_grid = None
        kpoint_shift = None
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
    full_kpoints = tuple(kpoints)
    full_kpoint_count = len(kpoints)
    full_to_irreducible = np.arange(full_kpoint_count, dtype=np.int32)
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
        coordinates, weights, full_to_irreducible = reduce_kpoints(
            np.array([point.crystal for point in kpoints]),
            np.array([point.weight for point in kpoints]),
            operations,
            # ``noinv`` disables QE's ordinary inversion/time-reversal
            # reduction in this scalar, nonmagnetic implementation.
            time_reversal=not no_time_reversal,
            return_mapping=True,
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
        full_kpoints=full_kpoints,
        full_to_irreducible=full_to_irreducible,
        kpoint_grid=kpoint_grid,
        kpoint_shift=kpoint_shift,
        symmetry_operations=operations,
        source=source_name,
        warnings=input_warnings,
    )
