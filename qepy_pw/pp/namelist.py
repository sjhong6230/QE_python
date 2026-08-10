"""Small strict parser for the scalar PP namelists."""

from __future__ import annotations

import ast
import re

from ..errors import QEInputError
from ..fortran_namelist import (
    expanded_array_keys,
    split_assignment_values,
    split_namelist_assignments,
    strip_namelist_comment,
)


_ONE_DIMENSIONAL_ARRAYS = {
    "inputpp": {"kpoint", "kband", "spin_component", "nc", "n0"},
    "plot": {"filepp", "weight", "e1", "e2", "e3", "x0"},
}


def _value(value: str) -> object:
    value = value.strip()
    lowered = value.lower()
    if lowered in {".true.", "true", ".t."}:
        return True
    if lowered in {".false.", "false", ".f."}:
        return False
    if value[:1] in {"'", '"'}:
        return ast.literal_eval(value)
    normalized = re.sub(r"([0-9.])d([+-]?\d+)", r"\1e\2", value, flags=re.I)
    try:
        return float(normalized) if any(c in normalized.lower() for c in ".e") else int(normalized)
    except ValueError:
        return value


def parse_namelist(text: str, expected: str) -> dict[str, object]:
    match = re.search(
        rf"&{re.escape(expected)}\b(.*?)(?:^|\s)/",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise QEInputError(f"namelist &{expected.upper()} not found")
    body = "\n".join(strip_namelist_comment(line) for line in match.group(1).splitlines())
    result: dict[str, object] = {}
    for assignment in split_namelist_assignments(body):
        if "=" not in assignment:
            raise QEInputError(
                f"bad line in namelist &{expected.upper()}: {assignment}"
            )
        raw_key, raw_value = assignment.split("=", 1)
        fields = split_assignment_values(raw_value)
        if not fields:
            raise QEInputError(
                f"bad line in namelist &{expected.upper()}: {assignment}"
            )
        try:
            base = raw_key.split("(", 1)[0].strip().lower()
            keys = expanded_array_keys(
                raw_key,
                len(fields),
                unindexed_array=(
                    "(" not in raw_key
                    and base in _ONE_DIMENSIONAL_ARRAYS.get(expected.lower(), set())
                ),
            )
        except ValueError as exc:
            raise QEInputError(
                f"bad line in namelist &{expected.upper()}: {assignment}"
            ) from exc
        for key, field in zip(keys, fields):
            result[key] = _value(field)
    return result
