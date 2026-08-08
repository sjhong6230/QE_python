"""Small strict parser for the scalar PP namelists."""

from __future__ import annotations

import ast
import re

from ..errors import QEInputError


def parse_namelist(text: str, expected: str) -> dict[str, object]:
    match = re.search(
        rf"&{re.escape(expected)}\b(.*?)(?:^|\s)/",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise QEInputError(f"namelist &{expected.upper()} not found")
    body = re.sub(r"[!#].*$", "", match.group(1), flags=re.MULTILINE)
    assignments = re.findall(
        r"([A-Za-z_]\w*(?:\(\d+\))?)\s*=\s*("
        r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|[^,\n]+)",
        body,
    )
    result: dict[str, object] = {}
    for raw_key, raw_value in assignments:
        key = raw_key.lower()
        value = raw_value.strip()
        lowered = value.lower()
        if lowered in {".true.", "true", ".t."}:
            parsed: object = True
        elif lowered in {".false.", "false", ".f."}:
            parsed = False
        elif value[:1] in {"'", '"'}:
            parsed = ast.literal_eval(value)
        else:
            normalized = re.sub(r"([0-9.])d([+-]?\d+)", r"\1e\2", value, flags=re.I)
            try:
                parsed = float(normalized) if any(c in normalized.lower() for c in ".e") else int(normalized)
            except ValueError:
                parsed = value
        result[key] = parsed
    return result
