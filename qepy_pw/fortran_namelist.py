"""Shared lexical helpers for QE-style Fortran namelist assignments."""

from __future__ import annotations

import re


_ASSIGNMENT_START = re.compile(
    r"^[A-Za-z_]\w*(?:\(\s*\d+(?:\s*,\s*\d+)*\s*\))?\s*="
)
_INDEXED_KEY = re.compile(
    r"^([A-Za-z_]\w*)\(\s*(\d+)(?:\s*,\s*(\d+)(?:\s*,\s*\d+)*)?\s*\)$"
)


def strip_namelist_comment(line: str) -> str:
    """Remove a QE comment without treating markers inside strings as comments."""
    quote = None
    output: list[str] = []
    for char in line:
        if char in {"'", '"'}:
            quote = None if quote == char else (char if quote is None else quote)
        if char in {"!", "#"} and quote is None:
            break
        output.append(char)
    return "".join(output)


def _lexical_fields(text: str, *, split_newlines: bool) -> list[str]:
    fields: list[str] = []
    token: list[str] = []
    quote = None
    depth = 0
    for char in text:
        if char in {"'", '"'}:
            quote = None if quote == char else (char if quote is None else quote)
        if quote is None:
            depth += char == "("
            depth -= char == ")"
        delimiter = char == "," or (split_newlines and char in "\r\n")
        if delimiter and quote is None and depth == 0:
            field = "".join(token).strip()
            if field:
                fields.append(field)
            token = []
        elif char != "\r":
            token.append(char)
    field = "".join(token).strip()
    if field:
        fields.append(field)
    return fields


def split_namelist_assignments(body: str) -> list[str]:
    """Split assignments while retaining comma-separated array continuations.

    For example, ``e1=1,0,0, nx=20`` becomes the two assignments
    ``e1=1,0,0`` and ``nx=20`` rather than four unrelated fields.
    """
    assignments: list[str] = []
    current: str | None = None
    for field in _lexical_fields(body, split_newlines=True):
        if _ASSIGNMENT_START.match(field):
            if current is not None:
                assignments.append(current)
            current = field
        elif current is None:
            assignments.append(field)
        else:
            current += "," + field
    if current is not None:
        assignments.append(current)
    return assignments


def split_assignment_values(raw: str) -> list[str]:
    """Return the scalar value fields on one assignment's right-hand side."""
    return _lexical_fields(raw, split_newlines=True)


def expanded_array_keys(
    raw_key: str,
    count: int,
    *,
    unindexed_array: bool = False,
) -> list[str]:
    """Expand a one-dimensional Fortran array assignment into indexed keys."""
    key = re.sub(r"\s+", "", raw_key.strip().lower())
    if count <= 1 and not unindexed_array:
        return [key]
    match = _INDEXED_KEY.fullmatch(key)
    if match is None:
        if "(" in key or ")" in key:
            raise ValueError("multi-dimensional array continuation is ambiguous")
        base, first = key, 1
    else:
        base, first_text, second = match.groups()
        if second is not None:
            raise ValueError("multi-dimensional array continuation is ambiguous")
        first = int(first_text)
    return [f"{base}({first + offset})" for offset in range(count)]


__all__ = [
    "expanded_array_keys",
    "split_assignment_values",
    "split_namelist_assignments",
    "strip_namelist_comment",
]
