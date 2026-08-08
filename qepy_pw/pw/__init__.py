"""The lazily imported pw.x program layer."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["PWInput", "SCFResult", "read_pw_input", "run_scf"]

_EXPORTS = {
    "PWInput": (".input", "PWInput"),
    "read_pw_input": (".input", "read_pw_input"),
    "SCFResult": (".scf", "SCFResult"),
    "run_scf": (".scf", "run_scf"),
}


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
