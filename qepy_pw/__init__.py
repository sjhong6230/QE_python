"""Scalar plane-wave DFT components derived from the design of QE pw.x.

The public objects are resolved lazily.  Importing a parser or utility module
must not eagerly import SciPy, HDF5, the eigensolver, and the complete SCF
driver in every MPI process.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .version import __version__

if TYPE_CHECKING:
    from .pw.input import PWInput, read_pw_input
    from .pw.save import resolve_save_directory, write_qe_save
    from .pw.scf import SCFResult, run_scf

__all__ = [
    "PWInput",
    "SCFResult",
    "read_pw_input",
    "resolve_save_directory",
    "run_scf",
    "write_qe_save",
]


_LAZY_EXPORTS = {
    "PWInput": (".pw.input", "PWInput"),
    "read_pw_input": (".pw.input", "read_pw_input"),
    "SCFResult": (".pw.scf", "SCFResult"),
    "run_scf": (".pw.scf", "run_scf"),
    "resolve_save_directory": (".pw.save", "resolve_save_directory"),
    "write_qe_save": (".pw.save", "write_qe_save"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
