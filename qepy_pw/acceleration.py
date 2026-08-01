"""Discovery and lazy loading for optional compiled acceleration kernels."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from types import ModuleType

from .errors import QEInputError


def numba_available() -> bool:
    """Return whether the optional Numba runtime can be imported."""
    return find_spec("numba") is not None


def resolve_numba(setting: object = "auto") -> bool:
    """Interpret ``py_numba`` and validate explicitly requested acceleration."""
    if isinstance(setting, str):
        normalized = setting.strip().lower()
        if normalized == "auto":
            return numba_available()
        if normalized in {"true", ".true.", "yes", "on", "1"}:
            requested = True
        elif normalized in {"false", ".false.", "no", "off", "0"}:
            requested = False
        else:
            raise QEInputError(
                "py_numba must be 'auto', .true., or .false."
            )
    else:
        requested = bool(setting)
    if requested and not numba_available():
        raise QEInputError(
            "py_numba=.true. requires the optional Numba package; "
            "install qepy-pw[jit]"
        )
    return requested


@lru_cache(maxsize=1)
def numba_kernels() -> ModuleType:
    """Import compiled kernels only after the JIT path has been selected."""
    from . import _numba_kernels

    return _numba_kernels
