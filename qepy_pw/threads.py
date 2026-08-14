"""Hybrid MPI/thread runtime controls."""

from __future__ import annotations

import os

from .errors import QEInputError


def _leading_thread_count(name: str, raw: str) -> int:
    """Parse the first OpenMP-style thread-list entry."""
    try:
        count = int(raw.split(",", 1)[0].strip())
    except ValueError as exc:
        raise QEInputError(f"{name} must begin with a positive integer") from exc
    if count < 1:
        raise QEInputError(f"{name} must be a positive integer")
    return count


def hybrid_thread_count() -> int:
    """Return threads per MPI rank from the environment, defaulting to one."""
    for name in ("QEPY_NUM_THREADS", "OMP_NUM_THREADS"):
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        return _leading_thread_count(name, raw)
    return 1


def thread_environment_warning() -> str | None:
    """Explain when QEPY_NUM_THREADS silently overrides an OpenMP setting."""
    qepy_raw = os.environ.get("QEPY_NUM_THREADS")
    omp_raw = os.environ.get("OMP_NUM_THREADS")
    if not qepy_raw or not qepy_raw.strip() or not omp_raw or not omp_raw.strip():
        return None
    qepy_threads = _leading_thread_count("QEPY_NUM_THREADS", qepy_raw)
    try:
        omp_threads = _leading_thread_count("OMP_NUM_THREADS", omp_raw)
    except QEInputError:
        # OMP_NUM_THREADS is not consumed by this project when the explicit
        # QEPY override is present.  Leave validation to the OpenMP runtime.
        return None
    if qepy_threads == omp_threads:
        return None
    return (
        f"QEPY_NUM_THREADS={qepy_threads} overrides "
        f"OMP_NUM_THREADS={omp_threads}; using {qepy_threads} thread(s) "
        "per MPI process"
    )
