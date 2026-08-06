"""Hybrid MPI/thread runtime controls."""

from __future__ import annotations

import os

from .errors import QEInputError


def hybrid_thread_count() -> int:
    """Return threads per MPI rank from the environment, defaulting to one."""
    for name in ("QEPY_NUM_THREADS", "OMP_NUM_THREADS"):
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            count = int(raw.split(",", 1)[0].strip())
        except ValueError as exc:
            raise QEInputError(f"{name} must begin with a positive integer") from exc
        if count < 1:
            raise QEInputError(f"{name} must be a positive integer")
        return count
    return 1
