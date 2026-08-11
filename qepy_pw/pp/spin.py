"""Collinear-spin layout helpers shared by QE post-processing programs."""

from __future__ import annotations

import numpy as np

from ..errors import QEInputError


def spin_labels(lsda: bool, count: int) -> np.ndarray:
    """Return QE's block-ordered spin label for every saved k point."""
    if count < 1:
        raise QEInputError("saved data contains no k points")
    if not lsda:
        return np.ones(count, dtype=np.int8)
    if count % 2:
        raise QEInputError("LSDA save contains an odd number of k points")
    spatial = count // 2
    return np.concatenate(
        (np.ones(spatial, dtype=np.int8), np.full(spatial, 2, dtype=np.int8))
    )


def normalize_spin_weights(weights: np.ndarray, spins: np.ndarray) -> np.ndarray:
    """Normalize each collinear spin block to one, as QE's ``wk`` does."""
    values = np.asarray(weights, dtype=float).copy()
    labels = np.asarray(spins, dtype=np.int8)
    if values.shape != labels.shape or not np.all(np.isfinite(values)):
        raise QEInputError("saved k-point weights are invalid")
    for spin in np.unique(labels):
        selected = labels == spin
        total = float(np.sum(values[selected]))
        if total <= 0.0:
            raise QEInputError("saved k-point weights are invalid")
        values[selected] /= total
    return values


def validate_spin_blocks(kpoints: np.ndarray, spins: np.ndarray) -> None:
    """Check that LSDA up/down blocks describe the same spatial k points."""
    points = np.asarray(kpoints, dtype=float)
    labels = np.asarray(spins, dtype=np.int8)
    if points.shape != (len(labels), 3):
        raise QEInputError("saved k points are missing or inconsistent")
    if np.max(labels) == 1:
        return
    up = points[labels == 1]
    down = points[labels == 2]
    if up.shape != down.shape or not np.allclose(up, down, rtol=0.0, atol=1.0e-10):
        raise QEInputError("LSDA spin blocks contain inconsistent spatial k points")


__all__ = ["normalize_spin_weights", "spin_labels", "validate_spin_blocks"]
