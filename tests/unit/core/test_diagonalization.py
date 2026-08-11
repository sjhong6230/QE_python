from __future__ import annotations

import numpy as np

from qepy_pw.diagonalization import parallel_orbital


def test_qe_paro_converges_lowest_matrix_eigenpairs() -> None:
    rng = np.random.default_rng(7321)
    dimension = 18
    roots = 4
    matrix = rng.normal(size=(dimension, dimension))
    matrix = 0.5 * (matrix + matrix.T)
    matrix += np.diag(np.linspace(1.0, 18.0, dimension))
    expected = np.linalg.eigvalsh(matrix)[:roots]
    initial = rng.normal(size=(dimension, roots))

    result = parallel_orbital(
        lambda vectors: matrix @ vectors,
        np.diag(matrix),
        roots,
        initial_vectors=initial,
        tolerance=1.0e-10,
        max_iterations=20,
    )

    assert result.converged
    assert result.number_unconverged == 0
    assert np.allclose(result.eigenvalues, expected, atol=1.0e-9)
    # QE ParO terminates on Ritz-value changes, not on the residual norm.
    assert np.max(result.residual_norms) < 3.0e-6


def test_qe_paro_uses_bpcg_hamiltonian_applications() -> None:
    rng = np.random.default_rng(91)
    diagonal = np.arange(1.0, 13.0)
    initial = rng.normal(size=(12, 3))

    result = parallel_orbital(
        lambda vectors: diagonal[:, None] * vectors,
        diagonal,
        3,
        initial_vectors=initial,
        tolerance=1.0e-8,
        max_iterations=2,
    )

    # The removed implementation made only one H application per correction;
    # QE bpcg_k performs an inner PCG trajectory (up to five applications).
    assert result.hamiltonian_applications > 2 * 3
