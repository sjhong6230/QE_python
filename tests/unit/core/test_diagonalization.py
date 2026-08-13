from __future__ import annotations

import numpy as np

from qepy_pw.basis import _load_native_fft
from qepy_pw.diagonalization import parallel_orbital


def test_native_davidson_projected_rows_match_numpy() -> None:
    rng = np.random.default_rng(824)
    basis = np.asfortranarray(
        rng.normal(size=(37, 9)) + 1j * rng.normal(size=(37, 9))
    )
    applied = np.asfortranarray(
        rng.normal(size=(37, 9)) + 1j * rng.normal(size=(37, 9))
    )
    new = basis[:, 5:]

    rows = _load_native_fft().davidson_projected_rows(new, applied, basis)

    assert rows is not None
    h_rows, s_rows = rows
    assert np.allclose(h_rows, new.conj().T @ applied, atol=2.0e-14)
    assert np.allclose(s_rows, new.conj().T @ basis, atol=2.0e-14)


def test_native_davidson_residual_matches_numpy() -> None:
    rng = np.random.default_rng(391)
    basis = np.asfortranarray(
        rng.normal(size=(41, 8)) + 1j * rng.normal(size=(41, 8))
    )
    applied = np.asfortranarray(
        rng.normal(size=(41, 8)) + 1j * rng.normal(size=(41, 8))
    )
    coefficients = np.asfortranarray(
        rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
    )
    roots = np.array([-1.3, -0.2, 0.7, 2.1])

    residual = _load_native_fft().davidson_residual(
        basis, applied, coefficients, roots
    )
    expected = applied @ coefficients - basis @ (
        coefficients * roots[None, :]
    )

    assert residual is not None
    assert residual.flags.f_contiguous
    assert np.allclose(residual, expected, atol=3.0e-14)


def test_native_selected_preconditioner_matches_separate_formula() -> None:
    rng = np.random.default_rng(552)
    residuals = np.asfortranarray(
        rng.normal(size=(53, 5)) + 1j * rng.normal(size=(53, 5))
    )
    eigenvalues = np.array([-2.1, -0.8, 0.3, 1.4, 3.0])
    diagonal = np.linspace(0.2, 18.0, 53)
    selected = np.array([0, 2, 4], dtype=np.int64)
    native = _load_native_fft()

    combined, squared_norms = native.qe_precondition_normalized_selected(
        residuals, eigenvalues, diagonal, selected
    )
    separate = native.qe_precondition(
        residuals[:, selected], eigenvalues[selected], diagonal
    )
    expected_norms = native.column_squared_norms(separate)
    expected = separate / np.sqrt(expected_norms)[None, :]

    assert np.allclose(squared_norms, expected_norms, rtol=3.0e-15)
    assert np.allclose(combined, expected, atol=3.0e-15)


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
