import numpy as np
import pytest
from scipy.linalg import eigh

from qepy_pw.basis import (
    FFTScratchPool,
    PlaneWaveBasis,
    LocalPotentialWorkspace,
    apply_local_potential,
    fft_shape,
    potential_matrix,
    make_basis,
)
from qepy_pw.diagonalization import (
    FactorizedProjectorTerm,
    PlaneWaveHamiltonian,
    davidson,
)


def test_fft_grid_prevents_density_convolution_aliasing():
    indices = np.array(
        [[-3, -2, -1], [2, 4, 1], [0, 0, 0]]
    )
    basis = PlaneWaveBasis(
        indices, indices.astype(float), np.zeros(len(indices))
    )
    shape = fft_shape([basis])
    span = np.max(indices, axis=0) - np.min(indices, axis=0)
    assert np.all(np.asarray(shape) >= 2 * span + 2)


def test_fft_grid_is_invariant_to_independent_kpoint_gauge_shifts():
    indices = np.array(
        [
            [-3, -2, -1],
            [2, 4, 1],
            [0, 0, 0],
            [1, -1, 2],
        ]
    )
    first = PlaneWaveBasis(
        indices, indices.astype(float), np.zeros(len(indices))
    )
    shifted_indices = indices + np.array([0, 0, 7])
    shifted = PlaneWaveBasis(
        shifted_indices,
        shifted_indices.astype(float),
        np.zeros(len(shifted_indices)),
    )
    assert fft_shape([first, shifted]) == fft_shape([first])


def test_fft_grid_from_charge_cutoff_matches_qe_grid_set():
    basis = PlaneWaveBasis(
        np.zeros((1, 3), dtype=int),
        np.zeros((1, 3)),
        np.zeros(1),
    )
    # |G|^2 <= 225 selects Miller indices through +/-15. QE grid_set
    # requests 31 and good_fft_order advances that to 32.
    assert fft_shape([basis], np.eye(3), 225.0) == (32, 32, 32)
    assert fft_shape(
        [basis], np.eye(3), 196.0, (4, 4, 4)
    ) == (32, 32, 32)


def test_make_basis_keeps_vectors_created_by_nonorthogonal_cancellation():
    reciprocal = np.array(
        [[1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.0, 0.0, 1.0]]
    )
    basis = make_basis(reciprocal, np.zeros(3), 1.1)
    assert np.any(np.all(basis.indices == np.array([5, -5, 0]), axis=1))


def test_fft_local_potential_application_matches_dense_convolution():
    rng = np.random.default_rng(12)
    shape = (5, 6, 5)
    indices = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 1, 0], [0, -2, 1], [2, 1, -1]]
    )
    real_potential = rng.normal(size=shape)
    potential_g = np.fft.fftn(real_potential) / np.prod(shape)
    vectors = rng.normal(size=(len(indices), 3)) + 1j * rng.normal(
        size=(len(indices), 3)
    )
    matrix_result = potential_matrix(potential_g, indices) @ vectors
    fft_result = apply_local_potential(potential_g, indices, vectors)
    assert np.allclose(fft_result, matrix_result, atol=1.0e-13)


def test_matrix_free_hamiltonian_matches_dense_local_and_nonlocal_terms():
    rng = np.random.default_rng(23)
    shape = (5, 5, 5)
    indices = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 1, 0], [0, -1, 1], [1, 1, -1]]
    )
    kinetic = np.linspace(0.0, 2.0, len(indices))
    basis = PlaneWaveBasis(indices, indices.astype(float), kinetic)
    real_potential = rng.normal(size=shape)
    potential_g = np.fft.fftn(real_potential) / np.prod(shape)
    beta = rng.normal(size=(len(indices), 2)) + 1j * rng.normal(
        size=(len(indices), 2)
    )
    coupling = np.array([[0.7, -0.2], [-0.2, 0.3]])
    operator = PlaneWaveHamiltonian(
        basis, potential_g, ((beta, coupling),)
    )
    dense = potential_matrix(potential_g, indices)
    dense[np.diag_indices_from(dense)] += kinetic
    dense += beta @ coupling @ beta.conj().T
    vectors = rng.normal(size=(len(indices), 2)) + 1j * rng.normal(
        size=(len(indices), 2)
    )
    assert np.allclose(operator.apply(vectors), dense @ vectors, atol=1.0e-12)
    assert np.allclose(operator.diagonal, np.real(np.diag(dense)))


def test_factorized_multiatom_projectors_match_materialized_operator():
    rng = np.random.default_rng(230)
    shape = (5, 5, 5)
    indices = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 1, 0], [0, -1, 1]]
    )
    basis = PlaneWaveBasis(
        indices,
        indices.astype(float),
        np.linspace(0.0, 1.0, len(indices)),
    )
    potential_g = np.zeros(shape, dtype=complex)
    beta = np.asfortranarray(
        rng.normal(size=(len(indices), 2))
        + 1j * rng.normal(size=(len(indices), 2))
    )
    phases = np.asfortranarray(
        np.exp(
            1j
            * rng.normal(size=(len(indices), 3))
        )
    )
    coupling = np.array([[0.7, -0.2], [-0.2, 0.3]])
    materialized = tuple(
        (np.asfortranarray(beta * phases[:, atom, None]), coupling)
        for atom in range(phases.shape[1])
    )
    factorized = (
        FactorizedProjectorTerm(beta, phases, coupling),
    )
    vectors = rng.normal(size=(len(indices), 3)) + 1j * rng.normal(
        size=(len(indices), 3)
    )
    reference = PlaneWaveHamiltonian(
        basis, potential_g, materialized
    )
    compact = PlaneWaveHamiltonian(
        basis, potential_g, factorized
    )
    assert np.allclose(
        compact.apply(vectors),
        reference.apply(vectors),
        atol=1.0e-12,
    )
    assert np.allclose(
        compact.diagonal,
        reference.diagonal,
        atol=1.0e-12,
    )


def test_block_davidson_matches_dense_lowest_eigenpairs():
    rng = np.random.default_rng(7)
    dimension = 40
    low_rank = rng.normal(size=(dimension, 6)) + 1j * rng.normal(
        size=(dimension, 6)
    )
    matrix = np.diag(np.linspace(0.0, 20.0, dimension))
    matrix = matrix + 0.08 * (low_rank @ low_rank.conj().T)
    matrix = 0.5 * (matrix + matrix.conj().T)
    expected = eigh(
        matrix, subset_by_index=(0, 4), check_finite=False
    )[0]
    result = davidson(
        lambda vectors: matrix @ vectors,
        np.real(np.diag(matrix)),
        5,
        tolerance=1.0e-10,
        max_iterations=100,
        subspace_multiplier=4,
    )
    assert result.converged
    assert np.max(result.residual_norms) < 1.0e-9
    assert np.allclose(result.eigenvalues, expected, atol=1.0e-11)
    assert np.allclose(
        result.eigenvectors.conj().T @ result.eigenvectors,
        np.eye(5),
        atol=1.0e-12,
    )


def test_davidson_energy_scaled_residual_safeguard():
    rng = np.random.default_rng(71)
    dimension = 32
    update = rng.normal(size=(dimension, 5))
    matrix = np.diag(np.linspace(-1.0, 12.0, dimension))
    matrix += 0.05 * update @ update.T
    tolerance = 1.0e-7
    result = davidson(
        lambda vectors: matrix @ vectors,
        np.diag(matrix),
        4,
        tolerance=tolerance,
        max_iterations=100,
        residual_factor=None,
        residual_energy_scale=1.0,
    )
    assert result.converged
    assert np.max(result.residual_norms) ** 2 < tolerance


def test_davidson_restart_refreshes_operator_and_stays_well_conditioned():
    rng = np.random.default_rng(711)
    dimension = 48
    update = rng.normal(size=(dimension, 7))
    matrix = np.diag(np.linspace(-2.0, 15.0, dimension))
    matrix += 0.08 * update @ update.T
    expected = eigh(matrix, subset_by_index=(0, 4))[0]
    applications = 0

    def apply(vectors):
        nonlocal applications
        applications += vectors.shape[1]
        return matrix @ vectors

    result = davidson(
        apply,
        np.diag(matrix),
        5,
        tolerance=1.0e-9,
        max_iterations=100,
        subspace_multiplier=2,
        residual_factor=None,
        residual_energy_scale=1.0,
    )
    assert result.converged
    assert result.hamiltonian_applications == applications
    assert np.allclose(result.eigenvalues, expected, atol=1.0e-10)
    assert np.allclose(
        result.eigenvectors.T @ result.eigenvectors,
        np.eye(5),
        atol=1.0e-11,
    )


def test_local_potential_workspace_reuses_block_buffers():
    rng = np.random.default_rng(45)
    shape = (5, 5, 5)
    indices = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 1, 0], [0, -1, 1]]
    )
    real_potential = rng.normal(size=shape)
    potential_g = np.fft.fftn(real_potential) / np.prod(shape)
    vectors = rng.normal(size=(len(indices), 2)) + 1j * rng.normal(
        size=(len(indices), 2)
    )
    workspace = LocalPotentialWorkspace(indices, shape)
    prepared = workspace.prepare_potential(potential_g)
    first = workspace.apply(prepared, vectors)
    assert workspace._numpy_input is not None
    buffer_id = id(workspace._numpy_input[1])
    second = workspace.apply(prepared, vectors)
    assert workspace._numpy_input is not None
    assert id(workspace._numpy_input[1]) == buffer_id
    workspace.apply(prepared, vectors[:, :1])
    assert workspace._numpy_input is not None
    assert workspace._numpy_input[0] == 1
    assert id(workspace._numpy_input[1]) != buffer_id
    expected = potential_matrix(potential_g, indices) @ vectors
    assert np.allclose(first, expected, atol=1.0e-13)
    assert np.allclose(second, expected, atol=1.0e-13)


def test_fft_scratch_pool_reuses_storage_and_indices_are_compact():
    shape = (5, 5, 5)
    indices = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 1, 0], [0, -1, 1]]
    )
    pool = FFTScratchPool()
    workspace = LocalPotentialWorkspace(
        indices, shape, scratch_pool=pool
    )
    coefficients = np.ones((len(indices), 2), dtype=complex)
    first = workspace.coefficients_to_grid(
        coefficients, use_scratch=True
    )
    buffer_id = id(pool._complex["real_slabs"])
    second = workspace.coefficients_to_grid(
        coefficients, use_scratch=True
    )
    assert id(pool._complex["real_slabs"]) == buffer_id
    assert np.allclose(first, second)
    assert workspace.indices.dtype == np.int32
    assert workspace.linear_slots.dtype == np.int32


def test_planned_pyfftw_workspace_matches_numpy_backend():
    pytest.importorskip("pyfftw")
    rng = np.random.default_rng(46)
    shape = (5, 6, 5)
    indices = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 1, 0], [0, -2, 1]]
    )
    potential_g = np.fft.fftn(
        rng.normal(size=shape)
    ) / np.prod(shape)
    vectors = rng.normal(size=(len(indices), 3)) + 1j * rng.normal(
        size=(len(indices), 3)
    )
    numpy_workspace = LocalPotentialWorkspace(indices, shape)
    fftw_workspace = LocalPotentialWorkspace(
        indices,
        shape,
        backend="pyfftw",
        planner_effort="estimate",
    )
    numpy_result = numpy_workspace.apply(
        numpy_workspace.prepare_potential(potential_g), vectors
    )
    fftw_result = fftw_workspace.apply(
        fftw_workspace.prepare_potential(potential_g), vectors
    )
    assert fftw_workspace._fftw_buffer is not None
    first_plan_ids = tuple(
        id(item) for item in fftw_workspace._fftw_buffer[1]
    )
    repeated = fftw_workspace.apply(
        fftw_workspace.prepare_potential(potential_g), vectors
    )
    assert fftw_workspace._fftw_buffer is not None
    assert tuple(
        id(item) for item in fftw_workspace._fftw_buffer[1]
    ) == first_plan_ids
    assert np.allclose(fftw_result, numpy_result, atol=1.0e-12)
    assert np.allclose(repeated, numpy_result, atol=1.0e-12)


def test_scipy_overwrite_workspace_matches_numpy_backend():
    rng = np.random.default_rng(146)
    shape = (7, 6, 5)
    indices = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 1, 0], [0, -2, 1]]
    )
    potential_g = np.fft.fftn(
        rng.normal(size=shape)
    ) / np.prod(shape)
    vectors = rng.normal(size=(len(indices), 3)) + 1j * rng.normal(
        size=(len(indices), 3)
    )
    reference = LocalPotentialWorkspace(indices, shape, backend="numpy")
    accelerated = LocalPotentialWorkspace(indices, shape, backend="scipy")
    expected = reference.apply(
        reference.prepare_potential(potential_g), vectors
    )
    actual = accelerated.apply(
        accelerated.prepare_potential(potential_g), vectors
    )
    assert np.allclose(actual, expected, atol=1.0e-12)


def test_numba_local_potential_workspace_matches_numpy_backend():
    pytest.importorskip("numba")
    shape = (8, 7, 6)
    indices = np.array(
        [[0, 0, 0], [1, 0, -1], [-2, 1, 0], [0, -2, 1]]
    )
    rng = np.random.default_rng(419)
    potential_g = np.fft.fftn(rng.normal(size=shape)) / np.prod(shape)
    vectors = rng.normal(size=(len(indices), 3)) + 1j * rng.normal(
        size=(len(indices), 3)
    )
    numpy_workspace = LocalPotentialWorkspace(indices, shape)
    numba_workspace = LocalPotentialWorkspace(
        indices, shape, use_numba=True
    )
    numpy_result = numpy_workspace.apply(
        numpy_workspace.prepare_potential(potential_g), vectors
    )
    numba_result = numba_workspace.apply(
        numba_workspace.prepare_potential(potential_g), vectors
    )
    assert np.allclose(numba_result, numpy_result, atol=1.0e-13)


def test_numba_and_pyfftw_backends_compose():
    pytest.importorskip("numba")
    pytest.importorskip("pyfftw")
    shape = (6, 5, 7)
    indices = np.array(
        [[0, 0, 0], [1, -1, 0], [-2, 0, 1], [0, 2, -1]]
    )
    rng = np.random.default_rng(420)
    potential_g = np.fft.fftn(rng.normal(size=shape)) / np.prod(shape)
    vectors = rng.normal(size=(len(indices), 2)) + 1j * rng.normal(
        size=(len(indices), 2)
    )
    reference = LocalPotentialWorkspace(indices, shape)
    accelerated = LocalPotentialWorkspace(
        indices,
        shape,
        backend="pyfftw",
        planner_effort="estimate",
        use_numba=True,
    )
    expected = reference.apply(
        reference.prepare_potential(potential_g), vectors
    )
    actual = accelerated.apply(
        accelerated.prepare_potential(potential_g), vectors
    )
    assert np.allclose(actual, expected, atol=1.0e-12)


def test_davidson_accepts_an_expanded_atomic_trial_subspace():
    rng = np.random.default_rng(31)
    dimension = 30
    matrix = np.diag(np.linspace(-2.0, 8.0, dimension))
    perturbation = rng.normal(size=(dimension, 4))
    matrix += 0.03 * perturbation @ perturbation.T
    expected = eigh(matrix, subset_by_index=(0, 2))[0]
    trials = rng.normal(size=(dimension, 7))
    result = davidson(
        lambda vectors: matrix @ vectors,
        np.diag(matrix),
        3,
        initial_vectors=trials,
        tolerance=1.0e-10,
        max_iterations=100,
        subspace_multiplier=4,
    )
    assert result.converged
    assert np.allclose(result.eigenvalues, expected, atol=1.0e-10)
