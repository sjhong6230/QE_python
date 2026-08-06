"""Matrix-free plane-wave Hamiltonian and block Davidson eigensolver."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Callable

import numpy as np

from .basis import LocalPotentialWorkspace, PlaneWaveBasis, _load_native_fft
from .mpi import MPIContext
from .timing import TimingRegistry


BlockOperator = Callable[[np.ndarray], np.ndarray]
MaterializedProjectorTerm = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class FactorizedProjectorTerm:
    """Species-centered beta functions plus atom-dependent phase columns."""

    beta: np.ndarray
    phases: np.ndarray
    coupling: np.ndarray


@dataclass(frozen=True)
class PackedProjectorTerm:
    """QE-style atom-packed beta matrix with a precomputed diagonal."""

    beta: np.ndarray
    coupling: np.ndarray
    diagonal: np.ndarray


ProjectorTerm = (
    MaterializedProjectorTerm
    | FactorizedProjectorTerm
    | PackedProjectorTerm
)


def _lowest_generalized_eigh(
    hamiltonian: np.ndarray,
    overlap: np.ndarray,
    roots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a small Hermitian generalized problem without importing SciPy.

    Davidson's reduced matrices are only a few tens of rows in normal SCF
    runs.  A Cholesky reduction followed by NumPy's LAPACK-backed ``eigh``
    avoids the sizeable per-rank SciPy extension-module baseline while doing
    the same Hermitian-definite transformation as LAPACK ``zhegv``.
    """
    native_result = _load_native_fft().generalized_eigh(
        hamiltonian, overlap, roots
    )
    if native_result is not None:
        return native_result
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    overlap = np.asarray(overlap, dtype=complex)
    cholesky = np.linalg.cholesky(overlap)
    left_reduced = np.linalg.solve(cholesky, hamiltonian)
    reduced = np.linalg.solve(
        np.conjugate(cholesky), left_reduced.T
    ).T
    reduced = 0.5 * (reduced + reduced.conj().T)
    values, transformed = np.linalg.eigh(reduced)
    vectors = np.linalg.solve(cholesky.conj().T, transformed)
    return values[:roots], vectors[:, :roots]


@dataclass(frozen=True)
class DavidsonResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    residual_norms: np.ndarray
    iterations: int
    converged: bool
    number_unconverged: int = 0
    hamiltonian_applications: int = 0


class PlaneWaveHamiltonian:
    """Apply scalar norm-conserving H to vectors without an Npw-square array."""

    def __init__(
        self,
        basis: PlaneWaveBasis,
        potential_g: np.ndarray,
        projector_terms: tuple[ProjectorTerm, ...] = (),
        local_workspace: LocalPotentialWorkspace | None = None,
        timers: TimingRegistry | None = None,
        real_potential: np.ndarray | None = None,
        native_potential_layout: bool = False,
        potential_average: float | None = None,
    ) -> None:
        self.basis = basis
        self.potential_g = potential_g
        self.local_workspace = (
            local_workspace
            if local_workspace is not None
            else LocalPotentialWorkspace(basis, potential_g.shape)
        )
        self.timers = timers
        self.real_potential = (
            np.asarray(real_potential)
            if real_potential is not None
            else self.local_workspace.prepare_potential(potential_g)
        )
        self.native_potential_layout = bool(native_potential_layout)
        self.potential_average = (
            float(potential_average)
            if potential_average is not None
            else float(np.real(self.potential_g.flat[0]))
        )
        self.mpi = self.local_workspace.mpi
        self.local_rows = self.local_workspace.local_plane_wave_indices
        kinetic = self.basis.kinetic
        self.local_kinetic = (
            kinetic[self.local_rows]
            if self.mpi.size > 1
            else kinetic
        )
        localized_terms: list[ProjectorTerm] = []
        for term in projector_terms:
            if isinstance(term, FactorizedProjectorTerm):
                if (
                    self.mpi.size > 1
                    and term.beta.shape[0] != self.local_rows.size
                ):
                    localized_terms.append(
                        FactorizedProjectorTerm(
                            np.asfortranarray(term.beta[self.local_rows]),
                            np.asfortranarray(term.phases[self.local_rows]),
                            term.coupling,
                        )
                    )
                else:
                    localized_terms.append(term)
            elif isinstance(term, PackedProjectorTerm):
                if (
                    self.mpi.size > 1
                    and term.beta.shape[0] != self.local_rows.size
                ):
                    localized_terms.append(
                        PackedProjectorTerm(
                            np.asfortranarray(term.beta[self.local_rows]),
                            term.coupling,
                            term.diagonal[self.local_rows],
                        )
                    )
                else:
                    localized_terms.append(term)
            else:
                beta, coupling = term
                if (
                    self.mpi.size > 1
                    and beta.shape[0] != self.local_rows.size
                ):
                    beta = np.asfortranarray(beta[self.local_rows])
                localized_terms.append((beta, coupling))
        self.projector_terms = tuple(localized_terms)

    @property
    def diagonal(self) -> np.ndarray:
        """Hamiltonian diagonal used by QE's ``usnldiag`` preconditioner."""
        result = self.local_kinetic + self.potential_average
        for term in self.projector_terms:
            if isinstance(term, PackedProjectorTerm):
                result = result + term.diagonal
                continue
            if isinstance(term, FactorizedProjectorTerm):
                beta = term.beta
                coupling = term.coupling
                multiplicity = term.phases.shape[1]
            else:
                beta, coupling = term
                multiplicity = 1
            result = result + multiplicity * np.real(
                np.einsum(
                    "gi,ij,gj->g",
                    beta,
                    coupling,
                    beta.conj(),
                    optimize=True,
                )
            )
        return result

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        if self.timers is not None:
            with self.timers.measure("h_psi"):
                return self._apply(coefficients)
        return self._apply(coefficients)

    def _apply(self, coefficients: np.ndarray) -> np.ndarray:
        vectors = np.asarray(coefficients)
        was_vector = vectors.ndim == 1
        if was_vector:
            vectors = vectors[:, None]
        result = self.local_kinetic[:, None] * vectors
        if self.timers is None:
            result += self.local_workspace.apply(
                self.real_potential,
                vectors,
                native_potential_layout=self.native_potential_layout,
            )
        else:
            with self.timers.measure("vloc_psi"):
                result += self.local_workspace.apply(
                    self.real_potential,
                    vectors,
                    native_potential_layout=self.native_potential_layout,
                )
        descriptors: list[
            tuple[np.ndarray, np.ndarray, np.ndarray | None]
        ] = []
        counts: list[int] = []
        all_overlaps: np.ndarray | None = None
        if self.projector_terms:
            overlap_blocks: list[np.ndarray] = []
            # One Npw-by-nband buffer serves both the conjugated overlap
            # input and the nonlocal contribution.  In particular, avoid
            # materializing beta.conj(), whose size grows with the number of
            # atoms/projectors and was previously hidden inside BLAS wrappers.
            nonlocal_buffer = np.empty_like(vectors)
            calbec_timer = (
                self.timers.measure("h_psi:calbec")
                if self.timers is not None
                else nullcontext()
            )
            with calbec_timer:
                for term in self.projector_terms:
                    if isinstance(term, FactorizedProjectorTerm):
                        for atom in range(term.phases.shape[1]):
                            phase = term.phases[:, atom]
                            np.conjugate(vectors, out=nonlocal_buffer)
                            np.multiply(
                                nonlocal_buffer,
                                phase[:, None],
                                out=nonlocal_buffer,
                            )
                            overlap = term.beta.T @ nonlocal_buffer
                            np.conjugate(overlap, out=overlap)
                            overlap_blocks.append(overlap)
                            descriptors.append(
                                (term.beta, term.coupling, phase)
                            )
                    else:
                        if isinstance(term, PackedProjectorTerm):
                            beta, coupling = term.beta, term.coupling
                        else:
                            beta, coupling = term
                        np.conjugate(vectors, out=nonlocal_buffer)
                        overlap = beta.T @ nonlocal_buffer
                        np.conjugate(overlap, out=overlap)
                        overlap_blocks.append(overlap)
                        descriptors.append((beta, coupling, None))
                counts = [block.shape[0] for block in overlap_blocks]
                packed_overlaps = (
                    overlap_blocks[0]
                    if len(overlap_blocks) == 1
                    else np.vstack(overlap_blocks)
                )
                all_overlaps = self.mpi.sum_array(
                    packed_overlaps
                )
        if self.projector_terms:
            assert all_overlaps is not None
            add_timer = (
                self.timers.measure("add_vuspsi")
                if self.timers is not None
                else nullcontext()
            )
            with add_timer:
                offset = 0
                for count, descriptor in zip(counts, descriptors):
                    beta, coupling, phase = descriptor
                    overlaps = all_overlaps[offset : offset + count]
                    coupled = coupling @ overlaps
                    np.matmul(beta, coupled, out=nonlocal_buffer)
                    if phase is not None:
                        nonlocal_buffer *= phase[:, None]
                    result += nonlocal_buffer
                    offset += count
        return result[:, 0] if was_vector else result


def _orthonormalize(
    vectors: np.ndarray,
    against: np.ndarray | None = None,
    threshold: float = 1.0e-12,
    mpi: MPIContext | None = None,
) -> np.ndarray:
    """Return a rank-revealing basis with one common-case MPI reduction.

    Projection and the unprojected Gram matrix are reduced together. Their
    Schur complement is the projected Gram matrix when ``against`` is
    orthonormal. A second projection/reduction is reserved for cancellation-
    dominated blocks, retaining the former numerical safety net.
    """
    block = np.asarray(vectors, dtype=complex).copy()
    if block.ndim == 1:
        block = block[:, None]
    mpi = mpi if mpi is not None else MPIContext()
    if block.shape[1] == 0:
        return np.empty((block.shape[0], 0), dtype=complex)
    if mpi.size > 1:
        if against is not None and against.ndim == 2 and against.shape[1]:
            projection_local = against.conj().T @ block
            raw_gram_local = block.conj().T @ block
            split = projection_local.size
            reduced = mpi.sum_array(
                np.concatenate(
                    (projection_local.ravel(), raw_gram_local.ravel())
                )
            )
            projection = reduced[:split].reshape(projection_local.shape)
            raw_gram = reduced[split:].reshape(raw_gram_local.shape)
            block -= against @ projection
            gram = raw_gram - projection.conj().T @ projection
            gram = 0.5 * (gram + gram.conj().T)
            raw_scale = max(
                float(np.max(np.abs(np.diag(raw_gram)))), 1.0e-300
            )
            remaining_scale = float(
                np.max(np.abs(np.diag(gram)), initial=0.0)
            )
            smallest = (
                float(np.real(gram[0, 0]))
                if gram.shape == (1, 1)
                else float(np.min(np.linalg.eigvalsh(gram), initial=0.0))
            )
            if (
                remaining_scale < 1.0e-8 * raw_scale
                or smallest < -1.0e-10 * raw_scale
            ):
                # Severe cancellation makes the Schur complement inaccurate.
                # Reproject and form the Gram matrix explicitly only then.
                projection = mpi.sum_array(against.conj().T @ block)
                block -= against @ projection
                gram = mpi.sum_array(block.conj().T @ block)
        else:
            gram = mpi.sum_array(block.conj().T @ block)
        gram = 0.5 * (gram + gram.conj().T)
        if gram.shape == (1, 1):
            squared_norm = float(np.real(gram[0, 0]))
            if squared_norm <= 0.0:
                return np.empty((block.shape[0], 0), dtype=complex)
            return block / np.sqrt(squared_norm)
        squared, rotation = np.linalg.eigh(gram)
        order = np.argsort(squared)[::-1]
        squared = np.maximum(np.real(squared[order]), 0.0)
        rotation = rotation[:, order]
        if not len(squared) or squared[0] == 0.0:
            return np.empty((block.shape[0], 0), dtype=complex)
        rank = int(
            np.count_nonzero(
                squared > (threshold**2) * squared[0]
            )
        )
        return (
            block
            @ rotation[:, :rank]
            / np.sqrt(squared[:rank])[None, :]
        )
    if against is not None and against.ndim == 2 and against.shape[1]:
        for _ in range(2):
            block -= against @ (against.conj().T @ block)
    if block.shape[1] == 1:
        norm = float(np.linalg.norm(block[:, 0]))
        if norm == 0.0:
            return np.empty((block.shape[0], 0), dtype=complex)
        return block / norm
    left, singular_values, _ = np.linalg.svd(block, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0.0:
        return np.empty((block.shape[0], 0), dtype=complex)
    rank = int(
        np.count_nonzero(
            singular_values > threshold * singular_values[0]
        )
    )
    return left[:, :rank]


def _qe_precondition(
    residuals: np.ndarray,
    eigenvalues: np.ndarray,
    diagonal: np.ndarray,
) -> np.ndarray:
    """QE ``g_psi`` smooth approximate inverse iteration."""
    # QE evaluates g_psi in Rydberg units.  The smoothing expression contains
    # the dimensionful constant 1 Ry, so simply feeding it Hartree values is
    # not a harmless common rescaling: it produces different correction
    # vectors and, at the deliberately loose first-iteration threshold,
    # different Ritz values.
    return _load_native_fft().qe_precondition(
        np.asarray(residuals, dtype=np.complex128),
        np.ascontiguousarray(eigenvalues, dtype=np.float64),
        np.ascontiguousarray(diagonal, dtype=np.float64),
    )


def _normalize_columns(
    vectors: np.ndarray,
    mpi: MPIContext,
    threshold: float = 1.0e-14,
) -> np.ndarray:
    """Normalize a correction block as QE ``cegterg`` does."""
    block = np.asarray(vectors, dtype=complex)
    if block.ndim == 1:
        block = block[:, None]
    native = _load_native_fft()
    squared_norms = np.real(
        mpi.sum_array(native.column_squared_norms(block))
    )
    if not len(squared_norms):
        return np.empty((block.shape[0], 0), dtype=complex)
    scale = max(float(np.max(squared_norms)), 1.0e-300)
    keep = squared_norms > threshold**2 * scale
    if not np.any(keep):
        return np.empty((block.shape[0], 0), dtype=complex)
    selected = np.ascontiguousarray(np.flatnonzero(keep), dtype=np.int64)
    return native.normalize_selected_columns(
        block,
        np.ascontiguousarray(squared_norms, dtype=np.float64),
        selected,
    )


def davidson(
    operator: BlockOperator,
    diagonal: np.ndarray,
    number_of_roots: int,
    *,
    initial_vectors: np.ndarray | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 40,
    subspace_multiplier: int = 4,
    residual_factor: float | None = 10.0,
    residual_energy_scale: float | None = None,
    occupied_roots: int | None = None,
    initial_is_ritz: bool = False,
    initial_is_orthonormal: bool = False,
    initial_eigenvalues: np.ndarray | None = None,
    mpi: MPIContext | None = None,
    global_dimension: int | None = None,
    global_row_indices: np.ndarray | None = None,
    timers: TimingRegistry | None = None,
) -> DavidsonResult:
    """Compute the lowest eigenpairs with restarted block Davidson iteration."""
    diagonal = np.asarray(diagonal, dtype=float)
    mpi = mpi if mpi is not None else MPIContext()
    local_dimension = len(diagonal)
    dimension = (
        int(global_dimension)
        if global_dimension is not None
        else local_dimension
    )
    if not 1 <= number_of_roots <= dimension:
        raise ValueError("number_of_roots must be between 1 and the matrix size")
    if tolerance <= 0.0:
        raise ValueError("Davidson tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("Davidson max_iterations must be at least 1")
    if subspace_multiplier < 2:
        raise ValueError("diago_david_ndim must be at least 2")
    if residual_factor is not None and residual_factor <= 0.0:
        raise ValueError("Davidson residual_factor must be positive or None")
    if residual_energy_scale is not None and residual_energy_scale <= 0.0:
        raise ValueError(
            "Davidson residual_energy_scale must be positive or None"
        )
    if occupied_roots is None:
        occupied_roots = number_of_roots
    if not 0 <= occupied_roots <= number_of_roots:
        raise ValueError(
            "occupied_roots must be between zero and number_of_roots"
        )
    root_tolerances = np.full(
        number_of_roots, max(5.0 * tolerance, 5.0e-6)
    )
    root_tolerances[:occupied_roots] = tolerance

    if initial_vectors is None:
        if global_row_indices is None:
            global_row_indices = np.arange(local_dimension)
        global_row_indices = np.asarray(global_row_indices, dtype=int)
        trial = np.zeros(
            (local_dimension, number_of_roots), dtype=complex
        )
        for column in range(number_of_roots):
            matches = np.flatnonzero(global_row_indices == column)
            if len(matches):
                trial[matches[0], column] = 1.0
    else:
        trial = np.asarray(initial_vectors, dtype=complex)
        if (
            trial.ndim != 2
            or trial.shape[0] != local_dimension
            or trial.shape[1] < number_of_roots
        ):
            raise ValueError(
                "initial_vectors must have shape "
                "(matrix size, at least number_of_roots)"
            )

    if initial_is_ritz or initial_is_orthonormal:
        # rotate_wfc has already returned orthonormal Ritz vectors in the
        # required band order; converged vectors from the preceding SCF step
        # are orthonormal as well. Reorthogonalizing either block adds a
        # global Gram reduction and may apply an arbitrary unitary rotation.
        basis = np.asarray(
            trial[:, :number_of_roots], dtype=complex
        ).copy()
    else:
        with (
            timers.measure("cegterg:over")
            if timers is not None
            else nullcontext()
        ):
            basis = _orthonormalize(trial, mpi=mpi)
    if basis.shape[1] < number_of_roots:
        raise ValueError("initial Davidson vectors are linearly dependent")
    maximum_subspace = min(
        dimension,
        max(
            basis.shape[1],
            2 * number_of_roots,
            subspace_multiplier * number_of_roots,
        ),
    )
    active_columns = basis.shape[1]
    basis_storage = np.empty(
        (local_dimension, maximum_subspace), dtype=complex, order="F"
    )
    applied_storage = np.empty_like(basis_storage)
    basis_storage[:, :active_columns] = basis
    basis = basis_storage[:, :active_columns]
    applied_storage[:, :active_columns] = operator(basis)
    applied_basis = applied_storage[:, :active_columns]
    hamiltonian_applications = active_columns
    projected_storage = np.zeros(
        (maximum_subspace, maximum_subspace), dtype=complex
    )
    overlap_storage = np.zeros_like(projected_storage)

    def update_projected(first_new: int) -> None:
        """Update only reduced-matrix rows/columns added since first_new."""
        with (
            timers.measure("cegterg:over")
            if timers is not None
            else nullcontext()
        ):
            new_basis = basis[:, first_new:active_columns]
            h_local = new_basis.conj().T @ applied_basis
            s_local = new_basis.conj().T @ basis
            split = h_local.size
            reduced = mpi.sum_array(
                np.concatenate((h_local.ravel(), s_local.ravel()))
            )
            h_rows = reduced[:split].reshape(h_local.shape)
            s_rows = reduced[split:].reshape(s_local.shape)
            projected_storage[
                first_new:active_columns, :active_columns
            ] = h_rows
            overlap_storage[
                first_new:active_columns, :active_columns
            ] = s_rows
            if first_new:
                projected_storage[
                    :first_new, first_new:active_columns
                ] = h_rows[:, :first_new].conj().T
                overlap_storage[
                    :first_new, first_new:active_columns
                ] = s_rows[:, :first_new].conj().T
            new_slice = slice(first_new, active_columns)
            projected_storage[new_slice, new_slice] = 0.5 * (
                projected_storage[new_slice, new_slice]
                + projected_storage[new_slice, new_slice].conj().T
            )
            overlap_storage[new_slice, new_slice] = 0.5 * (
                overlap_storage[new_slice, new_slice]
                + overlap_storage[new_slice, new_slice].conj().T
            )

    update_projected(0)

    def ritz_pairs() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        with (
            timers.measure("cdiaghg")
            if timers is not None
            else nullcontext()
        ):
            roots, coefficients = _lowest_generalized_eigh(
                projected_storage[:active_columns, :active_columns],
                overlap_storage[:active_columns, :active_columns],
                number_of_roots,
            )
        with (
            timers.measure("cegterg:upda")
            if timers is not None
            else nullcontext()
        ):
            vectors = basis @ coefficients
            applied = applied_basis @ coefficients
            residual = applied - vectors * roots[None, :]
        return roots, vectors, applied, residual

    if initial_is_ritz:
        # QE cegterg(lrot=.true.) trusts rotate_wfc: it does not perform a
        # second initial reduced diagonalization, but starts from the rotated
        # vectors and diagonal projected energies with vc=I.
        if initial_eigenvalues is None:
            projected_diagonal = mpi.sum_array(
                np.sum(basis.conj() * applied_basis, axis=0)
            )
            values = np.real(projected_diagonal[:number_of_roots])
        else:
            values = np.asarray(
                initial_eigenvalues[:number_of_roots], dtype=float
            )
        ritz_vectors = basis[:, :number_of_roots]
        applied_ritz = applied_basis[:, :number_of_roots]
        residuals = (
            applied_ritz - ritz_vectors * values[None, :]
        )
    else:
        values, ritz_vectors, applied_ritz, residuals = ritz_pairs()
    track_residual_norms = (
        residual_factor is not None or residual_energy_scale is not None
    )

    def reduced_residual_norms(block: np.ndarray) -> np.ndarray:
        local_squared = _load_native_fft().column_squared_norms(
            np.asarray(block, dtype=np.complex128)
        )
        return np.sqrt(
            np.real(mpi.sum_array(local_squared))
        )

    residual_norms = (
        reduced_residual_norms(residuals)
        if track_residual_norms
        else np.empty(number_of_roots, dtype=float)
    )
    completed_iterations = 0
    number_unconverged = number_of_roots
    unconverged = np.ones(number_of_roots, dtype=bool)

    for iteration in range(1, max_iterations + 1):
        completed_iterations = iteration
        previous_values = values
        with (
            timers.measure("g_psi")
            if timers is not None
            else nullcontext()
        ):
            corrections = _qe_precondition(
                residuals[:, unconverged],
                values[unconverged],
                diagonal,
            )
        # QE normalizes each preconditioned residual but deliberately keeps
        # the correction block nonorthogonal.  The generalized reduced
        # eigenproblem already contains its overlap matrix; projecting the
        # corrections against the full Davidson basis adds large BLAS work
        # and another collective without changing their useful span.
        corrections = _normalize_columns(corrections, mpi)
        if corrections.shape[1] == 0:
            break
        old_columns = active_columns
        active_columns += corrections.shape[1]
        basis_storage[:, old_columns:active_columns] = corrections
        basis = basis_storage[:, :active_columns]
        corrections = basis_storage[:, old_columns:active_columns]
        applied_corrections = operator(corrections)
        applied_storage[:, old_columns:active_columns] = (
            applied_corrections
        )
        applied_basis = applied_storage[:, :active_columns]
        hamiltonian_applications += corrections.shape[1]
        update_projected(old_columns)
        values, ritz_vectors, applied_ritz, residuals = ritz_pairs()
        if track_residual_norms:
            residual_norms = reduced_residual_norms(residuals)
        converged = (
            np.abs(values - previous_values) < root_tolerances
        )
        if residual_factor is not None:
            converged &= (
                residual_norms < residual_factor * root_tolerances
            )
        if residual_energy_scale is not None:
            # Ritz values can stagnate while the vectors still have large
            # residuals. Eigenvalue errors are second order in a normalized
            # residual, giving an energy-scaled safeguard without imposing
            # an excessively strict linear residual test.
            converged &= (
                residual_norms**2
                < residual_energy_scale * root_tolerances
            )
        unconverged = ~converged
        number_unconverged = int(np.count_nonzero(~converged))
        if number_unconverged == 0:
            if not track_residual_norms:
                residual_norms = reduced_residual_norms(residuals)
            return DavidsonResult(
                values,
                ritz_vectors,
                residual_norms,
                iteration,
                True,
                0,
                hamiltonian_applications,
            )
        # QE refreshes only after diagonalizing the expanded generalized
        # problem, when adding the next unconverged block would exceed nvecx.
        if active_columns + number_unconverged > maximum_subspace:
            with (
                timers.measure("cegterg:last")
                if timers is not None
                else nullcontext()
            ):
                active_columns = number_of_roots
                basis_storage[:, :active_columns] = ritz_vectors
                basis = basis_storage[:, :active_columns]
                # The Ritz vectors and their H images use the same reduced-
                # space rotation. Carry both through the restart as QE does;
                # reapplying H here only repeats an expensive FFT block.
                applied_storage[:, :active_columns] = applied_ritz
                applied_basis = applied_storage[:, :active_columns]
                projected_storage[:active_columns, :active_columns] = 0.0
                projected_storage[
                    np.arange(active_columns), np.arange(active_columns)
                ] = values
                overlap_storage[:active_columns, :active_columns] = np.eye(
                    active_columns, dtype=complex
                )

    if not track_residual_norms:
        residual_norms = reduced_residual_norms(residuals)
    return DavidsonResult(
        values,
        ritz_vectors,
        residual_norms,
        completed_iterations,
        False,
        number_unconverged,
        hamiltonian_applications,
    )


def _iterative_root_tolerances(
    number_of_roots: int,
    tolerance: float,
    occupied_roots: int | None,
    full_accuracy: bool,
) -> np.ndarray:
    if occupied_roots is None or full_accuracy:
        occupied_roots = number_of_roots
    if not 0 <= occupied_roots <= number_of_roots:
        raise ValueError(
            "occupied_roots must be between zero and number_of_roots"
        )
    tolerances = np.full(
        number_of_roots, max(5.0 * tolerance, 5.0e-6)
    )
    tolerances[:occupied_roots] = tolerance
    return tolerances


def _column_norms(vectors: np.ndarray, mpi: MPIContext) -> np.ndarray:
    local = _load_native_fft().column_squared_norms(
        np.asarray(vectors, dtype=np.complex128)
    )
    return np.sqrt(np.maximum(np.real(mpi.sum_array(local)), 0.0))


def _global_inner(
    left: np.ndarray, right: np.ndarray, mpi: MPIContext
) -> complex:
    return complex(
        mpi.sum_array(
            np.asarray([np.vdot(left, right)], dtype=np.complex128)
        )[0]
    )


def _initial_iterative_subspace(
    operator: BlockOperator,
    number_of_roots: int,
    initial_vectors: np.ndarray,
    mpi: MPIContext,
    *,
    initial_is_orthonormal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    trial = np.asarray(initial_vectors, dtype=complex)
    if trial.ndim != 2 or trial.shape[1] < number_of_roots:
        raise ValueError(
            "initial_vectors must contain at least number_of_roots columns"
        )
    vectors = (
        trial[:, :number_of_roots].copy()
        if initial_is_orthonormal
        else _orthonormalize(trial, mpi=mpi)[:, :number_of_roots]
    )
    if vectors.shape[1] < number_of_roots:
        raise ValueError("initial vectors are linearly dependent")
    applied = operator(vectors)
    projected = mpi.sum_array(vectors.conj().T @ applied)
    projected = 0.5 * (projected + projected.conj().T)
    values, rotation = np.linalg.eigh(projected)
    rotation = rotation[:, :number_of_roots]
    return (
        vectors @ rotation,
        np.real(values[:number_of_roots]),
        applied @ rotation,
        number_of_roots,
    )


def conjugate_gradient(
    operator: BlockOperator,
    diagonal: np.ndarray,
    number_of_roots: int,
    *,
    initial_vectors: np.ndarray,
    tolerance: float = 1.0e-8,
    max_iterations: int = 20,
    occupied_roots: int | None = None,
    full_accuracy: bool = False,
    initial_is_orthonormal: bool = False,
    mpi: MPIContext | None = None,
) -> DavidsonResult:
    """QE-style preconditioned band-by-band conjugate-gradient solver."""
    if tolerance <= 0.0:
        raise ValueError("diagonalization tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("diago_cg_maxiter must be at least 1")
    mpi = mpi if mpi is not None else MPIContext()
    vectors, values, applied, applications = _initial_iterative_subspace(
        operator,
        number_of_roots,
        initial_vectors,
        mpi,
        initial_is_orthonormal=initial_is_orthonormal,
    )
    root_tolerances = _iterative_root_tolerances(
        number_of_roots,
        tolerance,
        occupied_roots,
        full_accuracy,
    )
    unconverged = 0
    iteration_sum = 0
    for band in range(number_of_roots):
        psi = vectors[:, band].copy()
        hpsi = applied[:, band].copy()
        if band:
            lower = vectors[:, :band]
            coefficients = mpi.sum_array(lower.conj().T @ psi)
            psi -= lower @ coefficients
            hpsi -= applied[:, :band] @ coefficients
            norm = np.sqrt(
                max(float(np.real(_global_inner(psi, psi, mpi))), 1.0e-300)
            )
            psi /= norm
            hpsi /= norm
        energy = float(np.real(_global_inner(psi, hpsi, mpi)))
        previous_direction: np.ndarray | None = None
        converged = False
        for iteration in range(1, max_iterations + 1):
            residual = hpsi - energy * psi
            correction = _qe_precondition(
                residual[:, None], np.array([energy]), diagonal
            )[:, 0]
            candidates = (
                correction[:, None]
                if previous_direction is None
                else np.column_stack((correction, previous_direction))
            )
            against = np.column_stack((vectors[:, :band], psi))
            candidates = _orthonormalize(
                candidates, against=against, mpi=mpi
            )
            if not candidates.shape[1]:
                break
            h_candidates = operator(candidates)
            applications += candidates.shape[1]
            small_basis = np.column_stack((psi, candidates))
            h_small_basis = np.column_stack((hpsi, h_candidates))
            h_small = mpi.sum_array(
                small_basis.conj().T @ h_small_basis
            )
            s_small = mpi.sum_array(
                small_basis.conj().T @ small_basis
            )
            roots, coefficients = _lowest_generalized_eigh(
                h_small, s_small, 1
            )
            old_energy = energy
            coefficient = coefficients[:, 0]
            previous_direction = candidates @ coefficient[1:]
            psi = small_basis @ coefficient
            hpsi = h_small_basis @ coefficient
            energy = float(roots[0])
            if abs(energy - old_energy) < root_tolerances[band]:
                converged = True
                iteration_sum += iteration
                break
        if not converged:
            unconverged += 1
            iteration_sum += max_iterations
        vectors[:, band] = psi
        applied[:, band] = hpsi
        values[band] = energy
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    applied = applied[:, order]
    residuals = applied - vectors * values[None, :]
    return DavidsonResult(
        values,
        vectors,
        _column_norms(residuals, mpi),
        max(1, int(round(iteration_sum / number_of_roots))),
        unconverged == 0,
        unconverged,
        applications,
    )


def parallel_orbital(
    operator: BlockOperator,
    diagonal: np.ndarray,
    number_of_roots: int,
    *,
    initial_vectors: np.ndarray,
    tolerance: float = 1.0e-8,
    max_iterations: int = 20,
    occupied_roots: int | None = None,
    full_accuracy: bool = False,
    initial_is_orthonormal: bool = False,
    mpi: MPIContext | None = None,
) -> DavidsonResult:
    """Parallel orbital-updating iteration with a bounded 2N Ritz space."""
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("invalid ParO diagonalization controls")
    mpi = mpi if mpi is not None else MPIContext()
    vectors, values, applied, applications = _initial_iterative_subspace(
        operator,
        number_of_roots,
        initial_vectors,
        mpi,
        initial_is_orthonormal=initial_is_orthonormal,
    )
    root_tolerances = _iterative_root_tolerances(
        number_of_roots, tolerance, occupied_roots, full_accuracy
    )
    unconverged = np.ones(number_of_roots, dtype=bool)
    residuals = applied - vectors * values[None, :]
    for iteration in range(1, max_iterations + 1):
        corrections = _qe_precondition(
            residuals[:, unconverged],
            values[unconverged],
            diagonal,
        )
        corrections = _orthonormalize(
            corrections, against=vectors, mpi=mpi
        )
        if not corrections.shape[1]:
            break
        applied_corrections = operator(corrections)
        applications += corrections.shape[1]
        basis = np.column_stack((vectors, corrections))
        h_basis = np.column_stack((applied, applied_corrections))
        h_small = mpi.sum_array(basis.conj().T @ h_basis)
        s_small = mpi.sum_array(basis.conj().T @ basis)
        new_values, coefficients = _lowest_generalized_eigh(
            h_small, s_small, number_of_roots
        )
        old_values = values
        vectors = basis @ coefficients
        applied = h_basis @ coefficients
        values = new_values
        residuals = applied - vectors * values[None, :]
        unconverged = np.abs(values - old_values) >= root_tolerances
        if not np.any(unconverged):
            return DavidsonResult(
                values,
                vectors,
                _column_norms(residuals, mpi),
                iteration,
                True,
                0,
                applications,
            )
    return DavidsonResult(
        values,
        vectors,
        _column_norms(residuals, mpi),
        max_iterations,
        False,
        int(np.count_nonzero(unconverged)),
        applications,
    )


def _blocked_orthonormalize(
    vectors: np.ndarray, block_size: int, mpi: MPIContext
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for start in range(0, vectors.shape[1], block_size):
        against = np.column_stack(blocks) if blocks else None
        block = _orthonormalize(
            vectors[:, start : start + block_size],
            against=against,
            mpi=mpi,
        )
        if block.shape[1]:
            blocks.append(block)
    return np.column_stack(blocks) if blocks else np.empty(
        (vectors.shape[0], 0), dtype=complex
    )


def rmm_diis(
    operator: BlockOperator,
    diagonal: np.ndarray,
    number_of_roots: int,
    *,
    initial_vectors: np.ndarray,
    tolerance: float = 1.0e-8,
    max_iterations: int = 20,
    history_dimension: int = 4,
    converge: bool = False,
    gram_schmidt_block: int = 16,
    occupied_roots: int | None = None,
    full_accuracy: bool = False,
    initial_is_orthonormal: bool = False,
    mpi: MPIContext | None = None,
) -> DavidsonResult:
    """Residual-minimization method with per-band DIIS histories."""
    if history_dimension < 2:
        raise ValueError("diago_rmm_ndim must be at least 2")
    if gram_schmidt_block < 1:
        raise ValueError("diago_gs_nblock must be at least 1")
    mpi = mpi if mpi is not None else MPIContext()
    vectors, values, applied, applications = _initial_iterative_subspace(
        operator,
        number_of_roots,
        initial_vectors,
        mpi,
        initial_is_orthonormal=initial_is_orthonormal,
    )
    tolerances = _iterative_root_tolerances(
        number_of_roots, tolerance, occupied_roots, full_accuracy
    )
    vector_history: list[list[np.ndarray]] = [
        [] for _ in range(number_of_roots)
    ]
    residual_history: list[list[np.ndarray]] = [
        [] for _ in range(number_of_roots)
    ]
    residuals = applied - vectors * values[None, :]
    unconverged = np.ones(number_of_roots, dtype=bool)
    limit = max_iterations if converge else 1
    for iteration in range(1, limit + 1):
        trial = vectors.copy()
        for band in range(number_of_roots):
            vector_history[band].append(vectors[:, band].copy())
            residual_history[band].append(residuals[:, band].copy())
            if len(vector_history[band]) > history_dimension:
                vector_history[band].pop(0)
                residual_history[band].pop(0)
            count = len(vector_history[band])
            if count > 1:
                phi = np.column_stack(vector_history[band])
                res = np.column_stack(residual_history[band])
                residual_gram = mpi.sum_array(res.conj().T @ res)
                overlap = mpi.sum_array(phi.conj().T @ phi)
                _, coefficients = _lowest_generalized_eigh(
                    residual_gram, overlap, 1
                )
                trial[:, band] = phi @ coefficients[:, 0]
        trial = _blocked_orthonormalize(
            trial, gram_schmidt_block, mpi
        )
        if trial.shape[1] < number_of_roots:
            break
        trial = trial[:, :number_of_roots]
        h_trial = operator(trial)
        applications += number_of_roots
        h_small = mpi.sum_array(trial.conj().T @ h_trial)
        h_small = 0.5 * (h_small + h_small.conj().T)
        new_values, rotation = np.linalg.eigh(h_small)
        old_values = values
        values = np.real(new_values[:number_of_roots])
        rotation = rotation[:, :number_of_roots]
        vectors = trial @ rotation
        applied = h_trial @ rotation
        residuals = applied - vectors * values[None, :]
        corrections = _qe_precondition(residuals, values, diagonal)
        corrections = _orthonormalize(
            corrections, against=vectors, mpi=mpi
        )
        if corrections.shape[1]:
            h_corrections = operator(corrections)
            applications += corrections.shape[1]
            basis = np.column_stack((vectors, corrections))
            h_basis = np.column_stack((applied, h_corrections))
            h_small = mpi.sum_array(basis.conj().T @ h_basis)
            s_small = mpi.sum_array(basis.conj().T @ basis)
            values, coefficients = _lowest_generalized_eigh(
                h_small, s_small, number_of_roots
            )
            vectors = basis @ coefficients
            applied = h_basis @ coefficients
            residuals = applied - vectors * values[None, :]
        unconverged = np.abs(values - old_values) >= tolerances
        if not np.any(unconverged):
            return DavidsonResult(
                values,
                vectors,
                _column_norms(residuals, mpi),
                iteration,
                True,
                0,
                applications,
            )
    return DavidsonResult(
        values,
        vectors,
        _column_norms(residuals, mpi),
        limit,
        not converge,
        0 if not converge else int(np.count_nonzero(unconverged)),
        applications,
    )
