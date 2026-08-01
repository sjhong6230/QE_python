"""Matrix-free plane-wave Hamiltonian and block Davidson eigensolver."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Callable

import numpy as np
from scipy.linalg import eigh
from scipy.linalg.blas import zgemm

from .basis import LocalPotentialWorkspace, PlaneWaveBasis
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


ProjectorTerm = MaterializedProjectorTerm | FactorizedProjectorTerm


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
        potential_average: float | None = None,
    ) -> None:
        self.basis = basis
        self.potential_g = potential_g
        self.local_workspace = (
            local_workspace
            if local_workspace is not None
            else LocalPotentialWorkspace(basis.indices, potential_g.shape)
        )
        self.timers = timers
        self.real_potential = (
            np.asarray(real_potential)
            if real_potential is not None
            else self.local_workspace.prepare_potential(potential_g)
        )
        self.potential_average = (
            float(potential_average)
            if potential_average is not None
            else float(np.real(self.potential_g.flat[0]))
        )
        self.mpi = self.local_workspace.mpi
        self.local_rows = self.local_workspace.local_plane_wave_indices
        self.local_kinetic = (
            self.basis.kinetic[self.local_rows]
            if self.mpi.size > 1
            else self.basis.kinetic
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
                self.real_potential, vectors
            )
        else:
            with self.timers.measure("vloc_psi"):
                result += self.local_workspace.apply(
                    self.real_potential, vectors
                )
        if self.projector_terms:
            overlap_blocks: list[np.ndarray] = []
            descriptors: list[
                tuple[np.ndarray, np.ndarray, np.ndarray | None]
            ] = []
            modulated = np.empty_like(vectors)
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
                            np.multiply(
                                vectors,
                                np.conjugate(phase)[:, None],
                                out=modulated,
                            )
                            overlap_blocks.append(
                                zgemm(
                                    1.0,
                                    term.beta,
                                    modulated,
                                    trans_a=2,
                                )
                            )
                            descriptors.append(
                                (term.beta, term.coupling, phase)
                            )
                    else:
                        beta, coupling = term
                        overlap_blocks.append(
                            zgemm(
                                1.0,
                                beta,
                                vectors,
                                trans_a=2,
                            )
                        )
                        descriptors.append((beta, coupling, None))
                counts = [block.shape[0] for block in overlap_blocks]
                all_overlaps = self.mpi.sum_array(
                    np.vstack(overlap_blocks)
                )
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
                    contribution = zgemm(
                        1.0,
                        beta,
                        coupling @ overlaps,
                    )
                    if phase is not None:
                        contribution *= phase[:, None]
                    result += contribution
                    offset += count
        return result[:, 0] if was_vector else result


def _orthonormalize(
    vectors: np.ndarray,
    against: np.ndarray | None = None,
    threshold: float = 1.0e-12,
    mpi: MPIContext | None = None,
) -> np.ndarray:
    """Twice-project and return a rank-revealing orthonormal column basis."""
    block = np.asarray(vectors, dtype=complex).copy()
    if block.ndim == 1:
        block = block[:, None]
    mpi = mpi if mpi is not None else MPIContext()
    if against is not None and against.ndim == 2 and against.shape[1]:
        for _ in range(2):
            projection = against.conj().T @ block
            projection = mpi.sum_array(projection)
            block -= against @ projection
    if block.shape[1] == 0:
        return np.empty((block.shape[0], 0), dtype=complex)
    if mpi.size > 1:
        gram = mpi.sum_array(block.conj().T @ block)
        gram = 0.5 * (gram + gram.conj().T)
        squared, rotation = eigh(gram, check_finite=False)
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
    x_ry = 2.0 * (
        diagonal[:, None] - eigenvalues[None, :]
    )
    denominator_ry = 0.5 * (
        1.0
        + x_ry
        + np.sqrt(1.0 + (x_ry - 1.0) ** 2)
    )
    return (2.0 * residuals) / denominator_ry


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

    if initial_is_ritz:
        # rotate_wfc has already returned orthonormal Ritz vectors in the
        # required band order. SVD orthonormalization would apply another
        # arbitrary unitary rotation, invalidating cegterg's lrot assumption
        # that hc is diagonal and vc is the identity.
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
        (local_dimension, maximum_subspace), dtype=complex
    )
    applied_storage = np.empty_like(basis_storage)
    basis_storage[:, :active_columns] = basis
    basis = basis_storage[:, :active_columns]
    applied_storage[:, :active_columns] = operator(basis)
    applied_basis = applied_storage[:, :active_columns]
    hamiltonian_applications = active_columns

    def ritz_pairs() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        with (
            timers.measure("cdiaghg")
            if timers is not None
            else nullcontext()
        ):
            projected = mpi.sum_array(
                basis.conj().T @ applied_basis
            )
            projected = 0.5 * (
                projected + projected.conj().T
            )
            overlap = mpi.sum_array(basis.conj().T @ basis)
            overlap = 0.5 * (overlap + overlap.conj().T)
            roots, coefficients = eigh(
                projected,
                overlap,
                subset_by_index=(0, number_of_roots - 1),
                check_finite=False,
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
        projected_diagonal = mpi.sum_array(
            np.sum(basis.conj() * applied_basis, axis=0)
        )
        values = np.real(projected_diagonal[:number_of_roots])
        ritz_vectors = basis[:, :number_of_roots]
        applied_ritz = applied_basis[:, :number_of_roots]
        residuals = (
            applied_ritz - ritz_vectors * values[None, :]
        )
    else:
        values, ritz_vectors, applied_ritz, residuals = ritz_pairs()
    residual_norms = np.sqrt(
        np.real(mpi.sum_array(np.sum(np.abs(residuals) ** 2, axis=0)))
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
        # QE's cegterg retains a nonorthogonal correction space and solves a
        # generalized reduced problem.  That is safe with QE's incrementally
        # maintained overlap matrix, but in this matrix-free Python solver it
        # can become nearly rank deficient while Ritz values appear stationary.
        # A rank-revealing, twice-projected basis preserves the same correction
        # span and prevents false convergence from ill-conditioned overlaps.
        corrections = _orthonormalize(
            corrections,
            against=basis,
            mpi=mpi,
        )
        if corrections.shape[1] == 0:
            break
        old_columns = active_columns
        active_columns += corrections.shape[1]
        basis_storage[:, old_columns:active_columns] = corrections
        basis = basis_storage[:, :active_columns]
        applied_corrections = operator(corrections)
        applied_storage[:, old_columns:active_columns] = (
            applied_corrections
        )
        applied_basis = applied_storage[:, :active_columns]
        hamiltonian_applications += corrections.shape[1]
        values, ritz_vectors, applied_ritz, residuals = ritz_pairs()
        residual_norms = np.sqrt(
            np.real(
                mpi.sum_array(np.sum(np.abs(residuals) ** 2, axis=0))
            )
        )
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
                # Refresh H|psi> instead of carrying a long sequence of
                # projected linear combinations through Davidson restarts.
                # The extra application bounds accumulated FFT/BLAS roundoff
                # and keeps the reported residual tied to the actual operator.
                applied_storage[:, :active_columns] = operator(basis)
                applied_basis = applied_storage[:, :active_columns]
                hamiltonian_applications += basis.shape[1]

    return DavidsonResult(
        values,
        ritz_vectors,
        residual_norms,
        completed_iterations,
        False,
        number_unconverged,
        hamiltonian_applications,
    )
