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
BlockOperatorInto = Callable[[np.ndarray, np.ndarray], np.ndarray]
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


def _form_residual(
    applied: np.ndarray,
    vectors: np.ndarray,
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Return Hpsi-epsilon*psi with only its final band block allocated."""
    residual = np.array(applied, dtype=np.complex128, order="F", copy=True)
    _load_native_fft().subtract_band_energies(
        residual,
        np.asarray(vectors, dtype=np.complex128),
        np.ascontiguousarray(eigenvalues, dtype=np.float64),
    )
    return residual


def _column_expectations(
    left: np.ndarray,
    right: np.ndarray,
    mpi: MPIContext,
) -> np.ndarray:
    """Global columnwise <left|right> without an Npw-by-Nband temporary."""
    local = _load_native_fft().column_inner_products(
        np.asarray(left, dtype=np.complex128),
        np.asarray(right, dtype=np.complex128),
    )
    return mpi.sum_array(local)


def _matmul_f(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """BLAS product written directly into a column-major solver block."""
    result = np.empty(
        (left.shape[0], right.shape[1]), dtype=np.complex128, order="F"
    )
    np.matmul(left, right, out=result)
    return result


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
    iterations: float
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

    def apply_into(
        self, coefficients: np.ndarray, out: np.ndarray
    ) -> np.ndarray:
        """Apply H directly into solver-owned storage."""
        if self.timers is not None:
            with self.timers.measure("h_psi"):
                return self._apply(coefficients, out=out)
        return self._apply(coefficients, out=out)

    def _apply(
        self, coefficients: np.ndarray, out: np.ndarray | None = None
    ) -> np.ndarray:
        vectors = np.asarray(coefficients)
        was_vector = vectors.ndim == 1
        if was_vector:
            vectors = vectors[:, None]
        target = out
        if target is not None and was_vector and np.asarray(target).ndim == 1:
            target = np.asarray(target)[:, None]
        if self.timers is None:
            result = self.local_workspace.apply(
                self.real_potential,
                vectors,
                native_potential_layout=self.native_potential_layout,
                out=target,
                diagonal=self.local_kinetic,
            )
        else:
            with self.timers.measure("vloc_psi"):
                result = self.local_workspace.apply(
                    self.real_potential,
                    vectors,
                    native_potential_layout=self.native_potential_layout,
                    out=target,
                    diagonal=self.local_kinetic,
                )
        # The final FFT gather has already accumulated T_G*psi_G while both
        # source and destination rows were hot in the native kernel.
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
            nonlocal_buffer: np.ndarray | None = None
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
                            if nonlocal_buffer is None:
                                nonlocal_buffer = np.empty_like(vectors)
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
                        overlap = _load_native_fft().projector_overlaps(
                            beta, vectors
                        )
                        if overlap is None:
                            if nonlocal_buffer is None:
                                nonlocal_buffer = np.empty_like(vectors)
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
                    if result.flags.f_contiguous:
                        coupled = np.asfortranarray(coupled)
                    else:
                        coupled = np.ascontiguousarray(coupled)
                    added_directly = (
                        phase is None
                        and _load_native_fft().add_projector_product(
                            result, beta, coupled
                        )
                    )
                    if added_directly:
                        offset += count
                        continue
                    if nonlocal_buffer is None:
                        nonlocal_buffer = np.empty_like(vectors)
                    np.matmul(beta, coupled, out=nonlocal_buffer)
                    if phase is not None:
                        nonlocal_buffer *= phase[:, None]
                    result += nonlocal_buffer
                    offset += count
        return result[:, 0] if was_vector else result


class SpinorPlaneWaveHamiltonian:
    """Apply a norm-conserving two-component Pauli Hamiltonian.

    Coefficients follow QE's combined-index layout: all up-spin plane waves
    first, followed by all down-spin plane waves. The four real local fields
    are ordered ``(scalar, x, y, z)``.
    """

    def __init__(
        self,
        basis: PlaneWaveBasis,
        real_potential: np.ndarray,
        projector_terms: tuple[ProjectorTerm, ...] = (),
        *,
        local_workspace: LocalPotentialWorkspace | None = None,
        timers: TimingRegistry | None = None,
        potential_average: float = 0.0,
    ) -> None:
        self.basis = basis
        self.local_workspace = (
            local_workspace
            if local_workspace is not None
            else LocalPotentialWorkspace(basis, tuple(real_potential.shape[1:]))
        )
        self.mpi = self.local_workspace.mpi
        self.timers = timers
        fields = np.asarray(real_potential, dtype=np.float64)
        local_shape = (
            self.local_workspace.shape[0],
            self.local_workspace.shape[1],
            self.local_workspace.local_slab.stop
            - self.local_workspace.local_slab.start,
        )
        if fields.shape == (4, *self.local_workspace.shape):
            fields = fields[:, :, :, self.local_workspace.local_slab]
        if fields.shape != (4, *local_shape):
            raise ValueError("spinor local potential has the wrong shape")
        self.real_potential = np.ascontiguousarray(fields)
        scalar_rows = self.local_workspace.local_plane_wave_indices
        global_rows = len(basis)
        self.local_rows = np.concatenate(
            (scalar_rows, global_rows + scalar_rows)
        )
        kinetic = basis.kinetic
        local_kinetic = kinetic[scalar_rows] if self.mpi.size > 1 else kinetic
        self.local_kinetic = np.concatenate((local_kinetic, local_kinetic))
        self.potential_average = float(potential_average)
        self.projector_terms = projector_terms

    @property
    def diagonal(self) -> np.ndarray:
        result = self.local_kinetic + self.potential_average
        for term in self.projector_terms:
            if isinstance(term, (FactorizedProjectorTerm, PackedProjectorTerm)):
                raise ValueError(
                    "spinor Hamiltonian requires materialized projectors"
                )
            beta, coupling = term
            result = result + np.real(
                np.einsum(
                    "gi,ij,gj->g", beta, coupling, beta.conj(), optimize=True
                )
            )
        return result

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        return self._apply(coefficients)

    def apply_into(
        self, coefficients: np.ndarray, out: np.ndarray
    ) -> np.ndarray:
        return self._apply(coefficients, out=out)

    def _apply(
        self, coefficients: np.ndarray, out: np.ndarray | None = None
    ) -> np.ndarray:
        vectors = np.asarray(coefficients, dtype=np.complex128)
        was_vector = vectors.ndim == 1
        if was_vector:
            vectors = vectors[:, None]
        if vectors.shape[0] % 2:
            raise ValueError("spinor coefficient dimension must be even")
        plane_waves = vectors.shape[0] // 2
        up, down = vectors[:plane_waves], vectors[plane_waves:]
        wave_grid = self.local_workspace.coefficients_to_grid(
            np.concatenate((up, down), axis=1)
        )
        bands = vectors.shape[1]
        up_grid, down_grid = wave_grid[..., :bands], wave_grid[..., bands:]
        scalar, x_field, y_field, z_field = self.real_potential
        local_grid = np.empty_like(wave_grid)
        local_grid[..., :bands] = (
            (scalar + z_field)[..., None] * up_grid
            + (x_field - 1j * y_field)[..., None] * down_grid
        )
        local_grid[..., bands:] = (
            (x_field + 1j * y_field)[..., None] * up_grid
            + (scalar - z_field)[..., None] * down_grid
        )
        local_coefficients = self.local_workspace.grid_to_coefficients(
            local_grid
        )
        result = (
            np.empty_like(vectors, order="F")
            if out is None
            else np.asarray(out).reshape(vectors.shape)
        )
        result[:plane_waves] = local_coefficients[:, :bands]
        result[plane_waves:] = local_coefficients[:, bands:]
        result += self.local_kinetic[:, None] * vectors
        for term in self.projector_terms:
            if isinstance(term, (FactorizedProjectorTerm, PackedProjectorTerm)):
                raise ValueError(
                    "spinor Hamiltonian requires materialized projectors"
                )
            beta, coupling = term
            overlap = self.mpi.sum_array(beta.conj().T @ vectors)
            result += beta @ (coupling @ overlap)
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
    gram = block.conj().T @ block
    gram = 0.5 * (gram + gram.conj().T)
    squared, rotation = np.linalg.eigh(gram)
    order = np.argsort(squared)[::-1]
    squared = np.maximum(np.real(squared[order]), 0.0)
    if not len(squared) or squared[0] == 0.0:
        return np.empty((block.shape[0], 0), dtype=complex)
    rank = int(
        np.count_nonzero(
            squared > threshold**2 * squared[0]
        )
    )
    transform = rotation[:, order[:rank]]
    transform /= np.sqrt(squared[:rank])[None, :]
    return block @ transform


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
    #
    # The physical/numerical definition remains visible here:
    #   x_Gn = 2 * (H_GG - epsilon_n)                         [Ry]
    #   d_Gn = 1/2 * [1 + x_Gn + sqrt(1 + (x_Gn - 1)^2)]
    #   delta_psi_Gn = 2 * residual_Gn / d_Gn
    # Cython only fuses this elementwise matrix traversal and parallelizes
    # its plane-wave rows; it does not choose or alter the preconditioner.
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
    initial_applied: np.ndarray | None = None,
    mpi: MPIContext | None = None,
    global_dimension: int | None = None,
    global_row_indices: np.ndarray | None = None,
    timers: TimingRegistry | None = None,
    operator_into: BlockOperatorInto | None = None,
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
    if initial_applied is None:
        _apply_into(
            operator,
            operator_into,
            basis,
            applied_storage[:, :active_columns],
        )
        hamiltonian_applications = active_columns
    else:
        supplied_applied = np.asarray(initial_applied, dtype=np.complex128)
        if supplied_applied.shape != basis.shape:
            raise ValueError("initial_applied must match the initial Ritz block")
        applied_storage[:, :active_columns] = supplied_applied
        hamiltonian_applications = 0
    applied_basis = applied_storage[:, :active_columns]
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

    def ritz_pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            vectors = _matmul_f(basis, coefficients)
            # Form H|psi>-epsilon|psi> directly in its final owner.  The
            # former separate H|psi> Ritz matrix was only needed at a rare
            # Davidson restart and raised the large-Npw peak by one band
            # block on every rank.
            residual = _matmul_f(applied_basis, coefficients)
            _load_native_fft().subtract_band_energies(
                residual,
                vectors,
                np.ascontiguousarray(roots, dtype=np.float64),
            )
        return roots, vectors, residual

    if initial_is_ritz:
        # QE cegterg(lrot=.true.) trusts rotate_wfc: it does not perform a
        # second initial reduced diagonalization, but starts from the rotated
        # vectors and diagonal projected energies with vc=I.
        if initial_eigenvalues is None:
            projected_diagonal = _column_expectations(
                basis, applied_basis, mpi
            )
            values = np.real(projected_diagonal[:number_of_roots])
        else:
            values = np.asarray(
                initial_eigenvalues[:number_of_roots], dtype=float
            )
        ritz_vectors = basis[:, :number_of_roots]
        residuals = _form_residual(
            applied_basis[:, :number_of_roots], ritz_vectors, values
        )
    else:
        values, ritz_vectors, residuals = ritz_pairs()
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
        # The correction is now owned by basis_storage.  The preceding Ritz
        # pair is no longer used, so release both band blocks before Hpsi and
        # before constructing the replacement pair.  This prevents old and
        # new Ritz/residual blocks from overlapping at the Davidson peak.
        del ritz_vectors, residuals
        _apply_into(
            operator,
            operator_into,
            corrections,
            applied_storage[:, old_columns:active_columns],
        )
        applied_basis = applied_storage[:, :active_columns]
        hamiltonian_applications += corrections.shape[1]
        update_projected(old_columns)
        values, ritz_vectors, residuals = ritz_pairs()
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
                # Reconstruct H|psi> = residual + epsilon|psi> directly in
                # its destination.  This preserves QE's no-extra-H restart
                # while avoiding a permanently live H|psi> Ritz matrix.
                np.multiply(
                    ritz_vectors,
                    values[None, :],
                    out=applied_storage[:, :active_columns],
                )
                applied_storage[:, :active_columns] += residuals
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


def _apply_into(
    operator: BlockOperator,
    operator_into: BlockOperatorInto | None,
    vectors: np.ndarray,
    destination: np.ndarray,
) -> np.ndarray:
    """Evaluate H vectors in solver-owned storage when supported."""
    if operator_into is not None:
        return operator_into(vectors, destination)
    destination[...] = operator(vectors)
    return destination


def _initial_iterative_subspace(
    operator: BlockOperator,
    number_of_roots: int,
    initial_vectors: np.ndarray,
    mpi: MPIContext,
    *,
    initial_is_orthonormal: bool,
    operator_into: BlockOperatorInto | None = None,
    initial_applied: np.ndarray | None = None,
    initial_eigenvalues: np.ndarray | None = None,
    initial_is_ritz: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    trial = np.asarray(initial_vectors, dtype=complex)
    if trial.ndim != 2 or trial.shape[1] < number_of_roots:
        raise ValueError(
            "initial_vectors must contain at least number_of_roots columns"
        )
    vectors = (
        np.array(
            trial[:, :number_of_roots],
            dtype=np.complex128,
            order="F" if initial_is_ritz else "K",
            copy=True,
        )
        if initial_is_orthonormal
        else _orthonormalize(trial, mpi=mpi)[:, :number_of_roots]
    )
    if vectors.shape[1] < number_of_roots:
        raise ValueError("initial vectors are linearly dependent")
    if initial_is_ritz:
        if initial_applied is None or initial_eigenvalues is None:
            raise ValueError(
                "initial Ritz vectors require H-vectors and eigenvalues"
            )
        applied = np.array(
            initial_applied, dtype=np.complex128, order="F", copy=True
        )
        if applied.shape != vectors.shape:
            raise ValueError("initial_applied must match initial vectors")
        return (
            vectors,
            np.asarray(
                initial_eigenvalues[:number_of_roots], dtype=float
            ).copy(),
            applied,
            0,
        )
    applied = np.empty_like(vectors, order="F")
    _apply_into(operator, operator_into, vectors, applied)
    projected = mpi.sum_array(vectors.conj().T @ applied)
    projected = 0.5 * (projected + projected.conj().T)
    values, rotation = np.linalg.eigh(projected)
    rotation = rotation[:, :number_of_roots]
    return (
        _matmul_f(vectors, rotation),
        np.real(values[:number_of_roots]),
        _matmul_f(applied, rotation),
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
    operator_into: BlockOperatorInto | None = None,
    initial_applied: np.ndarray | None = None,
    initial_eigenvalues: np.ndarray | None = None,
    initial_is_ritz: bool = False,
) -> DavidsonResult:
    """QE ``ccgdiagg`` band-by-band conjugate-gradient solver.

    ``diagonal`` is the rank-local kinetic energy in Hartree.  QE builds its
    CG preconditioner from ``g2kin`` in Rydberg units and then performs an
    analytic two-vector line minimisation.  The orbital and H-orbital are
    views of their persistent band blocks, so only the five genuine work
    vectors are retained outside the band loop.
    """
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
        operator_into=operator_into,
        initial_applied=initial_applied,
        initial_eigenvalues=initial_eigenvalues,
        initial_is_ritz=initial_is_ritz,
    )
    root_tolerances = _iterative_root_tolerances(
        number_of_roots,
        tolerance,
        occupied_roots,
        full_accuracy,
    )
    kinetic_ry = 2.0 * np.asarray(diagonal, dtype=float)
    precondition = (
        1.0 + kinetic_ry + np.sqrt(1.0 + (kinetic_ry - 1.0) ** 2)
    )
    gradient = np.empty(vectors.shape[0], dtype=np.complex128)
    conjugate = np.empty_like(gradient)
    h_conjugate = np.empty_like(gradient)
    previous_metric_gradient = np.empty_like(gradient)
    projected_psi = np.empty_like(gradient)
    unconverged = 0
    iteration_sum = 0
    for band in range(number_of_roots):
        psi = vectors[:, band]
        hpsi = applied[:, band]
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
        converged = False
        previous_gg = 0.0
        previous_cg_norm = 0.0
        previous_sine = 0.0
        for iteration in range(1, max_iterations + 1):
            # QE applies P to H|psi> and |psi> separately, then removes the
            # component parallel to the current orbital in the P^2 metric.
            np.divide(hpsi, precondition, out=gradient)
            np.divide(psi, precondition, out=projected_psi)
            moments = mpi.sum_array(
                np.asarray(
                    [np.vdot(psi, gradient), np.vdot(psi, projected_psi)],
                    dtype=np.complex128,
                )
            )
            projected_energy = float(np.real(moments[0] / moments[1]))
            gradient -= projected_energy * projected_psi
            if band:
                coefficients = mpi.sum_array(
                    vectors[:, :band].conj().T @ gradient
                )
                gradient -= vectors[:, :band] @ coefficients

            if iteration > 1:
                cross = float(
                    np.real(_global_inner(
                        gradient, previous_metric_gradient, mpi
                    ))
                )
            np.multiply(gradient, precondition, out=previous_metric_gradient)
            gg = float(
                np.real(_global_inner(
                    gradient, previous_metric_gradient, mpi
                ))
            )
            if not np.isfinite(gg) or gg <= 1.0e-30:
                break
            if iteration == 1:
                conjugate[...] = gradient
            else:
                gamma = (gg - cross) / max(previous_gg, 1.0e-300)
                # After the previous line rotation the old conjugate vector
                # is no longer exactly perpendicular to psi.  QE removes the
                # known analytic component without another global reduction.
                conjugate *= gamma
                conjugate += gradient
                conjugate -= (
                    gamma * previous_cg_norm * previous_sine
                ) * psi
            previous_gg = gg

            _apply_into(
                operator, operator_into, conjugate, h_conjugate
            )
            applications += 1
            cg_squared = float(
                np.real(_global_inner(conjugate, conjugate, mpi))
            )
            if cg_squared <= 1.0e-30:
                break
            cg_norm = np.sqrt(cg_squared)
            line_terms = mpi.sum_array(
                np.asarray(
                    [
                        np.vdot(psi, h_conjugate),
                        np.vdot(conjugate, h_conjugate),
                    ],
                    dtype=np.complex128,
                )
            )
            a0 = 2.0 * float(np.real(line_terms[0])) / cg_norm
            b0 = float(np.real(line_terms[1])) / cg_squared
            denominator = energy - b0
            theta = 0.5 * np.arctan(
                a0 / denominator
                if denominator != 0.0
                else np.copysign(np.inf, a0)
            )
            cosine = float(np.cos(theta))
            sine = float(np.sin(theta))
            cos2 = cosine * cosine - sine * sine
            sin2 = 2.0 * cosine * sine
            roots = np.asarray(
                [
                    0.5 * ((energy - b0) * cos2 + a0 * sin2 + energy + b0),
                    0.5 * (-(energy - b0) * cos2 - a0 * sin2 + energy + b0),
                ]
            )
            if roots[1] < roots[0]:
                theta += 0.5 * np.pi
                cosine = float(np.cos(theta))
                sine = float(np.sin(theta))
            old_energy = energy
            energy = float(np.min(roots))
            psi *= cosine
            psi += (sine / cg_norm) * conjugate
            hpsi *= cosine
            hpsi += (sine / cg_norm) * h_conjugate
            previous_cg_norm = cg_norm
            previous_sine = sine
            if abs(energy - old_energy) < root_tolerances[band]:
                converged = True
                break
        if not converged:
            unconverged += 1
        # QE includes the initial H|psi> evaluation in avg_iter.
        iteration_sum += iteration + 1
        values[band] = energy
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    applied = applied[:, order]
    residuals = _form_residual(applied, vectors, values)
    return DavidsonResult(
        values,
        vectors,
        _column_norms(residuals, mpi),
        max(1, int(round(iteration_sum / number_of_roots))),
        unconverged == 0,
        unconverged,
        applications,
    )


def _paro_bpcg(
    operator: BlockOperator,
    diagonal: np.ndarray,
    reference_vectors: np.ndarray,
    initial_vectors: np.ndarray,
    initial_applied: np.ndarray,
    eigenvalues: np.ndarray,
    tolerance: float,
    mpi: MPIContext,
    *,
    max_iterations: int = 5,
    operator_into: BlockOperatorInto | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve QE ``bpcg_k`` corrections for norm-conserving ``S=I``."""
    count = initial_vectors.shape[1]
    corrections = np.zeros_like(initial_vectors, order="F")
    h_corrections = np.zeros_like(initial_vectors, order="F")
    if count == 0:
        return corrections, h_corrections, 0

    rhs = initial_vectors * eigenvalues[None, :] - initial_applied
    preconditioned = _qe_precondition(rhs, eigenvalues, diagonal)
    projection = mpi.sum_array(reference_vectors.conj().T @ preconditioned)
    preconditioned -= reference_vectors @ projection
    g0 = np.real(_column_expectations(preconditioned, rhs, mpi))
    thresholds = np.maximum(0.01 * tolerance, 0.01 * g0)
    directions = np.array(preconditioned, order="F", copy=True)
    previous_function = np.zeros(count, dtype=float)
    active = np.arange(count, dtype=int)
    applications = 0

    for cg_iteration in range(1, max_iterations + 1):
        if not len(active):
            break
        active_directions = directions[:, active]
        h_directions = np.empty_like(active_directions, order="F")
        _apply_into(
            operator, operator_into, active_directions, h_directions
        )
        applications += len(active)
        active_energies = eigenvalues[active]
        gamma = np.real(
            _column_expectations(active_directions, h_directions, mpi)
        ) - active_energies * np.real(
            _column_expectations(active_directions, active_directions, mpi)
        )
        safe_gamma = np.where(
            np.abs(gamma) > 1.0e-30,
            gamma,
            np.copysign(1.0e-30, gamma + (gamma == 0.0)),
        )
        alpha = g0[active] / safe_gamma
        corrections[:, active] += active_directions * alpha[None, :]
        h_corrections[:, active] += h_directions * alpha[None, :]

        gradient = (
            rhs[:, active]
            + corrections[:, active] * active_energies[None, :]
            - h_corrections[:, active]
        )
        old_preconditioned = preconditioned[:, active]
        g2 = np.real(
            _column_expectations(old_preconditioned, gradient, mpi)
        )
        new_preconditioned = _qe_precondition(
            gradient, active_energies, diagonal
        )
        projection = mpi.sum_array(
            reference_vectors.conj().T @ new_preconditioned
        )
        new_preconditioned -= reference_vectors @ projection
        g1 = np.real(
            _column_expectations(new_preconditioned, gradient, mpi)
        )

        correction_norm = np.real(
            _column_expectations(
                corrections[:, active], corrections[:, active], mpi
            )
        )
        correction_h = np.real(
            _column_expectations(
                corrections[:, active], h_corrections[:, active], mpi
            )
        )
        correction_rhs = np.real(
            _column_expectations(corrections[:, active], rhs[:, active], mpi)
        )
        function_value = 0.5 * (
            correction_h - active_energies * correction_norm
        ) - correction_rhs
        function_increased = function_value > previous_function[active]
        rollback_step = (
            function_increased & (previous_function[active] < 0.0)
        )
        if np.any(rollback_step):
            failed = active[rollback_step]
            corrections[:, failed] -= (
                directions[:, failed] * alpha[rollback_step][None, :]
            )
            h_corrections[:, failed] -= (
                h_directions[:, rollback_step]
                * alpha[rollback_step][None, :]
            )

        finished = (
            (np.abs(g1) < thresholds[active])
            | function_increased
            | (cg_iteration == max_iterations)
        )
        continuing_mask = ~finished
        if not np.any(continuing_mask):
            break
        continuing = active[continuing_mask]
        denominator = np.where(
            np.abs(g0[continuing]) > 1.0e-30,
            g0[continuing],
            np.copysign(
                1.0e-30,
                g0[continuing] + (g0[continuing] == 0.0),
            ),
        )
        beta = (
            g1[continuing_mask] - g2[continuing_mask]
        ) / denominator
        directions[:, continuing] = (
            new_preconditioned[:, continuing_mask]
            + directions[:, continuing] * beta[None, :]
        )
        preconditioned[:, continuing] = (
            new_preconditioned[:, continuing_mask]
        )
        g0[continuing] = g1[continuing_mask]
        previous_function[continuing] = function_value[continuing_mask]
        active = continuing

    return corrections, h_corrections, applications


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
    operator_into: BlockOperatorInto | None = None,
    initial_applied: np.ndarray | None = None,
    initial_eigenvalues: np.ndarray | None = None,
    initial_is_ritz: bool = False,
) -> DavidsonResult:
    """QE ``paro_k_new`` solver for the norm-conserving ``S=I`` case."""
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("invalid ParO diagonalization controls")
    mpi = mpi if mpi is not None else MPIContext()
    # paro_k_new always recomputes H|psi> and calls rotate_HSpsi, including
    # after wfcinit has supplied an already-rotated Ritz block.
    del initial_applied, initial_eigenvalues, initial_is_ritz
    vectors, values, applied, applications = _initial_iterative_subspace(
        operator,
        number_of_roots,
        initial_vectors,
        mpi,
        initial_is_orthonormal=initial_is_orthonormal,
        operator_into=operator_into,
        initial_applied=None,
        initial_eigenvalues=None,
        initial_is_ritz=False,
    )
    # QE's btype argument is unused by paro_k_new: all trusted roots use ethr.
    del occupied_roots, full_accuracy
    root_tolerances = np.full(number_of_roots, tolerance, dtype=float)
    maximum_subspace = number_of_roots + max(
        (number_of_roots + 1) // 2, 4
    )
    maximum_subspace = min(
        vectors.shape[0] * mpi.size, maximum_subspace
    )
    basis_storage = np.empty(
        (vectors.shape[0], maximum_subspace),
        dtype=np.complex128,
        order="F",
    )
    applied_storage = np.empty_like(basis_storage)
    basis_storage[:, :number_of_roots] = vectors
    applied_storage[:, :number_of_roots] = applied
    converged_roots = np.zeros(number_of_roots, dtype=bool)

    for _iteration in range(1, max_iterations + 1):
        converged_count = int(np.count_nonzero(converged_roots))
        active_count = min(
            (maximum_subspace - converged_count) // 2,
            maximum_subspace - number_of_roots,
        )
        base_count = max(converged_count + active_count, number_of_roots)
        trusted_count = min(
            converged_count + active_count, number_of_roots
        )
        if active_count <= 0:
            break

        trusted_unconverged = np.flatnonzero(
            ~converged_roots[:trusted_count]
        )
        candidate_columns = list(map(int, trusted_unconverged))
        candidate_energies = list(values[trusted_unconverged])
        if len(candidate_columns) < active_count:
            if not len(trusted_unconverged):
                break
            last_energy = float(values[trusted_unconverged[-1]])
            for column in range(number_of_roots, base_count):
                candidate_columns.append(column)
                candidate_energies.append(last_energy)
                if len(candidate_columns) == active_count:
                    break
        candidate_columns = candidate_columns[:active_count]
        candidate_energies_array = np.asarray(
            candidate_energies[:active_count], dtype=float
        )

        corrections, h_corrections, correction_applications = _paro_bpcg(
            operator,
            diagonal,
            basis_storage[:, :number_of_roots],
            basis_storage[:, candidate_columns],
            applied_storage[:, candidate_columns],
            candidate_energies_array,
            tolerance,
            mpi,
            operator_into=operator_into,
        )
        applications += correction_applications
        dimension = base_count + active_count
        basis_storage[:, base_count:dimension] = corrections
        applied_storage[:, base_count:dimension] = h_corrections
        basis = basis_storage[:, :dimension]
        h_basis = applied_storage[:, :dimension]
        h_small = mpi.sum_array(basis.conj().T @ h_basis)
        s_small = mpi.sum_array(basis.conj().T @ basis)
        new_values, coefficients = _lowest_generalized_eigh(
            h_small, s_small, dimension
        )
        rotated_basis = _matmul_f(basis, coefficients)
        rotated_applied = _matmul_f(h_basis, coefficients)
        basis_storage[:, :dimension] = rotated_basis
        applied_storage[:, :dimension] = rotated_applied

        old_values = values.copy()
        values = new_values[:number_of_roots].copy()
        converged_roots.fill(False)
        converged_roots[:trusted_count] = (
            np.abs(values[:trusted_count] - old_values[:trusted_count])
            < root_tolerances[:trusted_count]
        )
        if np.all(converged_roots):
            break

    vectors = np.array(
        basis_storage[:, :number_of_roots], order="F", copy=True
    )
    applied = np.array(
        applied_storage[:, :number_of_roots], order="F", copy=True
    )
    residuals = _form_residual(applied, vectors, values)
    number_unconverged = int(np.count_nonzero(~converged_roots))
    return DavidsonResult(
        values,
        vectors,
        _column_norms(residuals, mpi),
        applications / number_of_roots,
        number_unconverged == 0,
        number_unconverged,
        applications,
    )


def _blocked_orthonormalize(
    vectors: np.ndarray, block_size: int, mpi: MPIContext
) -> np.ndarray:
    output = np.empty_like(vectors)
    active = 0
    for start in range(0, vectors.shape[1], block_size):
        block = _orthonormalize(
            vectors[:, start : start + block_size],
            against=output[:, :active] if active else None,
            mpi=mpi,
        )
        if block.shape[1]:
            output[:, active : active + block.shape[1]] = block
            active += block.shape[1]
    return output[:, :active]


def _blocked_orthonormalize_pair(
    vectors: np.ndarray,
    applied: np.ndarray,
    block_size: int,
    mpi: MPIContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Blocked Gram--Schmidt while carrying ``H|psi>`` through rotations."""
    output = np.empty_like(vectors)
    h_output = np.empty_like(applied)
    active = 0
    for start in range(0, vectors.shape[1], block_size):
        stop = min(start + block_size, vectors.shape[1])
        width = stop - start
        block = vectors[:, start:stop].copy()
        h_block = applied[:, start:stop].copy()
        if active:
            for _ in range(2):
                projection = mpi.sum_array(
                    output[:, :active].conj().T @ block
                )
                block -= output[:, :active] @ projection
                h_block -= h_output[:, :active] @ projection
        gram = mpi.sum_array(block.conj().T @ block)
        gram = 0.5 * (gram + gram.conj().T)
        squared, rotation = np.linalg.eigh(gram)
        order = np.argsort(squared)[::-1]
        squared = np.maximum(np.real(squared[order]), 0.0)
        if not len(squared) or squared[0] <= 0.0:
            continue
        keep = squared > 1.0e-24 * squared[0]
        transform = rotation[:, order][:, keep]
        transform /= np.sqrt(squared[keep])[None, :]
        rank = transform.shape[1]
        output[:, active : active + rank] = block @ transform
        h_output[:, active : active + rank] = h_block @ transform
        active += rank
        if active >= vectors.shape[1]:
            break
    return output[:, :active], h_output[:, :active]


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
    operator_into: BlockOperatorInto | None = None,
    initial_applied: np.ndarray | None = None,
    initial_eigenvalues: np.ndarray | None = None,
    initial_is_ritz: bool = False,
) -> DavidsonResult:
    """QE ``crmmdiagg`` residual minimisation with bounded DIIS history."""
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
        operator_into=operator_into,
        initial_applied=initial_applied,
        initial_eigenvalues=initial_eigenvalues,
        initial_is_ritz=initial_is_ritz,
    )
    tolerances = _iterative_root_tolerances(
        number_of_roots, tolerance, occupied_roots, full_accuracy
    )
    local_dimension = vectors.shape[0]
    vector_history = np.empty(
        (local_dimension, number_of_roots, history_dimension),
        dtype=np.complex128,
        order="F",
    )
    applied_history = np.empty_like(vector_history)
    energy_history = np.empty(
        (number_of_roots, history_dimension), dtype=float
    )
    residuals = _form_residual(applied, vectors, values)
    unconverged = np.ones(number_of_roots, dtype=bool)
    kinetic_ry = 2.0 * np.asarray(diagonal, dtype=float)
    accumulated_iterations = 0
    completed = 0
    for idiis in range(history_dimension):
        completed = idiis + 1
        active = np.flatnonzero(unconverged)
        accumulated_iterations += len(active)
        if not len(active):
            break
        vector_history[:, active, idiis] = vectors[:, active]
        applied_history[:, active, idiis] = applied[:, active]
        energy_history[active, idiis] = values[active]

        # Minimise the residual norm independently in each band's DIIS
        # history.  Only tiny (ndiis x ndiis) matrices are materialised.
        for band in active:
            count = idiis + 1
            phi = vector_history[:, band, :count]
            hphi = applied_history[:, band, :count]
            history_energies = energy_history[band, :count]
            if count == 1:
                norm = np.sqrt(max(
                    float(np.real(_global_inner(phi[:, 0], phi[:, 0], mpi))),
                    1.0e-300,
                ))
                vectors[:, band] = phi[:, 0] / norm
                applied[:, band] = hphi[:, 0] / norm
                np.multiply(
                    phi[:, 0], history_energies[0],
                    out=residuals[:, band],
                )
                np.subtract(
                    hphi[:, 0], residuals[:, band],
                    out=residuals[:, band],
                )
                residuals[:, band] /= norm
            else:
                # R_i=Hphi_i-e_i phi_i.  Form R^H R from three tiny Gram
                # matrices instead of materializing Npw-by-ndiis residual
                # history for every band.  One packed reduction also replaces
                # the former separate residual/overlap collectives.
                hh_local = hphi.conj().T @ hphi
                hp_local = hphi.conj().T @ phi
                pp_local = phi.conj().T @ phi
                packed = mpi.sum_array(
                    np.concatenate(
                        (hh_local.ravel(), hp_local.ravel(), pp_local.ravel())
                    )
                )
                block_size = count * count
                hh = packed[:block_size].reshape(count, count)
                hp = packed[block_size : 2 * block_size].reshape(count, count)
                overlap = packed[2 * block_size :].reshape(count, count)
                residual_gram = hh.copy()
                residual_gram -= hp * history_energies[None, :]
                residual_gram -= (
                    history_energies[:, None] * hp.conj().T
                )
                residual_gram += (
                    history_energies[:, None]
                    * overlap
                    * history_energies[None, :]
                )
                try:
                    _, coefficients = _lowest_generalized_eigh(
                        residual_gram, overlap, 1
                    )
                    coefficient = coefficients[:, 0]
                except (np.linalg.LinAlgError, ValueError):
                    coefficient = np.zeros(count, dtype=np.complex128)
                    coefficient[-1] = 1.0
                vectors[:, band] = phi @ coefficient
                applied[:, band] = hphi @ coefficient
                np.multiply(
                    history_energies,
                    coefficient,
                    out=coefficient,
                )
                residuals[:, band] = (
                    applied[:, band] - phi @ coefficient
                )

        # Kresse--Furthmueller kinetic preconditioner and QE's bounded
        # quadratic line search (SREF=.5, SMIN=.05, SMAX=1).
        kinetic_expectation = np.real(
            mpi.sum_array(
                _load_native_fft().column_diagonal_expectations(
                    vectors, kinetic_ry
                )
            )[active]
        )
        kinetic_expectation = np.maximum(kinetic_expectation, 1.0e-14)
        # Kresse--Furthmueller diagonal, evaluated directly into correction:
        # x=G^2/(1.5<Ekin>), K=-4/(3<Ekin>) *
        # (27+18x+12x^2+8x^3)/(27+18x+12x^2+8x^3+16x^4).
        # The native elementwise loop avoids x, x^2, and kdiag matrices.
        correction = _load_native_fft().rmm_kinetic_precondition(
            residuals,
            kinetic_ry,
            np.ascontiguousarray(kinetic_expectation, dtype=np.float64),
            np.ascontiguousarray(active, dtype=np.int64),
        )
        h_correction = np.empty_like(correction, order="F")
        _apply_into(
            operator, operator_into, correction, h_correction
        )
        applications += len(active)
        for local_band, band in enumerate(active):
            psi = vectors[:, band]
            hpsi = applied[:, band]
            kpsi = correction[:, local_band]
            hkpsi = h_correction[:, local_band]
            elements = mpi.sum_array(np.asarray([
                np.vdot(psi, hpsi),
                np.vdot(kpsi, hpsi),
                np.vdot(kpsi, hkpsi),
                np.vdot(psi, psi),
                np.vdot(kpsi, psi),
                np.vdot(kpsi, kpsi),
            ], dtype=np.complex128))
            php, khp, khk, psp, ksp, ksk = np.real(elements)
            norm_ref = psp + 2.0 * ksp * 0.5 + ksk * 0.25
            energy0 = php / psp
            energy1 = (php + khp + 0.25 * khk) / norm_ref
            slope = 2.0 * (khp * psp - php * ksp) / (psp * psp)
            curvature = (energy1 - energy0 - 0.5 * slope) / 0.25
            if abs(curvature) > 1.0e-16:
                step = -0.5 * slope / curvature
            else:
                step = 1.0 if slope < 0.0 else 0.05
            step = float(np.clip(step, 0.05, 1.0))
            norm = np.sqrt(psp + 2.0 * ksp * step + ksk * step * step)
            first = 1.0 / norm
            second = step / norm
            psi *= first
            psi += second * kpsi
            hpsi *= first
            hpsi += second * hkpsi

        old_values = values.copy()
        values = np.real(_column_expectations(vectors, applied, mpi))
        residuals = _form_residual(applied, vectors, values)
        newly_converged = np.abs(values - old_values) < tolerances
        unconverged &= ~newly_converged

    vectors, applied = _blocked_orthonormalize_pair(
        vectors, applied, gram_schmidt_block, mpi
    )
    if vectors.shape[1] < number_of_roots:
        return DavidsonResult(
            values,
            vectors,
            _column_norms(residuals[:, : vectors.shape[1]], mpi),
            accumulated_iterations / max(number_of_roots, 1),
            False,
            number_of_roots - vectors.shape[1],
            applications,
        )
    vectors = vectors[:, :number_of_roots]
    applied = applied[:, :number_of_roots]
    h_small = mpi.sum_array(vectors.conj().T @ applied)
    h_small = 0.5 * (h_small + h_small.conj().T)
    values, rotation = np.linalg.eigh(h_small)
    values = np.real(values[:number_of_roots])
    rotation = rotation[:, :number_of_roots]
    vectors = _matmul_f(vectors, rotation)
    applied = _matmul_f(applied, rotation)
    residuals = _form_residual(applied, vectors, values)
    return DavidsonResult(
        values,
        vectors,
        _column_norms(residuals, mpi),
        accumulated_iterations / max(number_of_roots, 1),
        (not converge) or not np.any(unconverged),
        0 if not converge else int(np.count_nonzero(unconverged)),
        applications,
    )
