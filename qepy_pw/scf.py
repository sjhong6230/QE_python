"""Spinless Kohn--Sham SCF orchestration corresponding to ``pw.x``.

Numerical kernels remain in the focused ``basis``, ``diagonalization``,
``ewald``, ``mixing``, ``occupations``, and ``xc`` modules.  This module owns
the calculation phases, rank-local object lifetimes, and the small immutable
records passed to the reporting layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Sequence
import os
import time
from typing import Callable, Literal

import numpy as np

from .basis import (
    FFTGridDescriptor,
    FFTScratchPool,
    LocalPotentialWorkspace,
    PlaneWaveBasis,
    _load_native_fft,
    fft_shape,
    make_bases,
    potential_matrix,
)
from .diagonalization import (
    FactorizedProjectorTerm,
    PackedProjectorTerm,
    PlaneWaveHamiltonian,
    ProjectorTerm,
    _lowest_generalized_eigh,
    _matmul_f,
    conjugate_gradient,
    davidson,
    parallel_orbital,
    rmm_diis,
)
from .errors import QEInputError, UnsupportedFeatureError, not_implemented
from .ewald import ewald_energy, ewald_forces, ewald_stress
from .input import PWInput
from .mixing import (
    DistributedBroydenMixer,
    LinearMixer,
    PlainBroydenMixer,
)
from .memory import (
    available_memory_bytes,
    current_pss_bytes,
    current_rss_bytes,
    current_uss_bytes,
    peak_rss_bytes,
    trim_allocator,
)
from .occupations import (
    default_number_of_bands,
    smeared_occupations,
    smearing_order,
    tetrahedron_occupations,
)
from .timing import TimingEntry, TimingRegistry
from .mpi import MPIContext
from .upf import LocalPotential, read_upf
from .xc import (
    GGA_POINT_BLOCK_SIZE,
    GGA_FUNCTIONALS,
    canonical_xc_name,
    pbe_unpolarized_components,
    pw92_lda_unpolarized,
    pz81_unpolarized,
)
from .symmetry import (
    ReciprocalDensitySymmetrizer,
    fft_factors,
    symmetrize_forces,
    symmetrize_stress,
)
from .threads import hybrid_thread_count


def _canonical_diagonalization(value: object) -> str:
    requested = str(value).strip()
    aliases = {
        "david": "david",
        "davidson": "david",
        "cg": "cg",
        "paro": "paro",
        "rmm": "rmm-davidson",
        "rmm-diis": "rmm-davidson",
        "rmm-davidson": "rmm-davidson",
        "rmm-paro": "rmm-paro",
        "direct": "dense",
        "dense": "dense",
    }
    canonical = aliases.get(requested.lower())
    if canonical is not None:
        return canonical
    if requested.lower() == "ppcg":
        raise QEInputError(
            "PPCG diagonalization not supported anymore (Dec. 2024)",
            routine="iosys",
        )
    raise QEInputError(
        f"diagonalization {requested} not implemented", routine="iosys"
    )


@dataclass
class SCFIteration:
    """Stable reporting data captured after one electronic iteration."""

    number: int
    total_energy_ha: float
    estimated_accuracy_ha: float
    cutoff_ry: float = 0.0
    mixing_beta: float = 0.0
    diagonalization: str = "david"
    davidson_threshold_ha: float = 0.0
    average_diagonalization_iterations: float = 0.0
    average_hamiltonian_applications: float = 0.0
    maximum_eigen_residual_ha: float = 0.0
    cpu_seconds: float = 0.0
    memory_rss_bytes_per_rank: int = 0
    memory_pss_bytes_all_ranks: int = 0
    memory_available_bytes: int = 0
    memory_density_bytes_per_rank: int = 0
    memory_mixing_bytes_per_rank: int = 0
    memory_wavefunctions_bytes_per_rank: int = 0
    memory_davidson_bytes_per_rank: int = 0
    memory_fft_bytes_per_rank: int = 0
    memory_fft_scratch_bytes_per_rank: int = 0
    memory_fft_exchange_bytes_per_rank: int = 0
    memory_fft_plan_count_per_rank: int = 0


@dataclass(frozen=True)
class SCFEnergyTerms:
    """Converged total-energy decomposition in Hartree."""

    one_electron_ha: float
    hartree_ha: float
    xc_ha: float
    ewald_ha: float
    descf_ha: float = 0.0
    smearing_ha: float = 0.0


@dataclass(frozen=True)
class SCFSetup:
    """Read-only setup summary emitted once before the SCF loop."""

    kpoints: int
    min_plane_waves: int
    max_plane_waves: int
    total_plane_waves: int
    dense_work: int
    diagonalization: str = "david"
    davidson_subspace: int = 0
    number_of_bands: int = 0
    fft_shape: tuple[int, int, int] = (0, 0, 0)
    charge_gvectors: int = 0
    charge_sticks: int = 0
    wavefunction_sticks: int = 0
    atomic_orbitals: int = 0
    starting_charge: float = 0.0
    target_charge: float = 0.0
    starting_potential: str = "atomic"
    starting_wavefunctions: str = "atomic+random"
    starting_messages: tuple[str, ...] = ()
    mpi_processes: int = 1
    threads_per_process: int = 1
    estimated_persistent_bytes_per_rank: int = 0
    estimated_peak_workspace_bytes_per_rank: int = 0
    estimated_array_components_per_rank: tuple[tuple[str, int], ...] = ()
    estimated_array_bytes_all_ranks: int = 0
    estimated_peak_rss_bytes_per_rank: int = 0
    runtime_baseline_rss_bytes_per_rank: int = 0
    runtime_baseline_pss_bytes_all_ranks: int = 0
    runtime_baseline_uss_bytes_all_ranks: int = 0


@dataclass
class SCFResult:
    """Physical result and restart/reporting data returned by :func:`run_scf`."""

    converged: bool
    total_energy_ha: float
    eigenvalues_ha: list[np.ndarray]
    density: np.ndarray
    iterations: list[SCFIteration] = field(default_factory=list)
    wall_seconds: float = 0.0
    energy_terms: SCFEnergyTerms | None = None
    plane_waves_per_k: list[int] = field(default_factory=list)
    peak_rss_bytes_per_rank: int = 0
    peak_rss_bytes_all_ranks: int = 0
    peak_sampled_pss_bytes_all_ranks: int = 0
    timings: dict[str, TimingEntry] = field(default_factory=dict)
    forces_ha_per_bohr: np.ndarray | None = None
    stress_ha_per_bohr3: np.ndarray | None = None
    occupations: list[np.ndarray] = field(default_factory=list)
    fermi_energy_ha: float | None = None
    mpi_processes: int = 1
    wavefunctions: list[np.ndarray] = field(default_factory=list)
    wavefunction_miller_indices: list[np.ndarray] = field(default_factory=list)
    wavefunction_row_indices: list[np.ndarray] = field(default_factory=list)
    wavefunctions_distributed: bool = False


ProgressKind = Literal["setup", "iteration"]
ProgressPayload = SCFSetup | SCFIteration
ProgressCallback = Callable[[ProgressKind, ProgressPayload], None]
_PACKED_PROJECTOR_CHANNEL_LIMIT = 64


@dataclass(frozen=True)
class ReciprocalGrid:
    """Shared real/reciprocal FFT geometry for one density grid."""

    shape: tuple[int, int, int]
    reciprocal: np.ndarray
    g2: np.ndarray
    charge_slots: np.ndarray
    charge_indices: np.ndarray
    charge_vectors: np.ndarray
    charge_g2: np.ndarray

    @classmethod
    def build(
        cls,
        shape: tuple[int, int, int],
        reciprocal: np.ndarray,
        cutoff_ry: float,
    ) -> "ReciprocalGrid":
        shape = tuple(int(value) for value in shape)
        reciprocal = np.asarray(reciprocal, dtype=float)
        axes = [
            np.rint(np.fft.fftfreq(size) * size).astype(np.int32)
            for size in shape
        ]
        metric = reciprocal @ reciprocal.T
        x = axes[0][:, None, None]
        y = axes[1][None, :, None]
        z = axes[2][None, None, :]
        g2 = (
            metric[0, 0] * x * x
            + metric[1, 1] * y * y
            + metric[2, 2] * z * z
            + 2.0 * metric[0, 1] * x * y
            + 2.0 * metric[0, 2] * x * z
            + 2.0 * metric[1, 2] * y * z
        )
        g2 = np.asarray(g2, dtype=float)
        charge_slots = np.asarray(
            np.argwhere(g2 <= cutoff_ry + 1.0e-12), dtype=np.int32
        )
        charge_indices = np.column_stack(
            [axes[axis][charge_slots[:, axis]] for axis in range(3)]
        ).astype(np.int32, copy=False)
        charge_vectors = charge_indices @ reciprocal
        charge_g2 = g2[tuple(charge_slots.T)]
        return cls(
            shape,
            reciprocal.copy(),
            g2,
            charge_slots,
            charge_indices,
            charge_vectors,
            charge_g2,
        )


def _peak_memory_across_ranks(mpi: MPIContext) -> tuple[int, int]:
    local_peak = peak_rss_bytes()
    maximum = int(mpi.max_scalar(local_peak))
    total = int(mpi.sum_scalar(local_peak))
    return maximum, total


class _LazyWorkspaceSequence(Sequence[LocalPotentialWorkspace]):
    """Provide compact current-k maps around one shared FFT descriptor."""

    def __init__(
        self,
        bases: Sequence[PlaneWaveBasis],
        shape: tuple[int, int, int],
        *,
        mpi: MPIContext,
        timers: TimingRegistry,
        scratch_pool: FFTScratchPool,
        descriptor: FFTGridDescriptor | None,
        serial_fft_batch_size: int | None = None,
        cache_workspaces: bool = False,
        retain_serial_sticks: bool = True,
    ) -> None:
        self.bases = bases
        self.shape = shape
        self._cache: list[LocalPotentialWorkspace | None] | None = (
            [None] * len(bases)
            if cache_workspaces
            else None
        )
        self.options = dict(
            mpi=mpi,
            timers=timers,
            scratch_pool=scratch_pool,
            descriptor=descriptor,
            serial_fft_batch_size=serial_fft_batch_size,
            retain_serial_sticks=retain_serial_sticks,
        )

    def __len__(self) -> int:
        return len(self.bases)

    def __getitem__(
        self, index: int | slice
    ) -> LocalPotentialWorkspace | list[LocalPotentialWorkspace]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._cache is not None:
            cached = self._cache[index]
            if cached is not None:
                return cached
        workspace = LocalPotentialWorkspace(
            self.bases[index], self.shape, **self.options
        )
        if self._cache is not None:
            # Ownership was already distilled into the compact arrays used
            # by the hot FFT path. The full per-G stick and boolean owner maps
            # are setup intermediates; dropping them keeps the all-k cache
            # compact without rebuilding any operational metadata.
            workspace.stick_indices = np.empty(0, dtype=np.int32)
            workspace.owned_plane_waves = np.empty(0, dtype=bool)
            self._cache[index] = workspace
        return workspace


def _collect_wavefunctions_root(
    eigenvectors: list[np.ndarray],
    bases: list[PlaneWaveBasis],
    workspaces: Sequence[LocalPotentialWorkspace],
    mpi: MPIContext,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Collect final plane-wave coefficients on rank zero for persistence."""
    collected: list[np.ndarray] = []
    miller_indices: list[np.ndarray] = []
    for vectors, basis, workspace in zip(eigenvectors, bases, workspaces):
        indices = basis.indices
        if vectors.shape[0] == len(basis):
            gathered = vectors if mpi.is_root else np.empty((0, 0))
        else:
            gathered = mpi.gather_indexed_rows_root(
                vectors,
                workspace.local_plane_wave_indices,
                len(basis),
            )
        if mpi.is_root:
            collected.append(np.asarray(gathered))
            miller_indices.append(np.asarray(indices, dtype=np.int32).copy())
    return collected, miller_indices


def _final_wavefunction_payload(
    eigenvectors: list[np.ndarray],
    bases: list[PlaneWaveBasis],
    workspaces: Sequence[LocalPotentialWorkspace],
    mpi: MPIContext,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], bool]:
    """Return serial wavefunctions or rank-local rows for streaming MPI I/O."""
    if mpi.size == 1:
        # The serial eigenvectors already use the global basis order.  They
        # are no longer mutated after this point, so returning the existing
        # matrices avoids a second all-k resident copy for disk_io != 'none'.
        return (
            list(eigenvectors),
            [np.asarray(basis.indices, dtype=np.int32) for basis in bases],
            [],
            False,
        )

    # Do not gather every complete k-point matrix into rank zero.  The CLI
    # save path collectively gathers one matrix, writes it, and releases it
    # before advancing to the next k point.  Only rank zero needs the global
    # Miller order; all ranks retain their compact row maps.
    miller = (
        [np.asarray(basis.indices, dtype=np.int32) for basis in bases]
        if mpi.is_root
        else []
    )
    rows = [
        np.asarray(workspaces[index].local_plane_wave_indices, dtype=np.int32)
        for index in range(len(bases))
    ]
    return list(eigenvectors), miller, rows, True


class _QERandom:
    """Quantum ESPRESSO ``random_numbers:randy`` stream."""

    _modulus = 714025
    _multiplier = 1366
    _increment = 150889
    _table_size = 97

    def __init__(self, seed: int = 0) -> None:
        self._seed = min(abs(int(seed)), self._increment)
        self._table = np.empty(self._table_size, dtype=np.int64)
        self._seed = (self._increment - self._seed) % self._modulus
        for index in range(self._table_size):
            self._seed = (
                self._multiplier * self._seed + self._increment
            ) % self._modulus
            self._table[index] = self._seed
        self._seed = (
            self._multiplier * self._seed + self._increment
        ) % self._modulus
        self._current = self._seed

    def random(self) -> float:
        table_index = (
            self._table_size * self._current
        ) // self._modulus
        self._current = int(self._table[table_index])
        value = self._current / self._modulus
        self._seed = (
            self._multiplier * self._seed + self._increment
        ) % self._modulus
        self._table[table_index] = self._seed
        return value

    def pairs_by_band(
        self, plane_waves: int, bands: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw rr1,rr2 in QE's band-outer, G-inner loop order."""
        first, second, current, seed = _load_native_fft().qe_random_pairs(
            int(plane_waves),
            int(bands),
            self._table,
            self._current,
            self._seed,
        )
        self._current = int(current)
        self._seed = int(seed)
        return first, second

    def pairs_by_band_rows(
        self,
        plane_waves: int,
        bands: int,
        rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw the global QE stream while retaining selected PW rows only."""
        selected = np.asarray(rows, dtype=np.int64).reshape(-1)
        if (
            np.any(selected < 0)
            or np.any(selected >= plane_waves)
            or len(np.unique(selected)) != len(selected)
        ):
            raise ValueError("selected random plane-wave rows are invalid")
        # local FFT ownership rows are sorted.  The native recurrence scans
        # the same global stream but writes only matching rows, avoiding both
        # Python's O(npw*nband) loop and the former O(npw) positions table.
        order = np.argsort(selected, kind="stable")
        sorted_rows = np.ascontiguousarray(selected[order], dtype=np.int64)
        first, second, current, seed = (
            _load_native_fft().qe_random_pairs_rows(
                int(plane_waves),
                int(bands),
                sorted_rows,
                self._table,
                self._current,
                self._seed,
            )
        )
        if not np.array_equal(order, np.arange(len(selected))):
            restored_first = np.empty_like(first)
            restored_second = np.empty_like(second)
            restored_first[order] = first
            restored_second[order] = second
            first, second = restored_first, restored_second
        self._current = int(current)
        self._seed = int(seed)
        return first, second


def _randomize_atomic_trials(
    trials: np.ndarray,
    kinetic: np.ndarray,
    random_stream: _QERandom,
    strength: float = 0.05,
    global_plane_waves: int | None = None,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Apply QE's coefficient-wise 5% perturbation to atomic trials."""
    del kinetic  # Retained in the private signature for compatibility.
    if rows is not None and global_plane_waves is None:
        raise ValueError(
            "global plane-wave count is required with local rows"
        )
    amplitude, phase = (
        random_stream.pairs_by_band(trials.shape[0], trials.shape[1])
        if rows is None
        else random_stream.pairs_by_band_rows(
            int(global_plane_waves),
            trials.shape[1],
            rows,
        )
    )
    noise = amplitude * np.exp(2j * np.pi * phase)
    return trials * (1.0 + strength * noise)


def _ionic_potential(
    pw: PWInput,
    pseudos: dict[str, LocalPotential],
    shape: tuple[int, int, int],
    g2_cutoff: float,
    geometry: ReciprocalGrid | None = None,
) -> np.ndarray:
    geometry = (
        geometry
        if geometry is not None
        else ReciprocalGrid.build(shape, pw.reciprocal, g2_cutoff)
    )
    gvec = geometry.charge_vectors
    q = np.sqrt(geometry.charge_g2)
    coefficients = np.zeros(len(gvec), dtype=complex)
    for species in pw.species:
        pseudo = pseudos[species.label]
        radial = pseudo.fourier(q, pw.volume)
        positions = [atom.position for atom in pw.atoms if atom.label == species.label]
        structure = sum(
            np.exp(-1j * (gvec @ position)) for position in positions
        )
        coefficients += radial * structure
    potential = np.zeros(shape, dtype=complex)
    potential[tuple(geometry.charge_slots.T)] = coefficients
    return potential


def _nonlocal_projector_terms(
    pw: PWInput,
    pseudos: dict[str, LocalPotential],
    basis: PlaneWaveBasis,
    *,
    factorized: bool = False,
    packed: bool = False,
    local_rows: np.ndarray | None = None,
) -> tuple[ProjectorTerm, ...]:
    """Build the compact atom-projector representation of the nonlocal term."""
    basis_vectors = basis.vectors
    if factorized or packed:
        vectors = (
            basis_vectors
            if local_rows is None
            else basis_vectors[local_rows]
        )
        layouts = []
        for species in pw.species:
            pseudo = pseudos[species.label]
            positions = np.asarray(
                [
                    atom.position
                    for atom in pw.atoms
                    if atom.label == species.label
                ],
                dtype=np.float64,
            )
            if not len(positions) or not pseudo.projectors:
                continue
            layouts.append((pseudo, positions))

        total_channels = sum(
            pseudo.number_of_projector_channels * len(positions)
            for pseudo, positions in layouts
        )
        if (
            packed
            and total_channels
            and total_channels <= _PACKED_PROJECTOR_CHANNEL_LIMIT
        ):
            # Allocate the final atom-packed representation first and consume
            # one species at a time.  The former two-pass path retained every
            # species beta/phase pair while also constructing packed_beta,
            # making the setup peak grow with their sum.  Here it grows only
            # with the largest current species and the final packed owner.
            packed_beta = np.empty(
                (len(vectors), total_channels),
                dtype=complex,
                order="F",
            )
            packed_coupling = np.zeros(
                (total_channels, total_channels), dtype=float
            )
            packed_diagonal = np.zeros(len(vectors), dtype=float)
            offset = 0
            for pseudo, positions in layouts:
                beta, coupling = pseudo.projector_basis(
                    vectors, pw.volume
                )
                beta = np.asfortranarray(beta)
                phases = _load_native_fft().phase_matrix(
                    np.asarray(vectors, dtype=np.float64),
                    np.ascontiguousarray(positions, dtype=np.float64),
                )
                channels = beta.shape[1]
                first_block = slice(offset, offset + channels)
                # D_G = N_atom Re[beta_G D beta_G^H].  Use the first final
                # atom block as beta*D scratch, then immediately overwrite it
                # with its eventual phased beta values below.  This retains
                # BLAS speed without an extra Npw-by-projector owner.
                np.matmul(
                    beta,
                    coupling,
                    out=packed_beta[:, first_block],
                )
                _load_native_fft().accumulate_projector_diagonal(
                    packed_diagonal,
                    packed_beta[:, first_block],
                    beta,
                    float(len(positions)),
                )
                for atom in range(len(positions)):
                    block = slice(offset, offset + channels)
                    np.multiply(
                        beta,
                        phases[:, atom, None],
                        out=packed_beta[:, block],
                    )
                    packed_coupling[block, block] = coupling
                    offset += channels
            return (
                PackedProjectorTerm(
                    packed_beta, packed_coupling, packed_diagonal
                ),
            )

        terms: list[ProjectorTerm] = []
        for pseudo, positions in layouts:
            beta, coupling = pseudo.projector_basis(
                vectors, pw.volume
            )
            phases = _load_native_fft().phase_matrix(
                np.asarray(vectors, dtype=np.float64),
                np.ascontiguousarray(positions, dtype=np.float64),
            )
            terms.append(
                FactorizedProjectorTerm(
                    np.asfortranarray(beta),
                    np.asfortranarray(phases),
                    coupling,
                )
            )
        return tuple(terms)
    terms: list[ProjectorTerm] = []
    for atom in pw.atoms:
        pseudo = pseudos[atom.label]
        beta, coupling = pseudo.atomic_projectors(
            basis_vectors, atom.position, pw.volume
        )
        if beta.shape[1]:
            terms.append((beta, coupling))
    return tuple(terms)


def _local_projector_terms(
    terms: tuple[ProjectorTerm, ...],
    local_rows: np.ndarray,
    mpi: MPIContext,
) -> tuple[ProjectorTerm, ...]:
    """Retain only the projector rows owned by this plane-wave rank."""
    if mpi.size == 1:
        return terms
    localized: list[ProjectorTerm] = []
    for term in terms:
        if isinstance(term, FactorizedProjectorTerm):
            localized.append(
                FactorizedProjectorTerm(
                    np.asfortranarray(term.beta[local_rows]),
                    np.asfortranarray(term.phases[local_rows]),
                    term.coupling,
                )
            )
        elif isinstance(term, PackedProjectorTerm):
            localized.append(
                PackedProjectorTerm(
                    np.asfortranarray(term.beta[local_rows]),
                    term.coupling,
                    term.diagonal[local_rows],
                )
            )
        else:
            beta, coupling = term
            localized.append(
                (np.asfortranarray(beta[local_rows]), coupling)
            )
    return tuple(localized)


def _nonlocal_hamiltonian(
    size: int, terms: tuple[ProjectorTerm, ...]
) -> np.ndarray:
    """Materialize the nonlocal operator for the diagnostic dense solver."""
    result = np.zeros((size, size), dtype=complex)
    for term in terms:
        if isinstance(term, FactorizedProjectorTerm):
            for atom in range(term.phases.shape[1]):
                beta = term.beta * term.phases[:, atom, None]
                result += (beta @ term.coupling) @ beta.conj().T
            continue
        if isinstance(term, PackedProjectorTerm):
            beta, coupling = term.beta, term.coupling
        else:
            beta, coupling = term
        result += (beta @ coupling) @ beta.conj().T
    # Roundoff in transforms should not leak into the Hermitian eigensolver.
    return 0.5 * (result + result.conj().T)


def _atomic_starting_orbitals(
    pw: PWInput,
    pseudos: dict[str, LocalPotential],
    basis: PlaneWaveBasis,
    number_of_bands: int,
    random_stream: _QERandom,
    local_rows: np.ndarray | None = None,
    mpi: MPIContext | None = None,
    starting_wfc: str | None = None,
) -> np.ndarray:
    """Build QE ``atomic+random`` trial vectors for the first diagonalization."""
    starting_wfc = str(
        starting_wfc
        if starting_wfc is not None
        else pw.electrons.get("startingwfc", "atomic+random")
    ).lower()
    if starting_wfc not in {"atomic", "atomic+random", "random"}:
        raise UnsupportedFeatureError(
            f"startingwfc={starting_wfc!r} is not ported"
        )
    basis_vectors = (
        basis.vectors
        if local_rows is None
        else basis.vectors[np.asarray(local_rows, dtype=np.int32)]
    )
    kinetic = 0.5 * np.einsum(
        "ij,ij->i", basis_vectors, basis_vectors
    )
    blocks = []
    if starting_wfc != "random":
        centered_by_label: dict[str, np.ndarray] = {}
        for atom in pw.atoms:
            centered = centered_by_label.get(atom.label)
            if centered is None:
                centered = pseudos[
                    atom.label
                ].atomic_orbital_basis(
                    basis_vectors, pw.volume
                )
                centered_by_label[atom.label] = centered
            phase = np.exp(-1j * (basis_vectors @ atom.position))
            blocks.append(centered * phase[:, None])
    atomic = [
        block for block in blocks if block.shape[1] > 0
    ]
    trials = (
        np.column_stack(atomic)
        if atomic
        else np.empty((len(basis_vectors), 0), dtype=complex)
    )
    if trials.shape[1]:
        if local_rows is None:
            norms = np.linalg.norm(trials, axis=0)
        else:
            if mpi is None:
                raise ValueError("MPI context is required for local trial rows")
            norms = np.sqrt(
                mpi.sum_array(
                    np.sum(np.abs(trials) ** 2, axis=0)
                )
            )
        usable = norms > max(1.0e-14, 1.0e-12 * float(np.max(norms)))
        trials = trials[:, usable]
    if (
        starting_wfc == "atomic+random"
        and trials.shape[1] >= number_of_bands
    ):
        # QE's default ``atomic+random`` perturbs a complete atomic trial
        # space by five percent to avoid loss of symmetry-related states.
        trials = _randomize_atomic_trials(
            trials,
            kinetic,
            random_stream,
            global_plane_waves=(
                len(basis.vectors) if local_rows is not None else None
            ),
            rows=local_rows,
        )
    elif trials.shape[1] < number_of_bands:
        missing = number_of_bands - trials.shape[1]
        amplitude, phase_fraction = (
            random_stream.pairs_by_band(len(basis.vectors), missing)
            if local_rows is None
            else random_stream.pairs_by_band_rows(
                len(basis.vectors), missing, local_rows
            )
        )
        phase = np.exp(
            2j
            * np.pi
            * phase_fraction
        )
        random_trials = amplitude * phase / (
            1.0 + 2.0 * kinetic[:, None]
        )
        trials = np.column_stack((trials, random_trials))
    return trials


def _starting_charge_scales(
    pw: PWInput, pseudos: dict[str, LocalPotential]
) -> dict[str, float]:
    """Return QE atomic-rho scale factors from ``starting_charge(i)``."""
    scales: dict[str, float] = {}
    for index, species in enumerate(pw.species, start=1):
        raw = pw.system.get(f"starting_charge({index})", 0.0)
        charge = float(raw)
        if not np.isfinite(charge):
            raise QEInputError(
                f"starting_charge({index}) must be finite"
            )
        valence = float(pseudos[species.label].z_valence)
        scales[species.label] = (
            max(0.0, valence - charge) / valence
            if valence > 1.0e-8
            else 1.0
        )
    return scales


def _rotate_starting_subspace(
    operator: PlaneWaveHamiltonian,
    trials: np.ndarray,
    number_of_bands: int,
    mpi: MPIContext | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce QE ``wfcinit:rotate_wfc`` before the first Davidson call."""
    atomic_basis = np.asarray(trials, dtype=complex)
    mpi = mpi if mpi is not None else MPIContext()
    applied = np.empty_like(atomic_basis, order="F")
    operator.apply_into(atomic_basis, applied)
    projected_local = atomic_basis.conj().T @ applied
    overlap_local = atomic_basis.conj().T @ atomic_basis
    split = projected_local.size
    reduced = mpi.sum_array(
        np.concatenate((projected_local.ravel(), overlap_local.ravel()))
    )
    projected = reduced[:split].reshape(projected_local.shape)
    overlap = reduced[split:].reshape(overlap_local.shape)
    projected = 0.5 * (projected + projected.conj().T)
    overlap = 0.5 * (overlap + overlap.conj().T)
    values, rotation = _lowest_generalized_eigh(
        projected,
        overlap,
        number_of_bands,
    )
    return (
        _matmul_f(atomic_basis, rotation),
        values,
        _matmul_f(applied, rotation),
    )


def _atomic_starting_density(
    pw: PWInput,
    pseudos: dict[str, LocalPotential],
    shape: tuple[int, int, int],
    nelec: float,
    g2_cutoff: float,
    geometry: ReciprocalGrid | None = None,
    workspace: LocalPotentialWorkspace | None = None,
) -> tuple[np.ndarray, float]:
    """Return QE's superposition of UPF atomic valence charge densities."""
    geometry = (
        geometry
        if geometry is not None
        else ReciprocalGrid.build(shape, pw.reciprocal, g2_cutoff)
    )
    gvec = geometry.charge_vectors
    q = np.sqrt(geometry.charge_g2)
    coefficients = np.zeros(len(gvec), dtype=complex)
    charge_scales = _starting_charge_scales(pw, pseudos)
    for species in pw.species:
        pseudo = pseudos[species.label]
        scale = charge_scales[species.label]
        positions = [
            atom.position
            for atom in pw.atoms
            if atom.label == species.label
        ]
        if pseudo.has_atomic_density:
            radial = pseudo.atomic_density_fourier(q, pw.volume)
            structure = sum(
                np.exp(
                    -1j
                    * np.einsum("...j,j->...", gvec, position)
                )
                for position in positions
            )
            coefficients += scale * radial * structure
        else:
            # A UPF without PP_RHOATOM contributes the correct average
            # charge but no invented atom-centered Fourier components.
            coefficients[geometry.charge_g2 < 1.0e-14] += (
                len(positions)
                * scale
                * pseudo.z_valence
                / pw.volume
            )
    starting_charge = float(
        np.real(np.sum(coefficients[geometry.charge_g2 < 1.0e-14]))
    ) * pw.volume
    if abs(starting_charge) < 1.0e-14:
        coefficients.fill(0.0)
        coefficients[geometry.charge_g2 < 1.0e-14] = nelec / pw.volume
    else:
        coefficients *= nelec / starting_charge
    if workspace is None:
        raise ValueError("atomic density requires the native FFT workspace")
    transformed = workspace.coefficients_to_grid(
        coefficients, use_scratch=True
    )
    density = np.ascontiguousarray(np.real(transformed))
    return density, starting_charge


def _xc_terms(
    valence_density: np.ndarray,
    core_density: np.ndarray,
    volume: float,
    mpi: MPIContext | None = None,
    functional: str = "pz",
    workspace: LocalPotentialWorkspace | None = None,
    g_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Return Vxc, Exc[rho_v+rho_c], and integral Vxc*rho_v in Hartree."""
    mpi = mpi if mpi is not None else MPIContext()
    shape = valence_density.shape
    if valence_density.ndim != 3 or mpi.size == 1:
        total_density = valence_density + core_density
        epsilon_xc, potential_xc = _xc_energy_potential(
            total_density,
            functional,
            workspace=workspace,
            g_vectors=g_vectors,
        )
        energy_xc = volume * float(
            np.mean(total_density * epsilon_xc)
        )
        potential_valence = volume * float(
            np.mean(valence_density * potential_xc)
        )
        return potential_xc, energy_xc, potential_valence
    z_slab = mpi.slab(shape[2])
    valence_local = valence_density[:, :, z_slab]
    total_local = valence_local + core_density[:, :, z_slab]
    epsilon_xc, potential_local = _xc_energy_potential(
        total_local,
        functional,
        workspace=workspace,
        g_vectors=g_vectors,
    )
    scale = volume / float(np.prod(shape))
    energy_xc = scale * mpi.sum_scalar(
        float(np.sum(total_local * epsilon_xc))
    )
    potential_valence = scale * mpi.sum_scalar(
        float(np.sum(valence_local * potential_local))
    )
    potential_xc = mpi.gather_z_slabs(potential_local, shape)
    return potential_xc, energy_xc, potential_valence


def _xc_energy_potential(
    density: np.ndarray,
    functional: str,
    *,
    workspace: LocalPotentialWorkspace | None = None,
    g_vectors: np.ndarray | None = None,
    need_epsilon: bool = True,
    energy_density_out: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Evaluate the selected XC functional on a rank-local FFT slab."""
    if functional == "pz":
        epsilon, potential = pz81_unpolarized(density)
    elif functional == "pw":
        epsilon, potential = pw92_lda_unpolarized(density)
    else:
        epsilon = potential = None
    if potential is not None:
        if energy_density_out is not None:
            np.multiply(density, epsilon, out=energy_density_out)
        if not need_epsilon:
            epsilon = None
        return epsilon, potential
    if functional not in GGA_FUNCTIONALS:
        raise ValueError(f"unknown XC functional {functional!r}")
    if workspace is None or g_vectors is None:
        raise ValueError("GGA requires an FFT workspace and reciprocal vectors")

    epsilon, potential, _stress_tensor = _gga_energy_potential_data(
        density,
        workspace,
        g_vectors,
        functional,
        need_stress=False,
        need_epsilon=need_epsilon,
        energy_density_out=energy_density_out,
    )
    return epsilon, potential


def _gga_energy_potential_data(
    density: np.ndarray,
    workspace: LocalPotentialWorkspace,
    g_vectors: np.ndarray,
    functional: str,
    *,
    need_stress: bool = False,
    need_epsilon: bool = True,
    energy_density_out: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Return GGA energy/potential and an optional analytic-stress tensor."""
    coefficients = workspace.grid_to_coefficients(density)
    gradient_coefficients = np.multiply(
        g_vectors,
        coefficients[:, None],
        dtype=np.complex128,
    )
    gradient_coefficients *= 1j
    del coefficients
    gradient_grid = workspace.coefficients_to_grid(gradient_coefficients)
    del gradient_coefficients
    gradient = np.moveaxis(gradient_grid, -1, 0).real

    epsilon, local_potential, gradient_coefficient = (
        pbe_unpolarized_components(
            density,
            gradient,
            functional=functional,
            need_epsilon=need_epsilon,
            energy_density_out=energy_density_out,
        )
    )
    stress_tensor = (
        np.einsum(
            "...,i...,j...->ij",
            gradient_coefficient,
            gradient,
            gradient,
            optimize=True,
        )
        if need_stress
        else None
    )
    # The gradient is no longer needed after the optional 3x3 contraction.
    # Reuse its complex FFT grid as c*grad(rho), avoiding a second 3-component
    # real field. The following forward/inverse transforms are sequential, so
    # this also shortens the lifetime of every large GGA array.
    np.multiply(
        gradient_grid.real,
        gradient_coefficient[..., None],
        out=gradient_grid.real,
    )
    gradient_grid.imag.fill(0.0)
    del gradient
    flux_coefficients = workspace.grid_to_coefficients(gradient_grid)
    del gradient_grid
    np.multiply(
        flux_coefficients,
        g_vectors,
        out=flux_coefficients,
    )
    flux_coefficients *= 1j
    derivatives = workspace.coefficients_to_grid(flux_coefficients)
    del flux_coefficients
    np.add.reduce(
        derivatives.real,
        axis=-1,
        out=gradient_coefficient,
    )
    local_potential -= gradient_coefficient
    del gradient_coefficient
    del derivatives
    return epsilon, local_potential, stress_tensor


def _functional_family(name: str) -> str | None:
    canonical = canonical_xc_name(name)
    if canonical is not None:
        return canonical
    normalized = "".join(
        character for character in str(name).lower() if character.isalnum()
    )
    known_unsupported = (
        "blyp", "pw91", "olyp", "wc", "sogga", "tpss", "scan",
        "pbe0", "hse", "b3lyp", "vdw", "rvv10",
    )
    if any(value in normalized for value in known_unsupported):
        return "unsupported"
    return None


def _resolve_xc_functional(
    pw: PWInput, pseudos: dict[str, LocalPotential]
) -> str:
    """Resolve ``input_dft`` or, when absent, the UPF functional metadata."""
    if "input_dft" in pw.system:
        selected = _functional_family(str(pw.system["input_dft"]))
        if selected is None:
            raise UnsupportedFeatureError(
                f"input_dft={pw.system['input_dft']!r} is not ported"
            )
        return selected

    resolved_metadata = [
        (pseudo.functional, _functional_family(pseudo.functional))
        for pseudo in pseudos.values()
    ]
    unknown_metadata = sorted(
        {name for name, family in resolved_metadata if family is None}
    )
    if unknown_metadata:
        raise UnsupportedFeatureError(
            "cannot determine a supported XC functional from "
            f"pseudopotential metadata: {', '.join(unknown_metadata)}; "
            "set input_dft explicitly"
        )
    families = {family for _name, family in resolved_metadata}
    if "unsupported" in families:
        names = ", ".join(
            sorted({pseudo.functional for pseudo in pseudos.values()})
        )
        raise UnsupportedFeatureError(
            f"pseudopotential functional is not ported: {names}"
        )
    if len(families) > 1:
        raise UnsupportedFeatureError(
            "pseudopotentials use inconsistent exchange-correlation functionals"
        )
    if families:
        return families.pop()
    raise UnsupportedFeatureError(
        "cannot determine XC functional without input_dft or pseudopotentials"
    )


def _density_from_states(
    vectors: list[np.ndarray],
    bases: list[PlaneWaveBasis],
    weights: np.ndarray,
    occupations: list[np.ndarray],
    shape: tuple[int, int, int],
    volume: float,
    nelec: float,
    mpi: MPIContext | None = None,
    workspaces: Sequence[LocalPotentialWorkspace] | None = None,
    timers: TimingRegistry | None = None,
) -> np.ndarray:
    mpi = mpi if mpi is not None else MPIContext()
    z_slab = mpi.slab(shape[2])
    local_shape = (
        shape[0],
        shape[1],
        z_slab.stop - z_slab.start,
    )
    rho_local = np.zeros(local_shape, dtype=float)
    for index, (coeff, basis, weight, band_occupations) in enumerate(
        zip(vectors, bases, weights, occupations)
    ):
        active = np.flatnonzero(np.abs(band_occupations) > 1.0e-15)
        if active.size == 0:
            continue
        band_count = int(active[-1]) + 1
        workspace = (
            workspaces[index] if workspaces is not None else None
        )
        if workspace is None:
            workspace = LocalPotentialWorkspace(
                basis.indices, shape, mpi=mpi
            )
        transform_started = timers.start() if timers is not None else None
        workspace.accumulate_density(
            rho_local,
            coeff[:, :band_count],
            weight
            * np.asarray(band_occupations[:band_count], dtype=float)
            / volume,
        )
        if timers is not None and transform_started is not None:
            timers.stop("sum_band:wfc", transform_started)
    local_sum = float(np.sum(rho_local))
    total_sum = mpi.sum_scalar(local_sum)
    rho_local *= nelec * np.prod(shape) / (total_sum * volume)
    return rho_local


def _nonlocal_energy_and_forces(
    pw: PWInput,
    pseudos: dict[str, LocalPotential],
    bases: list[PlaneWaveBasis],
    eigenvectors: list[np.ndarray],
    weights: np.ndarray,
    occupations: list[np.ndarray],
    mpi: MPIContext,
    workspaces: Sequence[LocalPotentialWorkspace],
    *,
    forces: bool,
) -> tuple[float, np.ndarray | None]:
    """Evaluate norm-conserving nonlocal energy and HF atomic forces."""
    energy, result, _ = _nonlocal_derivatives(
        pw,
        pseudos,
        bases,
        eigenvectors,
        weights,
        occupations,
        mpi,
        workspaces,
        forces=forces,
        stress=False,
    )
    return energy, result


def _nonlocal_derivatives(
    pw: PWInput,
    pseudos: dict[str, LocalPotential],
    bases: list[PlaneWaveBasis],
    eigenvectors: list[np.ndarray],
    weights: np.ndarray,
    occupations: list[np.ndarray],
    mpi: MPIContext,
    workspaces: Sequence[LocalPotentialWorkspace],
    *,
    forces: bool,
    stress: bool,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Compute force/stress projector derivatives once per species and k."""
    energy = 0.0
    result = np.zeros((len(pw.atoms), 3)) if forces else None
    stress_tensor = np.zeros((3, 3)) if stress else None
    atoms_by_label: dict[str, list[tuple[int, object]]] = {}
    for atom_index, atom in enumerate(pw.atoms):
        atoms_by_label.setdefault(atom.label, []).append((atom_index, atom))
    for compact_basis, vectors, weight, workspace, band_occupations in zip(
        bases, eigenvectors, weights, workspaces, occupations
    ):
        basis = compact_basis.materialize()
        local_rows = (
            workspace.local_plane_wave_indices
            if mpi.size > 1
            else np.arange(len(basis.indices))
        )
        gk = basis.vectors[local_rows]
        active = np.flatnonzero(np.abs(band_occupations) > 1.0e-15)
        if active.size == 0:
            continue
        band_count = int(active[-1]) + 1
        occupied_vectors = vectors[:, :band_count]
        occupation_weights = np.asarray(band_occupations[:band_count])
        for label, indexed_atoms in atoms_by_label.items():
            if stress:
                beta, coupling, beta_gradient = pseudos[
                    label
                ].projector_basis_with_gradient(gk, pw.volume)
            else:
                beta, coupling = pseudos[label].projector_basis(
                    gk, pw.volume
                )
                beta_gradient = None
            if beta.shape[1] == 0:
                continue
            # These arrays are local derivative workspaces.  Store their
            # conjugated form in place once, instead of materializing a full
            # beta/beta-gradient conjugate for every atom and contraction.
            np.conjugate(beta, out=beta)
            if beta_gradient is not None:
                np.conjugate(beta_gradient, out=beta_gradient)
            for atom_index, atom in indexed_atoms:
                phase = np.exp(-1j * (gk @ atom.position))
                modulated = np.conjugate(phase)[:, None] * occupied_vectors
                overlap = mpi.sum_array(beta.T @ modulated)
                coupled = coupling @ overlap
                energy += float(weight) * float(
                    np.real(
                        np.sum(
                            np.conjugate(overlap)
                            * coupled
                            * occupation_weights[None, :]
                        )
                    )
                )
                if result is not None:
                    derivative_overlap = mpi.sum_array(
                        np.einsum(
                            "gp,gab->pab",
                            beta,
                            1j * gk[:, :, None] * modulated[:, None, :],
                            optimize=True,
                        )
                    )
                    result[atom_index] -= 2.0 * float(weight) * np.real(
                        np.einsum(
                            "pab,pb,b->a",
                            np.conjugate(derivative_overlap),
                            coupled,
                            occupation_weights,
                            optimize=True,
                        )
                    )
                if stress_tensor is not None:
                    assert beta_gradient is not None
                    derivative_overlap = mpi.sum_array(
                        np.einsum(
                            "gpl,grb->plrb",
                            beta_gradient,
                            gk[:, :, None] * modulated[:, None, :],
                            optimize=True,
                        )
                    )
                    stress_tensor += (
                        2.0
                        * float(weight)
                        / pw.volume
                        * np.real(
                            np.einsum(
                                "plrb,pb,b->lr",
                                np.conjugate(derivative_overlap),
                                coupled,
                                occupation_weights,
                                optimize=True,
                            )
                        )
                    )
    return energy, result, stress_tensor


def _local_and_core_forces(
    pw: PWInput,
    pseudos: dict[str, LocalPotential],
    density: np.ndarray,
    potential_xc: np.ndarray,
    workspace: LocalPotentialWorkspace,
    g_vectors: np.ndarray,
    mpi: MPIContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local-pseudopotential and NLCC Hellmann-Feynman forces."""
    density_g = workspace.grid_to_coefficients(density)
    potential_xc_g = workspace.grid_to_coefficients(potential_xc)
    q = np.linalg.norm(g_vectors, axis=1)
    local = np.zeros((len(pw.atoms), 3))
    core = np.zeros_like(local)
    for atom_index, atom in enumerate(pw.atoms):
        pseudo = pseudos[atom.label]
        phase = np.exp(-1j * (g_vectors @ atom.position))
        local_coefficient = pseudo.fourier(q, pw.volume) * phase
        local[atom_index] = pw.volume * np.real(
            np.sum(
                np.conjugate(density_g)[:, None]
                * (1j * g_vectors)
                * local_coefficient[:, None],
                axis=0,
            )
        )
        if pseudo.has_nlcc:
            core_coefficient = (
                pseudo.core_density_fourier(q, pw.volume) * phase
            )
            core[atom_index] = pw.volume * np.real(
                np.sum(
                    np.conjugate(potential_xc_g)[:, None]
                    * (1j * g_vectors)
                    * core_coefficient[:, None],
                    axis=0,
                )
            )
    return mpi.sum_array(local), mpi.sum_array(core)


def _hellmann_feynman_stress(
    pw: PWInput,
    pseudos: dict[str, LocalPotential],
    bases: list[PlaneWaveBasis],
    eigenvectors: list[np.ndarray],
    weights: np.ndarray,
    occupations: list[np.ndarray],
    density: np.ndarray,
    core_density: np.ndarray,
    shape: tuple[int, int, int],
    charge_vectors: np.ndarray,
    charge_workspace: LocalPotentialWorkspace,
    local_workspaces: Sequence[LocalPotentialWorkspace],
    ecutrho: float,
    xc_functional: str,
    mpi: MPIContext,
    charges: np.ndarray,
    nonlocal_energy: float,
    nonlocal_tensor: np.ndarray,
    real_workspace: np.ndarray | None = None,
) -> np.ndarray:
    """Return analytic compressive-positive HF stress in Ha/bohr^3."""
    volume = pw.volume
    stress = np.zeros((3, 3))

    # Kinetic term: fixed reduced-coordinate coefficients and k points imply
    # d(G+k)/d strain = -(G+k) strain.
    kinetic_stress = np.zeros((3, 3))
    for compact_basis, vectors, weight, workspace, band_occupations in zip(
        bases, eigenvectors, weights, local_workspaces, occupations
    ):
        basis = compact_basis.materialize()
        local_rows = (
            workspace.local_plane_wave_indices
            if mpi.size > 1
            else np.arange(len(basis.indices))
        )
        gk = basis.vectors[local_rows]
        band_count = len(band_occupations)
        plane_wave_weight = np.sum(
            np.abs(vectors[:, :band_count]) ** 2
            * np.asarray(band_occupations)[None, :],
            axis=1,
        )
        kinetic_stress += float(weight) * mpi.sum_array(
            np.einsum("g,gi,gj->ij", plane_wave_weight, gk, gk)
        ) / volume
    stress += kinetic_stress

    g_vectors = charge_vectors
    g2 = np.einsum("gi,gi->g", g_vectors, g_vectors)
    q = np.sqrt(g2)
    density_g = charge_workspace.grid_to_coefficients(density)

    # Local pseudopotential. Atomic phases are invariant when cell and ions
    # undergo the same homogeneous deformation; only Omega^-1 and |G| vary.
    local_energy = 0.0
    local_tensor = np.zeros((3, 3))
    radial_mask = q > 1.0e-14
    local_radial: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for atom in pw.atoms:
        if atom.label not in local_radial:
            local_radial[atom.label] = pseudos[
                atom.label
            ].fourier_with_derivative(q, volume)
        potential, radial_derivative = local_radial[atom.label]
        phase = np.exp(-1j * (g_vectors @ atom.position))
        coefficient = potential * phase
        local_energy += volume * mpi.sum_scalar(
            float(np.real(np.vdot(density_g, coefficient)))
        )
        radial_weight = np.zeros_like(q)
        radial_weight[radial_mask] = np.real(
            np.conjugate(density_g[radial_mask])
            * radial_derivative[radial_mask]
            * phase[radial_mask]
        ) / q[radial_mask]
        local_tensor += mpi.sum_array(
            np.einsum("g,gi,gj->ij", radial_weight, g_vectors, g_vectors)
        )
    stress += local_tensor + local_energy / volume * np.eye(3)
    del local_radial, potential, radial_derivative, phase, coefficient

    # Hartree term, obtained by differentiating 4*pi/G^2 and the two
    # Omega^-1 density coefficients.
    nonzero = g2 > 1.0e-14
    np.abs(density_g, out=radial_weight)
    np.square(radial_weight, out=radial_weight)
    np.divide(radial_weight, g2, out=radial_weight, where=nonzero)
    radial_weight[~nonzero] = 0.0
    hartree_energy = 2.0 * np.pi * volume * mpi.sum_scalar(
        float(np.sum(radial_weight))
    )
    np.divide(radial_weight, g2, out=radial_weight, where=nonzero)
    hartree_tensor = -4.0 * np.pi * mpi.sum_array(
        np.einsum(
            "g,gi,gj->ij",
            radial_weight,
            g_vectors,
            g_vectors,
        )
    )
    stress += hartree_tensor + hartree_energy / volume * np.eye(3)
    del density_g, radial_weight

    # Exchange-correlation contribution. QE's diagonal term contains the
    # valence-density potential integral, while Exc and every GGA gradient
    # are evaluated at rho_valence + rho_core.
    if real_workspace is None:
        total_density = density + core_density
    else:
        if real_workspace.shape != density.shape:
            raise ValueError("stress real workspace has the wrong shape")
        np.add(density, core_density, out=real_workspace)
        total_density = real_workspace
    if xc_functional in GGA_FUNCTIONALS:
        (
            epsilon_xc,
            potential_xc,
            gga_stress_tensor,
        ) = _gga_energy_potential_data(
            total_density,
            charge_workspace,
            g_vectors,
            xc_functional,
            need_stress=True,
        )
    else:
        epsilon_xc, potential_xc = _xc_energy_potential(
            total_density,
            xc_functional,
            workspace=charge_workspace,
            g_vectors=g_vectors,
        )
        gga_stress_tensor = None
    grid_scale = volume / np.prod(shape)
    xc_energy = grid_scale * mpi.sum_scalar(
        float(np.sum(total_density * epsilon_xc))
    )
    xc_potential_energy = grid_scale * mpi.sum_scalar(
        float(np.sum(density * potential_xc))
    )
    stress -= (xc_energy - xc_potential_energy) / volume * np.eye(3)
    if gga_stress_tensor is not None:
        stress += mpi.sum_array(gga_stress_tensor) / np.prod(shape)
    del epsilon_xc, total_density, gga_stress_tensor

    # Nonlinear core correction. This is QE's stres_cc: the diagonal term
    # restores the frozen-core part omitted from vtxc, and the radial tensor
    # differentiates each spherical core profile with respect to |G|.
    if any(pseudo.has_nlcc for pseudo in pseudos.values()):
        core_diagonal = mpi.sum_scalar(
            float(np.sum(core_density * potential_xc))
        ) / np.prod(shape)
        core_tensor = np.zeros((3, 3))
        potential_xc_g = charge_workspace.grid_to_coefficients(
            potential_xc
        )
        for species in pw.species:
            pseudo = pseudos[species.label]
            if not pseudo.has_nlcc:
                continue
            radial_derivative = pseudo.core_density_fourier_derivative(
                q, volume
            )
            positions = [
                atom.position
                for atom in pw.atoms
                if atom.label == species.label
            ]
            structure = sum(
                np.exp(-1j * (g_vectors @ position))
                for position in positions
            )
            radial_weight = np.zeros_like(q)
            radial_weight[radial_mask] = np.real(
                np.conjugate(potential_xc_g[radial_mask])
                * radial_derivative[radial_mask]
                * structure[radial_mask]
            ) / q[radial_mask]
            core_tensor += mpi.sum_array(
                np.einsum(
                    "g,gi,gj->ij",
                    radial_weight,
                    g_vectors,
                    g_vectors,
                )
            )
        stress += core_tensor + core_diagonal * np.eye(3)
        del potential_xc_g, radial_weight

    del potential_xc

    # Norm-conserving projector derivatives are evaluated jointly with
    # forces before entering this routine, once per species and k point.
    stress += (
        0.5 * (nonlocal_tensor + nonlocal_tensor.T)
        + nonlocal_energy / volume * np.eye(3)
    )

    stress += ewald_stress(
        pw.lattice,
        np.array([atom.position for atom in pw.atoms]),
        charges,
        ecutrho,
        reciprocal_vectors=charge_vectors,
        mpi=mpi,
    )
    return symmetrize_stress(
        0.5 * (stress + stress.T), pw.lattice, pw.symmetry_operations
    )


def _serial_blas_thread_count(
    pw: PWInput,
    mpi: MPIContext,
    requested_threads: int,
) -> int:
    """Use BLAS teams only when the short matrix dimension is substantial."""
    if mpi.size > 1:
        return 1
    explicit_bands = int(pw.system.get("nbnd", 0))
    # npwx and the number of irreducible k points are not BLAS work measures.
    # The dominant products have nbnd/Davidson block width as their short
    # dimension.  With four Si bands, symmetry reduction made the old policy
    # enable BLAS workers and paid a team launch in tens of thousands of
    # narrow GEMMs. QE's OpenMP CPU build likewise keeps its ordinary BLAS
    # independent of the FFT/OpenMP team. The native FFT, projector, and PW-
    # row kernels still use every requested rank-local thread.
    minimum_threaded_bands = max(32, 8 * requested_threads)
    return (
        requested_threads
        if explicit_bands >= minimum_threaded_bands
        else 1
    )


def run_scf(
    pw: PWInput,
    progress: ProgressCallback | None = None,
    mpi: MPIContext | None = None,
) -> SCFResult:
    """Run SCF with environment-selected threads per MPI rank."""
    mpi = mpi if mpi is not None else MPIContext.world()
    threads = hybrid_thread_count()
    # MPI already distributes the plane-wave rows, so keep BLAS single
    # threaded per rank and avoid nested teams. A single MPI rank has no such
    # spatial decomposition. A single rank enables BLAS workers only for an
    # explicitly large band dimension; symmetry and npwx do not make a narrow
    # GEMM wide. Native FFT/projector/PW kernels retain requested threads.
    blas_threads = _serial_blas_thread_count(pw, mpi, threads)
    # The CLI re-exec has already set every BLAS runtime to one thread before
    # NumPy import. MPI and narrow-band serial runs can skip threadpoolctl's
    # rank-private Python object graph. Library callers lack that guarantee,
    # while a wide-band serial run must raise the active BLAS limit.
    if (
        blas_threads == 1
        and os.environ.get("QEPY_RUNTIME_CONFIGURED") == "1"
    ):
        return _run_scf(pw, progress, mpi)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=blas_threads):
        return _run_scf(pw, progress, mpi)


def _run_scf(
    pw: PWInput,
    progress: ProgressCallback | None = None,
    mpi: MPIContext | None = None,
) -> SCFResult:
    mpi = mpi if mpi is not None else MPIContext()
    threads_per_process = hybrid_thread_count()
    baseline_rss_per_rank = int(mpi.max_scalar(current_rss_bytes()))
    baseline_pss_all_ranks = int(mpi.sum_scalar(current_pss_bytes()))
    baseline_uss_all_ranks = int(mpi.sum_scalar(current_uss_bytes()))
    peak_sampled_pss_all_ranks = baseline_pss_all_ranks
    start = time.perf_counter()
    cpu_start = time.process_time()
    timers = TimingRegistry()
    init_started = timers.start()
    calculation = str(
        pw.control.get("calculation", "scf")
    ).strip().lower()
    fixed_potential = calculation in {"nscf", "bands"}
    ecut = float(pw.system.get("ecutwfc", 0.0))
    if ecut <= 0:
        raise QEInputError("ecutwfc not set", routine="set_cutoff")
    ecutrho = float(pw.system.get("ecutrho", 4.0 * ecut))
    if ecutrho <= 0.0:
        raise QEInputError("ecutrho <= ecutwfc?!?", routine="set_cutoff")
    calculate_forces = bool(pw.control.get("tprnfor", False))
    calculate_stress = bool(pw.control.get("tstress", False))
    if fixed_potential and (calculate_forces or calculate_stress):
        raise QEInputError(
            "forces and stress are available only for calculation='scf'",
            routine="iosys",
        )
    restart_mode = str(
        pw.control.get("restart_mode", "from_scratch")
    ).strip().lower()
    if restart_mode not in {"from_scratch", "restart"}:
        raise QEInputError(
            f"unknown restart_mode {restart_mode}", routine="iosys"
        )
    starting_potential = str(
        pw.electrons.get("startingpot", "atomic")
    ).strip().lower()
    starting_wavefunctions = str(
        pw.electrons.get("startingwfc", "atomic+random")
    ).strip().lower()
    if starting_potential not in {"atomic", "file"}:
        raise QEInputError(
            f"invalid startingpot={starting_potential!r}; "
            "use 'atomic' or 'file'"
        )
    if starting_wavefunctions not in {
        "atomic", "atomic+random", "random", "file"
    }:
        raise QEInputError(
            f"invalid startingwfc={starting_wavefunctions!r}; use "
            "'atomic', 'atomic+random', 'random', or 'file'"
        )
    if restart_mode == "restart":
        starting_potential = "file"
        starting_wavefunctions = "file"
    if fixed_potential:
        starting_potential = "file"
    starting_messages: list[str] = []
    save_wavefunctions = (
        str(pw.control.get("disk_io", "low")).strip().lower() != "none"
    )
    retain_occupied_states = (
        calculate_forces or calculate_stress or save_wavefunctions
    )
    occupations_mode = str(
        pw.system.get("occupations", "fixed")
    ).strip().lower().replace("-", "_")
    tetrahedron_modes = {
        "tetrahedra", "tetrahedra_lin", "tetrahedra_opt"
    }
    if occupations_mode not in {"fixed", "smearing", *tetrahedron_modes}:
        raise UnsupportedFeatureError(
            not_implemented(f"occupations {occupations_mode}"),
            routine="set_occupations",
        )
    pseudo_payload: tuple[bool, object] | None = None
    if mpi.is_root:
        try:
            pseudo_payload = (
                True,
                {
                    species.label: read_upf(
                        pw.pseudo_dir / species.pseudo_file
                    )
                    for species in pw.species
                },
            )
        except Exception as exc:
            # Broadcast failures too, otherwise non-root ranks remain blocked
            # after rank zero exits this initialization phase.
            pseudo_payload = (False, f"{type(exc).__name__}: {exc}")
    pseudo_payload = mpi.broadcast(pseudo_payload)
    assert isinstance(pseudo_payload, tuple) and len(pseudo_payload) == 2
    pseudo_succeeded, pseudo_value = pseudo_payload
    if not pseudo_succeeded:
        raise QEInputError(str(pseudo_value))
    assert isinstance(pseudo_value, dict)
    pseudo_by_label = pseudo_value
    xc_functional = _resolve_xc_functional(pw, pseudo_by_label)
    pw.system["_resolved_xc"] = xc_functional
    z_by_label = {label: pseudo.z_valence for label, pseudo in pseudo_by_label.items()}
    nelec = sum(z_by_label[atom.label] for atom in pw.atoms) - float(pw.system.get("tot_charge", 0.0))
    if occupations_mode == "fixed" and (
        abs(nelec - round(nelec)) > 1e-10
        or int(round(nelec)) % 2
    ):
        raise UnsupportedFeatureError("fixed non-spin occupations require an even integer electron count")
    occupied = int(round(nelec)) // 2
    nbnd = int(
        pw.system.get(
            "nbnd",
            default_number_of_bands(nelec, occupations_mode),
        )
    )
    if nbnd <= 0:
        raise QEInputError("nbnd less than 1", routine="iosys")
    if occupations_mode == "fixed" and nbnd < occupied:
        raise QEInputError("nbnd is smaller than the number of occupied states")
    if occupations_mode == "smearing":
        degauss_ry = float(pw.system.get("degauss", 0.0))
        if degauss_ry <= 0.0:
            raise QEInputError(
                "smearing requires a value for gaussian broadening (degauss)",
                routine="set_occupations",
            )
        gaussian_order = smearing_order(
            pw.system.get("smearing", "gaussian")
        )
    else:
        degauss_ry = 0.0
        gaussian_order = 0
    if occupations_mode in tetrahedron_modes:
        if pw.kpoint_mode != "automatic" or pw.kpoint_grid is None:
            raise QEInputError(
                "tetrahedron occupations require K_POINTS automatic"
            )
        if len(pw.full_to_irreducible) != int(np.prod(pw.kpoint_grid)):
            raise QEInputError(
                "tetrahedron occupations require a complete automatic k-point grid"
            )
    pw.system["_number_of_bands"] = nbnd
    bases = make_bases(
        pw.reciprocal,
        np.array([point.crystal for point in pw.kpoints]),
        ecut,
        mpi=mpi,
    )
    sizes = [len(basis) for basis in bases]
    if min(sizes) < nbnd:
        raise QEInputError(
            f"nbnd={nbnd} exceeds the smallest plane-wave basis "
            f"dimension ({min(sizes)})"
        )
    diagonalization = _canonical_diagonalization(
        pw.electrons.get("diagonalization", "david")
    )
    if "diago_david_ndiim" in pw.electrons:
        raise QEInputError(
            "unknown ELECTRONS variable diago_david_ndiim; "
            "use QE's diago_david_ndim spelling"
        )
    davidson_ndim = int(pw.electrons.get("diago_david_ndim", 2))
    if davidson_ndim < 2:
        raise QEInputError("diago_david_ndim must be at least 2")
    cg_maxiter = int(pw.electrons.get("diago_cg_maxiter", 20))
    if cg_maxiter < 1:
        raise QEInputError("diago_cg_maxiter must be at least 1")
    rmm_ndim = int(pw.electrons.get("diago_rmm_ndim", 4))
    if rmm_ndim < 2:
        raise QEInputError("diago_rmm_ndim must be at least 2")
    rmm_converge = bool(pw.electrons.get("diago_rmm_conv", False))
    gs_nblock = int(pw.electrons.get("diago_gs_nblock", 16))
    if gs_nblock < 1:
        raise QEInputError("diago_gs_nblock must be at least 1")
    full_accuracy = bool(pw.electrons.get("diago_full_acc", False))
    shape = fft_shape(
        bases,
        pw.reciprocal,
        ecutrho,
        fft_factors(
            pw.symmetry_operations,
            use_all_frac=bool(pw.system.get("use_all_frac", False)),
        ),
    )
    if mpi.size > shape[2]:
        raise QEInputError(
            f"{mpi.size} MPI ranks exceed the third FFT dimension "
            f"({shape[2]}); use at most {shape[2]} ranks"
        )
    if restart_mode != "restart" and (
        starting_potential == "file" or starting_wavefunctions == "file"
    ):
        from .save import resolve_save_directory

        save_directory = resolve_save_directory(pw)
        density_path = save_directory / "charge-density.hdf5"
        if starting_potential == "file" and not density_path.is_file():
            if fixed_potential:
                raise QEInputError(
                    f"Cannot read rho: {density_path} not found; run an "
                    "SCF calculation with the same prefix and outdir first"
                )
            else:
                starting_messages.append(
                    f"Cannot read rho: {density_path} not found; "
                    "using atomic starting potential"
                )
                starting_potential = "atomic"
        if starting_wavefunctions == "file":
            missing_wavefunctions = [
                save_directory / f"wfc{index}.hdf5"
                for index in range(1, len(pw.kpoints) + 1)
                if not (save_directory / f"wfc{index}.hdf5").is_file()
            ]
            if missing_wavefunctions:
                starting_messages.append(
                    "Cannot read wfcs: file not found; recomputing them "
                    "from atomic+random wavefunctions"
                )
                starting_wavefunctions = "atomic+random"
    if starting_potential == "file" or starting_wavefunctions == "file":
        # Local import avoids coupling the persistence writer to SCF module
        # initialization while still sharing the exact QE save schema.
        from .save import validate_restart_metadata

        validate_restart_metadata(
            pw,
            shape,
            nbnd,
            electronic_states=not fixed_potential,
        )
    fft_scratch_pool = FFTScratchPool()
    if mpi.size > 1:
        catalog = bases[0].catalog
        wave_indices = (
            catalog.indices
            if catalog is not None
            else np.unique(
                np.vstack([basis.indices for basis in bases]), axis=0
            )
        )
        wave_fft_descriptor = FFTGridDescriptor.build(
            wave_indices, shape, mpi.size, local_rank=mpi.rank
        )
    else:
        wave_fft_descriptor = None
    local_workspaces = _LazyWorkspaceSequence(
        bases,
        shape,
        mpi=mpi,
        timers=timers,
        scratch_pool=fft_scratch_pool,
        descriptor=wave_fft_descriptor,
        # Bound serial FFT storage by the rank-local thread team. This keeps
        # memory O(threads), not O(bands), while independent band transforms
        # occupy the cores. MPI retains its block transform.
        serial_fft_batch_size=(
            threads_per_process if mpi.size == 1 else None
        ),
        # Every k point is revisited by c_bands and sum_band throughout SCF.
        # Cache only compact slot/owner maps; FFT payloads and plans remain in
        # the one shared scratch pool, so this does not create per-k grids.
        cache_workspaces=True,
        # Serial kernels address one flat FFT grid directly. QE-style stick
        # lists are distributed-FFT metadata and are never read in this path.
        retain_serial_sticks=mpi.size > 1,
    )
    reciprocal_grid = ReciprocalGrid.build(
        shape, pw.reciprocal, ecutrho
    )
    charge_slots = reciprocal_grid.charge_slots
    charge_indices = reciprocal_grid.charge_indices
    charge_fft_descriptor = (
        FFTGridDescriptor.build(
            charge_indices, shape, mpi.size, local_rank=mpi.rank
        )
        if mpi.size > 1
        else None
    )
    charge_workspace = LocalPotentialWorkspace(
        charge_indices,
        shape,
        mpi=mpi,
        timers=timers,
        scratch_pool=fft_scratch_pool,
        descriptor=charge_fft_descriptor,
        # Serial charge transforms use compact flat FFT slots.  The retained
        # QE stick catalog is needed only by the distributed transpose.
        retain_serial_sticks=mpi.size > 1,
    )
    symmetry_plan_started = timers.start()
    density_symmetrizer = ReciprocalDensitySymmetrizer(
        shape,
        pw.symmetry_operations,
        charge_indices,
        reciprocal_grid.charge_g2,
        charge_workspace,
        mpi,
    )
    timers.stop("sym_rho:init", symmetry_plan_started)
    # Star construction needs a dense lookup table and other one-shot index
    # vectors.  Return those freed pages before the first sampled PSS value;
    # otherwise glibc retention makes setup memory look proportional to the
    # persistent SCF arrays even though these buffers are no longer live.
    trim_allocator()
    local_charge_rows = charge_workspace.local_plane_wave_indices
    local_charge_vectors = reciprocal_grid.charge_vectors[local_charge_rows]
    local_charge_g2 = reciprocal_grid.charge_g2[local_charge_rows]
    charge_gvectors = len(charge_indices)
    charge_sticks = len(
        np.unique(charge_slots[:, :2], axis=0)
    )
    wavefunction_sticks = (
        len(wave_fft_descriptor.sticks)
        if wave_fft_descriptor is not None
        else max(
            len(np.unique(basis.indices[:, :2], axis=0))
            for basis in bases
        )
    )
    atomic_orbitals = sum(
        pseudo_by_label[atom.label].number_of_atomic_orbitals
        for atom in pw.atoms
    )
    starting_charge_scales = _starting_charge_scales(
        pw, pseudo_by_label
    )
    starting_charge = 0.0
    for atom in pw.atoms:
        pseudo = pseudo_by_label[atom.label]
        scale = starting_charge_scales[atom.label]
        if pseudo.has_atomic_density:
            starting_charge += float(
                scale
                * pseudo.atomic_density_fourier(
                    np.array([0.0]), pw.volume
                )[0]
                * pw.volume
            )
        else:
            starting_charge += scale * pseudo.z_valence
    grid_points = int(np.prod(shape))
    mixing_ndim_estimate = int(
        pw.electrons.get("mixing_ndim", 8)
    )
    maximum_block = max(nbnd, atomic_orbitals)
    local_plane_wave_counts = [
        len(local_workspaces[index].local_plane_wave_indices)
        for index in range(len(local_workspaces))
    ]
    local_plane_waves = max(local_plane_wave_counts)
    # This is an array working-set estimate, not an RSS prediction. It covers
    # persistent SCF grids/history, basis metadata, wavefunctions, and the
    # largest Davidson/FFT block. Python, BLAS, FFT, and MPI runtimes are
    # measured separately by the peak-RSS report.
    mixing_bytes = (
        2
        * (mixing_ndim_estimate + 1)
        * len(local_charge_rows)
        * 16
        if str(pw.electrons.get("mixing_mode", "plain"))
        .strip()
        .lower()
        .replace("_", "-")
        in {"plain", "default", "tf", "local-tf"}
        else 0
    )
    local_z_slab = mpi.slab(shape[2])
    local_real_points = (
        shape[0]
        * shape[1]
        * (local_z_slab.stop - local_z_slab.start)
    )
    # Count the actual grid representations rather than treating ten grids
    # as complex128.  At the SCF high-water point there are at most seven
    # rank-local float64 density/potential grids (core, ionic, Hartree, XC,
    # the shared Hxc/effective owner, output, and mixed/energy density). The
    # ReciprocalGrid geometry initially contains one float64 g2 grid, two
    # int32 triplet tables, one float64 G-vector triplet table, and one
    # float64 |G|^2 vector. The targeted MPI symmetry path releases those
    # replicated setup tables below; all other paths retain them. The final
    # term counts rank-local charge-vector and |G|^2 caches.
    density_potential_grid_bytes = 7 * local_real_points * 8
    release_replicated_reciprocal_grid = (
        mpi.size > 1
        and pw.full_kpoint_count > len(pw.kpoints)
    )
    reciprocal_grid_bytes = (
        len(local_charge_rows) * (3 * 8 + 8)
        + (
            0
            if release_replicated_reciprocal_grid
            else grid_points * 8
            + charge_gvectors * (3 * 4 + 3 * 4 + 3 * 8 + 8)
        )
    )
    # Per-k bases retain only int32 mappings into one shared global G catalog.
    # MPI FFT stick ownership/transposes live in one grid descriptor; compact
    # current-k maps are rebuilt lazily instead of being retained for all k.
    catalog_size = len(bases[0].catalog.indices) if bases[0].catalog else 0
    basis_metadata_bytes = (
        sum(
            basis.global_indices.nbytes
            for basis in bases
            if basis.global_indices is not None
        )
        + (
            (36 * catalog_size + mpi.shared_size - 1) // mpi.shared_size
            if bases[0].catalog is not None and bases[0].catalog.shared
            else 36 * catalog_size
        )
        + (wave_fft_descriptor.nbytes if wave_fft_descriptor is not None else 0)
        + (
            charge_fft_descriptor.nbytes
            if charge_fft_descriptor is not None
            else 0
        )
    )
    if local_workspaces._cache is not None:
        # Every serial k point, and every distributed irreducible k point, is
        # revisited in each SCF iteration. Retain only compact operational
        # maps; grids and FFT plans still live in the shared scratch pool.
        if mpi.size == 1:
            basis_metadata_bytes += sum(
                workspace.linear_slots.nbytes
                + workspace.sticks.nbytes
                + sum(values.nbytes for values in workspace.sticks_by_rank)
                + workspace.local_plane_wave_indices.nbytes
                for workspace in local_workspaces
            )
        else:
            basis_metadata_bytes += sum(
                workspace.stick_indices.nbytes
                + workspace.owned_plane_waves.nbytes
                + workspace.local_plane_wave_indices.nbytes
                + workspace.local_slots[2].nbytes
                + workspace.local_stick_positions.nbytes
                for workspace in local_workspaces
            )
    starting_and_saved_wfc_bytes = (
        (
            sum(local_plane_wave_counts) * nbnd
            + local_plane_waves * maximum_block
        )
        * 16
    )
    projector_species_layout = [
        (
            pseudo_by_label[species.label].number_of_projector_channels,
            sum(atom.label == species.label for atom in pw.atoms),
        )
        for species in pw.species
        if pseudo_by_label[species.label].number_of_projector_channels
    ]
    packed_projector_channels = sum(
        channels * atoms for channels, atoms in projector_species_layout
    )
    factorized_projector_columns = sum(
        channels + atoms for channels, atoms in projector_species_layout
    )
    # The packed path consumes one species at a time.  At its peak, final beta
    # coexists only with that species' beta, phase, and short-lived conjugated
    # beta used to form the diagonal.  The factorized fallback retains every
    # beta/phase pair and briefly conjugates the largest beta.
    largest_species_setup = max(
        (2 * channels + atoms for channels, atoms in projector_species_layout),
        default=0,
    )
    projector_peak_columns = (
        packed_projector_channels + largest_species_setup
        if packed_projector_channels <= _PACKED_PROJECTOR_CHANNEL_LIMIT
        else factorized_projector_columns
        + max(
            (channels for channels, _atoms in projector_species_layout),
            default=0,
        )
    )
    projector_workspace_bytes = max(
        (
            local_rows * projector_peak_columns * 16
            for local_rows in local_plane_wave_counts
        ),
        default=0,
    )
    estimated_persistent = (
        mixing_bytes
        + density_potential_grid_bytes
        + reciprocal_grid_bytes
        + basis_metadata_bytes
        + starting_and_saved_wfc_bytes
        + density_symmetrizer.mapping_bytes
    )
    persistent_components = [
        ("density/potential grids", density_potential_grid_bytes),
        ("mixing history", mixing_bytes),
        ("wavefunctions", starting_and_saved_wfc_bytes),
        (
            "basis/reciprocal metadata",
            reciprocal_grid_bytes + basis_metadata_bytes,
        ),
        ("symmetry metadata", density_symmetrizer.mapping_bytes),
    ]
    if diagonalization == "dense":
        estimated_workspace = 2 * max(sizes) ** 2 * 16
        workspace_components = [
            ("dense eigensolver workspace", estimated_workspace)
        ]
    else:
        solver_rows = (
            max(sizes) if mpi.size == 1 else local_plane_waves
        )
        davidson_subspace_columns = min(
            max(sizes), davidson_ndim * nbnd
        )
        davidson_bytes = (
            (
                2 * solver_rows * davidson_subspace_columns
                # The previous pair is released after its residual has been
                # preconditioned and copied into basis_storage.  Only the new
                # Ritz vectors and residuals coexist at the expansion peak.
                + 2 * solver_rows * nbnd
            )
            * 16
        )
        # psi and Hpsi are views into the two persistent band blocks; five
        # rank-local work vectors remain for CG.
        cg_bytes = (2 * solver_rows * nbnd + 5 * solver_rows) * 16
        paro_columns = min(
            max(sizes), nbnd + max(int(np.rint(0.5 * nbnd)), 4)
        )
        paro_bytes = (
            2 * solver_rows * paro_columns + 3 * solver_rows * nbnd
        ) * 16
        rmm_bytes = (
            2 * solver_rows * nbnd * rmm_ndim
            + 5 * solver_rows * nbnd
        ) * 16
        if diagonalization == "david":
            solver_bytes = davidson_bytes
            solver_label = "Davidson workspace"
        elif diagonalization == "cg":
            solver_bytes = cg_bytes
            solver_label = "CG workspace"
        elif diagonalization == "paro":
            solver_bytes = paro_bytes
            solver_label = "ParO workspace"
        elif diagonalization == "rmm-paro":
            solver_bytes = max(rmm_bytes, paro_bytes)
            solver_label = "RMM/ParO workspace"
        else:
            solver_bytes = max(rmm_bytes, davidson_bytes)
            solver_label = "RMM/Davidson workspace"
        local_stick_points = shape[2] * (
            len(wave_fft_descriptor.sticks_by_rank[mpi.rank])
            if wave_fft_descriptor is not None
            else wavefunction_sticks
        )
        if mpi.size == 1:
            # One full grid per concurrently transformed band, bounded by the
            # rank-local thread count, plus compact reciprocal results. GGA
            # differentiates all three Cartesian components in one FFT batch
            # even for a single-threaded calculation.
            serial_fft_batch = max(
                min(threads_per_process, maximum_block),
                3 if xc_functional in GGA_FUNCTIONALS else 1,
            )
            serial_grid_stride = local_real_points
            if threads_per_process > 1 and serial_fft_batch > 1:
                # Native band-parallel FFTs pad only the distance between
                # grids, by at most three complex values, so every FFTW
                # new-array pointer preserves 64-byte SIMD alignment.
                serial_grid_stride = (local_real_points + 3) // 4 * 4
            fft_bytes = (
                (serial_grid_stride + local_plane_waves)
                * serial_fft_batch
                * 16
            )
        elif mpi.size > 1:
            assert wave_fft_descriptor is not None
            assert charge_fft_descriptor is not None
            charge_local_stick_points = shape[2] * len(
                charge_fft_descriptor.sticks_by_rank[mpi.rank]
            )
            wave_slab_points = local_real_points // shape[0] // shape[1]
            wave_slab_points *= wavefunction_sticks
            charge_slab_points = (
                local_real_points // shape[0] // shape[1]
            ) * charge_sticks
            # One scratch allocation is shared by wave and charge Z sticks.
            # MPI owns grow-only send/receive buffers sized by the largest of
            # either transpose direction.  Count these actual maxima instead
            # of assuming perfectly balanced wave sticks, which understated
            # the runtime pool for the denser charge descriptor.
            stick_scratch = max(
                charge_local_stick_points,
                local_stick_points * maximum_block,
            )
            transpose_buffer = max(
                charge_local_stick_points,
                charge_slab_points,
                local_stick_points * maximum_block,
                wave_slab_points * maximum_block,
            )
            real_batch = 1
            fft_bytes = (
                stick_scratch
                + local_real_points * real_batch
                + 2 * transpose_buffer
            ) * 16
        else:
            fft_batch_size = 1
            # The inverse and forward Z-stick phases are sequential and use
            # one shared grow-only buffer.  The two MPI transpose buffers
            # add two more stick payloads; only one real-slab batch exists.
            fft_bytes = (
                (local_real_points + 3 * local_stick_points)
                * fft_batch_size
                * 16
            )
        estimated_workspace = (
            solver_bytes + fft_bytes + projector_workspace_bytes
        )
        workspace_components = [
            (solver_label, solver_bytes),
            ("FFT/MPI workspace", fft_bytes),
            ("nonlocal-projector workspace", projector_workspace_bytes),
        ]
    if xc_functional in GGA_FUNCTIONALS:
        # PW92 ec/vc and the four reusable PBE work arrays are pointwise and
        # therefore limited to one cache-sized tile.  Full-size epsilon,
        # potential, and gradient-coefficient outputs are already included in
        # the density/potential-grid high-water set above; the three-component
        # complex gradient is represented by the FFT workspace.
        gga_xc_workspace_bytes = (
            6 * min(local_real_points, GGA_POINT_BLOCK_SIZE) * 8
        )
        estimated_workspace += gga_xc_workspace_bytes
        workspace_components.append(
            ("GGA XC workspace", gga_xc_workspace_bytes)
        )
    if save_wavefunctions and mpi.size > 1:
        save_workspace = (
            max(sizes) * nbnd * np.dtype(np.complex128).itemsize
        )
        if save_workspace > estimated_workspace:
            estimated_workspace = save_workspace
            workspace_components = [
                ("wavefunction-save workspace", save_workspace)
            ]
    array_components = tuple(
        (label, int(size))
        for label, size in persistent_components + workspace_components
        if size
    )
    estimated_array_all_ranks = int(
        mpi.sum_scalar(estimated_persistent + estimated_workspace)
    )
    setup = SCFSetup(
        kpoints=len(bases),
        min_plane_waves=min(sizes),
        max_plane_waves=max(sizes),
        total_plane_waves=sum(sizes),
        dense_work=sum(size**3 for size in sizes),
        diagonalization=diagonalization,
        davidson_subspace=(
            max(min(size, davidson_ndim * nbnd) for size in sizes)
            if diagonalization == "david"
            else 0
        ),
        number_of_bands=nbnd,
        fft_shape=shape,
        charge_gvectors=charge_gvectors,
        charge_sticks=charge_sticks,
        wavefunction_sticks=wavefunction_sticks,
        atomic_orbitals=atomic_orbitals,
        starting_charge=(
            nelec if starting_potential == "file" else starting_charge
        ),
        target_charge=nelec,
        starting_potential=starting_potential,
        starting_wavefunctions=starting_wavefunctions,
        starting_messages=tuple(starting_messages),
        mpi_processes=mpi.size,
        threads_per_process=threads_per_process,
        estimated_persistent_bytes_per_rank=estimated_persistent,
        estimated_peak_workspace_bytes_per_rank=estimated_workspace,
        estimated_array_components_per_rank=array_components,
        estimated_array_bytes_all_ranks=estimated_array_all_ranks,
        runtime_baseline_rss_bytes_per_rank=baseline_rss_per_rank,
        runtime_baseline_pss_bytes_all_ranks=baseline_pss_all_ranks,
        runtime_baseline_uss_bytes_all_ranks=baseline_uss_all_ranks,
    )
    if release_replicated_reciprocal_grid:
        # From here onward MPI branches use only local_charge_vectors/g2 and
        # charge_workspace's compact descriptor.  The complete geometry is
        # referenced exclusively by serial branches, so release its replicated
        # grids after the QE-style symmetry plan has distilled its stars.
        del reciprocal_grid, charge_slots, charge_indices
        trim_allocator()
    if diagonalization == "dense":
        max_dense_pw = 2500
        if setup.max_plane_waves > max_dense_pw:
            raise UnsupportedFeatureError(
                f"dense reference diagonalization needs up to "
                f"{setup.max_plane_waves} plane waves, exceeding "
                f"the built-in limit of {max_dense_pw}"
            )
    hinit_started = timers.start()
    local_q = np.sqrt(local_charge_g2)
    v_ion_coefficients = np.zeros(
        len(local_charge_rows), dtype=complex
    )
    for species in pw.species:
        pseudo = pseudo_by_label[species.label]
        radial = pseudo.fourier(local_q, pw.volume)
        positions_species = [
            atom.position
            for atom in pw.atoms
            if atom.label == species.label
        ]
        structure = sum(
            np.exp(-1j * (local_charge_vectors @ position))
            for position in positions_species
        )
        v_ion_coefficients += radial * structure
    v_ion_local = np.ascontiguousarray(
        np.real(
            charge_workspace.coefficients_to_grid(
                v_ion_coefficients, use_scratch=True
            )
        )
    )
    v_ion_average = mpi.sum_scalar(
        float(np.real(np.sum(v_ion_coefficients[local_charge_g2 < 1e-14])))
    )
    local_z = mpi.slab(shape[2])
    core_coefficients = np.zeros(
        len(local_charge_rows), dtype=complex
    )
    for species in pw.species:
        pseudo = pseudo_by_label[species.label]
        if not pseudo.has_nlcc:
            continue
        radial = pseudo.core_density_fourier(local_q, pw.volume)
        positions_species = [
            atom.position
            for atom in pw.atoms
            if atom.label == species.label
        ]
        structure = sum(
            np.exp(-1j * (local_charge_vectors @ position))
            for position in positions_species
        )
        core_coefficients += radial * structure
    rho_core = np.ascontiguousarray(
        np.real(
            charge_workspace.coefficients_to_grid(
                core_coefficients, use_scratch=True
            )
        )
    )
    if diagonalization == "dense":
        init_us_started = timers.start()
        dense_nonlocal_terms = [
            _nonlocal_projector_terms(pw, pseudo_by_label, basis)
            for basis in bases
        ]
        timers.stop("init_us_2", init_us_started, calls=len(bases))
        nonlocal_matrices = [
            _nonlocal_hamiltonian(len(basis), terms)
            for basis, terms in zip(bases, dense_nonlocal_terms)
        ]
        del dense_nonlocal_terms
    else:
        nonlocal_matrices = []
    timers.stop("hinit0", hinit_started)
    potinit_started = timers.start()
    starting_coefficients: np.ndarray | None = None
    if starting_potential == "file":
        from .save import read_saved_density_coefficients

        # Load directly into the rank-local compact G basis. This avoids the
        # old root-only full complex grid and NumPy FFT temporaries, whose
        # lifetimes overlapped the persistent SCF FFT workspace.
        saved_coefficients = read_saved_density_coefficients(
            pw, charge_indices[local_charge_rows]
        )
        saved_charge = pw.volume * mpi.sum_scalar(
            float(
                np.real(
                    np.sum(
                        saved_coefficients[
                            local_charge_g2 < 1.0e-14
                        ]
                    )
                )
            )
        )
        if (
            not np.isfinite(saved_charge)
            or abs(saved_charge - nelec) > 1.0e-7 * max(1.0, nelec)
        ):
            raise QEInputError(
                f"saved density contains {saved_charge:.12g} electrons; "
                f"expected {nelec:.12g}"
            )
        rho = np.ascontiguousarray(
            np.real(
                charge_workspace.coefficients_to_grid(
                    saved_coefficients, use_scratch=True
                )
            )
        )
        _starting_charge = nelec
        del saved_coefficients
    elif mpi.size == 1:
        rho, _starting_charge = _atomic_starting_density(
            pw,
            pseudo_by_label,
            shape,
            nelec,
            ecutrho,
            reciprocal_grid,
            charge_workspace,
        )
    else:
        starting_coefficients = np.zeros(
            len(local_charge_rows), dtype=complex
        )
        for species in pw.species:
            pseudo = pseudo_by_label[species.label]
            scale = starting_charge_scales[species.label]
            positions_species = [
                atom.position
                for atom in pw.atoms
                if atom.label == species.label
            ]
            if pseudo.has_atomic_density:
                radial = pseudo.atomic_density_fourier(
                    local_q, pw.volume
                )
                structure = sum(
                    np.exp(
                        -1j * (local_charge_vectors @ position)
                    )
                    for position in positions_species
                )
                starting_coefficients += scale * radial * structure
            else:
                zero = local_charge_g2 < 1.0e-14
                starting_coefficients[zero] += (
                    len(positions_species)
                    * scale
                    * pseudo.z_valence
                    / pw.volume
                )
        _starting_charge = pw.volume * mpi.sum_scalar(
            float(
                np.real(
                    np.sum(
                        starting_coefficients[
                            local_charge_g2 < 1.0e-14
                        ]
                    )
                )
            )
        )
        if abs(_starting_charge) < 1.0e-14:
            starting_coefficients.fill(0.0)
            starting_coefficients[local_charge_g2 < 1.0e-14] = (
                nelec / pw.volume
            )
        else:
            starting_coefficients *= nelec / _starting_charge
        rho = np.ascontiguousarray(
            np.real(
                charge_workspace.coefficients_to_grid(
                    starting_coefficients, use_scratch=True
                )
            )
        )
    timers.stop("potinit", potinit_started)
    del local_q, v_ion_coefficients, core_coefficients, starting_coefficients
    trim_allocator()
    weights = np.array([point.weight for point in pw.kpoints])
    charges = np.array([z_by_label[atom.label] for atom in pw.atoms])
    positions = np.array([atom.position for atom in pw.atoms])
    e_ion = ewald_energy(pw.lattice, positions, charges, ecutrho)
    conv_thr_ry = float(pw.electrons.get("conv_thr", 1.0e-6))
    maxiter = int(pw.electrons.get("electron_maxstep", 100))
    mixing = float(pw.electrons.get("mixing_beta", 0.7))
    mixing_mode = (
        str(pw.electrons.get("mixing_mode", "plain"))
        .strip()
        .lower()
        .replace("_", "-")
    )
    mixing_ndim = int(pw.electrons.get("mixing_ndim", 8))
    mixing_pulay_frequency = int(
        pw.electrons.get("mixing_pulay_frequency", 1)
    )
    if not np.isfinite(mixing):
        raise QEInputError("mixing_beta is not finite", routine="iosys")
    if not 1 <= mixing_ndim <= 25:
        raise QEInputError("mixing_ndim must be between 1 and 25")
    if mixing_pulay_frequency < 1:
        raise QEInputError(
            "mixing_pulay_frequency must be at least 1"
        )
    if mixing_mode in {"plain", "default", "tf", "local-tf"}:
        # The compact rho(G) implementation is valid for one rank as well as
        # MPI.  Using it in serial removes PlainBroydenMixer's second full FFT
        # geometry and its NumPy/pocketfft full-grid temporaries.
        density_mixer = DistributedBroydenMixer(
            charge_workspace,
            local_charge_g2,
            beta=mixing,
            ndim=mixing_ndim,
            g2_cutoff=ecutrho,
            mode=mixing_mode,
            nelec=nelec,
            volume=pw.volume,
            pulay_frequency=mixing_pulay_frequency,
        )
    elif mixing_mode == "linear":
        density_mixer = LinearMixer(beta=mixing)
    else:
        raise UnsupportedFeatureError(
            f"mixing_mode={mixing_mode!r} is not supported; use 'plain', "
            "'TF', 'local-TF', or the Python-only diagnostic mode 'linear'"
        )
    iterations: list[SCFIteration] = []
    old_energy = None
    eigenvalues: list[np.ndarray] = []
    band_occupations: list[np.ndarray] = []
    fermi_energy: float | None = None
    smearing_energy = 0.0
    previous_eigenvectors: list[np.ndarray | None] = [None] * len(bases)
    loaded_wavefunctions = starting_wavefunctions == "file"
    if loaded_wavefunctions:
        from .save import read_saved_wavefunction

        for kpoint_index, (basis, workspace) in enumerate(
            zip(bases, local_workspaces)
        ):
            previous_eigenvectors[kpoint_index] = read_saved_wavefunction(
                pw,
                kpoint_index,
                basis.indices,
                nbnd,
                local_rows=(
                    workspace.local_plane_wave_indices
                    if mpi.size > 1 else None
                ),
            )
            # HDF5 selection and Miller-order work arrays are one-k-point
            # temporaries. Return them before loading the next persistent
            # wavefunction block so they do not stack at the restart peak.
            trim_allocator()
    starting_fft_scratch_pool: FFTScratchPool | None = None
    starting_local_workspaces: _LazyWorkspaceSequence | None = None
    if (
        mpi.size == 1
        and diagonalization != "dense"
        and not loaded_wavefunctions
    ):
        # Atomic-orbital initialization is a one-shot phase.  Keep its FFTW
        # plan and full-grid buffer out of the long-lived SCF pool, then drop
        # both after all starting subspaces have been rotated.
        starting_fft_scratch_pool = FFTScratchPool()
        starting_local_workspaces = _LazyWorkspaceSequence(
            bases,
            shape,
            mpi=mpi,
            timers=timers,
            scratch_pool=starting_fft_scratch_pool,
            descriptor=None,
            serial_fft_batch_size=threads_per_process,
            retain_serial_sticks=False,
        )
    qe_random = _QERandom()
    requested_diago_thr = pw.electrons.get("diago_thr_init")
    diago_thr_ry = (
        float(requested_diago_thr)
        if requested_diago_thr is not None
        else (1.0e-5 if starting_potential == "file" else 1.0e-2)
    )
    if not np.isfinite(diago_thr_ry) or diago_thr_ry < 0.0:
        raise QEInputError("diago_thr_init must be nonnegative")
    davidson_tolerance = (
        0.5 * diago_thr_ry if diago_thr_ry > 0.0 else 5.0e-3
    )
    previous_accuracy = np.inf
    energy_terms: SCFEnergyTerms | None = None
    estimated_peak_rss = int(
        mpi.max_scalar(
            max(
                peak_rss_bytes(),
                current_rss_bytes() + estimated_workspace,
            )
        )
    )
    setup_pss_all_ranks = int(mpi.sum_scalar(current_pss_bytes()))
    peak_sampled_pss_all_ranks = max(
        peak_sampled_pss_all_ranks, setup_pss_all_ranks
    )
    setup = replace(
        setup,
        estimated_peak_rss_bytes_per_rank=estimated_peak_rss,
    )
    if progress is not None:
        progress("setup", setup)
    timers.stop("init_run", init_started)
    electrons_started = timers.start()
    iteration = 1
    first_scf_diagonalization = True
    # The end of each completed iteration computes V_Hxc[rho_mixed] for the
    # total-energy correction.  Retain that grid as the next iteration's
    # input potential instead of repeating identical Hartree and GGA FFTs.
    input_hxc: np.ndarray | None = None
    while iteration <= maxiter:
        build_input_hxc = input_hxc is None
        potential_started = (
            timers.start() if build_input_hxc else None
        )
        if build_input_hxc:
            hartree_started = timers.start()
            rho_g_local = charge_workspace.grid_to_coefficients(rho)
            vh_g_local = np.zeros_like(rho_g_local)
            _load_native_fft().hartree_coefficients(
                rho_g_local,
                local_charge_g2,
                vh_g_local,
            )
            vh = np.ascontiguousarray(
                np.real(
                    charge_workspace.coefficients_to_grid(
                        vh_g_local, use_scratch=True
                    )
                )
            )
            del rho_g_local, vh_g_local
            timers.stop("v_h", hartree_started)
        else:
            vh = None
        # Allocate the effective-potential owner before XC and use it first as
        # the total-density workspace. This removes ``rho + rho_core`` as a
        # separate full-grid temporary on every SCF iteration.
        native_potential_layout = (
            (mpi.size > 1 or threads_per_process > 1)
            and diagonalization != "dense"
        )
        if build_input_hxc:
            if native_potential_layout:
                local_z_count = rho.shape[2]
                v_eff_local = np.empty(
                    (local_z_count, shape[0], shape[1]), dtype=np.float64
                )
                v_eff_grid = np.moveaxis(v_eff_local, 0, 2)
            else:
                v_eff_local = np.empty_like(v_ion_local)
                v_eff_grid = v_eff_local
        else:
            # The previous iteration left Hxc in the effective-potential
            # owner. Re-form its native z,x,y view instead of allocating a
            # separate Veff grid. The view recovers the contiguous owner that
            # the threaded local-potential kernel consumes.
            assert input_hxc is not None
            v_eff_grid = input_hxc
            v_eff_local = (
                np.moveaxis(v_eff_grid, 2, 0)
                if native_potential_layout
                else v_eff_grid
            )
        if build_input_hxc:
            np.add(rho, rho_core, out=v_eff_grid)
            xc_started = timers.start()
            _epsilon_input, vxc = _xc_energy_potential(
                v_eff_grid,
                xc_functional,
                workspace=charge_workspace,
                g_vectors=local_charge_vectors,
                need_epsilon=False,
            )
            timers.stop("v_xc", xc_started)
            assert vh is not None
            np.add(vh, vxc, out=vh)
            input_hxc = vh
            assert _epsilon_input is None
            del vxc, _epsilon_input
        # QE constructs vrs once per SCF iteration and shares it across the
        # entire k loop.  Keep the same real-space array in serial as well as
        # MPI instead of inverse-transforming identical v_eff(G) for every k.
        np.add(v_ion_local, input_hxc, out=v_eff_grid)
        potential_average = mpi.sum_scalar(
            float(np.sum(v_eff_local))
        ) / np.prod(shape)
        if diagonalization == "dense":
            v_eff_g = np.fft.fftn(v_eff_local) / np.prod(shape)
        else:
            v_eff_g = np.asarray([potential_average], dtype=complex)
        if potential_started is not None:
            timers.stop("v_of_rho", potential_started)
        eigenvalues, eigenvectors = [], []
        diagonalization_iterations: list[int] = []
        hamiltonian_applications: list[int] = []
        eigen_residuals: list[float] = []
        active_diagonalization = diagonalization
        if iteration > 1 and np.isfinite(previous_accuracy):
            # QE resets ethr to 1e-2 Ry at the second SCF iteration before
            # applying the dr2/nelec tightening rule. This matters when the
            # first Python solve was deliberately more accurate: retaining
            # that tiny threshold forever multiplies Davidson work without
            # improving the converged density.
            if iteration == 2:
                davidson_tolerance = 5.0e-3
            davidson_tolerance = max(
                5.0e-14,
                min(
                    davidson_tolerance,
                    0.1 * previous_accuracy / max(1.0, nelec),
                ),
            )
        memory_rss_report = int(
            mpi.max_scalar(current_rss_bytes())
        )
        memory_pss_all_ranks = int(mpi.sum_scalar(current_pss_bytes()))
        peak_sampled_pss_all_ranks = max(
            peak_sampled_pss_all_ranks, memory_pss_all_ranks
        )
        memory_available_report = available_memory_bytes()
        history_count_report = (
            len(density_mixer.delta_inputs)
            if isinstance(
                density_mixer,
                (PlainBroydenMixer, DistributedBroydenMixer),
            )
            else 0
        )
        wavefunction_memory_report = sum(
            vector.nbytes
            for vector in previous_eigenvectors
            if vector is not None
        ) + max(sizes) * maximum_block * 16 // mpi.size
        bands_started = timers.start()
        for kpoint_index, compact_basis in enumerate(bases):
            basis = compact_basis.materialize()
            local_workspace = local_workspaces[kpoint_index]
            if diagonalization == "dense":
                hamiltonian = potential_matrix(v_eff_g, basis.indices)
                hamiltonian[np.diag_indices_from(hamiltonian)] += (
                    basis.kinetic
                )
                hamiltonian += nonlocal_matrices[kpoint_index]
                # Dense diagonalization is a diagnostic/reference path.  Keep
                # SciPy's sizeable extension baseline out of the default
                # Davidson process until this path is explicitly requested.
                from scipy.linalg import eigh

                values, vectors = eigh(
                    hamiltonian,
                    subset_by_index=(0, nbnd - 1),
                    check_finite=False,
                )
                diagonalization_iterations.append(1)
                hamiltonian_applications.append(1)
                eigen_residuals.append(0.0)
            else:
                init_us_started = timers.start()
                projector_terms = _nonlocal_projector_terms(
                    pw,
                    pseudo_by_label,
                    basis,
                    factorized=True,
                    packed=True,
                    local_rows=(
                        local_workspace.local_plane_wave_indices
                        if mpi.size > 1
                        else None
                    ),
                )
                timers.stop("init_us_2", init_us_started)
                operator = PlaneWaveHamiltonian(
                    basis,
                    v_eff_g,
                    projector_terms,
                    local_workspace=local_workspace,
                    timers=timers,
                    real_potential=v_eff_local,
                    native_potential_layout=(
                        (mpi.size > 1 or threads_per_process > 1)
                        and diagonalization != "dense"
                    ),
                    # QE's usnldiag uses v_of_0: the G=0 component of the
                    # ionic local potential, not the average total effective
                    # potential used by H|psi>.
                    potential_average=v_ion_average,
                )
                trial_vectors = previous_eigenvectors[kpoint_index]
                trial_eigenvalues = None
                trial_applied = None
                if trial_vectors is None:
                    wfcinit_started = timers.start()
                    starting = _atomic_starting_orbitals(
                        pw,
                        pseudo_by_label,
                        basis,
                        nbnd,
                        random_stream=qe_random,
                        # Retain only rank-local coefficients, but advance the
                        # same global QE randy() stream on every rank.  Thus the
                        # distributed starting subspace is the exact row slice
                        # of the serial one, without allocating its global
                        # wavefunction matrix on every rank.
                        local_rows=(
                            operator.local_rows
                            if mpi.size > 1
                            and pw.full_kpoint_count > len(pw.kpoints)
                            else None
                        ),
                        mpi=(
                            mpi
                            if mpi.size > 1
                            and pw.full_kpoint_count > len(pw.kpoints)
                            else None
                        ),
                        starting_wfc=starting_wavefunctions,
                    )
                    if mpi.size > 1 and starting.shape[0] == len(basis):
                        starting = starting[
                            operator.local_rows
                        ]
                    rotation_operator = operator
                    if starting_local_workspaces is not None:
                        rotation_operator = PlaneWaveHamiltonian(
                            basis,
                            v_eff_g,
                            projector_terms,
                            local_workspace=(
                                starting_local_workspaces[kpoint_index]
                            ),
                            timers=timers,
                            real_potential=v_eff_local,
                            native_potential_layout=True,
                            potential_average=v_ion_average,
                        )
                    (
                        trial_vectors,
                        trial_eigenvalues,
                        trial_applied,
                    ) = _rotate_starting_subspace(
                        rotation_operator, starting, nbnd, mpi
                    )
                    timers.stop("wfcinit", wfcinit_started)
                    del starting, rotation_operator
                active_diagonalization = diagonalization
                if diagonalization == "rmm-davidson":
                    active_diagonalization = (
                        "david"
                        if iteration < 3 or iteration % 5 == 0
                        else "rmm"
                    )
                elif diagonalization == "rmm-paro":
                    active_diagonalization = (
                        "paro" if iteration % 5 == 1 else "rmm"
                    )
                diagonalization_started = timers.start()
                occupied_roots = (
                    occupied
                    if occupations_mode == "fixed" and iteration > 1
                    else nbnd
                )
                if active_diagonalization == "david":
                    solution = davidson(
                        operator.apply,
                        operator.diagonal,
                        nbnd,
                        initial_vectors=trial_vectors,
                        tolerance=davidson_tolerance,
                        max_iterations=20,
                        subspace_multiplier=davidson_ndim,
                        residual_factor=None,
                        residual_energy_scale=None,
                        occupied_roots=(
                            nbnd if full_accuracy else occupied_roots
                        ),
                        initial_is_ritz=(
                            iteration == 1 and not loaded_wavefunctions
                        ),
                        initial_is_orthonormal=True,
                        initial_eigenvalues=trial_eigenvalues,
                        initial_applied=trial_applied,
                        mpi=mpi,
                        global_dimension=len(basis),
                        global_row_indices=operator.local_rows,
                        timers=timers,
                        operator_into=operator.apply_into,
                    )
                    timer_name = "cegterg"
                elif active_diagonalization == "cg":
                    solution = conjugate_gradient(
                        operator.apply,
                        operator.local_kinetic,
                        nbnd,
                        initial_vectors=trial_vectors,
                        tolerance=davidson_tolerance,
                        max_iterations=cg_maxiter,
                        occupied_roots=occupied_roots,
                        full_accuracy=full_accuracy,
                        initial_is_orthonormal=True,
                        initial_applied=trial_applied,
                        initial_eigenvalues=trial_eigenvalues,
                        initial_is_ritz=(trial_applied is not None),
                        mpi=mpi,
                        operator_into=operator.apply_into,
                    )
                    timer_name = "ccgdiagg"
                elif active_diagonalization == "paro":
                    solution = parallel_orbital(
                        operator.apply,
                        operator.diagonal,
                        nbnd,
                        initial_vectors=trial_vectors,
                        tolerance=davidson_tolerance,
                        max_iterations=20,
                        occupied_roots=occupied_roots,
                        full_accuracy=full_accuracy,
                        initial_is_orthonormal=True,
                        initial_applied=trial_applied,
                        initial_eigenvalues=trial_eigenvalues,
                        initial_is_ritz=(trial_applied is not None),
                        mpi=mpi,
                        operator_into=operator.apply_into,
                    )
                    timer_name = "paro"
                else:
                    solution = rmm_diis(
                        operator.apply,
                        operator.local_kinetic,
                        nbnd,
                        initial_vectors=trial_vectors,
                        tolerance=davidson_tolerance,
                        max_iterations=20,
                        history_dimension=rmm_ndim,
                        converge=rmm_converge,
                        gram_schmidt_block=gs_nblock,
                        occupied_roots=occupied_roots,
                        full_accuracy=full_accuracy,
                        initial_is_orthonormal=True,
                        initial_applied=trial_applied,
                        initial_eigenvalues=trial_eigenvalues,
                        initial_is_ritz=(trial_applied is not None),
                        mpi=mpi,
                        operator_into=operator.apply_into,
                    )
                    timer_name = "rmm-diis"
                timers.stop(timer_name, diagonalization_started)
                davidson_iterations = solution.iterations
                # cegterg returns its best Ritz vectors after 20 iterations.
                # c_bands accepts that result unless more than five bands are
                # unconverged; do not turn QE's accepted small-band result
                # into a fatal Python error.
                if (
                    not solution.converged
                    and solution.number_unconverged > 5
                ):
                    raise QEInputError(
                        f"{active_diagonalization} diagonalization did not "
                        f"converge at k point "
                        f"{kpoint_index + 1} after {davidson_iterations} "
                        f"iterations (largest residual "
                        f"{np.max(solution.residual_norms):.3e} Ha)"
                    )
                values, vectors = (
                    solution.eigenvalues,
                    solution.eigenvectors,
                )
                previous_eigenvectors[kpoint_index] = vectors
                diagonalization_iterations.append(davidson_iterations)
                hamiltonian_applications.append(
                    solution.hamiltonian_applications
                )
                eigen_residuals.append(
                    float(np.max(solution.residual_norms))
                )
            eigenvalues.append(values)
            eigenvectors.append(vectors)
            if diagonalization != "dense":
                del (
                    operator,
                    projector_terms,
                    solution,
                    trial_vectors,
                    trial_eigenvalues,
                    trial_applied,
                )
        if starting_local_workspaces is not None:
            # The last temporary Hamiltonian/workspace was deleted above;
            # releasing these owners now also releases the atomic FFT plan
            # and its aligned full-grid allocation before density mixing.
            starting_local_workspaces = None
            starting_fft_scratch_pool = None
            trim_allocator()
        timers.stop("c_bands", bands_started)
        # All current-k Davidson bases, Hpsi images, and projector operands
        # have gone out of scope. Trim once per complete k loop—not per k
        # point—so freed solver heaps do not inflate the next iteration's PSS
        # sample while avoiding thousands of allocator calls.
        trim_allocator()
        if occupations_mode == "smearing":
            fermi_energy, band_occupations, smearing_energy = (
                smeared_occupations(
                    eigenvalues,
                    weights,
                    nelec,
                    0.5 * degauss_ry,
                    gaussian_order,
                )
            )
        elif occupations_mode in tetrahedron_modes:
            fermi_energy, band_occupations = tetrahedron_occupations(
                eigenvalues,
                weights,
                nelec,
                pw.kpoint_grid,
                pw.full_to_irreducible,
                pw.reciprocal,
                occupations_mode,
            )
            smearing_energy = 0.0
        else:
            band_occupations = []
            for values in eigenvalues:
                values_occupations = np.zeros(len(values))
                values_occupations[:occupied] = 2.0
                band_occupations.append(values_occupations)
            fermi_energy = None
            smearing_energy = 0.0
        if fixed_potential:
            # NSCF and bands calculations stop after diagonalizing the fixed
            # SCF potential.  In particular, do not construct rho_out, test a
            # density residual, or enter the mixing path.
            timers.stop("electrons", electrons_started)
            result_density = mpi.gather_z_slabs_root(rho, shape)
            if save_wavefunctions:
                (
                    final_wavefunctions,
                    final_miller_indices,
                    final_wavefunction_rows,
                    wavefunctions_distributed,
                ) = _final_wavefunction_payload(
                    eigenvectors,
                    bases,
                    local_workspaces,
                    mpi,
                )
            else:
                final_wavefunctions = []
                final_miller_indices = []
                final_wavefunction_rows = []
                wavefunctions_distributed = False
            peak_per_rank, peak_all_ranks = _peak_memory_across_ranks(mpi)
            final_pss = int(mpi.sum_scalar(current_pss_bytes()))
            peak_sampled_pss_all_ranks = max(
                peak_sampled_pss_all_ranks, final_pss
            )
            return SCFResult(
                converged=True,
                # A fixed-potential diagonalization does not define a new
                # self-consistent total energy.  Keep this field finite for
                # API compatibility; reporting suppresses it for these modes.
                total_energy_ha=0.0,
                eigenvalues_ha=eigenvalues,
                density=result_density,
                wall_seconds=time.perf_counter() - start,
                plane_waves_per_k=sizes,
                peak_rss_bytes_per_rank=peak_per_rank,
                peak_rss_bytes_all_ranks=peak_all_ranks,
                peak_sampled_pss_bytes_all_ranks=(
                    peak_sampled_pss_all_ranks
                ),
                timings=timers.snapshot(),
                occupations=[values.copy() for values in band_occupations],
                fermi_energy_ha=fermi_energy,
                mpi_processes=mpi.size,
                wavefunctions=final_wavefunctions,
                wavefunction_miller_indices=final_miller_indices,
                wavefunction_row_indices=final_wavefunction_rows,
                wavefunctions_distributed=wavefunctions_distributed,
            )
        sum_band_started = timers.start()
        rho_out = _density_from_states(
            eigenvectors,
            bases,
            weights,
            band_occupations,
            shape,
            pw.volume,
            nelec,
            mpi,
            local_workspaces,
            timers,
        )
        if not retain_occupied_states:
            del eigenvectors
        if pw.full_kpoint_count > len(pw.kpoints):
            symmetry_started = timers.start()
            rho_out = density_symmetrizer.apply(rho_out)
            total_density = mpi.sum_scalar(float(np.sum(rho_out)))
            rho_out *= (
                nelec * np.prod(shape) / (total_density * pw.volume)
            )
            timers.stop("sum_band:sym", symmetry_started)
        timers.stop("sum_band", sum_band_started)
        band_energy = sum(
            weight * float(np.sum(values * values_occupations))
            for weight, values, values_occupations in zip(
                weights, eigenvalues, band_occupations
            )
        )
        grid_scale = pw.volume / np.prod(shape)
        # Preserve the input-potential expectation before reusing the Veff
        # owner for the density residual below. Hxc = Veff - Vion.
        deband = -grid_scale * mpi.sum_scalar(
            float(
                np.einsum("ijk,ijk->", rho_out, v_eff_grid)
                - np.einsum("ijk,ijk->", rho_out, v_ion_local)
            )
        )
        input_hxc = None
        real_grid_workspace = v_eff_grid
        np.subtract(rho_out, rho, out=real_grid_workspace)
        # Keep the compact residual produced for QE's dr2 metric and hand it
        # to Broyden below. Previously mix_rho repeated this forward FFT.
        residual_g_local = charge_workspace.grid_to_coefficients(
            real_grid_workspace
        )
        accuracy = 0.5 * pw.volume * mpi.sum_scalar(
            _load_native_fft().hartree_residual_metric(
                residual_g_local,
                local_charge_g2,
            )
        )
        previous_accuracy = accuracy
        # QE estimates the first-solve eigensolver error as ethr*nelec. If
        # the newly measured dr2 is smaller, it repeats the same first SCF
        # iteration, without mixing, at 0.1*dr2/nelec. Both ``accuracy`` and
        # ``davidson_tolerance`` are in Hartree here, so the unit conversion
        # cancels in the comparison and update.
        if (
            iteration == 1
            and first_scf_diagonalization
            and accuracy < davidson_tolerance * max(1.0, nelec)
        ):
            davidson_tolerance = max(
                5.0e-14,
                0.1 * accuracy / max(1.0, nelec),
            )
            first_scf_diagonalization = False
            del rho_out, real_grid_workspace, v_eff_grid, v_eff_local
            del residual_g_local
            trim_allocator()
            continue
        first_scf_diagonalization = False
        # QE mix_rho decides convergence directly from dr2, including on the
        # first SCF iteration.  Requiring a previous total energy imposed an
        # extra, non-QE iteration when a deliberately loose conv_thr is used.
        density_converged = accuracy * 2.0 < conv_thr_ry
        maximum_eigen_residual = float(np.max(eigen_residuals))
        # Davidson has already enforced its residual safeguard.  As in QE's
        # mix_rho, SCF convergence itself is decided by the density residual;
        # feeding an already-converged residual back through Broyden changes
        # the fixed-point history and can move the calculation away again.
        converged = density_converged
        mix_started = timers.start()
        if converged:
            rho_energy = rho_out
        elif isinstance(density_mixer, DistributedBroydenMixer):
            rho_energy = density_mixer.mix(
                rho,
                rho_out,
                residual_coefficients=residual_g_local,
            )
        else:
            rho_energy = density_mixer.mix(rho, rho_out)
        timers.stop("mix_rho", mix_started)
        del residual_g_local
        # The mixer has produced the next input density.  The preceding input
        # grid is not part of the energy expression below; release it before
        # XC/Hartree workspaces are allocated rather than carrying eight real
        # grids through the energy high-water phase.
        del rho
        energy_potential_started = timers.start()
        rho_energy_g_local = charge_workspace.grid_to_coefficients(
            rho_energy
        )
        vh_energy_g_local = np.zeros_like(rho_energy_g_local)
        _load_native_fft().hartree_coefficients(
            rho_energy_g_local,
            local_charge_g2,
            vh_energy_g_local,
        )
        eh = pw.volume * mpi.sum_scalar(
            0.5
            * float(
                np.real(
                    np.vdot(
                        rho_energy_g_local,
                        vh_energy_g_local,
                    )
                )
            )
        )
        del rho_energy_g_local
        np.add(rho_energy, rho_core, out=real_grid_workspace)
        xc_started = timers.start()
        epsilon_energy, vxc_energy = _xc_energy_potential(
            real_grid_workspace,
            xc_functional,
            workspace=charge_workspace,
            g_vectors=local_charge_vectors,
            need_epsilon=False,
            energy_density_out=real_grid_workspace,
        )
        timers.stop("v_xc", xc_started)
        assert epsilon_energy is None
        del epsilon_energy
        exc = grid_scale * mpi.sum_scalar(
            float(np.sum(real_grid_workspace))
        )
        vh_energy = np.ascontiguousarray(
            np.real(
                charge_workspace.coefficients_to_grid(
                    vh_energy_g_local, use_scratch=True
                )
            )
        )
        del vh_energy_g_local
        timers.stop("v_of_rho", energy_potential_started)
        np.add(vh_energy, vxc_energy, out=real_grid_workspace)
        descf = -grid_scale * mpi.sum_scalar(
            float(
                np.einsum("ijk,ijk->", rho_energy, real_grid_workspace)
                - np.einsum("ijk,ijk->", rho_out, real_grid_workspace)
            )
        )
        # The Veff owner is now exactly V_Hxc[rho_energy]. Preserve its native
        # storage as next iteration's input potential without a copy.
        input_hxc = real_grid_workspace
        energy_terms = SCFEnergyTerms(
            one_electron_ha=band_energy + deband,
            hartree_ha=eh,
            xc_ha=exc,
            ewald_ha=e_ion,
            descf_ha=descf,
            smearing_ha=smearing_energy,
        )
        energy = (
            energy_terms.one_electron_ha
            + energy_terms.hartree_ha
            + energy_terms.xc_ha
            + energy_terms.ewald_ha
            + energy_terms.descf_ha
            + energy_terms.smearing_ha
        )
        iteration_density_bytes = (
            3
            * shape[0]
            * shape[1]
            * (local_z.stop - local_z.start)
            * 8
        )
        iteration_mixing_bytes = (
            2
            * (history_count_report + 1)
            * len(local_charge_rows)
            * 16
            if isinstance(
                density_mixer,
                (PlainBroydenMixer, DistributedBroydenMixer),
            )
            else 0
        )
        iteration_davidson_bytes = (
            estimated_workspace
            if diagonalization == "dense"
            else solver_bytes
        )
        iteration_fft_scratch_bytes = fft_scratch_pool.nbytes
        iteration_fft_exchange_bytes = mpi.exchange_buffer_bytes
        iteration_fft_bytes = (
            iteration_fft_scratch_bytes + iteration_fft_exchange_bytes
        )
        step = SCFIteration(
            iteration,
            energy,
            accuracy,
            cutoff_ry=ecut,
            mixing_beta=mixing,
            diagonalization=active_diagonalization,
            davidson_threshold_ha=davidson_tolerance,
            average_diagonalization_iterations=float(
                np.mean(diagonalization_iterations)
            ),
            average_hamiltonian_applications=float(
                np.mean(hamiltonian_applications)
            ),
            maximum_eigen_residual_ha=float(
                maximum_eigen_residual
            ),
            cpu_seconds=time.process_time() - cpu_start,
            memory_rss_bytes_per_rank=memory_rss_report,
            memory_pss_bytes_all_ranks=memory_pss_all_ranks,
            memory_available_bytes=memory_available_report,
            memory_density_bytes_per_rank=iteration_density_bytes,
            memory_mixing_bytes_per_rank=iteration_mixing_bytes,
            memory_wavefunctions_bytes_per_rank=wavefunction_memory_report,
            memory_davidson_bytes_per_rank=iteration_davidson_bytes,
            memory_fft_bytes_per_rank=iteration_fft_bytes,
            memory_fft_scratch_bytes_per_rank=iteration_fft_scratch_bytes,
            memory_fft_exchange_bytes_per_rank=iteration_fft_exchange_bytes,
            memory_fft_plan_count_per_rank=int(
                mpi.max_scalar(len(fft_scratch_pool._fftw_plans))
            ),
        )
        iterations.append(step)
        if progress is not None:
            progress("iteration", step)
        rho = rho_energy
        if converged:
            timers.stop("electrons", electrons_started)
            result_density = mpi.gather_z_slabs_root(rho, shape)
            derivative_real_workspace = (
                real_grid_workspace
                if calculate_forces or calculate_stress
                else None
            )
            # mix_rho and sym_rho are finished permanently at convergence.
            # Releasing their O(N_G) histories/maps before forces and stress
            # prevents the SCF and derivative high-water sets from stacking.
            del density_mixer, density_symmetrizer, previous_eigenvectors
            del v_ion_local
            del (
                rho_out,
                rho_energy,
                real_grid_workspace,
                v_eff_grid,
                v_eff_local,
                input_hxc,
            )
            del (
                vxc_energy,
                vh_energy,
            )
            if mpi.size == 1:
                del reciprocal_grid, charge_slots, charge_indices
            trim_allocator()
            forces_ha_per_bohr = None
            stress_ha_per_bohr3 = None
            nonlocal_energy = 0.0
            nonlocal_force = None
            nonlocal_tensor = None
            if calculate_forces or calculate_stress:
                derivative_started = timers.start()
                (
                    nonlocal_energy,
                    nonlocal_force,
                    nonlocal_tensor,
                ) = _nonlocal_derivatives(
                    pw,
                    pseudo_by_label,
                    bases,
                    eigenvectors,
                    weights,
                    band_occupations,
                    mpi,
                    local_workspaces,
                    forces=calculate_forces,
                    stress=calculate_stress,
                )
                timers.stop("nonlocal_derivatives", derivative_started)
            if calculate_forces:
                force_started = timers.start()
                assert derivative_real_workspace is not None
                np.add(
                    rho,
                    rho_core,
                    out=derivative_real_workspace,
                )
                final_epsilon, final_vxc = _xc_energy_potential(
                    derivative_real_workspace,
                    xc_functional,
                    workspace=charge_workspace,
                    g_vectors=local_charge_vectors,
                )
                del final_epsilon
                local_force, core_force = _local_and_core_forces(
                    pw,
                    pseudo_by_label,
                    rho,
                    final_vxc,
                    charge_workspace,
                    local_charge_vectors,
                    mpi,
                )
                assert nonlocal_force is not None
                ionic_force = ewald_forces(
                    pw.lattice,
                    positions,
                    charges,
                    ecutrho,
                    reciprocal_vectors=local_charge_vectors,
                    mpi=mpi,
                )
                forces_ha_per_bohr = (
                    local_force
                    + core_force
                    + nonlocal_force
                    + ionic_force
                )
                forces_ha_per_bohr = symmetrize_forces(
                    forces_ha_per_bohr,
                    pw.lattice,
                    positions @ np.linalg.inv(pw.lattice),
                    [atom.label for atom in pw.atoms],
                    pw.symmetry_operations,
                )
                # A periodic neutral cell is invariant under a rigid
                # translation. Enforce the corresponding acoustic sum rule
                # after assembling independently rounded FFT/projector terms.
                forces_ha_per_bohr -= np.mean(
                    forces_ha_per_bohr, axis=0, keepdims=True
                )
                del final_vxc
                trim_allocator()
                timers.stop("forces", force_started)
            if calculate_stress:
                stress_started = timers.start()
                assert nonlocal_tensor is not None
                stress_ha_per_bohr3 = _hellmann_feynman_stress(
                    pw,
                    pseudo_by_label,
                    bases,
                    eigenvectors,
                    weights,
                    band_occupations,
                    rho,
                    rho_core,
                    shape,
                    local_charge_vectors,
                    charge_workspace,
                    local_workspaces,
                    ecutrho,
                    xc_functional,
                    mpi,
                    charges,
                    nonlocal_energy,
                    nonlocal_tensor,
                    real_workspace=derivative_real_workspace,
                )
                trim_allocator()
                timers.stop("stress", stress_started)
            if save_wavefunctions:
                (
                    final_wavefunctions,
                    final_miller_indices,
                    final_wavefunction_rows,
                    wavefunctions_distributed,
                ) = _final_wavefunction_payload(
                    eigenvectors,
                    bases,
                    local_workspaces,
                    mpi,
                )
            else:
                final_wavefunctions = []
                final_miller_indices = []
                final_wavefunction_rows = []
                wavefunctions_distributed = False
            # The returned SCFResult owns only final observables (and optional
            # saved wavefunctions), not the FFT plans, k-point maps, projector
            # state, or occupied states used to form derivatives.
            if retain_occupied_states:
                del eigenvectors
            del derivative_real_workspace
            del basis, compact_basis, local_workspace, vectors
            del local_workspaces, charge_workspace, fft_scratch_pool, bases
            trim_allocator()
            peak_per_rank, peak_all_ranks = _peak_memory_across_ranks(mpi)
            final_pss = int(mpi.sum_scalar(current_pss_bytes()))
            peak_sampled_pss_all_ranks = max(
                peak_sampled_pss_all_ranks, final_pss
            )
            return SCFResult(
                converged=True,
                total_energy_ha=energy,
                eigenvalues_ha=eigenvalues,
                density=result_density,
                iterations=iterations,
                wall_seconds=time.perf_counter() - start,
                energy_terms=energy_terms,
                plane_waves_per_k=sizes,
                peak_rss_bytes_per_rank=peak_per_rank,
                peak_rss_bytes_all_ranks=peak_all_ranks,
                peak_sampled_pss_bytes_all_ranks=(
                    peak_sampled_pss_all_ranks
                ),
                timings=timers.snapshot(),
                forces_ha_per_bohr=forces_ha_per_bohr,
                stress_ha_per_bohr3=stress_ha_per_bohr3,
                occupations=[values.copy() for values in band_occupations],
                fermi_energy_ha=fermi_energy,
                mpi_processes=mpi.size,
                wavefunctions=final_wavefunctions,
                wavefunction_miller_indices=final_miller_indices,
                wavefunction_row_indices=final_wavefunction_rows,
                wavefunctions_distributed=wavefunctions_distributed,
            )
        old_energy = energy
        iteration += 1
        # None of these current-output/energy workspaces is an input to the
        # next diagonalization.  Leaving their Python names alive retained two
        # real grids until after the next iteration's memory sample. QE reuses
        # this storage at the iteration boundary; release it here and let the
        # shared FFT pool remain the one intentional grow-only owner.
        del (
            rho_out,
            rho_energy,
            real_grid_workspace,
            v_eff_grid,
            v_eff_local,
        )
        del (
            vxc_energy,
            vh_energy,
        )
        trim_allocator()
    timers.stop("electrons", electrons_started)
    result_density = mpi.gather_z_slabs_root(rho, shape)
    if save_wavefunctions:
        (
            final_wavefunctions,
            final_miller_indices,
            final_wavefunction_rows,
            wavefunctions_distributed,
        ) = _final_wavefunction_payload(
            eigenvectors,
            bases,
            local_workspaces,
            mpi,
        )
    else:
        final_wavefunctions = []
        final_miller_indices = []
        final_wavefunction_rows = []
        wavefunctions_distributed = False
    peak_per_rank, peak_all_ranks = _peak_memory_across_ranks(mpi)
    final_pss = int(mpi.sum_scalar(current_pss_bytes()))
    peak_sampled_pss_all_ranks = max(
        peak_sampled_pss_all_ranks, final_pss
    )
    return SCFResult(
        converged=False,
        total_energy_ha=float(old_energy),
        eigenvalues_ha=eigenvalues,
        density=result_density,
        iterations=iterations,
        wall_seconds=time.perf_counter() - start,
        energy_terms=energy_terms,
        plane_waves_per_k=sizes,
        peak_rss_bytes_per_rank=peak_per_rank,
        peak_rss_bytes_all_ranks=peak_all_ranks,
        peak_sampled_pss_bytes_all_ranks=peak_sampled_pss_all_ranks,
        timings=timers.snapshot(),
        occupations=[values.copy() for values in band_occupations],
        fermi_energy_ha=fermi_energy,
        mpi_processes=mpi.size,
        wavefunctions=final_wavefunctions,
        wavefunction_miller_indices=final_miller_indices,
        wavefunction_row_indices=final_wavefunction_rows,
        wavefunctions_distributed=wavefunctions_distributed,
    )
