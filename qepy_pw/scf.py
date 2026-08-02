"""Serial scalar Kohn-Sham SCF driver corresponding to pw.x/electrons."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from importlib.util import find_spec
import time
from typing import Callable

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from .acceleration import numba_kernels, resolve_numba
from .basis import (
    FFTGridDescriptor,
    FFTScratchPool,
    LocalPotentialWorkspace,
    PlaneWaveBasis,
    coefficients_to_grid,
    fft_shape,
    make_bases,
    potential_matrix,
)
from .diagonalization import (
    FactorizedProjectorTerm,
    PlaneWaveHamiltonian,
    ProjectorTerm,
    davidson,
)
from .errors import QEInputError, UnsupportedFeatureError
from .ewald import ewald_energy, ewald_forces, ewald_stress
from .input import PWInput
from .mixing import (
    DistributedBroydenMixer,
    LinearMixer,
    PlainBroydenMixer,
)
from .memory import available_memory_bytes, current_rss_bytes, peak_rss_bytes
from .occupations import (
    default_number_of_bands,
    smeared_occupations,
    smearing_order,
)
from .timing import TimingEntry, TimingRegistry
from .mpi import MPIContext
from .upf import LocalPotential, read_upf
from .xc import pbe_unpolarized_components, pz81_unpolarized
from .symmetry import (
    DensitySymmetrizer,
    fft_factors,
    symmetrize_forces,
    symmetrize_stress,
)


@dataclass
class SCFIteration:
    number: int
    total_energy_ha: float
    estimated_accuracy_ha: float
    cutoff_ry: float = 0.0
    mixing_beta: float = 0.0
    davidson_threshold_ha: float = 0.0
    average_diagonalization_iterations: float = 0.0
    average_hamiltonian_applications: float = 0.0
    maximum_eigen_residual_ha: float = 0.0
    cpu_seconds: float = 0.0
    memory_rss_bytes_per_rank: int = 0
    memory_available_bytes: int = 0
    memory_density_bytes_per_rank: int = 0
    memory_mixing_bytes_per_rank: int = 0
    memory_wavefunctions_bytes_per_rank: int = 0
    memory_davidson_bytes_per_rank: int = 0
    memory_fft_bytes_per_rank: int = 0


@dataclass(frozen=True)
class SCFEnergyTerms:
    one_electron_ha: float
    hartree_ha: float
    xc_ha: float
    ewald_ha: float
    descf_ha: float = 0.0
    smearing_ha: float = 0.0


@dataclass(frozen=True)
class SCFSetup:
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
    fft_backend: str = "numpy"
    fft_threads: int = 1
    fft_planner: str = "measure"
    numba_enabled: bool = False
    mpi_processes: int = 1
    estimated_persistent_bytes_per_rank: int = 0
    estimated_peak_workspace_bytes_per_rank: int = 0


@dataclass
class SCFResult:
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
    timings: dict[str, TimingEntry] = field(default_factory=dict)
    forces_ha_per_bohr: np.ndarray | None = None
    stress_ha_per_bohr3: np.ndarray | None = None
    occupations: list[np.ndarray] = field(default_factory=list)
    fermi_energy_ha: float | None = None
    mpi_processes: int = 1
    wavefunctions: list[np.ndarray] = field(default_factory=list)
    wavefunction_miller_indices: list[np.ndarray] = field(default_factory=list)


ProgressCallback = Callable[[str, SCFSetup | SCFIteration], None]


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
    """Rebuild compact current-k maps around one shared FFT descriptor."""

    def __init__(
        self,
        bases: Sequence[PlaneWaveBasis],
        shape: tuple[int, int, int],
        *,
        backend: str,
        threads: int,
        planner_effort: str,
        use_numba: bool,
        mpi: MPIContext,
        timers: TimingRegistry,
        scratch_pool: FFTScratchPool,
        descriptor: FFTGridDescriptor | None,
    ) -> None:
        self.bases = bases
        self.shape = shape
        self.options = dict(
            backend=backend,
            threads=threads,
            planner_effort=planner_effort,
            use_numba=use_numba,
            mpi=mpi,
            timers=timers,
            scratch_pool=scratch_pool,
            descriptor=descriptor,
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
        return LocalPotentialWorkspace(
            self.bases[index], self.shape, **self.options
        )


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
        count = 2 * plane_waves * bands
        values = np.empty(count, dtype=float)
        table = self._table
        table_size = self._table_size
        modulus = self._modulus
        multiplier = self._multiplier
        increment = self._increment
        current = self._current
        seed = self._seed
        # Keep the stateful shuffle recurrence in one tight loop. Calling
        # ``random`` through a generator adds millions of Python function
        # calls for dense k meshes, although the recurrence itself is tiny.
        for index in range(count):
            table_index = (table_size * current) // modulus
            current = int(table[table_index])
            values[index] = current / modulus
            seed = (multiplier * seed + increment) % modulus
            table[table_index] = seed
        self._current = current
        self._seed = seed
        first = values[0::2].reshape(bands, plane_waves).T
        second = values[1::2].reshape(bands, plane_waves).T
        return first, second


def _randomize_atomic_trials(
    trials: np.ndarray,
    kinetic: np.ndarray,
    random_stream: _QERandom,
    strength: float = 0.05,
) -> np.ndarray:
    """Apply QE's coefficient-wise 5% perturbation to atomic trials."""
    del kinetic  # Retained in the private signature for compatibility.
    amplitude, phase = random_stream.pairs_by_band(
        trials.shape[0], trials.shape[1]
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
    local_rows: np.ndarray | None = None,
) -> tuple[ProjectorTerm, ...]:
    """Build the compact atom-projector representation of the nonlocal term."""
    basis_vectors = basis.vectors
    if factorized:
        terms: list[ProjectorTerm] = []
        vectors = (
            basis_vectors
            if local_rows is None
            else basis_vectors[local_rows]
        )
        for species in pw.species:
            pseudo = pseudos[species.label]
            positions = [
                atom.position
                for atom in pw.atoms
                if atom.label == species.label
            ]
            if not positions or not pseudo.projectors:
                continue
            beta, coupling = pseudo.projector_basis(
                vectors, pw.volume
            )
            phases = np.exp(
                -1j
                * (
                    vectors
                    @ np.asarray(positions, dtype=float).T
                )
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
    for beta, coupling in terms:
        result += (beta @ coupling) @ beta.conj().T
    # Roundoff in transforms should not leak into the Hermitian eigensolver.
    return 0.5 * (result + result.conj().T)


def _atomic_starting_orbitals(
    pw: PWInput,
    pseudos: dict[str, LocalPotential],
    basis: PlaneWaveBasis,
    number_of_bands: int,
    random_stream: _QERandom,
) -> np.ndarray:
    """Build QE ``atomic+random`` trial vectors for the first diagonalization."""
    starting_wfc = str(
        pw.electrons.get("startingwfc", "atomic+random")
    ).lower()
    if starting_wfc not in {"atomic", "atomic+random", "random"}:
        raise UnsupportedFeatureError(
            f"startingwfc={starting_wfc!r} is not ported"
        )
    basis_vectors = basis.vectors
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
        else np.empty((len(basis), 0), dtype=complex)
    )
    if trials.shape[1]:
        norms = np.linalg.norm(trials, axis=0)
        usable = norms > max(1.0e-14, 1.0e-12 * float(np.max(norms)))
        trials = trials[:, usable]
    if (
        starting_wfc == "atomic+random"
        and trials.shape[1] >= number_of_bands
    ):
        # QE's default ``atomic+random`` perturbs a complete atomic trial
        # space by five percent to avoid loss of symmetry-related states.
        trials = _randomize_atomic_trials(
            trials, kinetic, random_stream
        )
    else:
        missing = number_of_bands - trials.shape[1]
        amplitude, phase_fraction = random_stream.pairs_by_band(
            len(basis), missing
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


def _rotate_starting_subspace(
    operator: PlaneWaveHamiltonian,
    trials: np.ndarray,
    number_of_bands: int,
    mpi: MPIContext | None = None,
) -> np.ndarray:
    """Reproduce QE ``wfcinit:rotate_wfc`` before the first Davidson call."""
    atomic_basis = np.asarray(trials, dtype=complex)
    mpi = mpi if mpi is not None else MPIContext()
    applied = operator.apply(atomic_basis)
    projected = mpi.sum_array(atomic_basis.conj().T @ applied)
    overlap = mpi.sum_array(atomic_basis.conj().T @ atomic_basis)
    projected = 0.5 * (projected + projected.conj().T)
    overlap = 0.5 * (overlap + overlap.conj().T)
    _, rotation = eigh(
        projected,
        overlap,
        subset_by_index=(0, number_of_bands - 1),
        check_finite=False,
    )
    return atomic_basis @ rotation


def _frozen_core_density(
    pw: PWInput,
    pseudos: dict[str, LocalPotential],
    shape: tuple[int, int, int],
    g2_cutoff: float,
    geometry: ReciprocalGrid | None = None,
) -> np.ndarray:
    """Superpose the UPF NLCC pseudo-core densities on the periodic FFT grid."""
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
        if not pseudo.has_nlcc:
            continue
        radial = pseudo.core_density_fourier(q, pw.volume)
        positions = [
            atom.position for atom in pw.atoms if atom.label == species.label
        ]
        structure = sum(
            np.exp(-1j * np.einsum("...j,j->...", gvec, position))
            for position in positions
        )
        coefficients += radial * structure
    core_g = np.zeros(shape, dtype=complex)
    core_g[tuple(geometry.charge_slots.T)] = coefficients
    return np.real(np.fft.ifftn(core_g * np.prod(shape)))


def _atomic_starting_density(
    pw: PWInput,
    pseudos: dict[str, LocalPotential],
    shape: tuple[int, int, int],
    nelec: float,
    g2_cutoff: float,
    geometry: ReciprocalGrid | None = None,
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
    for species in pw.species:
        pseudo = pseudos[species.label]
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
            coefficients += radial * structure
        else:
            # A UPF without PP_RHOATOM contributes the correct average
            # charge but no invented atom-centered Fourier components.
            coefficients[geometry.charge_g2 < 1.0e-14] += (
                len(positions) * pseudo.z_valence / pw.volume
            )
    starting_charge = float(
        np.real(np.sum(coefficients[geometry.charge_g2 < 1.0e-14]))
    ) * pw.volume
    if abs(starting_charge) < 1.0e-14:
        raise QEInputError("atomic starting density has zero total charge")
    coefficients *= nelec / starting_charge
    density_g = np.zeros(shape, dtype=complex)
    density_g[tuple(geometry.charge_slots.T)] = coefficients
    density = np.real(np.fft.ifftn(density_g * np.prod(shape)))
    return density, starting_charge


def _xc_terms(
    valence_density: np.ndarray,
    core_density: np.ndarray,
    volume: float,
    use_numba: bool = False,
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
            use_numba=use_numba,
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
        use_numba=use_numba,
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
    use_numba: bool = False,
    workspace: LocalPotentialWorkspace | None = None,
    g_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the selected XC functional on a rank-local FFT slab."""
    if functional == "pz":
        return pz81_unpolarized(density, use_numba=use_numba)
    if functional != "pbe":
        raise ValueError(f"unknown XC functional {functional!r}")
    if workspace is None or g_vectors is None:
        raise ValueError("PBE requires an FFT workspace and reciprocal vectors")

    epsilon, potential, _gradient, _coefficient = _pbe_energy_potential_data(
        density, workspace, g_vectors
    )
    return epsilon, potential


def _pbe_energy_potential_data(
    density: np.ndarray,
    workspace: LocalPotentialWorkspace,
    g_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return PBE energy/potential plus fields needed by GGA stress."""
    coefficients = workspace.grid_to_coefficients(density)
    gradient = np.moveaxis(
        workspace.coefficients_to_grid(
            1j * g_vectors * coefficients[:, None]
        ),
        -1,
        0,
    ).real

    def divergence(field: np.ndarray) -> np.ndarray:
        # Put Cartesian components last while entering the FFT workspace: its
        # C-order batch layout then keeps grid points contiguous in memory.
        flux_coefficients = workspace.grid_to_coefficients(
            np.moveaxis(field, 0, -1)
        )
        derivatives = workspace.coefficients_to_grid(
            1j * g_vectors * flux_coefficients
        )
        return np.sum(derivatives.real, axis=-1)

    epsilon, local_potential, gradient_coefficient = (
        pbe_unpolarized_components(density, gradient)
    )
    potential = local_potential - divergence(
        gradient_coefficient[None, ...] * gradient
    )
    return epsilon, potential, gradient, gradient_coefficient


def _hartree(
    density_g: np.ndarray,
    reciprocal: np.ndarray,
    mpi: MPIContext | None = None,
    geometry: ReciprocalGrid | None = None,
) -> tuple[np.ndarray, float]:
    """Distributed reciprocal-space Hartree kernel over contiguous G chunks."""
    mpi = mpi if mpi is not None else MPIContext()
    shape = density_g.shape
    if mpi.size == 1:
        g2 = (
            geometry.g2
            if geometry is not None
            else ReciprocalGrid.build(
                shape, reciprocal, float("inf")
            ).g2
        )
        vh = np.zeros(shape, dtype=complex)
        mask = g2 > 1.0e-14
        vh[mask] = 4.0 * np.pi * density_g[mask] / g2[mask]
        energy = 0.5 * float(
            np.real(np.sum(np.conj(density_g) * vh))
        )
        return vh, energy
    size = int(np.prod(shape))
    local_slice = mpi.slab(size)
    if geometry is not None:
        g2 = geometry.g2.ravel()[local_slice]
    else:
        linear = np.arange(local_slice.start, local_slice.stop)
        slots = np.column_stack(np.unravel_index(linear, shape))
        for axis, axis_size in enumerate(shape):
            slots[:, axis] = np.where(
                slots[:, axis] <= axis_size // 2,
                slots[:, axis],
                slots[:, axis] - axis_size,
            )
        gvec = slots @ reciprocal
        g2 = np.einsum("ij,ij->i", gvec, gvec)
    rho_local = density_g.ravel()[local_slice]
    vh_local = np.zeros(rho_local.size, dtype=complex)
    mask = g2 > 1.0e-14
    vh_local[mask] = 4.0 * np.pi * rho_local[mask] / g2[mask]
    energy_local = 0.5 * float(
        np.real(np.vdot(rho_local, vh_local))
    )
    vh = mpi.gather_flat_chunks(vh_local, size).reshape(shape)
    return vh, mpi.sum_scalar(energy_local)


def _iteration_energy(
    band_energy: float,
    e_ion: float,
    rho_out: np.ndarray,
    rho_energy: np.ndarray,
    input_hxc: np.ndarray,
    core_density: np.ndarray,
    reciprocal: np.ndarray,
    volume: float,
    use_numba: bool = False,
    mpi: MPIContext | None = None,
    functional: str = "pz",
    workspace: LocalPotentialWorkspace | None = None,
    g_vectors: np.ndarray | None = None,
    smearing_energy: float = 0.0,
    geometry: ReciprocalGrid | None = None,
) -> tuple[float, SCFEnergyTerms]:
    """QE ``delta_e`` plus ``delta_escf`` energy for one SCF iteration."""
    rho_energy_g = np.fft.fftn(rho_energy) / np.prod(rho_energy.shape)
    mpi = mpi if mpi is not None else MPIContext()
    z_slab = mpi.slab(rho_out.shape[2])
    grid_points = float(np.prod(rho_out.shape))
    deband = -volume / grid_points * mpi.sum_scalar(
        float(
            np.sum(
                rho_out[:, :, z_slab] * input_hxc[:, :, z_slab]
            )
        )
    )
    vh_energy_g, eh_density = _hartree(
        rho_energy_g, reciprocal, mpi, geometry
    )
    eh = volume * eh_density
    vxc_energy, exc, _ = _xc_terms(
        rho_energy,
        core_density,
        volume,
        use_numba,
        mpi,
        functional,
        workspace,
        g_vectors,
    )
    vh_energy = np.real(
        np.fft.ifftn(vh_energy_g * np.prod(rho_energy.shape))
    )
    descf = -volume / grid_points * mpi.sum_scalar(
        float(
            np.sum(
                (rho_energy[:, :, z_slab] - rho_out[:, :, z_slab])
                * (vh_energy[:, :, z_slab] + vxc_energy[:, :, z_slab])
            )
        )
    )
    terms = SCFEnergyTerms(
        one_electron_ha=band_energy + deband,
        hartree_ha=eh,
        xc_ha=exc,
        ewald_ha=e_ion,
        descf_ha=descf,
        smearing_ha=smearing_energy,
    )
    return (
        terms.one_electron_ha
        + terms.hartree_ha
        + terms.xc_ha
        + terms.ewald_ha
        + terms.descf_ha
        + terms.smearing_ha,
        terms,
    )


def _density_error_ry(
    density_in: np.ndarray,
    density_out: np.ndarray,
    reciprocal: np.ndarray,
    volume: float,
    g2_cutoff: float | None = None,
    mpi: MPIContext | None = None,
    geometry: ReciprocalGrid | None = None,
) -> float:
    """QE ``rho_ddot`` norm of the scalar density residual, in Ry."""
    residual_g = np.fft.fftn(density_out - density_in) / np.prod(
        density_in.shape
    )
    mpi = mpi if mpi is not None else MPIContext()
    shape = density_in.shape
    if mpi.size == 1:
        g2 = (
            geometry.g2
            if geometry is not None
            else ReciprocalGrid.build(
                shape,
                reciprocal,
                float("inf") if g2_cutoff is None else g2_cutoff,
            ).g2
        )
        mask = g2 > 1.0e-14
        if g2_cutoff is not None:
            mask &= g2 <= g2_cutoff + 1.0e-12
        return volume * float(
            np.sum(
                4.0 * np.pi * np.abs(residual_g[mask]) ** 2 / g2[mask]
            )
        )
    local_slice = mpi.slab(int(np.prod(shape)))
    if geometry is not None:
        g2 = geometry.g2.ravel()[local_slice]
    else:
        linear = np.arange(local_slice.start, local_slice.stop)
        indices = np.column_stack(np.unravel_index(linear, shape))
        for axis, size in enumerate(shape):
            indices[:, axis] = np.where(
                indices[:, axis] <= size // 2,
                indices[:, axis],
                indices[:, axis] - size,
            )
        vectors = indices @ reciprocal
        g2 = np.einsum("ij,ij->i", vectors, vectors)
    mask = g2 > 1.0e-14
    if g2_cutoff is not None:
        mask &= g2 <= g2_cutoff + 1.0e-12
    residual_local = residual_g.ravel()[local_slice]
    return volume * mpi.sum_scalar(
        float(
            np.sum(
                4.0
                * np.pi
                * np.abs(residual_local[mask]) ** 2
                / g2[mask]
            )
        )
    )


def _functional_family(name: str) -> str | None:
    normalized = (
        str(name)
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    if any(
        variant in normalized
        for variant in ("pbesol", "revpbe", "rpbe")
    ):
        return "unsupported"
    if (
        normalized == "pbe"
        or "perdewburkeernzerhof" in normalized
        or ("pbx" in normalized and "pbc" in normalized)
    ):
        return "pbe"
    if normalized in {"pz", "pz81", "lda"}:
        return "pz"
    if "pz" in normalized and "pbe" not in normalized:
        return "pz"
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

    families = {
        family
        for pseudo in pseudos.values()
        if (family := _functional_family(pseudo.functional)) is not None
    }
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
    return "pz"


def _density_from_states(
    vectors: list[np.ndarray],
    bases: list[PlaneWaveBasis],
    weights: np.ndarray,
    occupations: list[np.ndarray],
    shape: tuple[int, int, int],
    volume: float,
    nelec: float,
    use_numba: bool = False,
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
    jit = numba_kernels() if use_numba else None
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
        psi = workspace.coefficients_to_grid(
            coeff[:, :band_count],
            use_scratch=True,
        )
        psi /= np.sqrt(volume)
        if timers is not None and transform_started is not None:
            timers.stop("sum_band:wfc", transform_started)
        accumulate_started = timers.start() if timers is not None else None
        if jit is None:
            rho_local += weight * np.sum(
                np.abs(psi) ** 2
                * band_occupations[None, None, None, :band_count],
                axis=-1,
            )
        else:
            jit.accumulate_density_weighted_bands(
                rho_local.ravel(),
                psi.reshape(-1, band_count),
                np.asarray(band_occupations[:band_count]),
                float(weight),
            )
        if timers is not None and accumulate_started is not None:
            timers.stop("sum_band:rho", accumulate_started)
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
            for atom_index, atom in indexed_atoms:
                phase = np.exp(-1j * (gk @ atom.position))
                modulated = np.conjugate(phase)[:, None] * occupied_vectors
                overlap = mpi.sum_array(beta.conj().T @ modulated)
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
                            beta.conj(),
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
                            beta_gradient.conj(),
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
    use_numba: bool,
    mpi: MPIContext,
    charges: np.ndarray,
    nonlocal_energy: float,
    nonlocal_tensor: np.ndarray,
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

    # Hartree term, obtained by differentiating 4*pi/G^2 and the two
    # Omega^-1 density coefficients.
    nonzero = g2 > 1.0e-14
    hartree_energy = 2.0 * np.pi * volume * mpi.sum_scalar(
        float(np.sum(np.abs(density_g[nonzero]) ** 2 / g2[nonzero]))
    )
    hartree_tensor = np.zeros((3, 3))
    if np.any(nonzero):
        hartree_tensor = -4.0 * np.pi * mpi.sum_array(
            np.einsum(
                "g,gi,gj->ij",
                np.abs(density_g[nonzero]) ** 2 / g2[nonzero] ** 2,
                g_vectors[nonzero],
                g_vectors[nonzero],
            )
        )
    stress += hartree_tensor + hartree_energy / volume * np.eye(3)

    # Exchange-correlation contribution. QE's diagonal term contains the
    # valence-density potential integral, while Exc and every GGA gradient
    # are evaluated at rho_valence + rho_core.
    total_density = density + core_density
    if xc_functional == "pbe":
        (
            epsilon_xc,
            potential_xc,
            density_gradient,
            gradient_coefficient,
        ) = _pbe_energy_potential_data(
            total_density, charge_workspace, g_vectors
        )
    else:
        epsilon_xc, potential_xc = _xc_energy_potential(
            total_density,
            xc_functional,
            use_numba=use_numba,
            workspace=charge_workspace,
            g_vectors=g_vectors,
        )
        density_gradient = gradient_coefficient = None
    grid_scale = volume / np.prod(shape)
    xc_energy = grid_scale * mpi.sum_scalar(
        float(np.sum(total_density * epsilon_xc))
    )
    xc_potential_energy = grid_scale * mpi.sum_scalar(
        float(np.sum(density * potential_xc))
    )
    stress -= (xc_energy - xc_potential_energy) / volume * np.eye(3)
    if density_gradient is not None and gradient_coefficient is not None:
        stress += mpi.sum_array(
            np.einsum(
                "...,i...,j...->ij",
                gradient_coefficient,
                density_gradient,
                density_gradient,
                optimize=True,
            )
        ) / np.prod(shape)

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
    )
    return symmetrize_stress(
        0.5 * (stress + stress.T), pw.lattice, pw.symmetry_operations
    )


def run_scf(
    pw: PWInput,
    progress: ProgressCallback | None = None,
    mpi: MPIContext | None = None,
) -> SCFResult:
    """Run SCF with one numerical-library thread per rank by default."""
    mpi = mpi if mpi is not None else MPIContext.world()
    numerical_threads = int(
        pw.electrons.get("py_blas_threads", 1)
    )
    if numerical_threads < 1:
        raise QEInputError("py_blas_threads must be at least 1")
    # With no user_api filter threadpoolctl also constrains OpenMP runtimes
    # loaded by SciPy/Numba, preventing MPI ranks from silently inheriting
    # the machine-wide OpenBLAS, MKL, or OpenMP thread count.
    with threadpool_limits(limits=numerical_threads):
        return _run_scf(pw, progress, mpi)


def _run_scf(
    pw: PWInput,
    progress: ProgressCallback | None = None,
    mpi: MPIContext | None = None,
) -> SCFResult:
    mpi = mpi if mpi is not None else MPIContext()
    start = time.perf_counter()
    cpu_start = time.process_time()
    timers = TimingRegistry()
    init_started = timers.start()
    ecut = float(pw.system.get("ecutwfc", 0.0))
    if ecut <= 0:
        raise QEInputError("ecutwfc must be positive")
    ecutrho = float(pw.system.get("ecutrho", 4.0 * ecut))
    if ecutrho <= 0.0:
        raise QEInputError("ecutrho must be positive")
    calculate_forces = bool(pw.control.get("tprnfor", False))
    calculate_stress = bool(pw.control.get("tstress", False))
    restart_mode = str(
        pw.control.get("restart_mode", "from_scratch")
    ).strip().lower()
    if restart_mode not in {"from_scratch", "restart"}:
        raise QEInputError(
            "restart_mode must be 'from_scratch' or 'restart'"
        )
    starting_potential = str(
        pw.electrons.get("startingpot", "atomic")
    ).strip().lower()
    starting_wavefunctions = str(
        pw.electrons.get("startingwfc", "atomic+random")
    ).strip().lower()
    if starting_potential not in {"atomic", "file"}:
        raise UnsupportedFeatureError(
            f"startingpot={starting_potential!r} is not ported; use 'atomic' or 'file'"
        )
    if starting_wavefunctions not in {
        "atomic", "atomic+random", "random", "file"
    }:
        raise UnsupportedFeatureError(
            f"startingwfc={starting_wavefunctions!r} is not ported"
        )
    if restart_mode == "restart":
        starting_potential = "file"
        starting_wavefunctions = "file"
    save_wavefunctions = (
        str(pw.control.get("disk_io", "low")).strip().lower() != "none"
    )
    retain_occupied_states = (
        calculate_forces or calculate_stress or save_wavefunctions
    )
    occupations_mode = str(
        pw.system.get("occupations", "fixed")
    ).strip().lower()
    if occupations_mode not in {"fixed", "smearing"}:
        raise UnsupportedFeatureError(
            f"occupations={occupations_mode!r} is not ported; use "
            "'fixed' or 'smearing'"
        )
    pseudo_by_label = {}
    for species in pw.species:
        pseudo_by_label[species.label] = read_upf(pw.pseudo_dir / species.pseudo_file)
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
        raise QEInputError("nbnd must be positive")
    if occupations_mode == "fixed" and nbnd < occupied:
        raise QEInputError("nbnd is smaller than the number of occupied states")
    if occupations_mode == "smearing":
        degauss_ry = float(pw.system.get("degauss", 0.0))
        if degauss_ry <= 0.0:
            raise QEInputError(
                "degauss must be positive when occupations='smearing'"
            )
        gaussian_order = smearing_order(
            pw.system.get("smearing", "gaussian")
        )
    else:
        degauss_ry = 0.0
        gaussian_order = 0
    pw.system["_number_of_bands"] = nbnd
    bases = make_bases(
        pw.reciprocal,
        np.array([point.crystal for point in pw.kpoints]),
        ecut,
    )
    sizes = [len(basis) for basis in bases]
    if min(sizes) < nbnd:
        raise QEInputError(
            f"nbnd={nbnd} exceeds the smallest plane-wave basis "
            f"dimension ({min(sizes)})"
        )
    diagonalization = str(
        pw.electrons.get("diagonalization", "david")
    ).lower()
    if diagonalization == "davidson":
        diagonalization = "david"
    if diagonalization not in {"david", "dense"}:
        raise UnsupportedFeatureError(
            f"diagonalization={diagonalization!r} is not ported; use "
            "'david' or the Python-only diagnostic mode 'dense'"
        )
    davidson_ndim = int(pw.electrons.get("diago_david_ndim", 2))
    if diagonalization == "david" and davidson_ndim < 2:
        raise QEInputError("diago_david_ndim must be at least 2")
    if "py_cache_projectors" in pw.electrons:
        raise QEInputError(
            "py_cache_projectors has been removed; nonlocal projectors "
            "are now built for the current k point like QE"
        )
    shape = fft_shape(
        bases,
        pw.reciprocal,
        ecutrho,
        fft_factors(pw.symmetry_operations),
    )
    if mpi.size > shape[2]:
        raise QEInputError(
            f"{mpi.size} MPI ranks exceed the third FFT dimension "
            f"({shape[2]}); use at most {shape[2]} ranks"
        )
    if starting_wavefunctions == "file" and diagonalization == "dense":
        raise UnsupportedFeatureError(
            "startingwfc='file' requires the Davidson diagonalization path"
        )
    if starting_potential == "file" or starting_wavefunctions == "file":
        # Local import avoids coupling the persistence writer to SCF module
        # initialization while still sharing the exact QE save schema.
        from .save import validate_restart_metadata

        validate_restart_metadata(pw, shape, nbnd)
    requested_fft_backend = pw.electrons.get("py_fft_backend")
    fft_backend = str(
        requested_fft_backend
        if requested_fft_backend is not None
        else ("pyfftw" if find_spec("pyfftw") is not None else "scipy")
    ).lower()
    fft_threads = int(pw.electrons.get("py_fft_threads", 1))
    fft_planner = str(
        pw.electrons.get("py_fft_planner", "measure")
    ).lower()
    numba_enabled = resolve_numba(
        pw.electrons.get("py_numba", False)
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
            wave_indices, shape, mpi.size
        )
    else:
        wave_fft_descriptor = None
    local_workspaces = _LazyWorkspaceSequence(
        bases,
        shape,
        backend=fft_backend,
        threads=fft_threads,
        planner_effort=fft_planner,
        use_numba=numba_enabled,
        mpi=mpi,
        timers=timers,
        scratch_pool=fft_scratch_pool,
        descriptor=wave_fft_descriptor,
    )
    symmetry_plan_started = timers.start()
    density_symmetrizer = DensitySymmetrizer(
        shape, pw.symmetry_operations, mpi
    )
    timers.stop("sym_rho:init", symmetry_plan_started)
    reciprocal_grid = ReciprocalGrid.build(
        shape, pw.reciprocal, ecutrho
    )
    charge_slots = reciprocal_grid.charge_slots
    charge_indices = reciprocal_grid.charge_indices
    charge_fft_descriptor = (
        FFTGridDescriptor.build(charge_indices, shape, mpi.size)
        if mpi.size > 1
        else None
    )
    charge_workspace = LocalPotentialWorkspace(
        charge_indices,
        shape,
        backend=fft_backend,
        threads=fft_threads,
        planner_effort=fft_planner,
        use_numba=numba_enabled,
        mpi=mpi,
        timers=timers,
        scratch_pool=fft_scratch_pool,
        descriptor=charge_fft_descriptor,
    )
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
        else max(len(workspace.sticks) for workspace in local_workspaces)
    )
    atomic_orbitals = sum(
        pseudo_by_label[atom.label].number_of_atomic_orbitals
        for atom in pw.atoms
    )
    starting_charge = 0.0
    for atom in pw.atoms:
        pseudo = pseudo_by_label[atom.label]
        if pseudo.has_atomic_density:
            starting_charge += float(
                pseudo.atomic_density_fourier(
                    np.array([0.0]), pw.volume
                )[0]
                * pw.volume
            )
        else:
            starting_charge += pseudo.z_valence
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
    # largest Davidson/FFT block. Python, BLAS, FFT, MPI, and JIT runtimes are
    # measured separately by the peak-RSS report.
    mixing_bytes = (
        2
        * (mixing_ndim_estimate + 1)
        * int(np.ceil(charge_gvectors / mpi.size))
        * 16
        if str(pw.electrons.get("mixing_mode", "plain")).lower()
        in {"plain", "default"}
        else 0
    )
    replicated_grid_bytes = (
        10 * grid_points * 16
        if mpi.size == 1
        else 10 * int(np.ceil(grid_points / mpi.size)) * 16
    )
    # Per-k bases retain only int32 mappings into one shared global G catalog.
    # MPI FFT stick ownership/transposes live in one grid descriptor; compact
    # current-k maps are rebuilt lazily instead of being retained for all k.
    catalog_size = len(bases[0].catalog.indices) if bases[0].catalog else 0
    basis_metadata_bytes = (
        4 * sum(sizes)
        + 36 * catalog_size
        + (wave_fft_descriptor.nbytes if wave_fft_descriptor is not None else 0)
        + (
            charge_fft_descriptor.nbytes
            if charge_fft_descriptor is not None
            else 0
        )
    )
    starting_and_saved_wfc_bytes = (
        (
            sum(sizes) * nbnd
            + max(sizes) * maximum_block
        )
        * 16
        // mpi.size
    )
    projector_species_layout = [
        (
            pseudo_by_label[species.label].number_of_projector_channels,
            sum(atom.label == species.label for atom in pw.atoms),
        )
        for species in pw.species
        if pseudo_by_label[species.label].number_of_projector_channels
    ]
    projector_workspace_bytes = max(
        (
            local_rows
            * sum(
                channels + atoms
                for channels, atoms in projector_species_layout
            )
            * 16
            for local_rows in local_plane_wave_counts
        ),
        default=0,
    )
    estimated_persistent = (
        mixing_bytes
        + replicated_grid_bytes
        + basis_metadata_bytes
        + starting_and_saved_wfc_bytes
        + density_symmetrizer.mapping_bytes
    )
    if diagonalization == "dense":
        estimated_workspace = 2 * max(sizes) ** 2 * 16
    else:
        davidson_rows = (
            max(sizes) if mpi.size == 1 else local_plane_waves
        )
        davidson_bytes = (
            2
            * davidson_rows
            * min(max(sizes), davidson_ndim * nbnd)
            * 16
        )
        local_real_points = int(np.ceil(grid_points / mpi.size))
        local_stick_points = shape[2] * (
            len(wave_fft_descriptor.sticks_by_rank[mpi.rank])
            if wave_fft_descriptor is not None
            else wavefunction_sticks
        )
        if mpi.size == 1 and fft_backend == "pyfftw":
            # QE-style in-place G->R->G transform: one full FFT grid plus
            # the compact reciprocal-space result block.
            fft_bytes = (
                (local_real_points + local_plane_waves)
                * maximum_block
                * 16
            )
        else:
            fft_bytes = (
                2
                * (local_real_points + local_stick_points)
                * maximum_block
                * 16
            )
        estimated_workspace = (
            davidson_bytes + fft_bytes + projector_workspace_bytes
        )
    if save_wavefunctions and mpi.size > 1:
        estimated_workspace = max(
            estimated_workspace,
            sum(sizes) * nbnd * np.dtype(np.complex128).itemsize,
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
        fft_backend=fft_backend,
        fft_threads=fft_threads,
        fft_planner=fft_planner,
        numba_enabled=numba_enabled,
        mpi_processes=mpi.size,
        estimated_persistent_bytes_per_rank=estimated_persistent,
        estimated_peak_workspace_bytes_per_rank=estimated_workspace,
    )
    if progress is not None:
        progress("setup", setup)
    if diagonalization == "dense":
        max_dense_pw = int(pw.system.get("py_max_dense_pw", 2500))
        if setup.max_plane_waves > max_dense_pw:
            raise UnsupportedFeatureError(
                f"dense reference diagonalization needs up to "
                f"{setup.max_plane_waves} plane waves, exceeding "
                f"py_max_dense_pw={max_dense_pw}"
            )
        max_dense_kpoints = int(
            pw.system.get("py_max_dense_kpoints", 64)
        )
        if setup.kpoints > max_dense_kpoints:
            raise UnsupportedFeatureError(
                f"dense reference solver received {setup.kpoints} active k "
                f"points, exceeding py_max_dense_kpoints="
                f"{max_dense_kpoints}; use Davidson or set the limit "
                f"explicitly to acknowledge the computational cost"
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
    v_ion_local = np.real(
        charge_workspace.coefficients_to_grid(v_ion_coefficients)
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
    rho_core = np.real(
        charge_workspace.coefficients_to_grid(core_coefficients)
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
    if starting_potential == "file":
        from .save import read_saved_density

        saved_density = None
        density_error = None
        if mpi.is_root:
            try:
                saved_density = read_saved_density(pw, shape, nelec)
            except QEInputError as exc:
                density_error = str(exc)
        density_error = mpi.broadcast(density_error)
        if density_error is not None:
            raise QEInputError(density_error)
        rho = mpi.scatter_z_slabs_root(saved_density, shape)
        _starting_charge = nelec
        del saved_density
    elif mpi.size == 1:
        rho, _starting_charge = _atomic_starting_density(
            pw,
            pseudo_by_label,
            shape,
            nelec,
            ecutrho,
            reciprocal_grid,
        )
    else:
        starting_coefficients = np.zeros(
            len(local_charge_rows), dtype=complex
        )
        for species in pw.species:
            pseudo = pseudo_by_label[species.label]
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
                starting_coefficients += radial * structure
            else:
                zero = local_charge_g2 < 1.0e-14
                starting_coefficients[zero] += (
                    len(positions_species)
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
        starting_coefficients *= nelec / _starting_charge
        rho = np.real(
            charge_workspace.coefficients_to_grid(
                starting_coefficients
            )
        )
    timers.stop("potinit", potinit_started)
    weights = np.array([point.weight for point in pw.kpoints])
    charges = np.array([z_by_label[atom.label] for atom in pw.atoms])
    positions = np.array([atom.position for atom in pw.atoms])
    e_ion = ewald_energy(pw.lattice, positions, charges, ecutrho)
    conv_thr_ry = float(pw.electrons.get("conv_thr", 1.0e-6))
    maxiter = int(pw.electrons.get("electron_maxstep", 100))
    mixing = float(pw.electrons.get("mixing_beta", 0.7))
    mixing_mode = str(pw.electrons.get("mixing_mode", "plain")).lower()
    mixing_ndim = int(pw.electrons.get("mixing_ndim", 8))
    if mixing_mode in {"plain", "default"}:
        density_mixer = (
            DistributedBroydenMixer(
                charge_workspace,
                local_charge_g2,
                beta=mixing,
                ndim=mixing_ndim,
                g2_cutoff=ecutrho,
            )
            if mpi.size > 1
            else PlainBroydenMixer(
                shape,
                pw.reciprocal,
                beta=mixing,
                ndim=mixing_ndim,
                g2_cutoff=ecutrho,
                use_numba=numba_enabled,
                mpi=mpi,
            )
        )
    elif mixing_mode == "linear":
        density_mixer = LinearMixer(beta=mixing)
    else:
        raise UnsupportedFeatureError(
            f"mixing_mode={mixing_mode!r} is not ported; use 'plain' "
            "(Broyden) or the Python-only diagnostic mode 'linear'"
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
    qe_random = _QERandom()
    requested_diago_thr = pw.electrons.get("diago_thr_init")
    diago_thr_ry = (
        float(requested_diago_thr)
        if requested_diago_thr is not None
        else 1.0e-2
    )
    if diago_thr_ry < 0.0:
        raise QEInputError("diago_thr_init must be nonnegative")
    davidson_tolerance = (
        0.5 * diago_thr_ry if diago_thr_ry > 0.0 else 5.0e-3
    )
    previous_accuracy = np.inf
    davidson_maxiter = int(
        pw.electrons.get("py_davidson_maxiter", 20)
    )
    residual_factor_value = pw.electrons.get(
        "py_davidson_residual_factor"
    )
    davidson_residual_factor = (
        None
        if residual_factor_value is None
        else float(residual_factor_value)
    )
    residual_energy_scale_value = pw.electrons.get(
        "py_davidson_residual_energy_scale"
    )
    if residual_energy_scale_value is None:
        davidson_residual_energy_scale = None
    else:
        residual_energy_scale = float(residual_energy_scale_value)
        davidson_residual_energy_scale = (
            None if residual_energy_scale == 0.0 else residual_energy_scale
        )
    energy_terms: SCFEnergyTerms | None = None
    timers.stop("init_run", init_started)
    electrons_started = timers.start()
    iteration = 1
    first_scf_diagonalization = True
    while iteration <= maxiter:
        potential_started = timers.start()
        hartree_started = timers.start()
        if mpi.size == 1:
            rho_g = np.fft.fftn(rho) / np.prod(shape)
            vh_g, _ = _hartree(
                rho_g, pw.reciprocal, mpi, reciprocal_grid
            )
            vh = np.real(np.fft.ifftn(vh_g * np.prod(shape)))
        else:
            rho_g_local = charge_workspace.grid_to_coefficients(rho)
            vh_g_local = np.zeros_like(rho_g_local)
            hartree_mask = local_charge_g2 > 1.0e-14
            vh_g_local[hartree_mask] = (
                4.0
                * np.pi
                * rho_g_local[hartree_mask]
                / local_charge_g2[hartree_mask]
            )
            vh = np.real(
                charge_workspace.coefficients_to_grid(vh_g_local)
            )
        timers.stop("v_h", hartree_started)
        xc_started = timers.start()
        _, vxc = _xc_energy_potential(
            rho + rho_core,
            xc_functional,
            use_numba=numba_enabled,
            workspace=charge_workspace,
            g_vectors=local_charge_vectors,
        )
        timers.stop("v_xc", xc_started)
        input_hxc = vh + vxc
        # QE constructs vrs once per SCF iteration and shares it across the
        # entire k loop.  Keep the same real-space array in serial as well as
        # MPI instead of inverse-transforming identical v_eff(G) for every k.
        v_eff_local = v_ion_local + input_hxc
        potential_average = mpi.sum_scalar(
            float(np.sum(v_eff_local))
        ) / np.prod(shape)
        if diagonalization == "dense":
            v_eff_g = np.fft.fftn(v_eff_local) / np.prod(shape)
        else:
            v_eff_g = np.asarray([potential_average], dtype=complex)
        timers.stop("v_of_rho", potential_started)
        eigenvalues, eigenvectors = [], []
        diagonalization_iterations: list[int] = []
        hamiltonian_applications: list[int] = []
        eigen_residuals: list[float] = []
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
                    # QE's usnldiag uses v_of_0: the G=0 component of the
                    # ionic local potential, not the average total effective
                    # potential used by H|psi>.
                    potential_average=v_ion_average,
                )
                trial_vectors = previous_eigenvectors[kpoint_index]
                if trial_vectors is None:
                    wfcinit_started = timers.start()
                    starting = _atomic_starting_orbitals(
                        pw,
                        pseudo_by_label,
                        basis,
                        nbnd,
                        random_stream=qe_random,
                    )
                    if mpi.size > 1:
                        starting = starting[
                            operator.local_rows
                        ]
                    trial_vectors = _rotate_starting_subspace(
                        operator,
                        starting,
                        nbnd,
                        mpi,
                    )
                    timers.stop("wfcinit", wfcinit_started)
                    del starting
                davidson_started = timers.start()
                solution = davidson(
                    operator.apply,
                    operator.diagonal,
                    nbnd,
                    initial_vectors=trial_vectors,
                    tolerance=davidson_tolerance,
                    max_iterations=davidson_maxiter,
                    subspace_multiplier=davidson_ndim,
                    residual_factor=davidson_residual_factor,
                    residual_energy_scale=davidson_residual_energy_scale,
                    # QE initializes every btype as occupied. Empty-band
                    # tolerances apply only after occupations are known from
                    # the first diagonalization.
                    occupied_roots=(
                        occupied
                        if occupations_mode == "fixed" and iteration > 1
                        else nbnd
                    ),
                    # Atomic trials have just been Rayleigh-Ritz rotated.
                    # File orbitals may accompany a different starting
                    # potential and must therefore be projected again.
                    initial_is_ritz=iteration == 1 and not loaded_wavefunctions,
                    mpi=mpi,
                    global_dimension=len(basis),
                    global_row_indices=operator.local_rows,
                    timers=timers,
                )
                timers.stop("cegterg", davidson_started)
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
                        f"Davidson diagonalization did not converge at k point "
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
                del operator, projector_terms, solution, trial_vectors
        timers.stop("c_bands", bands_started)
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
        else:
            band_occupations = []
            for values in eigenvalues:
                values_occupations = np.zeros(len(values))
                values_occupations[:occupied] = 2.0
                band_occupations.append(values_occupations)
            fermi_energy = None
            smearing_energy = 0.0
        sum_band_started = timers.start()
        rho_out = _density_from_states(
            eigenvectors,
            bases,
            weights,
            band_occupations,
            shape,
            pw.volume,
            nelec,
            numba_enabled,
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
        if mpi.size == 1:
            accuracy = 0.5 * _density_error_ry(
                rho,
                rho_out,
                pw.reciprocal,
                pw.volume,
                ecutrho,
                mpi,
                reciprocal_grid,
            )
        else:
            residual_g_local = charge_workspace.grid_to_coefficients(
                rho_out - rho
            )
            error_mask = local_charge_g2 > 1.0e-14
            accuracy = 0.5 * pw.volume * mpi.sum_scalar(
                float(
                    np.sum(
                        4.0
                        * np.pi
                        * np.abs(residual_g_local[error_mask]) ** 2
                        / local_charge_g2[error_mask]
                    )
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
        rho_energy = (
            rho_out
            if converged
            else density_mixer.mix(rho, rho_out)
        )
        timers.stop("mix_rho", mix_started)
        if mpi.size == 1:
            energy, energy_terms = _iteration_energy(
                band_energy,
                e_ion,
                rho_out,
                rho_energy,
                input_hxc,
                rho_core,
                pw.reciprocal,
                pw.volume,
                numba_enabled,
                mpi,
                xc_functional,
                charge_workspace,
                local_charge_vectors,
                smearing_energy,
                reciprocal_grid,
            )
        else:
            grid_scale = pw.volume / np.prod(shape)
            deband = -grid_scale * mpi.sum_scalar(
                float(np.sum(rho_out * input_hxc))
            )
            rho_energy_g_local = (
                charge_workspace.grid_to_coefficients(rho_energy)
            )
            vh_energy_g_local = np.zeros_like(rho_energy_g_local)
            energy_mask = local_charge_g2 > 1.0e-14
            vh_energy_g_local[energy_mask] = (
                4.0
                * np.pi
                * rho_energy_g_local[energy_mask]
                / local_charge_g2[energy_mask]
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
            epsilon_energy, vxc_energy = _xc_energy_potential(
                rho_energy + rho_core,
                xc_functional,
                use_numba=numba_enabled,
                workspace=charge_workspace,
                g_vectors=local_charge_vectors,
            )
            exc = grid_scale * mpi.sum_scalar(
                float(
                    np.sum(
                        (rho_energy + rho_core) * epsilon_energy
                    )
                )
            )
            vh_energy = np.real(
                charge_workspace.coefficients_to_grid(
                    vh_energy_g_local
                )
            )
            descf = -grid_scale * mpi.sum_scalar(
                float(
                    np.sum(
                        (rho_energy - rho_out)
                        * (vh_energy + vxc_energy)
                    )
                )
            )
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
            * int(np.ceil(charge_gvectors / mpi.size))
            * 16
            if isinstance(
                density_mixer,
                (PlainBroydenMixer, DistributedBroydenMixer),
            )
            else 0
        )
        iteration_davidson_bytes = (
            2
            * (
                max(sizes)
                if mpi.size == 1
                else local_plane_waves
            )
            * min(max(sizes), davidson_ndim * nbnd)
            * 16
        )
        if mpi.size == 1 and fft_backend == "pyfftw":
            iteration_fft_bytes = (
                (grid_points + max(sizes))
                * maximum_block
                * 16
            )
        else:
            iteration_fft_bytes = (
                2
                * (
                    int(np.ceil(grid_points / mpi.size))
                    + shape[2]
                    * (
                        len(wave_fft_descriptor.sticks_by_rank[mpi.rank])
                        if wave_fft_descriptor is not None
                        else wavefunction_sticks
                    )
                )
                * maximum_block
                * 16
            )
        step = SCFIteration(
            iteration,
            energy,
            accuracy,
            cutoff_ry=ecut,
            mixing_beta=mixing,
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
            memory_available_bytes=memory_available_report,
            memory_density_bytes_per_rank=iteration_density_bytes,
            memory_mixing_bytes_per_rank=iteration_mixing_bytes,
            memory_wavefunctions_bytes_per_rank=wavefunction_memory_report,
            memory_davidson_bytes_per_rank=iteration_davidson_bytes,
            memory_fft_bytes_per_rank=iteration_fft_bytes,
        )
        iterations.append(step)
        if progress is not None:
            progress("iteration", step)
        rho = rho_energy
        if converged:
            timers.stop("electrons", electrons_started)
            result_density = mpi.gather_z_slabs_root(rho, shape)
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
                _, final_vxc = _xc_energy_potential(
                    rho + rho_core,
                    xc_functional,
                    use_numba=numba_enabled,
                    workspace=charge_workspace,
                    g_vectors=local_charge_vectors,
                )
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
                    pw.lattice, positions, charges, ecutrho
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
                    numba_enabled,
                    mpi,
                    charges,
                    nonlocal_energy,
                    nonlocal_tensor,
                )
                timers.stop("stress", stress_started)
            if save_wavefunctions:
                final_wavefunctions, final_miller_indices = (
                    _collect_wavefunctions_root(
                        eigenvectors,
                        bases,
                        local_workspaces,
                        mpi,
                    )
                )
            else:
                final_wavefunctions, final_miller_indices = [], []
            peak_per_rank, peak_all_ranks = _peak_memory_across_ranks(mpi)
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
                timings=timers.snapshot(),
                forces_ha_per_bohr=forces_ha_per_bohr,
                stress_ha_per_bohr3=stress_ha_per_bohr3,
                occupations=[values.copy() for values in band_occupations],
                fermi_energy_ha=fermi_energy,
                mpi_processes=mpi.size,
                wavefunctions=final_wavefunctions,
                wavefunction_miller_indices=final_miller_indices,
            )
        old_energy = energy
        iteration += 1
    timers.stop("electrons", electrons_started)
    result_density = mpi.gather_z_slabs_root(rho, shape)
    if save_wavefunctions:
        final_wavefunctions, final_miller_indices = (
            _collect_wavefunctions_root(
                eigenvectors,
                bases,
                local_workspaces,
                mpi,
            )
        )
    else:
        final_wavefunctions, final_miller_indices = [], []
    peak_per_rank, peak_all_ranks = _peak_memory_across_ranks(mpi)
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
        timings=timers.snapshot(),
        occupations=[values.copy() for values in band_occupations],
        fermi_energy_ha=fermi_energy,
        mpi_processes=mpi.size,
        wavefunctions=final_wavefunctions,
        wavefunction_miller_indices=final_miller_indices,
    )
