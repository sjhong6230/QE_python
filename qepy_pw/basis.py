"""Reciprocal-space basis and FFT-grid utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
import ctypes
import glob
from math import prod
from pathlib import Path

import numpy as np

from .errors import QEInputError
from .fft_engine import PencilFFT3D, _partition_owner
from .mpi import MPIContext
from .threads import hybrid_thread_count
from .timing import TimingRegistry


_FFTW_ALIGNMENT = 64
_FFT_THREAD_WORK_THRESHOLD = 1 << 18
_BATCHED_FFT_THREAD_WORK_THRESHOLD = 1 << 15
_BLAS_LIBRARY_HANDLE: object | None = None
_NATIVE_FFT_MODULE: object | None = None


def _next_fast_len_real(target: int) -> int:
    """Return SciPy/FFTW's next 2-, 3-, 5-smooth real-FFT length."""
    candidate = max(1, int(target))
    while True:
        remainder = candidate
        for factor in (2, 3, 5):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return candidate
        candidate += 1


def _aligned_complex_empty(size: int, alignment: int) -> np.ndarray:
    """Allocate aligned complex storage without importing pyFFTW at runtime."""
    itemsize = np.dtype(np.complex128).itemsize
    raw = np.empty(size * itemsize + alignment - 1, dtype=np.uint8)
    offset = (-int(raw.ctypes.data)) % alignment
    return raw[offset : offset + size * itemsize].view(np.complex128)


def _load_native_fft() -> object:
    """Import the extension with its build-selected FFTW/MPI provider."""
    global _BLAS_LIBRARY_HANDLE, _NATIVE_FFT_MODULE
    if _NATIVE_FFT_MODULE is not None:
        return _NATIVE_FFT_MODULE
    if _BLAS_LIBRARY_HANDLE is None:
        # NumPy normally maps its private OpenBLAS with RTLD_LOCAL.  Promote
        # that already-mapped object to the global namespace so the native
        # extension can resolve ILP64 LAPACK/CBLAS symbols without linking a
        # second BLAS implementation (which would add code pages and another
        # worker pool to every MPI rank).
        numpy_library_dir = Path(np.__file__).resolve().parent.parent / "numpy.libs"
        blas_candidates = sorted(
            glob.glob(str(numpy_library_dir / "lib*openblas*.so*"))
        )
        if blas_candidates:
            _BLAS_LIBRARY_HANDLE = ctypes.CDLL(
                blas_candidates[-1], mode=ctypes.RTLD_GLOBAL
            )
    # setup.py records either the system FFTW/MPI stack or pyFFTW's bundled
    # fallback as DT_NEEDED dependencies. Let the dynamic loader select that
    # single provider; preloading pyFFTW here would put two FFTW copies into a
    # system-FFTW-MPI process and make plans/wisdom unsafe to exchange.
    from . import _native_fft

    _NATIVE_FFT_MODULE = _native_fft
    return _NATIVE_FFT_MODULE


def _compact_unsigned(values: np.ndarray, upper_bound: int) -> np.ndarray:
    """Store nonnegative table indices in the narrowest practical dtype."""
    dtype = np.uint16 if upper_bound <= np.iinfo(np.uint16).max + 1 else np.uint32
    return np.asarray(values, dtype=dtype)


@dataclass(frozen=True)
class PlaneWaveCatalog:
    """One QE-ordered global G catalog shared by every k-point basis."""

    indices: np.ndarray
    reciprocal: np.ndarray
    _vectors: np.ndarray | None = field(default=None, repr=False)
    shared: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self._vectors is None:
            object.__setattr__(
                self,
                "_vectors",
                np.asarray(self.indices @ self.reciprocal, dtype=float),
            )

    @property
    def vectors(self) -> np.ndarray:
        assert self._vectors is not None
        return self._vectors


@dataclass(frozen=True, init=False)
class PlaneWaveBasis:
    """Compact per-k mapping into a shared global reciprocal catalog."""

    catalog: PlaneWaveCatalog | None
    global_indices: np.ndarray | None
    k_crystal: np.ndarray | None
    _legacy_indices: np.ndarray | None
    _legacy_vectors: np.ndarray | None
    _legacy_kinetic: np.ndarray | None

    def __init__(
        self,
        catalog_or_indices: PlaneWaveCatalog | np.ndarray,
        mapping_or_vectors: np.ndarray,
        kpoint_or_kinetic: np.ndarray,
    ) -> None:
        # Retain the small public array constructor used by diagnostic tests
        # and downstream callers while make_bases uses the compact catalog.
        if isinstance(catalog_or_indices, PlaneWaveCatalog):
            object.__setattr__(self, "catalog", catalog_or_indices)
            mapping = np.asarray(mapping_or_vectors)
            if np.any(mapping < 0) or np.any(mapping >= len(catalog_or_indices.indices)):
                raise ValueError("global plane-wave mapping is out of range")
            object.__setattr__(
                self,
                "global_indices",
                _compact_unsigned(mapping, len(catalog_or_indices.indices)),
            )
            object.__setattr__(
                self, "k_crystal", np.asarray(kpoint_or_kinetic, dtype=float)
            )
            object.__setattr__(self, "_legacy_indices", None)
            object.__setattr__(self, "_legacy_vectors", None)
            object.__setattr__(self, "_legacy_kinetic", None)
        else:
            object.__setattr__(self, "catalog", None)
            object.__setattr__(self, "global_indices", None)
            object.__setattr__(self, "k_crystal", None)
            object.__setattr__(
                self,
                "_legacy_indices",
                np.asarray(catalog_or_indices, dtype=np.int32),
            )
            object.__setattr__(
                self,
                "_legacy_vectors",
                np.asarray(mapping_or_vectors, dtype=float),
            )
            object.__setattr__(
                self,
                "_legacy_kinetic",
                np.asarray(kpoint_or_kinetic, dtype=float),
            )

    @property
    def indices(self) -> np.ndarray:
        if self.catalog is None:
            assert self._legacy_indices is not None
            return self._legacy_indices
        assert self.global_indices is not None
        return self.catalog.indices[self.global_indices]

    @property
    def vectors(self) -> np.ndarray:
        if self.catalog is None:
            assert self._legacy_vectors is not None
            return self._legacy_vectors
        assert self.k_crystal is not None
        assert self.global_indices is not None
        return (
            self.catalog.vectors[self.global_indices]
            + self.k_crystal @ self.catalog.reciprocal
        )

    @property
    def kinetic(self) -> np.ndarray:
        if self.catalog is None:
            assert self._legacy_kinetic is not None
            return self._legacy_kinetic
        vectors = self.vectors
        return 0.5 * np.einsum("ij,ij->i", vectors, vectors)

    def __len__(self) -> int:
        return len(self.indices)

    def materialize(self) -> "PlaneWaveBasis":
        """Return current-k arrays without retaining them across k points."""
        if self.catalog is None:
            return self
        indices = self.indices
        vectors = self.vectors
        kinetic = 0.5 * np.einsum("ij,ij->i", vectors, vectors)
        return PlaneWaveBasis(indices, vectors, kinetic)


def _displacements(counts: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(counts[:-1]))
    )


@dataclass
class FFTTransposePlan:
    """Rank-local cached metadata for QE-style stick/slab transposes."""

    g2r_send_base: np.ndarray
    g2r_recv_base: np.ndarray
    slab_point_indices: np.ndarray
    native_slab_point_indices: np.ndarray
    _scaled: dict[
        tuple[int, bool], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = field(default_factory=dict, repr=False)
    _scaled_native: dict[
        tuple[int, bool], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = field(default_factory=dict, repr=False)

    def counts(
        self, number_of_vectors: int, *, reverse: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = (int(number_of_vectors), bool(reverse))
        cached = self._scaled.get(key)
        if cached is not None:
            return cached
        if reverse:
            send = self.g2r_recv_base * number_of_vectors
            recv = self.g2r_send_base * number_of_vectors
        else:
            send = self.g2r_send_base * number_of_vectors
            recv = self.g2r_recv_base * number_of_vectors
        cached = (send, _displacements(send), recv, _displacements(recv))
        self._scaled[key] = cached
        return cached

    def native_counts(
        self, number_of_vectors: int, *, reverse: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return C-MPI compatible int count/displacement arrays."""
        key = (int(number_of_vectors), bool(reverse))
        cached = self._scaled_native.get(key)
        if cached is None:
            cached = tuple(
                np.asarray(array, dtype=np.int32)
                for array in self.counts(
                    number_of_vectors, reverse=reverse
                )
            )
            self._scaled_native[key] = cached
        return cached


@dataclass(frozen=True)
class FFTGridDescriptor:
    """One shared FFT stick distribution for a grid and communicator size."""

    shape: tuple[int, int, int]
    processes: int
    sticks: np.ndarray
    stick_owners: np.ndarray
    sticks_by_rank: tuple[np.ndarray, ...]
    stick_lookup: np.ndarray
    row_stick_indices: np.ndarray
    row_z_slots: np.ndarray
    transpose_plans: tuple[FFTTransposePlan | None, ...]
    decomposition: str
    process_grid: tuple[int, int] | None

    @classmethod
    def build(
        cls,
        indices: np.ndarray,
        shape: tuple[int, int, int],
        processes: int,
        *,
        local_rank: int | None = None,
        decomposition: str = "slab",
        process_grid: tuple[int, int] | None = None,
    ) -> "FFTGridDescriptor":
        miller = np.asarray(indices, dtype=np.int32)
        shape = tuple(int(value) for value in shape)
        if miller.ndim != 2 or miller.shape[1] != 3:
            raise ValueError("indices must have shape (nplane, 3)")
        if processes < 1:
            raise ValueError("FFT descriptor needs at least one process")
        if local_rank is not None and not 0 <= local_rank < processes:
            raise ValueError("local FFT-descriptor rank is out of range")
        if decomposition not in {"slab", "pencil"}:
            raise ValueError("FFT decomposition must be slab or pencil")
        pairs = np.column_stack(
            (miller[:, 0] % shape[0], miller[:, 1] % shape[1])
        )
        sticks, inverse = np.unique(pairs, axis=0, return_inverse=True)
        sticks = np.asarray(sticks, dtype=np.int32)
        counts = np.bincount(inverse, minlength=len(sticks))
        owners = np.zeros(len(sticks), dtype=np.int32)
        if decomposition == "pencil":
            if process_grid is None or prod(process_grid) != processes:
                raise ValueError(
                    "pencil descriptor requires a matching process grid"
                )
            _px, py = process_grid
            for stick, (x_slot, y_slot) in enumerate(sticks):
                x_owner = _partition_owner(shape[0], process_grid[0], int(x_slot))
                y_owner = _partition_owner(shape[1], process_grid[1], int(y_slot))
                owners[stick] = x_owner * py + y_owner
        else:
            g_loads = np.zeros(processes, dtype=np.int64)
            stick_loads = np.zeros(processes, dtype=np.int64)
            # Match QE's primary G-count balance and stick-count tie breaker.
            for stick in np.argsort(-counts, kind="stable"):
                owner = min(
                    range(processes),
                    key=lambda rank: (g_loads[rank], stick_loads[rank]),
                )
                owners[stick] = owner
                g_loads[owner] += counts[stick]
                stick_loads[owner] += 1
        by_rank = tuple(
            np.asarray(np.flatnonzero(owners == rank), dtype=np.int32)
            for rank in range(processes)
        )
        lookup = np.full(shape[0] * shape[1], -1, dtype=np.int32)
        lookup[sticks[:, 0] * shape[1] + sticks[:, 1]] = np.arange(
            len(sticks), dtype=np.int32
        )
        z_counts = np.asarray(
            [
                shape[2] * (rank + 1) // processes
                - shape[2] * rank // processes
                for rank in range(processes)
            ],
            dtype=np.int64,
        )
        stick_counts = np.asarray(
            [len(value) for value in by_rank], dtype=np.int64
        )
        xy = sticks[:, 0].astype(np.int64) * shape[1] + sticks[:, 1]
        plans: list[FFTTransposePlan | None] = []
        for rank in range(processes):
            if decomposition == "pencil":
                plans.append(None)
                continue
            if local_rank is not None and rank != local_rank:
                plans.append(None)
                continue
            local_z = int(z_counts[rank])
            point_blocks = []
            native_point_blocks = []
            for peer_sticks in by_rank:
                peer_xy = xy[peer_sticks]
                point_blocks.append(
                    (
                        peer_xy[None, :] * local_z
                        + np.arange(local_z, dtype=np.int64)[:, None]
                    ).ravel()
                )
                # QE's local real-space layout keeps each XY plane
                # contiguous.  The generic NumPy grid remains (x,y,z), but
                # the fused native Hpsi/density kernels use (z,x,y) scratch
                # so FFTW does not traverse XY planes with stride local_z.
                native_point_blocks.append(
                    (
                        np.arange(local_z, dtype=np.int64)[:, None]
                        * (shape[0] * shape[1])
                        + peer_xy[None, :]
                    ).ravel()
                )
            points = (
                np.concatenate(point_blocks)
                if point_blocks
                else np.empty(0, dtype=np.int64)
            )
            native_points = (
                np.concatenate(native_point_blocks)
                if native_point_blocks
                else np.empty(0, dtype=np.int64)
            )
            plans.append(
                FFTTransposePlan(
                    g2r_send_base=z_counts * stick_counts[rank],
                    g2r_recv_base=local_z * stick_counts,
                    slab_point_indices=_compact_unsigned(
                        points, shape[0] * shape[1] * local_z
                    ),
                    native_slab_point_indices=_compact_unsigned(
                        native_points, shape[0] * shape[1] * local_z
                    ),
                )
            )
        return cls(
            shape,
            int(processes),
            sticks,
            owners,
            by_rank,
            lookup,
            _compact_unsigned(inverse, len(sticks)),
            _compact_unsigned(miller[:, 2] % shape[2], shape[2]),
            tuple(plans),
            decomposition,
            process_grid,
        )

    @property
    def nbytes(self) -> int:
        return int(
            self.sticks.nbytes
            + self.stick_owners.nbytes
            + self.stick_lookup.nbytes
            + self.row_stick_indices.nbytes
            + self.row_z_slots.nbytes
            + sum(array.nbytes for array in self.sticks_by_rank)
            + sum(
                plan.g2r_send_base.nbytes
                + plan.g2r_recv_base.nbytes
                + plan.slab_point_indices.nbytes
                + plan.native_slab_point_indices.nbytes
                for plan in self.transpose_plans
                if plan is not None
            )
        )


class FFTScratchPool:
    """Process-local reusable storage for sequential distributed FFT calls."""

    def __init__(self) -> None:
        self._complex: dict[str, np.ndarray] = {}
        self._complex_alignment: dict[str, int] = {}
        self._complex_pointers: set[int] = set()
        self._fftw_plans: dict[
            tuple[object, ...], tuple[object, ...]
        ] = {}

    def complex_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        zero: bool = False,
        alignment: int | None = None,
    ) -> np.ndarray:
        size = prod(shape)
        buffer = self._complex.get(name)
        requested_alignment = 0 if alignment is None else int(alignment)
        if (
            buffer is None
            or buffer.size < size
            or self._complex_alignment.get(name, 0) < requested_alignment
        ):
            if buffer is not None:
                old_pointer = int(buffer.ctypes.data)
                self._complex_pointers.discard(old_pointer)
                for key in list(self._fftw_plans):
                    if key and key[-1] == old_pointer:
                        del self._fftw_plans[key]
            if requested_alignment:
                buffer = _aligned_complex_empty(size, requested_alignment)
            else:
                buffer = np.empty(size, dtype=np.complex128)
            self._complex[name] = buffer
            self._complex_alignment[name] = requested_alignment
            self._complex_pointers.add(int(buffer.ctypes.data))
        view = buffer[:size].reshape(shape)
        if zero:
            view.fill(0.0)
        return view

    @property
    def nbytes(self) -> int:
        return sum(buffer.nbytes for buffer in self._complex.values())


class LocalPotentialWorkspace:
    """Reusable FFT buffers and optional plans for repeated local Hpsi calls."""

    def __init__(
        self,
        indices: np.ndarray | PlaneWaveBasis,
        shape: tuple[int, int, int],
        *,
        mpi: MPIContext | None = None,
        timers: TimingRegistry | None = None,
        scratch_pool: FFTScratchPool | None = None,
        descriptor: FFTGridDescriptor | None = None,
        pencil_fft: PencilFFT3D | None = None,
        serial_fft_batch_size: int | None = None,
        distributed_fft_batch_size: int | None = None,
        retain_serial_sticks: bool = True,
    ) -> None:
        compact_basis = (
            indices
            if isinstance(indices, PlaneWaveBasis)
            and indices.catalog is not None
            else None
        )
        if isinstance(indices, PlaneWaveBasis):
            miller = (
                None if compact_basis is not None else indices.indices
            )
            self.plane_waves = len(indices)
        else:
            miller = np.asarray(indices, dtype=np.int32)
            self.plane_waves = len(miller)
        if miller is not None and (
            miller.ndim != 2 or miller.shape[1] != 3
        ):
            raise ValueError("indices must have shape (nplane, 3)")
        self.shape = tuple(shape)
        self.size = int(np.prod(shape))
        slots = None
        self.mpi = mpi if mpi is not None else MPIContext()
        self.timers = timers
        self.scratch_pool = (
            scratch_pool if scratch_pool is not None else FFTScratchPool()
        )
        self.serial_fft_batch_size = (
            None
            if serial_fft_batch_size is None
            else int(serial_fft_batch_size)
        )
        if (
            self.serial_fft_batch_size is not None
            and self.serial_fft_batch_size < 1
        ):
            raise QEInputError("serial_fft_batch_size must be at least 1")
        self.distributed_fft_batch_size = (
            None
            if distributed_fft_batch_size is None
            else int(distributed_fft_batch_size)
        )
        if (
            self.distributed_fft_batch_size is not None
            and self.distributed_fft_batch_size < 1
        ):
            raise QEInputError(
                "distributed_fft_batch_size must be at least 1"
            )
        # QE stores complete Z-sticks indexed by (Gx,Gy), then transposes
        # them into real-space Z-plane slabs for the two-dimensional XY FFT.
        self.local_slab = self.mpi.slab(self.shape[2])
        self.pencil_fft = pencil_fft
        if self.mpi.size == 1:
            if miller is None:
                assert compact_basis is not None
                miller = compact_basis.indices
            slots = tuple(
                np.asarray(
                    miller[:, axis] % shape[axis], dtype=np.int32
                )
                for axis in range(3)
            )
            self.linear_slots = np.asarray(
                np.ravel_multi_index(slots, self.shape), dtype=np.int32
            )
            plane_size = self.shape[0] * self.shape[1]
            xy_slots = slots[0] * self.shape[1] + slots[1]
            self.z_major_slots = _compact_unsigned(
                slots[2] * plane_size + xy_slots, self.size
            )
            self.serial_stick_positions = _compact_unsigned(
                np.unique(xy_slots), plane_size
            )
            self.descriptor = None
            self.transpose_plan = None
            self.sticks = (
                np.unique(
                    np.column_stack((slots[0], slots[1])), axis=0
                ).astype(np.int32, copy=False)
                if retain_serial_sticks
                else np.empty((0, 2), dtype=np.int32)
            )
            # Serial transforms need only compact linear FFT slots. Avoid
            # retaining the distributed stick ownership/maps for every k.
            self.stick_indices = np.empty(0, dtype=np.int32)
            self.stick_owners = np.empty(0, dtype=np.int32)
            self.sticks_by_rank = [
                np.arange(len(self.sticks), dtype=np.int32)
            ]
            self.owned_plane_waves = np.empty(0, dtype=bool)
            self.local_plane_wave_indices = np.arange(
                self.plane_waves, dtype=np.int32
            )
            self.local_slots = tuple(
                np.empty(0, dtype=np.int32) for _ in range(3)
            )
            self.local_stick_positions = np.empty(0, dtype=np.int32)
            self.local_pencil_slots = tuple(
                np.empty(0, dtype=np.int32) for _ in range(3)
            )
        else:
            if descriptor is None:
                if miller is None:
                    assert compact_basis is not None
                    miller = compact_basis.indices
                descriptor = FFTGridDescriptor.build(
                    miller, self.shape, self.mpi.size
                )
            if (
                descriptor.shape != self.shape
                or descriptor.processes != self.mpi.size
            ):
                raise ValueError(
                    "FFT descriptor does not match the grid/communicator"
                )
            self.descriptor = descriptor
            self.transpose_plan = descriptor.transpose_plans[self.mpi.rank]
            if descriptor.decomposition == "pencil" and pencil_fft is None:
                raise ValueError("pencil descriptor requires a pencil FFT executor")
            if descriptor.decomposition == "slab" and self.transpose_plan is None:
                raise ValueError(
                    "FFT descriptor does not contain this rank's transpose plan"
                )
            if pencil_fft is not None and (
                descriptor.decomposition != "pencil"
                or pencil_fft.shape != self.shape
                or pencil_fft.topology.fft_size != self.mpi.size
                or pencil_fft.topology.fft_rank != self.mpi.rank
            ):
                raise ValueError("pencil FFT executor does not match the descriptor")
            self.sticks = descriptor.sticks
            self.stick_owners = descriptor.stick_owners
            self.sticks_by_rank = descriptor.sticks_by_rank
            if compact_basis is not None:
                assert compact_basis.global_indices is not None
                catalog_rows = compact_basis.global_indices
                self.stick_indices = descriptor.row_stick_indices[
                    catalog_rows
                ]
                z_slots = descriptor.row_z_slots[catalog_rows]
            else:
                if miller is None:
                    raise ValueError("distributed FFT indices are missing")
                if len(miller) == len(descriptor.row_stick_indices):
                    self.stick_indices = descriptor.row_stick_indices
                    z_slots = descriptor.row_z_slots
                else:
                    slots = tuple(
                        np.asarray(
                            miller[:, axis] % shape[axis],
                            dtype=np.int32,
                        )
                        for axis in range(3)
                    )
                    pair_slots = slots[0] * self.shape[1] + slots[1]
                    self.stick_indices = descriptor.stick_lookup[pair_slots]
                    z_slots = slots[2]
            if np.any(self.stick_indices < 0):
                raise ValueError(
                    "FFT descriptor does not cover the plane-wave basis"
                )
            self.owned_plane_waves = (
                self.stick_owners[self.stick_indices] == self.mpi.rank
            )
            self.local_plane_wave_indices = np.asarray(
                _compact_unsigned(
                    np.flatnonzero(self.owned_plane_waves), self.plane_waves
                )
            )
            empty_slots = np.empty(0, dtype=np.int32)
            self.local_slots = (
                empty_slots,
                empty_slots,
                _compact_unsigned(
                    z_slots[self.owned_plane_waves], self.shape[2]
                ),
            )
            self.linear_slots = np.empty(0, dtype=np.int32)
            self.z_major_slots = np.empty(0, dtype=np.int32)
            self.serial_stick_positions = np.empty(0, dtype=np.int32)
            local_stick_lookup = np.full(
                len(self.sticks), -1, dtype=np.int32
            )
            local_stick_lookup[
                self.sticks_by_rank[self.mpi.rank]
            ] = np.arange(len(self.sticks_by_rank[self.mpi.rank]))
            self.local_stick_positions = _compact_unsigned(
                local_stick_lookup[
                    self.stick_indices[self.owned_plane_waves]
                ],
                len(self.sticks_by_rank[self.mpi.rank]),
            )
            if pencil_fft is None:
                self.local_pencil_slots = tuple(
                    np.empty(0, dtype=np.int32) for _ in range(3)
                )
            else:
                owned_sticks = self.stick_indices[self.owned_plane_waves]
                owned_pairs = self.sticks[owned_sticks]
                self.local_pencil_slots = (
                    _compact_unsigned(
                        owned_pairs[:, 0] - pencil_fft.x_initial.start,
                        pencil_fft.local_z_shape[1],
                    ),
                    _compact_unsigned(
                        owned_pairs[:, 1] - pencil_fft.y_initial.start,
                        pencil_fft.local_z_shape[2],
                    ),
                    self.local_slots[2],
                )
        self._native_fft = _load_native_fft()
        self._fft_alignment = _FFTW_ALIGNMENT
        self.thread_count = hybrid_thread_count()
        self._native_fft.configure_openmp_threads(self.thread_count)
        # Preserve the validated execution policy: serial plans are reused
        # enough to amortize MEASURE, while MPI's changing Davidson block
        # shapes use bounded-cost ESTIMATE plans.  A 200 Ry end-to-end check
        # found all-ESTIMATE reduced VmHWM by only a few MiB but increased FFT
        # and total wall time materially.
        self.planner_flag = (
            "FFTW_MEASURE" if self.mpi.size == 1 else "FFTW_ESTIMATE"
        )

    def _scratch(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        zero: bool = False,
    ) -> np.ndarray:
        return self.scratch_pool.complex_buffer(
            name,
            shape,
            zero=zero,
            alignment=(self._fft_alignment or None),
        )

    def _parallel_band_fft(self, number_of_vectors: int) -> bool:
        """Whether independent band FFTs can occupy the rank-local team."""
        return (
            self.thread_count > 1
            and number_of_vectors > 1
        )

    def _serial_grid_stride(self, number_of_vectors: int) -> int:
        """Return a 64-byte-aligned band stride when bands run concurrently."""
        if not self._parallel_band_fft(number_of_vectors):
            return self.size
        complex_per_alignment = (
            _FFTW_ALIGNMENT // np.dtype(np.complex128).itemsize
        )
        return (
            (self.size + complex_per_alignment - 1)
            // complex_per_alignment
            * complex_per_alignment
        )

    def _native_plan(
        self,
        values: np.ndarray,
        lengths: tuple[int, ...],
        howmany: int,
        stride: int,
        distance: int,
        *,
        parallel_bands: bool = False,
        planner_flag: str | None = None,
        preserve_values: bool = True,
    ) -> object:
        selected_planner_flag = planner_flag or self.planner_flag
        pointer = int(values.ctypes.data)
        thread_threshold = (
            _BATCHED_FFT_THREAD_WORK_THRESHOLD
            if int(howmany) > 1
            else _FFT_THREAD_WORK_THRESHOLD
        )
        plan_threads = 1 if parallel_bands else (
            self.thread_count
            if prod(lengths) * int(howmany) >= thread_threshold
            else 1
        )
        cache_plan = pointer in self.scratch_pool._complex_pointers
        key = (
            "native",
            values.shape,
            lengths,
            int(howmany),
            int(stride),
            int(distance),
            selected_planner_flag,
            plan_threads,
            pointer,
        )
        cached = (
            self.scratch_pool._fftw_plans.get(key)
            if cache_plan
            else None
        )
        if cached is None:
            flag = {
                "FFTW_ESTIMATE": 64,
                "FFTW_MEASURE": 0,
                "FFTW_PATIENT": 32,
            }[selected_planner_flag]
            preserved = (
                values.copy()
                if flag != 64 and preserve_values
                else None
            )
            plan = self._native_fft.NativeFFTWPlan(
                values,
                lengths,
                howmany,
                stride,
                distance,
                flag,
                plan_threads,
            )
            if preserved is not None:
                values[...] = preserved
            cached = (plan,)
            if cache_plan:
                self.scratch_pool._fftw_plans[key] = cached
        return cached[0]

    def _serial_spatial_plans(self, grid: np.ndarray) -> tuple[object, object]:
        """Single-transform plans used by the shared-memory stick path."""
        plane_size = self.shape[0] * self.shape[1]
        z_plan = self._native_plan(
            grid,
            (self.shape[2],),
            1,
            plane_size,
            1,
            parallel_bands=True,
            preserve_values=False,
        )
        xy_plan = self._native_plan(
            grid,
            (self.shape[0], self.shape[1]),
            1,
            1,
            plane_size,
            parallel_bands=True,
            preserve_values=False,
        )
        return z_plan, xy_plan

    def prepare_potential(self, potential_g: np.ndarray) -> np.ndarray:
        """Transform a fixed local potential once per Hamiltonian."""
        # The one-shot FFT is destructive. Always copy even when the caller's
        # reciprocal array is already contiguous complex128; SCF reuses that
        # array for energies and subsequent Hamiltonians.
        values = np.array(
            potential_g, dtype=np.complex128, order="C", copy=True
        )
        plan_threads = (
            self.thread_count
            if values.size >= _FFT_THREAD_WORK_THRESHOLD
            else 1
        )
        return self._native_fft.inverse_real(
            values, self.shape, 64, plan_threads
        )

    def _backward_fft(
        self, values: np.ndarray, axes: tuple[int, ...]
    ) -> np.ndarray:
        lengths = tuple(values.shape[axis] for axis in axes)
        trailing = prod(values.shape[len(axes) :])
        if axes != tuple(range(len(axes))) or not values.flags.c_contiguous:
            raise ValueError("native local FFT requires leading contiguous axes")
        plan = self._native_plan(
            values, lengths, trailing, trailing, 1
        )
        plan.execute("backward")
        return values

    def _forward_fft(
        self, values: np.ndarray, axes: tuple[int, ...]
    ) -> np.ndarray:
        lengths = tuple(values.shape[axis] for axis in axes)
        trailing = prod(values.shape[len(axes) :])
        if axes != tuple(range(len(axes))) or not values.flags.c_contiguous:
            raise ValueError("native local FFT requires leading contiguous axes")
        plan = self._native_plan(
            values, lengths, trailing, trailing, 1
        )
        plan.execute("forward")
        return values

    def _coefficients_to_z_pencil(self, vectors: np.ndarray) -> np.ndarray:
        if self.pencil_fft is None:
            raise RuntimeError("pencil FFT executor is unavailable")
        result = np.zeros(
            self.pencil_fft.local_z_shape + (vectors.shape[1],),
            dtype=np.complex128,
        )
        x_slots, y_slots, z_slots = self.local_pencil_slots
        result[z_slots, x_slots, y_slots, :] = vectors
        return result

    def _z_pencil_to_coefficients(self, values: np.ndarray) -> np.ndarray:
        x_slots, y_slots, z_slots = self.local_pencil_slots
        return np.asfortranarray(values[z_slots, x_slots, y_slots, :])

    def slab_to_x_pencil(self, values: np.ndarray) -> np.ndarray:
        """Convert a public SCF Z-slab field to the local X pencil."""
        if self.pencil_fft is None:
            return np.asarray(values)
        return self.pencil_fft.slab_to_x_pencil(values)

    def x_pencil_to_slab(self, values: np.ndarray) -> np.ndarray:
        """Convert a local X-pencil field to the public SCF Z slab."""
        if self.pencil_fft is None:
            return np.asarray(values)
        return self.pencil_fft.x_pencil_to_slab(values)

    def coefficients_to_grid(
        self,
        coefficients: np.ndarray,
        *,
        use_scratch: bool = False,
    ) -> np.ndarray:
        """Inverse FFT using sparse reciprocal sticks and real-space slabs."""
        vectors = np.asarray(coefficients)
        was_vector = vectors.ndim == 1
        if was_vector:
            vectors = vectors[:, None]
        expected_rows = (
            len(self.local_plane_wave_indices)
            if self.mpi.size > 1
            else self.plane_waves
        )
        if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
            raise ValueError(
                "coefficients must contain the rank-local G-vector rows"
            )
        number_of_vectors = vectors.shape[1]
        if self.mpi.size == 1:
            grid_stride = self._serial_grid_stride(number_of_vectors)
            real_grid = self._scratch(
                "serial_fft_grid",
                (number_of_vectors, grid_stride),
            )
            parallel_bands = self._parallel_band_fft(number_of_vectors)
            plan = self._native_plan(
                real_grid,
                self.shape,
                1 if parallel_bands else number_of_vectors,
                1,
                grid_stride,
                parallel_bands=parallel_bands,
            )
            timer = (
                self.timers.measure("fftw")
                if self.timers is not None
                else nullcontext()
            )
            with timer:
                self._native_fft.inverse_serial(
                    real_grid,
                    vectors,
                    self.linear_slots,
                    plan,
                    self.size,
                    grid_stride,
                )
            result = np.moveaxis(
                real_grid[:, : self.size].reshape(
                    (number_of_vectors,) + self.shape
                ),
                0,
                -1,
            )
            if not use_scratch:
                result = result.copy()
            return result[..., 0] if was_vector else result
        if self.pencil_fft is not None:
            timer = (
                self.timers.measure("fftw")
                if self.timers is not None
                else nullcontext()
            )
            with timer:
                z_pencil = self._coefficients_to_z_pencil(vectors)
                x_pencil = self.pencil_fft.backward(z_pencil)
                real_slab = self.pencil_fft.x_pencil_to_slab(x_pencil)
            if not use_scratch:
                real_slab = real_slab.copy()
            return real_slab[..., 0] if was_vector else real_slab
        local_sticks = self.sticks_by_rank[self.mpi.rank]
        stick_shape = (
            self.shape[2],
            len(local_sticks),
            number_of_vectors,
        )
        data = self._scratch(
            "wave_sticks",
            stick_shape,
            zero=True,
        )
        data[
            self.local_slots[2],
            self.local_stick_positions,
        ] = vectors
        timer = (
            self.timers.measure("fftw")
            if self.timers is not None
            else nullcontext()
        )
        with timer:
            data = self._backward_fft(data, (0,))
            real_slab = self._sticks_to_real_slabs(
                data,
                out=self._scratch(
                    "real_slabs",
                    (
                        self.shape[0],
                        self.shape[1],
                        self.local_slab.stop
                        - self.local_slab.start,
                        number_of_vectors,
                    ),
                    zero=True,
                ),
            )
            real_slab = self._backward_fft(real_slab, (0, 1))
        if not use_scratch:
            real_slab = real_slab.copy()
        return real_slab[..., 0] if was_vector else real_slab

    def _sticks_to_real_slabs(
        self,
        stick_data: np.ndarray,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Transpose complete reciprocal Z-sticks to real-space Z slabs."""
        number_of_vectors = stick_data.shape[-1]
        assert self.transpose_plan is not None
        send_counts, send_displacements, recv_counts, recv_displacements = (
            self.transpose_plan.counts(number_of_vectors)
        )
        send = np.ascontiguousarray(stick_data).reshape(-1)
        received = self.mpi.exchange_complex_planned(
            send,
            send_counts,
            send_displacements,
            recv_counts,
            recv_displacements,
        )
        local_z = self.local_slab.stop - self.local_slab.start
        expected_shape = (
            self.shape[0],
            self.shape[1],
            local_z,
            number_of_vectors,
        )
        if out is None:
            result = np.zeros(expected_shape, dtype=complex)
        else:
            if out.shape != expected_shape:
                raise ValueError("real-slab scratch buffer has wrong shape")
            result = out
            result.fill(0.0)
        source_rows = received.reshape(-1, number_of_vectors)
        destination_rows = result.reshape(-1, number_of_vectors)
        destination_rows[
            self.transpose_plan.slab_point_indices
        ] = source_rows
        return result

    def _real_slabs_to_sticks(
        self,
        real_data: np.ndarray,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Transpose real-space Z slabs to complete reciprocal Z-sticks."""
        number_of_vectors = real_data.shape[-1]
        assert self.transpose_plan is not None
        send_counts, send_displacements, recv_counts, recv_displacements = (
            self.transpose_plan.counts(number_of_vectors, reverse=True)
        )
        local_sticks = self.sticks_by_rank[self.mpi.rank]
        expected_shape = (
            self.shape[2],
            len(local_sticks),
            number_of_vectors,
        )
        if out is None:
            result = self._scratch("wave_sticks", expected_shape)
        else:
            if out.shape != expected_shape:
                raise ValueError("stick scratch buffer has wrong shape")
            result = out
        send_size = (
            int(send_displacements[-1] + send_counts[-1])
            if send_counts.size
            else 0
        )
        send, _unused_recv = self.mpi.complex_exchange_buffers(
            send_size, 0
        )
        source_rows = np.asarray(real_data).reshape(-1, number_of_vectors)
        send_rows = send.reshape(-1, number_of_vectors)
        np.take(
            source_rows,
            self.transpose_plan.slab_point_indices,
            axis=0,
            out=send_rows,
        )
        self.mpi.exchange_complex_planned(
            send,
            send_counts,
            send_displacements,
            recv_counts,
            recv_displacements,
            recv_buffer=result,
        )
        return result

    def grid_to_coefficients(self, real_slab: np.ndarray) -> np.ndarray:
        """Forward distributed FFT from this rank's real slab to owned G rows."""
        values = np.asarray(real_slab)
        expected = (
            self.shape[0],
            self.shape[1],
            self.local_slab.stop - self.local_slab.start,
        )
        was_scalar = values.ndim == 3
        if was_scalar:
            values = values[..., None]
        if values.shape[:3] != expected:
            raise ValueError("real data does not match this rank's FFT slab")
        if self.mpi.size == 1:
            number_of_vectors = values.shape[-1]
            grid_stride = self._serial_grid_stride(number_of_vectors)
            transformed = self._scratch(
                "serial_fft_grid",
                (number_of_vectors, grid_stride),
            )
            transformed[:, : self.size].reshape(
                (number_of_vectors,) + self.shape
            )[...] = np.moveaxis(values, -1, 0)
            parallel_bands = self._parallel_band_fft(number_of_vectors)
            plan = self._native_plan(
                transformed,
                self.shape,
                1 if parallel_bands else number_of_vectors,
                1,
                grid_stride,
                parallel_bands=parallel_bands,
            )
            result = self._native_fft.forward_serial(
                transformed,
                self.linear_slots,
                plan,
                self.size,
                grid_stride,
            )
            return result[:, 0] if was_scalar else result
        if self.pencil_fft is not None:
            timer = (
                self.timers.measure("fftw")
                if self.timers is not None
                else nullcontext()
            )
            with timer:
                x_pencil = self.pencil_fft.slab_to_x_pencil(values)
                z_pencil = self.pencil_fft.forward(x_pencil)
                result = self._z_pencil_to_coefficients(z_pencil)
            return result[:, 0] if was_scalar else result
        if not (
            values.dtype == np.complex128
            and values.flags.c_contiguous
            and int(values.ctypes.data)
            in self.scratch_pool._complex_pointers
        ):
            transformed = self._scratch(
                "real_slabs",
                values.shape,
            )
            transformed[...] = values
        else:
            transformed = values
        transformed = self._forward_fft(transformed, (0, 1))
        transformed = self._real_slabs_to_sticks(transformed)
        transformed = self._forward_fft(transformed, (0,)) / self.size
        result = transformed[
            self.local_slots[2], self.local_stick_positions
        ]
        return result[:, 0] if was_scalar else result

    def _distributed_apply(
        self,
        real_potential: np.ndarray,
        vectors: np.ndarray,
        *,
        native_potential_layout: bool = False,
        out: np.ndarray | None = None,
        diagonal: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.pencil_fft is not None:
            return self._distributed_apply_pencil(
                real_potential,
                vectors,
                native_potential_layout=native_potential_layout,
                out=out,
                diagonal=diagonal,
            )
        return self._distributed_apply_native(
            real_potential,
            vectors,
            native_potential_layout=native_potential_layout,
            out=out,
            diagonal=diagonal,
        )

    def _distributed_apply_pencil(
        self,
        real_potential: np.ndarray,
        vectors: np.ndarray,
        *,
        native_potential_layout: bool = False,
        out: np.ndarray | None = None,
        diagonal: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply Vloc using the production sparse-to-dense pencil boundary."""
        assert self.pencil_fft is not None
        result = (
            np.empty(vectors.shape, dtype=np.complex128)
            if out is None
            else np.asarray(out)
        )
        if result.shape != vectors.shape or result.dtype != np.complex128:
            raise ValueError("local-potential output has the wrong shape or dtype")
        potential_values = np.asarray(real_potential, dtype=np.float64)
        if potential_values.shape == self.pencil_fft.local_x_shape:
            potential_x = potential_values
        else:
            local_z = self.local_slab.stop - self.local_slab.start
            if native_potential_layout:
                expected = (local_z, self.shape[0], self.shape[1])
                if potential_values.shape != expected:
                    raise ValueError("native local potential has the wrong shape")
                potential_slab = np.moveaxis(potential_values, 0, 2)
            else:
                expected = (self.shape[0], self.shape[1], local_z)
                if potential_values.shape != expected:
                    raise ValueError("local potential has the wrong slab shape")
                potential_slab = potential_values
            potential_x = self.pencil_fft.slab_to_x_pencil(potential_slab)
        timer = (
            self.timers.measure("fftw")
            if self.timers is not None
            else nullcontext()
        )
        with timer:
            z_pencil = self._coefficients_to_z_pencil(vectors)
            real_x = self.pencil_fft.backward(z_pencil)
            real_x *= potential_x[..., None]
            transformed = self.pencil_fft.forward(real_x)
            result[...] = self._z_pencil_to_coefficients(transformed)
            if diagonal is not None:
                result += np.asarray(diagonal)[:, None] * vectors
        return result

    def _distributed_apply_native(
        self,
        real_potential: np.ndarray,
        vectors: np.ndarray,
        *,
        native_potential_layout: bool = False,
        out: np.ndarray | None = None,
        diagonal: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run the fused Cython/MPI/FFTW local-potential kernel."""
        assert self._native_fft is not None
        assert self.transpose_plan is not None
        number_of_vectors = vectors.shape[1]
        result = (
            np.empty(vectors.shape, dtype=np.complex128)
            if out is None
            else np.asarray(out)
        )
        if result.shape != vectors.shape or result.dtype != np.complex128:
            raise ValueError("local-potential output has the wrong shape or dtype")
        local_sticks = self.sticks_by_rank[self.mpi.rank]
        local_z = self.local_slab.stop - self.local_slab.start
        sticks = self._scratch(
            "wave_sticks",
            (self.shape[2], len(local_sticks), number_of_vectors),
        )
        slab = self._scratch(
            "real_slabs", (local_z, self.shape[0], self.shape[1], 1)
        )
        native_potential_shape = (
            local_z,
            self.shape[0],
            self.shape[1],
        )
        if native_potential_layout:
            if real_potential.shape != native_potential_shape:
                raise ValueError("native local potential has the wrong shape")
            potential = np.ascontiguousarray(
                real_potential, dtype=np.float64
            )
        else:
            potential_view = (
                real_potential
                if real_potential.shape[2] == local_z
                else real_potential[:, :, self.local_slab]
            )
            potential = np.ascontiguousarray(
                np.moveaxis(potential_view, 2, 0), dtype=np.float64
            )
        forward = self.transpose_plan.native_counts(number_of_vectors)
        reverse = self.transpose_plan.native_counts(
            number_of_vectors, reverse=True
        )
        inverse_receive_size = int(
            forward[3][-1] + forward[2][-1]
        )
        reverse_send_size = int(reverse[1][-1] + reverse[0][-1])
        if reverse_send_size == inverse_receive_size:
            # The forward receive payload has been consumed by the time the
            # reverse transpose is packed.  Reuse it in place, cutting one
            # grow-only MPI buffer without changing either collective.
            _unused, inverse_receive = self.mpi.complex_exchange_buffers(
                0, inverse_receive_size
            )
            reverse_send = inverse_receive
        else:
            reverse_send, inverse_receive = self.mpi.complex_exchange_buffers(
                reverse_send_size, inverse_receive_size
            )
        threaded_sticks = self.thread_count > 1
        z_plan = self._native_plan(
            sticks,
            (self.shape[2],),
            1 if threaded_sticks else len(local_sticks) * number_of_vectors,
            len(local_sticks) * number_of_vectors,
            1,
            parallel_bands=threaded_sticks,
            preserve_values=False,
        )
        threaded_planes = self.thread_count > 1
        xy_plan = self._native_plan(
            slab,
            (self.shape[0], self.shape[1]),
            1 if threaded_planes else local_z,
            1,
            self.shape[0] * self.shape[1],
            parallel_bands=threaded_planes,
            planner_flag=(
                "FFTW_MEASURE"
                if self.thread_count > 1
                and self.shape[0] * self.shape[1] * local_z
                >= _BATCHED_FFT_THREAD_WORK_THRESHOLD
                else None
            ),
            preserve_values=False,
        )
        timer = (
            self.timers.measure("fftw")
            if self.timers is not None
            else nullcontext()
        )
        with timer:
            return self._native_fft.apply_streamed(
                sticks,
                slab,
                reverse_send,
                inverse_receive,
                vectors,
                potential,
                self.local_slots[2],
                self.local_stick_positions,
                self.transpose_plan.native_slab_point_indices,
                *forward,
                *reverse,
                self.mpi.comm,
                z_plan,
                xy_plan,
                result,
                self.size,
                diagonal,
            )

    def accumulate_density(
        self,
        density: np.ndarray,
        coefficients: np.ndarray,
        band_weights: np.ndarray,
    ) -> None:
        """Accumulate occupied-state density in the native FFT kernel."""
        vectors = np.asarray(coefficients)
        weights = np.asarray(band_weights, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != len(weights):
            raise ValueError("density coefficients and band weights disagree")
        expected_density_shape = (
            self.shape[0],
            self.shape[1],
            self.local_slab.stop - self.local_slab.start,
        )
        if density.shape != expected_density_shape:
            raise ValueError("density accumulator has the wrong local shape")
        number_of_vectors = vectors.shape[1]
        timer = (
            self.timers.measure("fftw")
            if self.timers is not None
            else nullcontext()
        )
        if self.mpi.size == 1:
            batch_size = min(
                number_of_vectors,
                self.serial_fft_batch_size or number_of_vectors,
            )
            for start in range(0, number_of_vectors, batch_size):
                stop = min(start + batch_size, number_of_vectors)
                active = stop - start
                grid_stride = self._serial_grid_stride(active)
                grid = self._scratch(
                    "serial_fft_grid", (active, grid_stride)
                )
                if self.thread_count > 1 and active < self.thread_count:
                    z_plan, xy_plan = self._serial_spatial_plans(grid)
                    with timer:
                        self._native_fft.accumulate_density_serial_spatial(
                            density,
                            grid,
                            vectors[:, start:stop],
                            weights[start:stop],
                            self.z_major_slots,
                            self.serial_stick_positions,
                            z_plan,
                            xy_plan,
                            self.size,
                            grid_stride,
                        )
                else:
                    parallel_bands = self._parallel_band_fft(active)
                    plan = self._native_plan(
                        grid,
                        self.shape,
                        1 if parallel_bands else active,
                        1,
                        grid_stride,
                        parallel_bands=parallel_bands,
                    )
                    with timer:
                        self._native_fft.accumulate_density_serial(
                            density,
                            grid,
                            vectors[:, start:stop],
                            weights[start:stop],
                            self.linear_slots,
                            plan,
                            self.size,
                            grid_stride,
                        )
            return
        if self.pencil_fft is not None:
            batch_size = min(
                number_of_vectors,
                self.distributed_fft_batch_size or number_of_vectors,
            )
            for start in range(0, number_of_vectors, batch_size):
                stop = min(start + batch_size, number_of_vectors)
                with timer:
                    z_pencil = self._coefficients_to_z_pencil(
                        vectors[:, start:stop]
                    )
                    real_x = self.pencil_fft.backward(z_pencil)
                    partial_x = np.sum(
                        np.abs(real_x) ** 2
                        * weights[None, None, None, start:stop],
                        axis=-1,
                    )
                    density += np.real(
                        self.pencil_fft.x_pencil_to_slab(partial_x)
                    )
            return
        if self.mpi.size > 1:
            assert self.transpose_plan is not None
            local_sticks = self.sticks_by_rank[self.mpi.rank]
            local_z = self.local_slab.stop - self.local_slab.start
            batch_size = min(
                number_of_vectors,
                self.distributed_fft_batch_size or number_of_vectors,
            )
            for start in range(0, number_of_vectors, batch_size):
                stop = min(start + batch_size, number_of_vectors)
                active = stop - start
                sticks = self._scratch(
                    "wave_sticks",
                    (self.shape[2], len(local_sticks), active),
                    zero=True,
                )
                slab = self._scratch(
                    "real_slabs",
                    (
                        local_z,
                        self.shape[0],
                        self.shape[1],
                        1,
                    ),
                )
                counts = self.transpose_plan.native_counts(active)
                receive_size = int(counts[3][-1] + counts[2][-1])
                _send, receive = self.mpi.complex_exchange_buffers(
                    0, receive_size
                )
                threaded_sticks = self.thread_count > 1
                z_plan = self._native_plan(
                    sticks,
                    (self.shape[2],),
                    1 if threaded_sticks else len(local_sticks) * active,
                    len(local_sticks) * active,
                    1,
                    parallel_bands=threaded_sticks,
                    preserve_values=False,
                )
                xy_plan = self._native_plan(
                    slab,
                    (self.shape[0], self.shape[1]),
                    local_z,
                    1,
                    self.shape[0] * self.shape[1],
                    planner_flag=(
                        "FFTW_MEASURE"
                        if self.thread_count > 1
                        and self.shape[0] * self.shape[1] * local_z
                        >= _BATCHED_FFT_THREAD_WORK_THRESHOLD
                        else None
                    ),
                    preserve_values=False,
                )
                with timer:
                    self._native_fft.accumulate_density_distributed(
                        density,
                        sticks,
                        slab,
                        receive,
                        vectors[:, start:stop],
                        weights[start:stop],
                        self.local_slots[2],
                        self.local_stick_positions,
                        self.transpose_plan.native_slab_point_indices,
                        *counts,
                        self.mpi.comm,
                        z_plan,
                        xy_plan,
                    )
            return

    def apply(
        self,
        real_potential: np.ndarray,
        coefficients: np.ndarray,
        *,
        native_potential_layout: bool = False,
        out: np.ndarray | None = None,
        diagonal: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply a prepared real-space potential using reusable FFT storage."""
        vectors = np.asarray(coefficients)
        was_vector = vectors.ndim == 1
        if was_vector:
            vectors = vectors[:, None]
        expected_rows = (
            len(self.local_plane_wave_indices)
            if self.mpi.size > 1
            else self.plane_waves
        )
        if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
            raise ValueError(
                "coefficients must have rank-local shape "
                "(nplane_local,) or (nplane_local, nvec)"
            )
        number_of_vectors = vectors.shape[1]
        if self.mpi.size > 1:
            batch_size = min(
                number_of_vectors,
                self.distributed_fft_batch_size or number_of_vectors,
            )
            result = (
                np.empty(vectors.shape, dtype=np.complex128)
                if out is None
                else np.asarray(out)
            )
            if result.shape != vectors.shape or result.dtype != np.complex128:
                raise ValueError(
                    "local-potential output has the wrong shape or dtype"
                )
            # Bound all band-dependent stick and collective buffers by the
            # tile size.  This is the distributed analogue of QE's small
            # many_fft blocks and keeps peak FFT storage O(tile), not
            # O(total bands).
            for start in range(0, number_of_vectors, batch_size):
                stop = min(start + batch_size, number_of_vectors)
                self._distributed_apply(
                    real_potential,
                    vectors[:, start:stop],
                    native_potential_layout=native_potential_layout,
                    out=result[:, start:stop],
                    diagonal=diagonal,
                )
            return result[:, 0] if was_vector else result
        # QE's serial vloc_psi transforms a small number of states at a time.
        # Keep one native result owner and one timing scope around the bounded
        # batches instead of recursively re-entering this Python method for
        # every batch. The grow-only full-grid scratch remains O(batch size).
        batch_size = min(
            number_of_vectors,
            self.serial_fft_batch_size or number_of_vectors,
        )
        potential = np.ascontiguousarray(real_potential, dtype=np.float64)
        result = (
            np.empty(vectors.shape, dtype=np.complex128)
            if out is None
            else np.asarray(out)
        )
        if result.shape != vectors.shape or result.dtype != np.complex128:
            raise ValueError("local-potential output has the wrong shape or dtype")
        timer = (
            self.timers.measure("fftw")
            if self.timers is not None
            else nullcontext()
        )
        with timer:
            for start in range(0, number_of_vectors, batch_size):
                stop = min(start + batch_size, number_of_vectors)
                active = stop - start
                grid_stride = self._serial_grid_stride(active)
                grid = self._scratch(
                    "serial_fft_grid", (active, grid_stride)
                )
                if self.thread_count > 1 and native_potential_layout:
                    z_plan, xy_plan = self._serial_spatial_plans(grid)
                    self._native_fft.apply_serial_spatial(
                        grid,
                        vectors[:, start:stop],
                        potential,
                        self.z_major_slots,
                        self.serial_stick_positions,
                        z_plan,
                        xy_plan,
                        self.size,
                        grid_stride,
                        result,
                        start,
                        diagonal,
                    )
                    continue
                parallel_bands = self._parallel_band_fft(active)
                plan = self._native_plan(
                    grid,
                    self.shape,
                    1 if parallel_bands else active,
                    1,
                    grid_stride,
                    parallel_bands=parallel_bands,
                )
                self._native_fft.apply_serial(
                    grid,
                    vectors[:, start:stop],
                    potential,
                    self.linear_slots,
                    plan,
                    self.size,
                    grid_stride,
                    result,
                    start,
                    diagonal,
                )
        return result[:, 0] if was_vector else result


def _basis_bound(
    reciprocal: np.ndarray,
    k_crystal: np.ndarray,
    ecut_ha: float,
    minimum_stretch: float,
) -> int:
    return int(
        np.ceil(
            (
                np.sqrt(2.0 * ecut_ha)
                + np.linalg.norm(k_crystal @ reciprocal)
            )
            / minimum_stretch
        )
    ) + 1


def _miller_grid(bound: int) -> np.ndarray:
    """Return the lexicographic Miller cube without Python tuple objects."""
    side = 2 * bound + 1
    grid = np.indices((side, side, side), dtype=np.int32)
    grid -= bound
    return np.moveaxis(grid, 0, -1).reshape(-1, 3)


def make_bases(
    reciprocal: np.ndarray,
    k_crystal: np.ndarray,
    ecut_ry: float,
    *,
    mpi: MPIContext | None = None,
) -> list[PlaneWaveBasis]:
    """Build multiple k-point bases from one globally ordered G catalog."""
    reciprocal = np.asarray(reciprocal, dtype=float)
    kpoints = np.asarray(k_crystal, dtype=float)
    if kpoints.ndim == 1:
        kpoints = kpoints[None, :]
    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError("k_crystal must have shape (3,) or (nk, 3)")
    if not len(kpoints):
        raise ValueError("at least one k point is required")
    ecut_ha = 0.5 * ecut_ry
    # For a non-orthogonal reciprocal basis, large integer coefficients can
    # cancel.  The shortest row norm is therefore not a safe Miller-index
    # bound.  Since ||n B|| >= sigma_min(B)||n||, the smallest singular value
    # provides a rigorous search radius for every vector satisfying the
    # kinetic cutoff.
    minimum_stretch = float(np.linalg.svd(reciprocal, compute_uv=False)[-1])
    bounds = np.array(
        [
            _basis_bound(
                reciprocal, point, ecut_ha, minimum_stretch
            )
            for point in kpoints
        ],
        dtype=int,
    )
    maximum_bound = int(np.max(bounds))
    integer = _miller_grid(maximum_bound)
    # QE ggen first orders the global reciprocal vectors by |G|^2, with
    # their original Miller-grid index breaking exact degeneracies. gk_sort
    # subsequently orders the retained vectors by |G+k|^2 and carries that
    # global-G index through ties. Starting directly from Miller order gives
    # a different deterministic atomic+random perturbation at nonzero k.
    global_vectors = integer @ reciprocal
    global_g2 = np.einsum("ij,ij->i", global_vectors, global_vectors)
    global_order = _qe_energy_order(global_g2)
    integer = integer[global_order]
    del global_vectors, global_g2, global_order
    selected_global_indices: list[np.ndarray] = []
    for point, bound in zip(kpoints, bounds):
        if bound == maximum_bound:
            candidate_global = np.arange(len(integer), dtype=np.int32)
        else:
            candidate_global = np.flatnonzero(
                np.all(np.abs(integer) <= bound, axis=1)
            ).astype(np.int32)
        point_integer = integer[candidate_global]
        vectors = (point_integer + point) @ reciprocal
        kinetic = 0.5 * np.einsum("ij,ij->i", vectors, vectors)
        keep = kinetic <= ecut_ha + 1.0e-12
        candidate_global = candidate_global[keep]
        kinetic = kinetic[keep]
        # Stable sorting retains QE's global-G precedence for degenerate G+k.
        order = _qe_energy_order(kinetic)
        if not len(order):
            raise QEInputError("ecutwfc produces an empty plane-wave basis")
        selected_global_indices.append(
            np.asarray(candidate_global[order], dtype=np.int32)
        )
    used = np.zeros(len(integer), dtype=bool)
    for selected in selected_global_indices:
        used[selected] = True
    compact_global = np.flatnonzero(used).astype(np.int32)
    remap = np.full(len(integer), -1, dtype=np.int32)
    remap[compact_global] = np.arange(len(compact_global), dtype=np.int32)
    catalog_indices = np.asarray(integer[compact_global], dtype=np.int32)
    catalog_vectors = np.asarray(catalog_indices @ reciprocal, dtype=float)
    share_catalog = (
        mpi is not None
        and mpi.size > 1
        and catalog_indices.nbytes + catalog_vectors.nbytes >= (1 << 20)
    )
    if share_catalog:
        catalog_indices = mpi.shared_readonly(catalog_indices)
        catalog_vectors = mpi.shared_readonly(catalog_vectors)
    catalog = PlaneWaveCatalog(
        catalog_indices,
        reciprocal.copy(),
        catalog_vectors,
        share_catalog,
    )
    return [
        PlaneWaveBasis(
            catalog,
            np.asarray(remap[selected], dtype=np.int32),
            np.asarray(point, dtype=float),
        )
        for point, selected in zip(kpoints, selected_global_indices)
    ]


def make_basis(
    reciprocal: np.ndarray,
    k_crystal: np.ndarray,
    ecut_ry: float,
) -> PlaneWaveBasis:
    return make_bases(reciprocal, k_crystal, ecut_ry)[0]


def _qe_energy_order(
    values: np.ndarray, tolerance: float = 1.0e-8
) -> np.ndarray:
    """Approximate QE ``hpsort_eps`` ordering for G and G+k energies.

    Energies separated by less than eps are treated as equal and retain
    their incoming global-G precedence. Raw NumPy sorting instead orders
    roundoff-split symmetry partners by their insignificant last bits, which
    changes the deterministic atomic+random starting wavefunctions.
    """
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    if len(order) < 2 or np.all(
        np.diff(array[order]) >= tolerance
    ):
        return order
    start = 0
    while start < len(order):
        stop = start + 1
        reference = array[order[start]]
        while (
            stop < len(order)
            and abs(array[order[stop]] - reference) < tolerance
        ):
            stop += 1
        if stop - start > 1:
            order[start:stop] = np.sort(order[start:stop])
        start = stop
    return order


def fft_shape(
    bases: list[PlaneWaveBasis],
    reciprocal: np.ndarray | None = None,
    ecutrho_ry: float | None = None,
    fft_factors: tuple[int, int, int] = (1, 1, 1),
) -> tuple[int, int, int]:
    if not bases:
        raise ValueError("at least one plane-wave basis is required")
    if (reciprocal is None) != (ecutrho_ry is None):
        raise ValueError("reciprocal and ecutrho_ry must be supplied together")
    if reciprocal is not None and ecutrho_ry is not None:
        # QE grid_set first selects the complete G sphere at the continuous
        # charge-density cutoff, obtains the largest Miller index on each
        # axis, and uses 2*nb+1 before good_fft_order.  Deriving the grid from
        # wavefunction G-G' spans loses the empty cutoff boundary and produced
        # 30^3 instead of QE's 32^3 for the 40-Ry Si test.
        charge_basis = make_basis(
            np.asarray(reciprocal, dtype=float),
            np.zeros(3),
            float(ecutrho_ry),
        )
        maximum_indices = np.max(np.abs(charge_basis.indices), axis=0)
        shape = []
        for value, factor in zip(maximum_indices, fft_factors):
            order = int(2 * value + 1)
            while True:
                candidate = _next_fast_len_real(order)
                if candidate % factor == 0:
                    shape.append(candidate)
                    break
                order = candidate + 1
        return tuple(shape)
    # A reciprocal-lattice shift of k is compensated by the opposite shift
    # of every G index in that k-point basis.  It therefore changes the
    # absolute integer labels but not any G-G' difference entering a local
    # potential convolution.  Combining global minima and maxima from
    # differently shifted bases spuriously enlarges selected FFT axes and can
    # break the grid symmetry.  Size each axis from the largest *per-k* span.
    spans = np.stack(
        [
            np.max(basis.indices, axis=0)
            - np.min(basis.indices, axis=0)
            for basis in bases
        ]
    )
    span = np.max(spans, axis=0)
    # rho(G) and V(G) contain differences of wavefunction G vectors. A
    # circular FFT convolution therefore needs more than twice the complete
    # wavefunction-index span to keep wrapped high-frequency products from
    # aliasing back into the retained plane-wave basis.  QE estimates the
    # grid from the continuous charge-density cutoff before selecting a good
    # FFT order.  The selected PW span can lie just below that boundary, so
    # 2*span+1 is one point too small in cases such as 12-Ry Si (15 instead
    # of QE's 16).  Include the unoccupied cutoff boundary point.
    shape = []
    for value in span:
        order = _next_fast_len_real(int(2 * value + 2))
        # QE's cutoff-based estimate gives an even dense-grid order for the
        # ordinary PW grids supported here.  Do not accept an intervening odd
        # fast length (e.g. scipy chooses 15 where QE chooses 16).
        if order % 2:
            order = _next_fast_len_real(order + 1)
        shape.append(order)
    return tuple(shape)


def potential_matrix(vg: np.ndarray, indices: np.ndarray) -> np.ndarray:
    delta = indices[:, None, :] - indices[None, :, :]
    slots = tuple(delta[:, :, axis] % vg.shape[axis] for axis in range(3))
    return vg[slots]


def apply_local_potential(
    vg: np.ndarray, indices: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """Apply a periodic local potential by FFT without forming its matrix."""
    workspace = LocalPotentialWorkspace(indices, vg.shape)
    return workspace.apply(
        workspace.prepare_potential(vg), coefficients
    )
