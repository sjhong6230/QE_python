"""Reciprocal-space basis and FFT-grid utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext

import numpy as np
from scipy.fft import next_fast_len
from scipy import fft as scipy_fft

from .errors import QEInputError
from .mpi import MPIContext
from .timing import TimingRegistry


@dataclass(frozen=True)
class PlaneWaveCatalog:
    """One QE-ordered global G catalog shared by every k-point basis."""

    indices: np.ndarray
    reciprocal: np.ndarray

    @property
    def vectors(self) -> np.ndarray:
        return self.indices @ self.reciprocal


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
            object.__setattr__(
                self,
                "global_indices",
                np.asarray(mapping_or_vectors, dtype=np.int32),
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
        return (
            self.indices + self.k_crystal[None, :]
        ) @ self.catalog.reciprocal

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
        vectors = (
            indices + self.k_crystal[None, :]
        ) @ self.catalog.reciprocal
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
    _scaled: dict[
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


@dataclass(frozen=True)
class FFTGridDescriptor:
    """One shared FFT stick distribution for a grid and communicator size."""

    shape: tuple[int, int, int]
    processes: int
    sticks: np.ndarray
    stick_owners: np.ndarray
    sticks_by_rank: tuple[np.ndarray, ...]
    stick_lookup: np.ndarray
    transpose_plans: tuple[FFTTransposePlan, ...]

    @classmethod
    def build(
        cls,
        indices: np.ndarray,
        shape: tuple[int, int, int],
        processes: int,
    ) -> "FFTGridDescriptor":
        miller = np.asarray(indices, dtype=np.int32)
        shape = tuple(int(value) for value in shape)
        if miller.ndim != 2 or miller.shape[1] != 3:
            raise ValueError("indices must have shape (nplane, 3)")
        if processes < 1:
            raise ValueError("FFT descriptor needs at least one process")
        pairs = np.column_stack(
            (miller[:, 0] % shape[0], miller[:, 1] % shape[1])
        )
        sticks, inverse = np.unique(pairs, axis=0, return_inverse=True)
        sticks = np.asarray(sticks, dtype=np.int32)
        counts = np.bincount(inverse, minlength=len(sticks))
        owners = np.zeros(len(sticks), dtype=np.int32)
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
        plans: list[FFTTransposePlan] = []
        for rank in range(processes):
            local_z = int(z_counts[rank])
            point_blocks = []
            for peer_sticks in by_rank:
                peer_xy = xy[peer_sticks]
                point_blocks.append(
                    (
                        peer_xy[None, :] * local_z
                        + np.arange(local_z, dtype=np.int64)[:, None]
                    ).ravel()
                )
            points = (
                np.concatenate(point_blocks)
                if point_blocks
                else np.empty(0, dtype=np.int64)
            )
            plans.append(
                FFTTransposePlan(
                    g2r_send_base=z_counts * stick_counts[rank],
                    g2r_recv_base=local_z * stick_counts,
                    slab_point_indices=points,
                )
            )
        return cls(
            shape,
            int(processes),
            sticks,
            owners,
            by_rank,
            lookup,
            tuple(plans),
        )

    @property
    def nbytes(self) -> int:
        return int(
            self.sticks.nbytes
            + self.stick_owners.nbytes
            + self.stick_lookup.nbytes
            + sum(array.nbytes for array in self.sticks_by_rank)
            + sum(
                plan.g2r_send_base.nbytes
                + plan.g2r_recv_base.nbytes
                + plan.slab_point_indices.nbytes
                for plan in self.transpose_plans
            )
        )


class FFTScratchPool:
    """Process-local reusable storage for sequential distributed FFT calls."""

    def __init__(self) -> None:
        self._complex: dict[str, np.ndarray] = {}
        self._complex_alignment: dict[str, int] = {}
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
        size = int(np.prod(shape))
        buffer = self._complex.get(name)
        requested_alignment = 0 if alignment is None else int(alignment)
        if (
            buffer is None
            or buffer.size < size
            or self._complex_alignment.get(name, 0) < requested_alignment
        ):
            if buffer is not None:
                old_pointer = int(buffer.ctypes.data)
                for key in list(self._fftw_plans):
                    if (
                        key
                        and key[0] == "local"
                        and key[-1] == old_pointer
                    ):
                        del self._fftw_plans[key]
            if requested_alignment:
                import pyfftw

                buffer = pyfftw.empty_aligned(
                    size, dtype="complex128", n=requested_alignment
                )
            else:
                buffer = np.empty(size, dtype=np.complex128)
            self._complex[name] = buffer
            self._complex_alignment[name] = requested_alignment
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
        backend: str = "numpy",
        threads: int = 1,
        planner_effort: str = "measure",
        use_numba: bool = False,
        mpi: MPIContext | None = None,
        timers: TimingRegistry | None = None,
        scratch_pool: FFTScratchPool | None = None,
        descriptor: FFTGridDescriptor | None = None,
    ) -> None:
        if isinstance(indices, PlaneWaveBasis):
            miller = indices.indices
        else:
            miller = np.asarray(indices, dtype=np.int32)
        if miller.ndim != 2 or miller.shape[1] != 3:
            raise ValueError("indices must have shape (nplane, 3)")
        self.plane_waves = len(miller)
        self.shape = tuple(shape)
        self.size = int(np.prod(shape))
        slots = tuple(
            np.asarray(
                miller[:, axis] % shape[axis],
                dtype=np.int32,
            )
            for axis in range(3)
        )
        self.linear_slots = np.asarray(
            np.ravel_multi_index(slots, self.shape),
            dtype=np.int32,
        )
        self.use_numba = bool(use_numba)
        self.mpi = mpi if mpi is not None else MPIContext()
        self.timers = timers
        self.scratch_pool = (
            scratch_pool if scratch_pool is not None else FFTScratchPool()
        )
        # QE stores complete Z-sticks indexed by (Gx,Gy), then transposes
        # them into real-space Z-plane slabs for the two-dimensional XY FFT.
        self.local_slab = self.mpi.slab(self.shape[2])
        if self.mpi.size == 1:
            self.descriptor = None
            self.transpose_plan = None
            self.sticks = np.unique(
                np.column_stack((slots[0], slots[1])), axis=0
            ).astype(np.int32, copy=False)
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
        else:
            if descriptor is None:
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
            self.sticks = descriptor.sticks
            self.stick_owners = descriptor.stick_owners
            self.sticks_by_rank = descriptor.sticks_by_rank
            pair_slots = slots[0] * self.shape[1] + slots[1]
            self.stick_indices = descriptor.stick_lookup[pair_slots]
            if np.any(self.stick_indices < 0):
                raise ValueError(
                    "FFT descriptor does not cover the plane-wave basis"
                )
            self.owned_plane_waves = (
                self.stick_owners[self.stick_indices] == self.mpi.rank
            )
            self.local_plane_wave_indices = np.asarray(
                np.flatnonzero(self.owned_plane_waves),
                dtype=np.int32,
            )
            self.local_slots = tuple(
                axis[self.owned_plane_waves] for axis in slots
            )
            local_stick_lookup = np.full(
                len(self.sticks), -1, dtype=np.int32
            )
            local_stick_lookup[
                self.sticks_by_rank[self.mpi.rank]
            ] = np.arange(len(self.sticks_by_rank[self.mpi.rank]))
            self.local_stick_positions = local_stick_lookup[
                self.stick_indices[self.owned_plane_waves]
            ]
        if self.use_numba:
            from .acceleration import numba_kernels

            self._jit = numba_kernels()
        self.backend = backend.lower()
        self.threads = int(threads)
        if self.threads < 1:
            raise QEInputError("py_fft_threads must be at least 1")
        effort = planner_effort.lower()
        efforts = {
            "estimate": "FFTW_ESTIMATE",
            "measure": "FFTW_MEASURE",
            "patient": "FFTW_PATIENT",
        }
        if effort not in efforts:
            raise QEInputError(
                "py_fft_planner must be 'estimate', 'measure', or 'patient'"
        )
        self.planner_flag = efforts[effort]
        if self.backend == "pyfftw":
            try:
                import pyfftw
            except ImportError as exc:
                raise QEInputError(
                    "py_fft_backend='pyfftw' requires the optional pyFFTW "
                    "package; install qepy-pw[fft]"
                ) from exc
            self._pyfftw = pyfftw
            self._fft_alignment = int(pyfftw.simd_alignment)
        elif self.backend not in {"numpy", "scipy"}:
            raise QEInputError(
                "py_fft_backend must be 'scipy', 'numpy', or 'pyfftw'"
            )
        else:
            self._fft_alignment = 0

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

    def _planned_local_transform(
        self,
        values: np.ndarray,
        axes: tuple[int, ...],
        direction: str,
    ) -> np.ndarray | None:
        """Run an in-place pyFFTW plan when the reusable array is aligned."""
        if (
            self.backend != "pyfftw"
            or values.dtype != np.complex128
            or not values.flags.c_contiguous
            or not any(
                int(values.ctypes.data) == int(buffer.ctypes.data)
                for buffer in self.scratch_pool._complex.values()
            )
            or not self._pyfftw.is_byte_aligned(
                values, n=self._fft_alignment
            )
        ):
            return None
        plan_values = values
        if (
            axes == tuple(range(len(axes)))
            and values.ndim > len(axes) + 1
        ):
            # The remaining dimensions are independent transform batches.
            # Flatten them into one dimension.  Besides reducing planner
            # metadata, this avoids a pyFFTW/FFTW limitation seen for
            # rank-3 one-dimensional and rank-4 two-dimensional in-place
            # plans when SciPy's native libraries are already loaded.
            plan_values = values.reshape(
                values.shape[: len(axes)] + (-1,)
            )
        cache_key = (
            "local",
            direction,
            plan_values.shape,
            plan_values.strides,
            axes,
            self.threads,
            self.planner_flag,
            int(values.ctypes.data),
        )
        cached = self.scratch_pool._fftw_plans.get(cache_key)
        if cached is None:
            # Keep at most one shape per direction on this reusable buffer.
            # Otherwise each transient Davidson block shape leaves a plan
            # object retaining a view of the shared allocation.
            for key in list(self.scratch_pool._fftw_plans):
                if (
                    key
                    and key[0] == "local"
                    and key[1] == direction
                    and key[-1] == int(values.ctypes.data)
                ):
                    del self.scratch_pool._fftw_plans[key]
            # FFTW_MEASURE/PATIENT may overwrite the planning array.
            preserved = values.copy()
            try:
                plan = self._pyfftw.FFTW(
                    plan_values,
                    plan_values,
                    axes=axes,
                    direction=(
                        "FFTW_FORWARD"
                        if direction == "forward"
                        else "FFTW_BACKWARD"
                    ),
                    flags=(self.planner_flag,),
                    threads=self.threads,
                )
            except RuntimeError:
                # Some FFTW builds reject a SIMD plan after native libraries
                # have changed alignment assumptions (notably under pytest's
                # MPI runner).  The unaligned plan remains genuinely in-place
                # and is preferable to silently allocating SciPy temporaries.
                plan = self._pyfftw.FFTW(
                    plan_values,
                    plan_values,
                    axes=axes,
                    direction=(
                        "FFTW_FORWARD"
                        if direction == "forward"
                        else "FFTW_BACKWARD"
                    ),
                    flags=(self.planner_flag, "FFTW_UNALIGNED"),
                    threads=self.threads,
                )
            values[...] = preserved
            cached = (plan,)
            self.scratch_pool._fftw_plans[cache_key] = cached
        plan = cached[0]
        # Keep NumPy/FFTW's conventional pair: backward transforms carry the
        # 1/N normalization and forward transforms are unscaled.  In pyFFTW,
        # passing ``normalise_idft=False`` instead normalizes the forward
        # direction, so the option must remain true for both plan directions.
        plan(normalise_idft=True)
        return values

    def prepare_potential(self, potential_g: np.ndarray) -> np.ndarray:
        """Transform a fixed local potential once per Hamiltonian."""
        values = potential_g * self.size
        if self.backend in {"scipy", "pyfftw"}:
            return np.real(
                scipy_fft.ifftn(
                    values,
                    axes=(0, 1, 2),
                    workers=self.threads,
                    overwrite_x=True,
                )
            )
        return np.real(np.fft.ifftn(values))

    def _backward_fft(
        self, values: np.ndarray, axes: tuple[int, ...]
    ) -> np.ndarray:
        planned = self._planned_local_transform(values, axes, "backward")
        if planned is not None:
            return planned
        if self.backend in {"scipy", "pyfftw"}:
            return scipy_fft.ifftn(
                values,
                axes=axes,
                workers=self.threads,
                overwrite_x=True,
            )
        return np.fft.ifftn(values, axes=axes)

    def _forward_fft(
        self, values: np.ndarray, axes: tuple[int, ...]
    ) -> np.ndarray:
        planned = self._planned_local_transform(values, axes, "forward")
        if planned is not None:
            return planned
        if self.backend in {"scipy", "pyfftw"}:
            return scipy_fft.fftn(
                values,
                axes=axes,
                workers=self.threads,
                overwrite_x=True,
            )
        return np.fft.fftn(values, axes=axes)

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
            if self.backend == "pyfftw" and use_scratch:
                real_grid, backward, _forward = self._planned_buffers(
                    number_of_vectors
                )
                real_grid.fill(0.0)
                real_grid.reshape(number_of_vectors, self.size)[
                    :, self.linear_slots
                ] = vectors.T * self.size
                timer = (
                    self.timers.measure("fftw")
                    if self.timers is not None
                    else nullcontext()
                )
                with timer:
                    backward(normalise_idft=True)
            else:
                reciprocal_input = (
                    self._numpy_buffers(number_of_vectors)
                    if use_scratch
                    else np.zeros(
                        (number_of_vectors,) + self.shape,
                        dtype=complex,
                    )
                )
                reciprocal_input.reshape(number_of_vectors, self.size)[
                    :, self.linear_slots
                ] = vectors.T * self.size
                timer = (
                    self.timers.measure("fftw")
                    if self.timers is not None
                    else nullcontext()
                )
                with timer:
                    real_grid = self._backward_fft(
                        reciprocal_input, (1, 2, 3)
                    )
            result = np.moveaxis(real_grid, 0, -1)
            return result[..., 0] if was_vector else result
        local_sticks = self.sticks_by_rank[self.mpi.rank]
        stick_shape = (
            self.shape[2],
            len(local_sticks),
            number_of_vectors,
        )
        data = (
            self._scratch(
                "inverse_sticks", stick_shape, zero=True
            )
            if use_scratch
            else np.zeros(stick_shape, dtype=complex)
        )
        data[
            self.local_slots[2],
            self.local_stick_positions,
        ] = vectors * self.size
        timer = (
            self.timers.measure("fftw")
            if self.timers is not None
            else nullcontext()
        )
        with timer:
            data = self._backward_fft(data, (0,))
            real_slab = self._sticks_to_real_slabs(
                data,
                out=(
                    self._scratch(
                        "real_slabs",
                        (
                            self.shape[0],
                            self.shape[1],
                            self.local_slab.stop
                            - self.local_slab.start,
                            number_of_vectors,
                        ),
                        zero=True,
                    )
                    if use_scratch
                    else None
                ),
            )
            real_slab = self._backward_fft(real_slab, (0, 1))
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
        if self.use_numba:
            self._jit.scatter_indexed_rows(
                source_rows,
                self.transpose_plan.slab_point_indices,
                destination_rows,
            )
        else:
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
            result = self._scratch("grid_forward_sticks", expected_shape)
        else:
            if out.shape != expected_shape:
                raise ValueError("stick scratch buffer has wrong shape")
            result = out
        send_size = int(np.sum(send_counts))
        send, _unused_recv = self.mpi.complex_exchange_buffers(
            send_size, 0
        )
        source_rows = np.asarray(real_data).reshape(-1, number_of_vectors)
        send_rows = send.reshape(-1, number_of_vectors)
        if self.use_numba:
            self._jit.gather_indexed_rows(
                source_rows,
                self.transpose_plan.slab_point_indices,
                send_rows,
            )
        else:
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
            transformed = self._forward_fft(values, (0, 1))
            transformed = self._forward_fft(transformed, (2,)) / self.size
            result = transformed.reshape(
                self.size, transformed.shape[-1]
            )[self.linear_slots]
            return result[:, 0] if was_scalar else result
        if self.backend == "pyfftw" and not (
            values.dtype == np.complex128
            and values.flags.c_contiguous
            and any(
                int(values.ctypes.data) == int(buffer.ctypes.data)
                for buffer in self.scratch_pool._complex.values()
            )
            and self._pyfftw.is_byte_aligned(
                values, n=self._fft_alignment
            )
        ):
            transformed = self._scratch(
                "grid_real_slabs", values.shape
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
        self, real_potential: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        real_wavefunctions = self.coefficients_to_grid(
            vectors, use_scratch=True
        )
        if real_wavefunctions.ndim == 3:
            real_wavefunctions = real_wavefunctions[..., None]
        potential = (
            real_potential
            if real_potential.shape[2] == real_wavefunctions.shape[2]
            else real_potential[:, :, self.local_slab]
        )
        np.multiply(
            potential[..., None],
            real_wavefunctions,
            out=real_wavefunctions,
        )
        timer = (
            self.timers.measure("fftw")
            if self.timers is not None
            else nullcontext()
        )
        with timer:
            transformed = self._forward_fft(real_wavefunctions, (0, 1))
            transformed = self._real_slabs_to_sticks(
                transformed,
                out=self._scratch(
                    "forward_sticks",
                    (
                        self.shape[2],
                        len(self.sticks_by_rank[self.mpi.rank]),
                        vectors.shape[1],
                    ),
                ),
            )
            transformed = self._forward_fft(transformed, (0,)) / self.size
        return transformed[
            self.local_slots[2],
            self.local_stick_positions,
        ]

    def _numpy_buffers(
        self, number_of_vectors: int
    ) -> np.ndarray:
        block_shape = (number_of_vectors,) + self.shape
        return self._scratch(
            "serial_fft_input", block_shape, zero=True
        )

    def _planned_buffers(
        self, number_of_vectors: int
    ) -> tuple[object, ...]:
        cache_key = (
            self.shape,
            number_of_vectors,
            self.threads,
            self.planner_flag,
        )
        cached = self.scratch_pool._fftw_plans.get(cache_key)
        if cached is not None:
            return cached
        # Serial plans own their full-grid work array. Retain only the active
        # block size so an earlier, smaller Davidson block is released.
        for key in list(self.scratch_pool._fftw_plans):
            if key and key[0] != "local":
                del self.scratch_pool._fftw_plans[key]
        pyfftw = self._pyfftw
        block_shape = (number_of_vectors,) + self.shape
        # QE's vloc_psi uses one ``psic`` grid for G->R, multiplies the
        # potential into it, then transforms that same grid R->G.  Mirror
        # that layout here: an in-place complex FFT needs only one full-grid
        # block instead of separate input/output blocks for both directions.
        grid = pyfftw.empty_aligned(
            block_shape, dtype="complex128"
        )
        backward = pyfftw.FFTW(
            grid,
            grid,
            axes=(1, 2, 3),
            direction="FFTW_BACKWARD",
            flags=(self.planner_flag,),
            threads=self.threads,
        )
        forward = pyfftw.FFTW(
            grid,
            grid,
            axes=(1, 2, 3),
            direction="FFTW_FORWARD",
            flags=(self.planner_flag,),
            threads=self.threads,
        )
        buffers = (grid, backward, forward)
        self.scratch_pool._fftw_plans[cache_key] = buffers
        return buffers

    def apply(
        self, real_potential: np.ndarray, coefficients: np.ndarray
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
            result = self._distributed_apply(real_potential, vectors)
            return result[:, 0] if was_vector else result
        if self.backend in {"numpy", "scipy"}:
            reciprocal_input = self._numpy_buffers(number_of_vectors)
            if self.use_numba:
                self._jit.scatter_scaled(
                    reciprocal_input.reshape(number_of_vectors, self.size).T,
                    self.linear_slots,
                    vectors,
                    float(self.size),
                )
            else:
                reciprocal_input.fill(0.0)
                reciprocal_input.reshape(number_of_vectors, self.size)[
                    :, self.linear_slots
                ] = vectors.T * self.size
            timer = (
                self.timers.measure("fftw")
                if self.timers is not None
                else nullcontext()
            )
            with timer:
                real_wavefunctions = self._backward_fft(
                    reciprocal_input, (1, 2, 3)
                )
            if self.use_numba:
                self._jit.multiply_real_complex(
                    real_potential.reshape(self.size),
                    real_wavefunctions.reshape(
                        number_of_vectors, self.size
                    ).T,
                    real_wavefunctions.reshape(
                        number_of_vectors, self.size
                    ).T,
                )
            else:
                np.multiply(
                    real_potential[None, ...],
                    real_wavefunctions,
                    out=real_wavefunctions,
                )
            timer = (
                self.timers.measure("fftw")
                if self.timers is not None
                else nullcontext()
            )
            with timer:
                transformed = self._forward_fft(
                    real_wavefunctions, (1, 2, 3)
                )
                transformed /= self.size
            if self.use_numba:
                result = np.empty_like(vectors)
                self._jit.gather_scaled(
                    transformed.reshape(number_of_vectors, self.size).T,
                    self.linear_slots,
                    result,
                    1.0,
                )
            else:
                result = transformed.reshape(
                    number_of_vectors, self.size
                )[:, self.linear_slots].T
        else:
            grid, backward, forward = self._planned_buffers(
                number_of_vectors
            )
            if self.use_numba:
                self._jit.scatter_scaled(
                    grid.reshape(number_of_vectors, self.size).T,
                    self.linear_slots,
                    vectors,
                    float(self.size),
                )
            else:
                grid.fill(0.0)
                grid.reshape(number_of_vectors, self.size)[
                    :, self.linear_slots
                ] = vectors.T * self.size
            timer = (
                self.timers.measure("fftw")
                if self.timers is not None
                else nullcontext()
            )
            with timer:
                backward(normalise_idft=True)
            if self.use_numba:
                self._jit.multiply_real_complex(
                    real_potential.reshape(self.size),
                    grid.reshape(number_of_vectors, self.size).T,
                    grid.reshape(number_of_vectors, self.size).T,
                )
            else:
                np.multiply(
                    real_potential[None, ...],
                    grid,
                    out=grid,
                )
            timer = (
                self.timers.measure("fftw")
                if self.timers is not None
                else nullcontext()
            )
            with timer:
                forward()
            if self.use_numba:
                result = np.empty_like(vectors)
                self._jit.gather_scaled(
                    grid.reshape(number_of_vectors, self.size).T,
                    self.linear_slots,
                    result,
                    1.0 / self.size,
                )
            else:
                result = grid.reshape(number_of_vectors, self.size)[
                    :, self.linear_slots
                ].T
                result /= self.size
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
    catalog = PlaneWaveCatalog(
        np.asarray(integer[compact_global], dtype=np.int32),
        reciprocal.copy(),
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
                candidate = next_fast_len(order, real=True)
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
        order = next_fast_len(int(2 * value + 2), real=True)
        # QE's cutoff-based estimate gives an even dense-grid order for the
        # ordinary PW grids supported here.  Do not accept an intervening odd
        # fast length (e.g. scipy chooses 15 where QE chooses 16).
        if order % 2:
            order = next_fast_len(order + 1, real=True)
        shape.append(order)
    return tuple(shape)


def coefficients_to_grid(coefficients: np.ndarray, indices: np.ndarray, shape: tuple[int, ...], volume: float) -> np.ndarray:
    data = np.zeros(shape, dtype=complex)
    slots = tuple((indices[:, axis] % shape[axis]) for axis in range(3))
    data[slots] = coefficients * np.prod(shape) / np.sqrt(volume)
    return np.fft.ifftn(data)


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
