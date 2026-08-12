"""Memory-bounded distributed FFT execution policies and pencil transforms.

The production plane-wave path in :mod:`qepy_pw.basis` uses a sparse Z-stick
slab decomposition.  This module provides the decomposition-independent
building blocks needed beyond that slab limit: a two-dimensional MPI process
grid, dense Z/Y/X pencils, task-group communicators, bounded band tiling, and
the algebra for QE's Gamma half-spectrum/two-real-orbital representation.

All three local transform axes are stored first.  Consequently every FFTW
``plan_many`` has one regular transform stride, while MPI packing is explicit
and accounted for separately.  Forward transforms carry the global 1/N
normalization used by QE and by ``LocalPotentialWorkspace``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt, prod
from typing import Iterable

import numpy as np

from .errors import QEInputError
from .threads import hybrid_thread_count


def _partition(length: int, parts: int, index: int) -> slice:
    """Return one balanced contiguous partition."""
    if parts < 1 or not 0 <= index < parts:
        raise ValueError("invalid partition")
    return slice(length * index // parts, length * (index + 1) // parts)


def _partition_owner(length: int, parts: int, slot: int) -> int:
    """Return the owner of ``slot`` under :func:`_partition`."""
    if not 0 <= slot < length:
        raise ValueError("partition slot is outside the global dimension")
    # This formula also handles length not divisible by parts.  Empty
    # partitions are excluded by the topology constructor.
    return min(parts - 1, ((slot + 1) * parts - 1) // length)


def choose_process_grid(
    shape: tuple[int, int, int], processes: int
) -> tuple[int, int]:
    """Choose a valid near-square ``(P_x, P_y)`` pencil process grid.

    The first process dimension partitions X in the Z/Y layouts and Y in the
    final X-pencil layout.  The second partitions Y initially and Z finally.
    Candidate grids that would create an empty pencil are rejected.
    """
    if processes < 1:
        raise ValueError("FFT process count must be positive")
    nx, ny, nz = map(int, shape)
    candidates: list[tuple[float, int, int]] = []
    for first in range(1, isqrt(processes) + 1):
        if processes % first:
            continue
        for px, py in ((first, processes // first), (processes // first, first)):
            if px > nx or px > ny or py > ny or py > nz:
                continue
            # Prefer balanced communicator sizes and local pencil cross
            # sections, with a small penalty for cuts that do not divide the
            # physical dimensions exactly.
            local_x = nx / px
            local_y_initial = ny / py
            local_y_final = ny / px
            local_z = nz / py
            aspect = max(local_x, local_y_initial, local_y_final, local_z) / max(
                1.0, min(local_x, local_y_initial, local_y_final, local_z)
            )
            uneven = sum(
                dimension % parts != 0
                for dimension, parts in ((nx, px), (ny, py), (ny, px), (nz, py))
            )
            score = abs(px - py) + 0.25 * aspect + 0.1 * uneven
            candidates.append((score, px, py))
    if not candidates:
        raise QEInputError(
            f"{processes} FFT ranks cannot form nonempty pencils for grid {shape}"
        )
    _score, px, py = min(candidates)
    return px, py


def choose_fft_decomposition(
    shape: tuple[int, int, int],
    processes: int,
    *,
    slab_bytes_per_rank: int | None = None,
    memory_limit_bytes: int | None = None,
) -> str:
    """Choose ``slab`` when it scales and fits, otherwise ``pencil``.

    This policy is intentionally conservative: slab needs one collective per
    direction and normally wins at modest rank counts. Pencil is mandatory
    beyond the Z-plane count and is selected earlier when a caller's exact
    slab memory estimate exceeds its hard budget.
    """
    if processes < 1:
        raise ValueError("FFT process count must be positive")
    needs_pencil = processes > int(shape[2])
    if memory_limit_bytes is not None and slab_bytes_per_rank is not None:
        needs_pencil = needs_pencil or slab_bytes_per_rank > memory_limit_bytes
    if not needs_pencil:
        return "slab"
    # Fail here, during setup, rather than much later inside an MPI transpose.
    choose_process_grid(shape, processes)
    return "pencil"


@dataclass
class FFTTaskTopology:
    """MPI task-group and two-dimensional FFT communicator hierarchy."""

    world_comm: object
    fft_comm: object
    row_comm: object
    column_comm: object
    homologous_comm: object
    world_rank: int
    world_size: int
    task_groups: int
    task_group: int
    fft_rank: int
    fft_size: int
    process_grid: tuple[int, int]
    coordinate: tuple[int, int]

    @classmethod
    def build(
        cls,
        world_comm: object,
        shape: tuple[int, int, int],
        *,
        task_groups: int = 1,
        process_grid: tuple[int, int] | None = None,
    ) -> "FFTTaskTopology":
        world_rank = int(world_comm.Get_rank())
        world_size = int(world_comm.Get_size())
        task_groups = int(task_groups)
        if task_groups < 1 or world_size % task_groups:
            raise QEInputError(
                "FFT task-group count must be a positive divisor of MPI size"
            )
        fft_size = world_size // task_groups
        task_group = world_rank // fft_size
        fft_rank = world_rank % fft_size
        fft_comm = world_comm.Split(color=task_group, key=fft_rank)
        selected_grid = (
            choose_process_grid(shape, fft_size)
            if process_grid is None
            else tuple(map(int, process_grid))
        )
        px, py = selected_grid
        if px * py != fft_size:
            raise QEInputError("pencil process grid does not match FFT group size")
        # Validate explicitly even when the user supplied the grid.
        nx, ny, nz = shape
        if px > nx or px > ny or py > ny or py > nz:
            raise QEInputError("pencil process grid creates an empty local pencil")
        coordinate = (fft_rank // py, fft_rank % py)
        row, column = coordinate
        row_comm = fft_comm.Split(color=row, key=column)
        column_comm = fft_comm.Split(color=column, key=row)
        homologous_comm = world_comm.Split(color=fft_rank, key=task_group)
        return cls(
            world_comm,
            fft_comm,
            row_comm,
            column_comm,
            homologous_comm,
            world_rank,
            world_size,
            task_groups,
            task_group,
            fft_rank,
            fft_size,
            selected_grid,
            coordinate,
        )

    def band_slice(self, bands: int) -> slice:
        """Bands owned by this task group under a balanced distribution."""
        return _partition(int(bands), self.task_groups, self.task_group)


@dataclass(frozen=True)
class FFTMemoryPlan:
    """A hard upper bound for band-dependent FFT scratch."""

    band_tile: int
    fixed_bytes: int
    bytes_per_band: int
    memory_limit_bytes: int | None
    double_buffer: bool

    @classmethod
    def choose(
        cls,
        *,
        bands: int,
        fixed_bytes: int,
        bytes_per_band: int,
        memory_limit_bytes: int | None = None,
        preferred_tile: int | None = None,
        gamma_pairs: bool = False,
        allow_double_buffer: bool = True,
    ) -> "FFTMemoryPlan":
        if bands < 1 or bytes_per_band < 1 or fixed_bytes < 0:
            raise ValueError("invalid FFT memory-plan dimensions")
        preferred = int(preferred_tile or hybrid_thread_count())
        preferred = max(1, min(int(bands), preferred))
        if memory_limit_bytes is None:
            tile = preferred
            double_buffer = False
        else:
            available = int(memory_limit_bytes) - int(fixed_bytes)
            if available < bytes_per_band:
                raise QEInputError(
                    "FFT memory limit cannot hold one distributed band tile"
                )
            tile = min(preferred, max(1, available // bytes_per_band))
            double_buffer = bool(
                allow_double_buffer
                and available >= 2 * bytes_per_band * tile
            )
        if gamma_pairs and bands > 1 and tile > 1:
            tile -= tile % 2
            tile = max(2, tile)
        return cls(
            tile,
            int(fixed_bytes),
            int(bytes_per_band),
            None if memory_limit_bytes is None else int(memory_limit_bytes),
            double_buffer,
        )


class PencilFFT3D:
    """Two-dimensional MPI pencil FFT with bounded band tiles.

    Input/output reciprocal arrays use Z pencils with shape
    ``(Nz, local_Nx, local_Ny, bands)``.  Real-space arrays use X pencils with
    shape ``(Nx, local_Ny, local_Nz, bands)``.  The implementation is dense;
    sparse plane-wave scatter/gather can wrap the Z-pencil boundary without
    changing either MPI transpose.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        topology: FFTTaskTopology,
        *,
        planner_flag: str = "FFTW_ESTIMATE",
        threads: int | None = None,
    ) -> None:
        self.shape = tuple(map(int, shape))
        self.topology = topology
        self.threads = int(threads or hybrid_thread_count())
        if self.threads < 1:
            raise ValueError("FFT thread count must be positive")
        try:
            self._flag = {
                "FFTW_ESTIMATE": 64,
                "FFTW_MEASURE": 0,
                "FFTW_PATIENT": 32,
            }[planner_flag]
        except KeyError as exc:
            raise ValueError("unsupported FFTW planner flag") from exc
        px, py = topology.process_grid
        row, column = topology.coordinate
        nx, ny, nz = self.shape
        self.x_initial = _partition(nx, px, row)
        self.y_initial = _partition(ny, py, column)
        self.y_final = _partition(ny, px, row)
        self.z_final = _partition(nz, py, column)
        self._arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._plans: dict[int, tuple[object, object, object]] = {}
        self._send = np.empty(0, dtype=np.complex128)
        self._receive = np.empty(0, dtype=np.complex128)

    @property
    def local_z_shape(self) -> tuple[int, int, int]:
        return (
            self.shape[2],
            self.x_initial.stop - self.x_initial.start,
            self.y_initial.stop - self.y_initial.start,
        )

    @property
    def local_x_shape(self) -> tuple[int, int, int]:
        return (
            self.shape[0],
            self.y_final.stop - self.y_final.start,
            self.z_final.stop - self.z_final.start,
        )

    def estimated_scratch_bytes(self, bands: int, *, double_buffer: bool = False) -> int:
        """Return the exact dense array plus transpose-buffer allocation."""
        z_points = prod(self.local_z_shape)
        y_points = (
            self.shape[1]
            * (self.x_initial.stop - self.x_initial.start)
            * (self.z_final.stop - self.z_final.start)
        )
        x_points = prod(self.local_x_shape)
        # At execution all three plan-owned layouts are retained.  One
        # send/receive pair is reused by both transposes.
        exchange_points = 2 * max(z_points, y_points, x_points)
        if double_buffer:
            exchange_points *= 2
        return int((z_points + y_points + x_points + exchange_points) * bands * 16)

    @staticmethod
    def _mpi_datatype(dtype: np.dtype) -> object:
        from mpi4py import MPI

        if dtype == np.dtype(np.float64):
            return MPI.DOUBLE
        if dtype == np.dtype(np.complex128):
            return MPI.C_DOUBLE_COMPLEX
        raise ValueError("pencil field redistribution requires float64 or complex128")

    @staticmethod
    def _intersection(first: slice, second: slice) -> slice:
        start = max(int(first.start), int(second.start))
        stop = min(int(first.stop), int(second.stop))
        return slice(start, max(start, stop))

    def slab_to_x_pencil(self, slab: np.ndarray) -> np.ndarray:
        """Redistribute a conventional Z slab into unique X pencils."""
        values = np.asarray(slab)
        scalar = values.ndim == 3
        if scalar:
            values = values[..., None]
        nx, ny, nz = self.shape
        source_z = _partition(nz, self.topology.fft_size, self.topology.fft_rank)
        expected = (nx, ny, source_z.stop - source_z.start)
        if values.shape[:3] != expected:
            raise ValueError("field does not match the rank-local Z slab")
        channels = values.shape[-1]
        blocks: list[np.ndarray] = []
        send_counts: list[int] = []
        px, py = self.topology.process_grid
        for destination in range(self.topology.fft_size):
            row, column = divmod(destination, py)
            target_y = _partition(ny, px, row)
            target_z = _partition(nz, py, column)
            overlap = self._intersection(source_z, target_z)
            block = np.ascontiguousarray(
                values[
                    :,
                    target_y,
                    overlap.start - source_z.start : overlap.stop - source_z.start,
                    :,
                ]
            ).reshape(-1)
            blocks.append(block)
            send_counts.append(block.size)
        receive_counts = np.asarray(
            self.topology.fft_comm.alltoall(send_counts), dtype=np.int64
        )
        receive, displacements = self._exchange_typed(
            self.topology.fft_comm, blocks, receive_counts, values.dtype
        )
        result = np.empty(self.local_x_shape + (channels,), dtype=values.dtype)
        for source in range(self.topology.fft_size):
            source_z = _partition(nz, self.topology.fft_size, source)
            overlap = self._intersection(source_z, self.z_final)
            count = int(receive_counts[source])
            block = receive[
                int(displacements[source]) : int(displacements[source]) + count
            ].reshape(
                nx,
                self.y_final.stop - self.y_final.start,
                overlap.stop - overlap.start,
                channels,
            )
            result[
                :,
                :,
                overlap.start - self.z_final.start : overlap.stop - self.z_final.start,
                :,
            ] = block
        return result[..., 0] if scalar else result

    def x_pencil_to_slab(self, pencil: np.ndarray) -> np.ndarray:
        """Redistribute unique X pencils into the conventional Z slabs."""
        values = np.asarray(pencil)
        scalar = values.ndim == 3
        if scalar:
            values = values[..., None]
        if values.shape[:3] != self.local_x_shape:
            raise ValueError("field does not match the rank-local X pencil")
        nx, ny, nz = self.shape
        channels = values.shape[-1]
        blocks: list[np.ndarray] = []
        send_counts: list[int] = []
        for destination in range(self.topology.fft_size):
            target_z = _partition(nz, self.topology.fft_size, destination)
            overlap = self._intersection(self.z_final, target_z)
            block = np.ascontiguousarray(
                values[
                    :,
                    :,
                    overlap.start - self.z_final.start : overlap.stop - self.z_final.start,
                    :,
                ]
            ).reshape(-1)
            blocks.append(block)
            send_counts.append(block.size)
        receive_counts = np.asarray(
            self.topology.fft_comm.alltoall(send_counts), dtype=np.int64
        )
        receive, displacements = self._exchange_typed(
            self.topology.fft_comm, blocks, receive_counts, values.dtype
        )
        target_z = _partition(nz, self.topology.fft_size, self.topology.fft_rank)
        result = np.empty(
            (nx, ny, target_z.stop - target_z.start, channels),
            dtype=values.dtype,
        )
        _px, py = self.topology.process_grid
        for source in range(self.topology.fft_size):
            row, column = divmod(source, py)
            source_y = _partition(ny, self.topology.process_grid[0], row)
            source_z = _partition(nz, self.topology.process_grid[1], column)
            overlap = self._intersection(source_z, target_z)
            count = int(receive_counts[source])
            block = receive[
                int(displacements[source]) : int(displacements[source]) + count
            ].reshape(
                nx,
                source_y.stop - source_y.start,
                overlap.stop - overlap.start,
                channels,
            )
            result[
                :,
                source_y,
                overlap.start - target_z.start : overlap.stop - target_z.start,
                :,
            ] = block
        return result[..., 0] if scalar else result

    def _exchange_typed(
        self,
        communicator: object,
        blocks: Iterable[np.ndarray],
        receive_counts: np.ndarray,
        dtype: np.dtype,
    ) -> tuple[np.ndarray, np.ndarray]:
        dtype = np.dtype(dtype)
        flat_blocks = [np.asarray(block, dtype=dtype).reshape(-1) for block in blocks]
        send_counts = np.asarray([block.size for block in flat_blocks], dtype=np.int64)
        send_displacements = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(send_counts[:-1]))
        )
        receive_counts = np.asarray(receive_counts, dtype=np.int64)
        receive_displacements = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(receive_counts[:-1]))
        )
        send = np.empty(int(np.sum(send_counts)), dtype=dtype)
        receive = np.empty(int(np.sum(receive_counts)), dtype=dtype)
        for block, start, count in zip(flat_blocks, send_displacements, send_counts):
            send[int(start) : int(start + count)] = block
        if int(communicator.Get_size()) == 1:
            receive[:] = send
        else:
            mpi_type = self._mpi_datatype(dtype)
            communicator.Alltoallv(
                [send, send_counts, send_displacements, mpi_type],
                [receive, receive_counts, receive_displacements, mpi_type],
            )
        return receive, receive_displacements

    def _storage(self, bands: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._arrays.get(bands)
        if cached is not None:
            return cached
        nx, ny, nz = self.shape
        local_x = self.x_initial.stop - self.x_initial.start
        local_y_initial = self.y_initial.stop - self.y_initial.start
        local_y_final = self.y_final.stop - self.y_final.start
        local_z = self.z_final.stop - self.z_final.start
        z_values = np.empty(
            (nz, local_x, local_y_initial, bands), dtype=np.complex128
        )
        y_values = np.empty(
            (ny, local_x, local_z, bands), dtype=np.complex128
        )
        x_values = np.empty(
            (nx, local_y_final, local_z, bands), dtype=np.complex128
        )
        from . import _native_fft

        plans = (
            _native_fft.NativeFFTWPlan(
                z_values,
                (nz,),
                local_x * local_y_initial * bands,
                local_x * local_y_initial * bands,
                1,
                self._flag,
                self.threads,
            ),
            _native_fft.NativeFFTWPlan(
                y_values,
                (ny,),
                local_x * local_z * bands,
                local_x * local_z * bands,
                1,
                self._flag,
                self.threads,
            ),
            _native_fft.NativeFFTWPlan(
                x_values,
                (nx,),
                local_y_final * local_z * bands,
                local_y_final * local_z * bands,
                1,
                self._flag,
                self.threads,
            ),
        )
        self._arrays[bands] = (z_values, y_values, x_values)
        self._plans[bands] = plans
        return self._arrays[bands]

    def _exchange(
        self,
        communicator: object,
        blocks: Iterable[np.ndarray],
        receive_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        flat_blocks = [np.asarray(block).reshape(-1) for block in blocks]
        send_counts = np.asarray([block.size for block in flat_blocks], dtype=np.int64)
        send_displacements = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(send_counts[:-1]))
        )
        receive_counts = np.asarray(receive_counts, dtype=np.int64)
        receive_displacements = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(receive_counts[:-1]))
        )
        send_size = int(np.sum(send_counts))
        receive_size = int(np.sum(receive_counts))
        if self._send.size < send_size:
            self._send = np.empty(send_size, dtype=np.complex128)
        if self._receive.size < receive_size:
            self._receive = np.empty(receive_size, dtype=np.complex128)
        send = self._send[:send_size]
        receive = self._receive[:receive_size]
        for block, start, count in zip(flat_blocks, send_displacements, send_counts):
            send[int(start) : int(start + count)] = block
        if int(communicator.Get_size()) == 1:
            receive[:] = send
        else:
            from mpi4py import MPI

            communicator.Alltoallv(
                [send, send_counts, send_displacements, MPI.C_DOUBLE_COMPLEX],
                [
                    receive,
                    receive_counts,
                    receive_displacements,
                    MPI.C_DOUBLE_COMPLEX,
                ],
            )
        return receive, receive_displacements

    def _z_to_y(self, z_values: np.ndarray, y_values: np.ndarray) -> None:
        _px, py = self.topology.process_grid
        bands = z_values.shape[-1]
        nx_local = z_values.shape[1]
        blocks = [
            z_values[_partition(self.shape[2], py, destination), :, :, :]
            for destination in range(py)
        ]
        nz_local = self.z_final.stop - self.z_final.start
        receive_counts = np.asarray(
            [
                nz_local
                * nx_local
                * (_partition(self.shape[1], py, source).stop
                   - _partition(self.shape[1], py, source).start)
                * bands
                for source in range(py)
            ],
            dtype=np.int64,
        )
        receive, displacements = self._exchange(
            self.topology.row_comm, blocks, receive_counts
        )
        for source in range(py):
            y_slice = _partition(self.shape[1], py, source)
            count = int(receive_counts[source])
            block = receive[
                int(displacements[source]) : int(displacements[source]) + count
            ].reshape(nz_local, nx_local, y_slice.stop - y_slice.start, bands)
            y_values[y_slice, :, :, :] = block.transpose(2, 1, 0, 3)

    def _y_to_x(self, y_values: np.ndarray, x_values: np.ndarray) -> None:
        px, _py = self.topology.process_grid
        bands = y_values.shape[-1]
        nz_local = y_values.shape[2]
        blocks = [
            y_values[_partition(self.shape[1], px, destination), :, :, :]
            for destination in range(px)
        ]
        ny_local = self.y_final.stop - self.y_final.start
        receive_counts = np.asarray(
            [
                ny_local
                * (_partition(self.shape[0], px, source).stop
                   - _partition(self.shape[0], px, source).start)
                * nz_local
                * bands
                for source in range(px)
            ],
            dtype=np.int64,
        )
        receive, displacements = self._exchange(
            self.topology.column_comm, blocks, receive_counts
        )
        for source in range(px):
            x_slice = _partition(self.shape[0], px, source)
            count = int(receive_counts[source])
            block = receive[
                int(displacements[source]) : int(displacements[source]) + count
            ].reshape(ny_local, x_slice.stop - x_slice.start, nz_local, bands)
            x_values[x_slice, :, :, :] = block.transpose(1, 0, 2, 3)

    def _x_to_y(self, x_values: np.ndarray, y_values: np.ndarray) -> None:
        px, _py = self.topology.process_grid
        bands = x_values.shape[-1]
        ny_local = x_values.shape[1]
        nz_local = x_values.shape[2]
        blocks = [
            x_values[_partition(self.shape[0], px, destination), :, :, :]
            for destination in range(px)
        ]
        nx_local = self.x_initial.stop - self.x_initial.start
        receive_counts = np.asarray(
            [
                nx_local
                * (_partition(self.shape[1], px, source).stop
                   - _partition(self.shape[1], px, source).start)
                * nz_local
                * bands
                for source in range(px)
            ],
            dtype=np.int64,
        )
        receive, displacements = self._exchange(
            self.topology.column_comm, blocks, receive_counts
        )
        for source in range(px):
            y_slice = _partition(self.shape[1], px, source)
            count = int(receive_counts[source])
            block = receive[
                int(displacements[source]) : int(displacements[source]) + count
            ].reshape(nx_local, y_slice.stop - y_slice.start, nz_local, bands)
            y_values[y_slice, :, :, :] = block.transpose(1, 0, 2, 3)

    def _y_to_z(self, y_values: np.ndarray, z_values: np.ndarray) -> None:
        _px, py = self.topology.process_grid
        bands = y_values.shape[-1]
        nx_local = y_values.shape[1]
        nz_local = y_values.shape[2]
        blocks = [
            y_values[_partition(self.shape[1], py, destination), :, :, :]
            for destination in range(py)
        ]
        ny_local = self.y_initial.stop - self.y_initial.start
        receive_counts = np.asarray(
            [
                ny_local
                * nx_local
                * (_partition(self.shape[2], py, source).stop
                   - _partition(self.shape[2], py, source).start)
                * bands
                for source in range(py)
            ],
            dtype=np.int64,
        )
        receive, displacements = self._exchange(
            self.topology.row_comm, blocks, receive_counts
        )
        for source in range(py):
            z_slice = _partition(self.shape[2], py, source)
            count = int(receive_counts[source])
            block = receive[
                int(displacements[source]) : int(displacements[source]) + count
            ].reshape(ny_local, nx_local, z_slice.stop - z_slice.start, bands)
            z_values[z_slice, :, :, :] = block.transpose(2, 1, 0, 3)

    def backward(self, reciprocal_z: np.ndarray) -> np.ndarray:
        """Return the unnormalized inverse transform in local X pencils."""
        values = np.asarray(reciprocal_z, dtype=np.complex128)
        if values.ndim == 3:
            values = values[..., None]
        expected = self.local_z_shape + (values.shape[-1],)
        if values.shape != expected:
            raise ValueError(f"Z pencil has shape {values.shape}, expected {expected}")
        bands = values.shape[-1]
        z_values, y_values, x_values = self._storage(bands)
        z_values[...] = values
        z_plan, y_plan, x_plan = self._plans[bands]
        z_plan.execute("backward")
        self._z_to_y(z_values, y_values)
        y_plan.execute("backward")
        self._y_to_x(y_values, x_values)
        x_plan.execute("backward")
        return x_values

    def forward(self, real_x: np.ndarray) -> np.ndarray:
        """Return the normalized forward transform in local Z pencils."""
        values = np.asarray(real_x, dtype=np.complex128)
        if values.ndim == 3:
            values = values[..., None]
        expected = self.local_x_shape + (values.shape[-1],)
        if values.shape != expected:
            raise ValueError(f"X pencil has shape {values.shape}, expected {expected}")
        bands = values.shape[-1]
        z_values, y_values, x_values = self._storage(bands)
        x_values[...] = values
        z_plan, y_plan, x_plan = self._plans[bands]
        x_plan.execute("forward")
        self._x_to_y(x_values, y_values)
        y_plan.execute("forward")
        self._y_to_z(y_values, z_values)
        z_plan.execute("forward")
        z_values /= prod(self.shape)
        return z_values

    def apply_local_potential(
        self,
        reciprocal_z: np.ndarray,
        potential_x: np.ndarray,
        *,
        band_tile: int = 1,
    ) -> np.ndarray:
        """Apply a real local potential with band-bounded pencil storage."""
        vectors = np.asarray(reciprocal_z, dtype=np.complex128)
        if vectors.ndim == 3:
            vectors = vectors[..., None]
        if vectors.shape[:3] != self.local_z_shape:
            raise ValueError("reciprocal coefficients do not match the local Z pencil")
        potential = np.asarray(potential_x, dtype=np.float64)
        if potential.shape != self.local_x_shape:
            raise ValueError("local potential does not match the local X pencil")
        if band_tile < 1:
            raise ValueError("band tile must be positive")
        result = np.empty_like(vectors)
        for start in range(0, vectors.shape[-1], band_tile):
            stop = min(start + band_tile, vectors.shape[-1])
            real_values = self.backward(vectors[..., start:stop])
            real_values *= potential[..., None]
            result[..., start:stop] = self.forward(real_values)
        return result


@dataclass(frozen=True)
class GammaHalfSpectrum:
    """Canonical half of a conjugate-symmetric reciprocal basis."""

    shape: tuple[int, int, int]
    indices: np.ndarray
    half_rows: np.ndarray
    partner_rows: np.ndarray
    weights: np.ndarray

    @classmethod
    def build(
        cls, indices: np.ndarray, shape: tuple[int, int, int]
    ) -> "GammaHalfSpectrum":
        miller = np.asarray(indices, dtype=np.int32)
        if miller.ndim != 2 or miller.shape[1] != 3:
            raise ValueError("Gamma indices must have shape (nplane, 3)")
        slots = miller % np.asarray(shape, dtype=np.int32)
        lookup = {tuple(map(int, slot)): row for row, slot in enumerate(slots)}
        if len(lookup) != len(miller):
            raise ValueError("Gamma basis contains duplicate FFT slots")
        selected: list[int] = []
        partners: list[int] = []
        weights: list[float] = []
        for row, slot in enumerate(slots):
            partner_slot = tuple(
                map(int, (-slot) % np.asarray(shape, dtype=np.int32))
            )
            if partner_slot not in lookup:
                raise ValueError("Gamma basis is not closed under G -> -G")
            partner = lookup[partner_slot]
            if row <= partner:
                selected.append(row)
                partners.append(partner)
                weights.append(1.0 if row == partner else 2.0)
        return cls(
            tuple(map(int, shape)),
            miller,
            np.asarray(selected, dtype=np.int32),
            np.asarray(partners, dtype=np.int32),
            np.asarray(weights, dtype=np.float64),
        )

    def compress(self, full: np.ndarray, *, check: bool = True) -> np.ndarray:
        values = np.asarray(full, dtype=np.complex128)
        if values.shape[0] != len(self.indices):
            raise ValueError("full Gamma coefficients have the wrong row count")
        half = values[self.half_rows].copy()
        if check:
            partner = values[self.partner_rows]
            error = np.max(np.abs(partner - np.conj(half))) if half.size else 0.0
            scale = max(1.0, float(np.max(np.abs(values))) if values.size else 0.0)
            if error > 1.0e-12 * scale:
                raise ValueError("Gamma coefficients are not conjugate symmetric")
        self_conjugate = self.half_rows == self.partner_rows
        half[self_conjugate] = half[self_conjugate].real
        return half

    def expand(self, half: np.ndarray) -> np.ndarray:
        values = np.asarray(half, dtype=np.complex128)
        if values.shape[0] != len(self.half_rows):
            raise ValueError("half-G coefficients have the wrong row count")
        full = np.empty((len(self.indices),) + values.shape[1:], dtype=np.complex128)
        full[self.half_rows] = values
        full[self.partner_rows] = np.conj(values)
        self_conjugate = self.half_rows == self.partner_rows
        full[self.half_rows[self_conjugate]] = values[self_conjugate].real
        return full

    def pack_two_real(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Pack two half-G real orbitals into one full complex spectrum."""
        return self.expand(first) + 1j * self.expand(second)

    def unpack_two_real(self, packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Recover two half-G orbitals after one complex forward FFT."""
        values = np.asarray(packed, dtype=np.complex128)
        if values.shape[0] != len(self.indices):
            raise ValueError("packed Gamma spectrum has the wrong row count")
        plus = values[self.half_rows]
        minus_conjugate = np.conj(values[self.partner_rows])
        first = 0.5 * (plus + minus_conjugate)
        second = (plus - minus_conjugate) / (2.0j)
        self_conjugate = self.half_rows == self.partner_rows
        first = first.copy()
        second = second.copy()
        first[self_conjugate] = first[self_conjugate].real
        second[self_conjugate] = second[self_conjugate].real
        return first, second

    def real_inner(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """QE Gamma metric for one or more columns of half-G coefficients."""
        lhs = np.asarray(left, dtype=np.complex128)
        rhs = np.asarray(right, dtype=np.complex128)
        if lhs.shape[0] != len(self.weights) or rhs.shape[0] != len(self.weights):
            raise ValueError("Gamma inner-product row count is inconsistent")
        if lhs.ndim == 1:
            lhs = lhs[:, None]
        if rhs.ndim == 1:
            rhs = rhs[:, None]
        weighted = self.weights[:, None] * np.conj(lhs)
        return np.real(weighted.T @ rhs)
