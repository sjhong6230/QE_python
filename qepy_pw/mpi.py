"""MPI runtime used by the mandatory distributed Cython FFT path."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys

import numpy as np

from .errors import QEInputError


@dataclass
class MPIContext:
    """Process metadata and collective helpers with a serial fallback."""

    comm: object | None = None
    rank: int = 0
    size: int = 1
    _exchange_send_buffer: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.complex128),
        repr=False,
    )
    _exchange_recv_buffer: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.complex128),
        repr=False,
    )
    _shared_comm: object | None = field(default=None, repr=False)
    _shared_windows: list[object] = field(default_factory=list, repr=False)

    @classmethod
    def world(cls) -> "MPIContext":
        try:
            import mpi4py

            if "mpi4py.MPI" not in sys.modules:
                # Native OpenMP/FFTW workers never call MPI. FUNNELED is the
                # strongest level required and avoids THREAD_MULTIPLE state in
                # every rank on MPI implementations that distinguish them.
                mpi4py.rc.threads = True
                mpi4py.rc.thread_level = "funneled"
                if "OMPI_COMM_WORLD_SIZE" in os.environ:
                    # Open MPI's UCX OSC probe emits warnings under WSL for a
                    # node-local shared window; the dedicated sm component is
                    # the appropriate implementation for COMM_TYPE_SHARED.
                    os.environ.setdefault("OMPI_MCA_osc", "sm")
            from mpi4py import MPI
        except ImportError:
            launched = any(
                int(os.environ.get(name, "1")) > 1
                for name in (
                    "OMPI_COMM_WORLD_SIZE",
                    "PMI_SIZE",
                    "PMIX_SIZE",
                    "MV2_COMM_WORLD_SIZE",
                )
                if os.environ.get(name, "").isdigit()
            )
            if launched:
                raise QEInputError(
                    "MPI execution requires the mandatory mpi4py dependency; "
                    "reinstall qepy-pw with pip"
                )
            return cls()
        comm = MPI.COMM_WORLD
        return cls(comm, int(comm.Get_rank()), int(comm.Get_size()))

    def _node_comm(self):
        if self.size == 1:
            return None
        if self._shared_comm is None:
            from mpi4py import MPI

            self._shared_comm = self.comm.Split_type(
                MPI.COMM_TYPE_SHARED, key=self.rank
            )
        return self._shared_comm

    @property
    def shared_size(self) -> int:
        node_comm = self._node_comm()
        return 1 if node_comm is None else int(node_comm.Get_size())

    def shared_readonly(self, value: np.ndarray) -> np.ndarray:
        """Keep one physical copy of an identical read-only array per node."""
        array = np.ascontiguousarray(value)
        node_comm = self._node_comm()
        if node_comm is None:
            array.flags.writeable = False
            return array
        from mpi4py import MPI

        local_rank = int(node_comm.Get_rank())
        metadata = (
            (array.shape, array.dtype.str) if local_rank == 0 else None
        )
        shape, dtype_string = node_comm.bcast(metadata, root=0)
        dtype = np.dtype(dtype_string)
        number_of_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        window = MPI.Win.Allocate_shared(
            number_of_bytes if local_rank == 0 else 0,
            dtype.itemsize,
            comm=node_comm,
        )
        shared_buffer, _disp_unit = window.Shared_query(0)
        shared = np.ndarray(shape, dtype=dtype, buffer=shared_buffer)
        if local_rank == 0:
            shared[...] = array
        node_comm.Barrier()
        shared.flags.writeable = False
        self._shared_windows.append(window)
        return shared

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def slab(self, length: int) -> slice:
        start = length * self.rank // self.size
        stop = length * (self.rank + 1) // self.size
        return slice(start, stop)

    def slab_for_rank(self, length: int, rank: int) -> slice:
        start = length * rank // self.size
        stop = length * (rank + 1) // self.size
        return slice(start, stop)

    def sum_array(self, value: np.ndarray) -> np.ndarray:
        array = np.ascontiguousarray(value)
        if self.size == 1:
            return array
        from mpi4py import MPI

        self.comm.Allreduce(MPI.IN_PLACE, array)
        return array

    def gather_z_slabs(
        self,
        local_slab: np.ndarray,
        shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Collect disjoint real-space Z slabs without a full-grid reduction."""
        local = np.asarray(local_slab)
        if local.shape != (
            shape[0],
            shape[1],
            self.slab(shape[2]).stop - self.slab(shape[2]).start,
        ):
            raise ValueError("local array does not match this rank's Z slab")
        if self.size == 1:
            return np.ascontiguousarray(local)
        # Z-major packing makes every rank's slab one contiguous Allgatherv
        # block. In the normal FFT decomposition the slabs are disjoint, so
        # QE does not need an intra-band-group sum here either.
        send = np.ascontiguousarray(local.transpose(2, 0, 1)).ravel()
        plane_points = shape[0] * shape[1]
        counts = np.asarray(
            [
                (
                    self.slab_for_rank(shape[2], rank).stop
                    - self.slab_for_rank(shape[2], rank).start
                )
                * plane_points
                for rank in range(self.size)
            ],
            dtype=np.int64,
        )
        displacements = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(counts[:-1]))
        )
        recv = np.empty(int(np.prod(shape)), dtype=local.dtype)
        from mpi4py import MPI

        mpi_type = MPI.DOUBLE if local.dtype == np.float64 else MPI.FLOAT
        self.comm.Allgatherv(
            send,
            [recv, counts, displacements, mpi_type],
        )
        return recv.reshape(shape[2], shape[0], shape[1]).transpose(1, 2, 0)

    def gather_z_slabs_root(
        self,
        local_slab: np.ndarray,
        shape: tuple[int, int, int],
        root: int = 0,
    ) -> np.ndarray:
        """Gather a distributed Z-slab field only on ``root``."""
        if self.size == 1:
            return np.ascontiguousarray(local_slab)
        send = np.ascontiguousarray(
            np.asarray(local_slab).transpose(2, 0, 1)
        ).ravel()
        plane_points = shape[0] * shape[1]
        counts = np.asarray(
            [
                (
                    self.slab_for_rank(shape[2], rank).stop
                    - self.slab_for_rank(shape[2], rank).start
                )
                * plane_points
                for rank in range(self.size)
            ],
            dtype=np.int64,
        )
        displacements = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(counts[:-1]))
        )
        recv = (
            np.empty(int(np.prod(shape)), dtype=send.dtype)
            if self.rank == root
            else None
        )
        from mpi4py import MPI

        mpi_type = MPI.DOUBLE if send.dtype == np.float64 else MPI.FLOAT
        self.comm.Gatherv(
            send,
            [recv, counts, displacements, mpi_type]
            if self.rank == root
            else None,
            root=root,
        )
        if self.rank != root:
            return np.empty((0,), dtype=send.dtype)
        return recv.reshape(shape[2], shape[0], shape[1]).transpose(1, 2, 0)

    def scatter_z_slabs_root(
        self,
        full_grid: np.ndarray | None,
        shape: tuple[int, int, int],
        root: int = 0,
    ) -> np.ndarray:
        """Scatter a root-owned real-space grid into disjoint Z slabs."""
        if self.size == 1:
            if full_grid is None:
                raise ValueError("root grid is required")
            return np.ascontiguousarray(full_grid)
        plane_points = shape[0] * shape[1]
        counts = np.asarray(
            [
                (self.slab_for_rank(shape[2], rank).stop
                 - self.slab_for_rank(shape[2], rank).start) * plane_points
                for rank in range(self.size)
            ],
            dtype=np.int64,
        )
        displacements = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(counts[:-1]))
        )
        local = np.empty(int(counts[self.rank]), dtype=np.float64)
        send = None
        if self.rank == root:
            if full_grid is None or np.asarray(full_grid).shape != shape:
                raise ValueError("root grid does not match the FFT shape")
            send = np.ascontiguousarray(
                np.asarray(full_grid, dtype=np.float64).transpose(2, 0, 1)
            ).ravel()
        from mpi4py import MPI

        self.comm.Scatterv(
            [send, counts, displacements, MPI.DOUBLE]
            if self.rank == root else None,
            local,
            root=root,
        )
        nz = self.slab(shape[2]).stop - self.slab(shape[2]).start
        return local.reshape(nz, shape[0], shape[1]).transpose(1, 2, 0)

    def gather_flat_chunks(
        self, local_chunk: np.ndarray, total_size: int
    ) -> np.ndarray:
        """All-gather contiguous pieces made with ``slab(total_size)``."""
        local = np.ascontiguousarray(local_chunk).ravel()
        if self.size == 1:
            return local
        counts = np.asarray(
            [
                self.slab_for_rank(total_size, rank).stop
                - self.slab_for_rank(total_size, rank).start
                for rank in range(self.size)
            ],
            dtype=np.int64,
        )
        displacements = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(counts[:-1]))
        )
        result = np.empty(total_size, dtype=local.dtype)
        from mpi4py import MPI

        mpi_type = (
            MPI.C_DOUBLE_COMPLEX
            if np.issubdtype(local.dtype, np.complexfloating)
            else MPI.DOUBLE
        )
        self.comm.Allgatherv(
            local, [result, counts, displacements, mpi_type]
        )
        return result

    def gather_indexed_rows_root(
        self,
        local_values: np.ndarray,
        global_rows: np.ndarray,
        total_rows: int,
        root: int = 0,
    ) -> np.ndarray:
        """Gather disjoint, non-contiguous matrix rows only on ``root``."""
        values = np.asarray(local_values)
        rows = np.asarray(global_rows, dtype=np.int64)
        if values.ndim != 2 or values.shape[0] != len(rows):
            raise ValueError("local matrix rows do not match global indices")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("global row indices contain duplicates")
        if np.any(rows < 0) or np.any(rows >= total_rows):
            raise ValueError("global row index is outside the matrix")
        if self.size == 1:
            if len(rows) != total_rows:
                raise ValueError("local rows do not cover the global matrix")
            result = np.empty(
                (total_rows, values.shape[1]), dtype=values.dtype
            )
            result[rows] = values
            return result

        from mpi4py import MPI

        local_count = len(rows)
        gathered_counts = self.comm.gather(local_count, root=root)
        if self.rank == root:
            row_counts = np.asarray(gathered_counts, dtype=np.int64)
            row_displacements = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.cumsum(row_counts[:-1]),
                )
            )
            gathered_rows = np.empty(int(np.sum(row_counts)), dtype=np.int64)
        else:
            row_counts = row_displacements = gathered_rows = None
        self.comm.Gatherv(
            np.ascontiguousarray(rows),
            [gathered_rows, row_counts, row_displacements, MPI.LONG_LONG]
            if self.rank == root
            else None,
            root=root,
        )

        bands = values.shape[1]
        if self.rank == root:
            value_counts = row_counts * bands
            value_displacements = row_displacements * bands
            gathered_values = np.empty(
                int(np.sum(value_counts)), dtype=values.dtype
            )
        else:
            value_counts = value_displacements = gathered_values = None
        mpi_type = (
            MPI.C_DOUBLE_COMPLEX
            if values.dtype == np.complex128
            else MPI.C_FLOAT_COMPLEX
        )
        self.comm.Gatherv(
            np.ascontiguousarray(values).ravel(),
            [
                gathered_values,
                value_counts,
                value_displacements,
                mpi_type,
            ]
            if self.rank == root
            else None,
            root=root,
        )
        if self.rank != root:
            return np.empty((0, bands), dtype=values.dtype)
        if len(gathered_rows) != total_rows or len(np.unique(gathered_rows)) != total_rows:
            raise ValueError("gathered rows do not cover the global matrix")
        result = np.empty((total_rows, bands), dtype=values.dtype)
        result[gathered_rows] = gathered_values.reshape(-1, bands)
        return result

    def sum_scalar(self, value: float) -> float:
        if self.size == 1:
            return float(value)
        return float(self.comm.allreduce(float(value)))

    def max_scalar(self, value: float) -> float:
        if self.size == 1:
            return float(value)
        from mpi4py import MPI

        return float(self.comm.allreduce(float(value), op=MPI.MAX))

    def exchange_complex(
        self, send_blocks: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Exchange packed complex blocks and return one flat block per source."""
        if self.size == 1:
            return [np.ascontiguousarray(send_blocks[0]).ravel()]
        if len(send_blocks) != self.size:
            raise ValueError("one MPI exchange block is required per rank")
        send_counts = np.asarray(
            [np.asarray(block).size for block in send_blocks],
            dtype=np.int64,
        )
        recv_counts = np.asarray(
            self.comm.alltoall(send_counts.tolist()), dtype=np.int64
        )
        send_displacements = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(send_counts[:-1]))
        )
        recv_displacements = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(recv_counts[:-1]))
        )
        send_size = int(np.sum(send_counts))
        if self._exchange_send_buffer.size < send_size:
            self._exchange_send_buffer = np.empty(
                send_size, dtype=np.complex128
            )
        send_buffer = self._exchange_send_buffer[:send_size]
        for block, displacement, count in zip(
            send_blocks, send_displacements, send_counts
        ):
            start = int(displacement)
            stop = start + int(count)
            send_buffer[start:stop] = np.asarray(
                block, dtype=np.complex128
            ).reshape(-1)
        recv_size = int(np.sum(recv_counts))
        if self._exchange_recv_buffer.size < recv_size:
            self._exchange_recv_buffer = np.empty(
                recv_size, dtype=np.complex128
            )
        recv_buffer = self._exchange_recv_buffer[:recv_size]
        from mpi4py import MPI

        self.comm.Alltoallv(
            [
                send_buffer,
                send_counts,
                send_displacements,
                MPI.C_DOUBLE_COMPLEX,
            ],
            [
                recv_buffer,
                recv_counts,
                recv_displacements,
                MPI.C_DOUBLE_COMPLEX,
            ],
        )
        return [
            recv_buffer[
                int(displacement) : int(displacement + count)
            ]
            for displacement, count in zip(
                recv_displacements, recv_counts
            )
        ]

    def complex_exchange_buffers(
        self, send_size: int, recv_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return grow-only buffers for a planned complex exchange."""
        if self._exchange_send_buffer.size < send_size:
            self._exchange_send_buffer = np.empty(
                send_size, dtype=np.complex128
            )
        if self._exchange_recv_buffer.size < recv_size:
            self._exchange_recv_buffer = np.empty(
                recv_size, dtype=np.complex128
            )
        return (
            self._exchange_send_buffer[:send_size],
            self._exchange_recv_buffer[:recv_size],
        )

    @property
    def exchange_buffer_bytes(self) -> int:
        """Bytes retained by the grow-only complex collective buffers."""
        return int(
            self._exchange_send_buffer.nbytes
            + self._exchange_recv_buffer.nbytes
        )

    def exchange_complex_planned(
        self,
        send_buffer: np.ndarray,
        send_counts: np.ndarray,
        send_displacements: np.ndarray,
        recv_counts: np.ndarray,
        recv_displacements: np.ndarray,
        *,
        recv_buffer: np.ndarray | None = None,
    ) -> np.ndarray:
        """Execute one descriptor-driven complex Alltoallv.

        Counts and displacements are prepared by the FFT descriptor, so the
        hot path needs neither a Python-object count exchange nor temporary
        block lists.
        """
        send = np.asarray(send_buffer, dtype=np.complex128).reshape(-1)
        send_counts = np.asarray(send_counts, dtype=np.int64)
        send_displacements = np.asarray(
            send_displacements, dtype=np.int64
        )
        recv_counts = np.asarray(recv_counts, dtype=np.int64)
        recv_displacements = np.asarray(
            recv_displacements, dtype=np.int64
        )
        # Planned displacement vectors are prefix sums.  Reading their last
        # entry avoids two NumPy reductions in every distributed FFT
        # transpose (tens of thousands of calls on an unreduced k mesh).
        send_size = (
            int(send_displacements[-1] + send_counts[-1])
            if send_counts.size
            else 0
        )
        recv_size = (
            int(recv_displacements[-1] + recv_counts[-1])
            if recv_counts.size
            else 0
        )
        if send.size != send_size:
            raise ValueError("planned MPI send buffer has the wrong size")
        if recv_buffer is None:
            _scratch_send, recv = self.complex_exchange_buffers(
                0, recv_size
            )
        else:
            recv = np.asarray(
                recv_buffer, dtype=np.complex128
            ).reshape(-1)
            if recv.size != recv_size:
                raise ValueError(
                    "planned MPI receive buffer has the wrong size"
                )
        if self.size == 1:
            recv[:] = send
            return recv
        from mpi4py import MPI

        self.comm.Alltoallv(
            [
                send,
                send_counts,
                send_displacements,
                MPI.C_DOUBLE_COMPLEX,
            ],
            [
                recv,
                recv_counts,
                recv_displacements,
                MPI.C_DOUBLE_COMPLEX,
            ],
        )
        return recv

    def barrier(self) -> None:
        if self.size > 1:
            self.comm.Barrier()

    def broadcast(self, value: object, root: int = 0) -> object:
        if self.size == 1:
            return value
        return self.comm.bcast(value, root=root)
