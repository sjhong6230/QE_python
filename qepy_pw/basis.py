"""Reciprocal-space basis and FFT-grid utilities."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import itertools

import numpy as np
from scipy.fft import next_fast_len
from scipy import fft as scipy_fft

from .errors import QEInputError
from .mpi import MPIContext
from .timing import TimingRegistry


@dataclass(frozen=True)
class PlaneWaveBasis:
    indices: np.ndarray
    vectors: np.ndarray
    kinetic: np.ndarray


class FFTScratchPool:
    """Process-local reusable storage for sequential distributed FFT calls."""

    def __init__(self) -> None:
        self._complex: dict[str, np.ndarray] = {}
        self._fftw_plans: dict[
            tuple[object, ...], tuple[object, ...]
        ] = {}

    def complex_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        zero: bool = False,
    ) -> np.ndarray:
        size = int(np.prod(shape))
        buffer = self._complex.get(name)
        if buffer is None or buffer.size < size:
            buffer = np.empty(size, dtype=np.complex128)
            self._complex[name] = buffer
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
        indices: np.ndarray,
        shape: tuple[int, int, int],
        *,
        backend: str = "numpy",
        threads: int = 1,
        planner_effort: str = "measure",
        use_numba: bool = False,
        mpi: MPIContext | None = None,
        timers: TimingRegistry | None = None,
        scratch_pool: FFTScratchPool | None = None,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int32)
        self.shape = tuple(shape)
        self.size = int(np.prod(shape))
        self.slots = tuple(
            np.asarray(
                self.indices[:, axis] % shape[axis],
                dtype=np.int32,
            )
            for axis in range(3)
        )
        self.linear_slots = np.asarray(
            np.ravel_multi_index(self.slots, self.shape),
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
        stick_pairs = np.column_stack((self.slots[0], self.slots[1]))
        self.sticks, self.stick_indices = np.unique(
            stick_pairs, axis=0, return_inverse=True
        )
        self.sticks = np.asarray(self.sticks, dtype=np.int32)
        self.stick_indices = np.asarray(
            self.stick_indices, dtype=np.int32
        )
        stick_counts = np.bincount(
            self.stick_indices, minlength=len(self.sticks)
        )
        stick_owners = np.zeros(len(self.sticks), dtype=np.int32)
        stick_loads = np.zeros(self.mpi.size, dtype=np.int64)
        for stick in np.argsort(-stick_counts, kind="stable"):
            owner = int(np.argmin(stick_loads))
            stick_owners[stick] = owner
            stick_loads[owner] += stick_counts[stick]
        self.stick_owners = stick_owners
        self.sticks_by_rank = [
            np.asarray(
                np.flatnonzero(stick_owners == rank),
                dtype=np.int32,
            )
            for rank in range(self.mpi.size)
        ]
        self.owned_plane_waves = (
            stick_owners[self.stick_indices] == self.mpi.rank
        )
        self.local_plane_wave_indices = np.asarray(
            np.flatnonzero(self.owned_plane_waves),
            dtype=np.int32,
        )
        self.local_slots = tuple(
            axis[self.owned_plane_waves] for axis in self.slots
        )
        local_stick_lookup = np.full(
            len(self.sticks), -1, dtype=np.int32
        )
        local_stick_lookup[self.sticks_by_rank[self.mpi.rank]] = np.arange(
            len(self.sticks_by_rank[self.mpi.rank])
        )
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
        elif self.backend not in {"numpy", "scipy"}:
            raise QEInputError(
                "py_fft_backend must be 'scipy', 'numpy', or 'pyfftw'"
            )

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
            else len(self.indices)
        )
        if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
            raise ValueError(
                "coefficients must contain the rank-local G-vector rows"
            )
        number_of_vectors = vectors.shape[1]
        if self.mpi.size == 1:
            band_slots = (slice(None),) + self.slots
            if self.backend == "pyfftw" and use_scratch:
                (
                    reciprocal_input,
                    real_grid,
                    _product,
                    _transformed,
                    backward,
                    _forward,
                ) = self._planned_buffers(number_of_vectors)
                reciprocal_input.fill(0.0)
                reciprocal_input[band_slots] = vectors.T * self.size
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
                reciprocal_input[band_slots] = vectors.T * self.size
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
            self.scratch_pool.complex_buffer(
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
                    self.scratch_pool.complex_buffer(
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
        received = self.mpi.exchange_complex(
            [
                stick_data[
                    self.mpi.slab_for_rank(self.shape[2], rank)
                ]
                for rank in range(self.mpi.size)
            ]
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
        for source, block in enumerate(received):
            source_sticks = self.sticks[self.sticks_by_rank[source]]
            if len(source_sticks):
                values = block.reshape(
                    local_z, len(source_sticks), number_of_vectors
                )
                result[
                    source_sticks[:, 0],
                    source_sticks[:, 1],
                    :,
                    :,
                ] = np.transpose(values, (1, 0, 2))
        return result

    def _real_slabs_to_sticks(
        self,
        real_data: np.ndarray,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Transpose real-space Z slabs to complete reciprocal Z-sticks."""
        number_of_vectors = real_data.shape[-1]
        send_blocks = []
        for destination in range(self.mpi.size):
            destination_sticks = self.sticks[
                self.sticks_by_rank[destination]
            ]
            send_blocks.append(
                np.transpose(
                    real_data[
                    destination_sticks[:, 0],
                    destination_sticks[:, 1],
                    :,
                    :,
                    ],
                    (1, 0, 2),
                )
            )
        received = self.mpi.exchange_complex(send_blocks)
        local_sticks = self.sticks_by_rank[self.mpi.rank]
        expected_shape = (
            self.shape[2],
            len(local_sticks),
            number_of_vectors,
        )
        if out is None:
            result = np.empty(expected_shape, dtype=complex)
        else:
            if out.shape != expected_shape:
                raise ValueError("stick scratch buffer has wrong shape")
            result = out
        for source, block in enumerate(received):
            z_slice = self.mpi.slab_for_rank(self.shape[2], source)
            result[z_slice] = block.reshape(
                z_slice.stop - z_slice.start,
                len(local_sticks),
                number_of_vectors,
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
        transformed = self._forward_fft(values, (0, 1))
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
                out=self.scratch_pool.complex_buffer(
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
        return self.scratch_pool.complex_buffer(
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
        pyfftw = self._pyfftw
        block_shape = (number_of_vectors,) + self.shape
        reciprocal_input = pyfftw.empty_aligned(
            block_shape, dtype="complex128"
        )
        real_output = pyfftw.empty_aligned(
            block_shape, dtype="complex128"
        )
        product_input = pyfftw.empty_aligned(
            block_shape, dtype="complex128"
        )
        reciprocal_output = pyfftw.empty_aligned(
            block_shape, dtype="complex128"
        )
        backward = pyfftw.FFTW(
            reciprocal_input,
            real_output,
            axes=(1, 2, 3),
            direction="FFTW_BACKWARD",
            flags=(self.planner_flag,),
            threads=self.threads,
        )
        forward = pyfftw.FFTW(
            product_input,
            reciprocal_output,
            axes=(1, 2, 3),
            direction="FFTW_FORWARD",
            flags=(self.planner_flag,),
            threads=self.threads,
        )
        buffers = (
            reciprocal_input,
            real_output,
            product_input,
            reciprocal_output,
            backward,
            forward,
        )
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
            else len(self.indices)
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
                reciprocal_input[(slice(None),) + self.slots] = vectors.T * self.size
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
                result = transformed[(slice(None),) + self.slots].T
        else:
            (
                reciprocal_input,
                real_wavefunctions,
                product,
                transformed,
                backward,
                forward,
            ) = self._planned_buffers(number_of_vectors)
            if self.use_numba:
                self._jit.scatter_scaled(
                    reciprocal_input.reshape(number_of_vectors, self.size).T,
                    self.linear_slots,
                    vectors,
                    float(self.size),
                )
            else:
                reciprocal_input.fill(0.0)
                reciprocal_input[(slice(None),) + self.slots] = vectors.T * self.size
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
                    real_wavefunctions.reshape(
                        number_of_vectors, self.size
                    ).T,
                    product.reshape(number_of_vectors, self.size).T,
                )
            else:
                np.multiply(
                    real_potential[None, ...],
                    real_wavefunctions,
                    out=product,
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
                    transformed.reshape(number_of_vectors, self.size).T,
                    self.linear_slots,
                    result,
                    1.0 / self.size,
                )
            else:
                result = transformed[(slice(None),) + self.slots].T / self.size
        return result[:, 0] if was_vector else result


def make_basis(reciprocal: np.ndarray, k_crystal: np.ndarray, ecut_ry: float) -> PlaneWaveBasis:
    ecut_ha = 0.5 * ecut_ry
    # For a non-orthogonal reciprocal basis, large integer coefficients can
    # cancel.  The shortest row norm is therefore not a safe Miller-index
    # bound.  Since ||n B|| >= sigma_min(B)||n||, the smallest singular value
    # provides a rigorous search radius for every vector satisfying the
    # kinetic cutoff.
    minimum_stretch = float(np.linalg.svd(reciprocal, compute_uv=False)[-1])
    bound = int(
        np.ceil(
            (
                np.sqrt(2.0 * ecut_ha)
                + np.linalg.norm(k_crystal @ reciprocal)
            )
            / minimum_stretch
        )
    ) + 1
    integer = np.array(list(itertools.product(range(-bound, bound + 1), repeat=3)), dtype=int)
    # QE ggen first orders the global reciprocal vectors by |G|^2, with
    # their original Miller-grid index breaking exact degeneracies. gk_sort
    # subsequently orders the retained vectors by |G+k|^2 and carries that
    # global-G index through ties. Starting directly from Miller order gives
    # a different deterministic atomic+random perturbation at nonzero k.
    global_vectors = integer @ reciprocal
    global_g2 = np.einsum("ij,ij->i", global_vectors, global_vectors)
    global_order = _qe_energy_order(global_g2)
    integer = integer[global_order]
    vectors = (integer + k_crystal) @ reciprocal
    kinetic = 0.5 * np.einsum("ij,ij->i", vectors, vectors)
    keep = kinetic <= ecut_ha + 1.0e-12
    integer, vectors, kinetic = integer[keep], vectors[keep], kinetic[keep]
    # Stable sorting retains QE's global-G precedence for degenerate G+k.
    order = _qe_energy_order(kinetic)
    if not len(order):
        raise QEInputError("ecutwfc produces an empty plane-wave basis")
    return PlaneWaveBasis(
        np.asarray(integer[order], dtype=np.int32),
        vectors[order],
        kinetic[order],
    )


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
