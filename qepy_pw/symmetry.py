"""Small crystallographic symmetry engine for scalar periodic calculations.

Fractional coordinates are row vectors. A space-group operation is represented
as ``x' = x @ matrix + translation`` modulo lattice vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import TYPE_CHECKING
import warnings

import numpy as np

if TYPE_CHECKING:
    from .basis import LocalPotentialWorkspace
    from .mpi import MPIContext


@dataclass(frozen=True)
class SymmetryOperation:
    matrix: np.ndarray
    translation: np.ndarray


class DensitySymmetrizer:
    """Memory-bounded scalar-density projection for one FFT grid.

    A full flat permutation for every space-group operation scales as
    ``O(nsym * nrxx)`` and can exceed the rest of a symmetry-reduced SCF
    calculation.  Instead retain only one-dimensional integer contributions
    to the transformed coordinates and build one flat permutation at a time
    in two reusable int32 grid buffers.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        operations: tuple[SymmetryOperation, ...],
        mpi: "MPIContext | None" = None,
    ) -> None:
        self.shape = tuple(int(size) for size in shape)
        self.operations = operations
        self.mpi = mpi
        distributed_grid = mpi is not None and mpi.size > 1
        local_z = (
            mpi.slab(self.shape[2])
            if distributed_grid
            else slice(0, self.shape[2])
        )
        self.local_shape = (
            self.shape[0],
            self.shape[1],
            local_z.stop - local_z.start,
        )
        target_indices = (
            np.arange(self.shape[0], dtype=np.int32),
            np.arange(self.shape[1], dtype=np.int32),
            np.arange(
                0 if distributed_grid else local_z.start,
                self.shape[2] if distributed_grid else local_z.stop,
                dtype=np.int32,
            ),
        )
        self._operation_terms: list[
            tuple[tuple[int, tuple[np.ndarray, ...], int], ...]
        ] = []
        self._identity_operations: list[bool] = []
        for operation in operations if len(operations) > 1 else ():
            transformed_axes = []
            for axis, destination_size in enumerate(self.shape):
                translated = operation.translation[axis] * destination_size
                rounded_translation = int(np.rint(translated))
                if not np.isclose(
                    translated,
                    rounded_translation,
                    rtol=0.0,
                    atol=1.0e-7,
                ):
                    raise ValueError(
                        "symmetry operation is incompatible with FFT grid"
                    )
                contributions = []
                for source_axis, source_size in enumerate(self.shape):
                    scaled = (
                        target_indices[source_axis].astype(float)
                        * operation.matrix[source_axis, axis]
                        * destination_size
                        / source_size
                    )
                    rounded = np.rint(scaled)
                    if not np.allclose(
                        scaled, rounded, rtol=0.0, atol=1.0e-7
                    ):
                        raise ValueError(
                            "symmetry operation is incompatible with FFT grid"
                        )
                    contributions.append(rounded.astype(np.int32))
                transformed_axes.append(
                    (
                        rounded_translation,
                        tuple(contributions),
                        destination_size,
                    )
                )
            self._operation_terms.append(tuple(transformed_axes))
            self._identity_operations.append(
                np.array_equal(operation.matrix, np.eye(3, dtype=int))
                and np.allclose(
                    operation.translation % 1.0,
                    0.0,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
        buffer_shape = (
            self.shape
            if len(operations) > 1 and distributed_grid
            else self.local_shape
            if len(operations) > 1
            else (0,)
        )
        self._mapping = np.empty(buffer_shape, dtype=np.int32)
        self._slot = np.empty(buffer_shape, dtype=np.int32)
        self._permuted = np.empty(
            int(np.prod(self.shape))
            if len(operations) > 1
            else 0,
            dtype=float,
        )

    @staticmethod
    def _fill_slot(
        terms: tuple[int, tuple[np.ndarray, ...], int],
        output: np.ndarray,
    ) -> None:
        """Materialize one transformed coordinate in a reusable buffer."""
        offset, contributions, size = terms
        output.fill(offset)
        for axis, contribution in enumerate(contributions):
            broadcast_shape = [1, 1, 1]
            broadcast_shape[axis] = len(contribution)
            np.add(
                output,
                contribution.reshape(broadcast_shape),
                out=output,
            )
        np.remainder(output, size, out=output)

    def _linear_xy_and_z(
        self,
        terms: tuple[tuple[int, tuple[np.ndarray, ...], int], ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return reusable flat XY indices and transformed global Z slots."""
        self._fill_slot(terms[0], self._mapping)
        self._mapping *= self.shape[1]
        self._fill_slot(terms[1], self._slot)
        self._mapping += self._slot
        self._fill_slot(terms[2], self._slot)
        return self._mapping.ravel(), self._slot.ravel()

    @property
    def mapping_bytes(self) -> int:
        return int(
            self._mapping.nbytes
            + self._slot.nbytes
            + self._permuted.nbytes
            + sum(
                contribution.nbytes
                for operation in self._operation_terms
                for _offset, contributions, _size in operation
                for contribution in contributions
            )
        )

    def apply(self, density: np.ndarray) -> np.ndarray:
        array = np.asarray(density, dtype=float)
        if (
            self.mpi is not None
            and self.mpi.size > 1
            and array.shape
            == (
                self.shape[0],
                self.shape[1],
                self.mpi.slab(self.shape[2]).stop
                - self.mpi.slab(self.shape[2]).start,
            )
        ):
            if len(self.operations) <= 1:
                return array
            full_density = self.mpi.gather_z_slabs(array, self.shape)
            source = full_density.ravel()
            averaged = np.zeros_like(source)
            for operation_index in range(
                self.mpi.rank,
                len(self._operation_terms),
                self.mpi.size,
            ):
                identity = self._identity_operations[operation_index]
                terms = self._operation_terms[operation_index]
                if identity:
                    averaged += source
                    continue
                xy, z_slots = self._linear_xy_and_z(terms)
                xy *= self.shape[2]
                xy += z_slots
                np.take(source, xy, out=self._permuted)
                averaged += self._permuted
            averaged = self.mpi.sum_array(averaged)
            averaged /= len(self.operations)
            local_z = self.mpi.slab(self.shape[2])
            return np.ascontiguousarray(
                averaged.reshape(self.shape)[:, :, local_z]
            )
        if array.shape != self.shape:
            raise ValueError("density does not match symmetry FFT grid")
        if len(self.operations) <= 1:
            return array
        source = array.ravel()
        averaged = np.zeros(source.size, dtype=float)
        for identity, terms in zip(
            self._identity_operations, self._operation_terms
        ):
            if identity:
                averaged += source
                continue
            xy, z_slots = self._linear_xy_and_z(terms)
            xy *= self.shape[2]
            xy += z_slots
            np.take(source, xy, out=self._permuted)
            averaged += self._permuted
        result = averaged.reshape(self.shape)
        if self.mpi is not None and self.mpi.size > 1:
            result = self.mpi.sum_array(result)
        result /= len(self.operations)
        return result


class ReciprocalDensitySymmetrizer:
    """QE-style scalar-density symmetrization on reciprocal-space stars.

    In MPI runs QE does not permute a replicated real-space FFT grid.  It
    first transforms ``rho(r)`` to the distributed compact ``rho(G)`` list,
    redistributes complete radial shells with ``MPI_Alltoallv``, applies the
    serial star projector on each shell owner, and reverses the exchange.
    This class follows that data flow while retaining the FFT stick ownership
    used by :class:`LocalPotentialWorkspace`.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        operations: tuple[SymmetryOperation, ...],
        reciprocal_indices: np.ndarray,
        reciprocal_g2: np.ndarray,
        workspace: "LocalPotentialWorkspace",
        mpi: "MPIContext",
    ) -> None:
        self.shape = tuple(int(size) for size in shape)
        self.operations = operations
        self.workspace = workspace
        self.mpi = mpi
        indices = np.asarray(reciprocal_indices, dtype=np.int32)
        g2 = np.asarray(reciprocal_g2, dtype=float)
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("reciprocal indices must have shape (ng, 3)")
        if g2.shape != (len(indices),):
            raise ValueError("reciprocal G squared values do not match indices")
        if workspace.shape != self.shape:
            raise ValueError("density FFT workspace has the wrong shape")
        if workspace.plane_waves != len(indices):
            raise ValueError("density FFT workspace has the wrong G-vector list")

        # Keep the already-fast nosym path free of reciprocal-star setup.
        # The object still presents the same accounting interface, while
        # apply() returns the caller's local slab without an FFT or collective.
        if len(operations) <= 1:
            self._set_empty_exchange()
            self._star_offsets = np.zeros(1, dtype=np.int32)
            self._star_members = np.empty(0, dtype=np.uint16)
            return

        if mpi.size == 1:
            # A serial projection operates directly on the compact rho(G)
            # vector.  MPI pack/shell/return buffers used to be allocated
            # here as well, even though apply() never touched them.  Besides
            # two complex vectors this also retained several full-size owner
            # and permutation tables.  Keep only the actual star projector.
            self._set_empty_exchange()
            (
                self._star_offsets,
                self._star_members,
                self._star_weight_table,
                self._star_weight_indices,
                self._star_fill_table,
                self._star_fill_indices,
            ) = self._build_star_metadata(
                indices, distributed=False
            )
            return

        # QE estimates equally populated spherical slices from Gcut with
        # gcut(i) = gcutm * (i/nproc)^(2/3).  Quantizing |G|^2 before
        # assigning the owner keeps a roundoff-split degenerate shell on one
        # rank, as required for a complete symmetry star.
        maximum_g2 = float(np.max(g2))
        radial_boundaries = maximum_g2 * (
            np.arange(1, mpi.size, dtype=float) / float(mpi.size)
        ) ** (2.0 / 3.0)
        radial_tolerance = max(maximum_g2, 1.0) * 1.0e-11
        radial_keys = np.rint(g2 / radial_tolerance).astype(np.int64)
        boundary_keys = np.rint(
            radial_boundaries / radial_tolerance
        ).astype(np.int64)
        shell_owner = np.asarray(
            np.searchsorted(boundary_keys, radial_keys, side="right"),
            dtype=np.int32,
        )
        # The distributed FFT owns complete (Gx,Gy) sticks.  This is the
        # original QE-like distribution to which sym_rho must return.
        fft_owner = np.asarray(
            workspace.stick_owners[workspace.stick_indices],
            dtype=np.int32,
        )
        local_global_rows = np.asarray(
            workspace.local_plane_wave_indices, dtype=np.int32
        )
        local_destinations = shell_owner[local_global_rows]
        send_positions = [
            np.flatnonzero(local_destinations == rank).astype(np.int32)
            for rank in range(mpi.size)
        ]
        self._send_order = (
            np.concatenate(send_positions)
            if send_positions
            else np.empty(0, dtype=np.int32)
        )
        self._send_counts = np.asarray(
            [len(rows) for rows in send_positions], dtype=np.int64
        )
        self._send_displacements = self._displacements(self._send_counts)
        self._recv_counts = np.asarray(
            [
                np.count_nonzero(
                    (fft_owner == source_rank)
                    & (shell_owner == mpi.rank)
                )
                for source_rank in range(mpi.size)
            ],
            dtype=np.int64,
        )
        self._recv_displacements = self._displacements(self._recv_counts)
        received_global_rows = np.concatenate(
            [
                np.flatnonzero(
                    (fft_owner == source_rank)
                    & (shell_owner == mpi.rank)
                ).astype(np.int32)
                for source_rank in range(mpi.size)
            ]
        )

        # Build the G->row map only after shell redistribution.  In MPI this
        # is one rank's radial slice, not the complete charge sphere.
        (
            self._star_offsets,
            self._star_members,
            self._star_weight_table,
            self._star_weight_indices,
            self._star_fill_table,
            self._star_fill_indices,
        ) = self._build_star_metadata(
            indices[received_global_rows], distributed=True
        )
        # The FFT transpose and sym_rho are sequential.  Reuse MPIContext's
        # existing grow-only exchange pair in apply() instead of retaining a
        # second pair of complex charge vectors solely for symmetry.
        self._packed_send = np.empty(0, dtype=np.complex128)
        self._shell_coefficients = np.empty(0, dtype=np.complex128)
        self._packed_return = self._packed_send
        self._packed_size = len(local_global_rows)
        self._shell_size = len(received_global_rows)

    def _set_empty_exchange(self) -> None:
        """Install the zero-storage descriptor used outside distributed MPI."""
        self._send_order = np.empty(0, dtype=np.int32)
        self._send_counts = np.empty(0, dtype=np.int64)
        self._send_displacements = np.empty(0, dtype=np.int64)
        self._recv_counts = np.empty(0, dtype=np.int64)
        self._recv_displacements = np.empty(0, dtype=np.int64)
        self._packed_send = np.empty(0, dtype=np.complex128)
        self._shell_coefficients = np.empty(0, dtype=np.complex128)
        self._packed_return = self._packed_send
        self._packed_size = 0
        self._shell_size = 0
        self._star_weight_table = np.empty(0, dtype=np.complex128)
        self._star_weight_indices = np.empty(0, dtype=np.uint8)
        self._star_fill_table = np.empty(0, dtype=np.complex128)
        self._star_fill_indices = np.empty(0, dtype=np.uint8)

    def _build_star_metadata(
        self, local_indices: np.ndarray, *, distributed: bool
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Build compact star tables without retaining per-star duplicates."""
        local_indices = np.asarray(local_indices, dtype=np.int32)
        number_of_rows = len(local_indices)
        member_dtype = (
            np.uint16
            if number_of_rows <= np.iinfo(np.uint16).max + 1
            else np.uint32
        )

        # Form flat FFT keys in one int64 vector.  np.mod(G, shape) followed
        # by ravel_multi_index formerly created an additional (N_G, 3) int64
        # array during symmetry initialization.
        nx, ny, nz = self.shape
        local_keys = np.remainder(local_indices[:, 0], nx).astype(np.int64)
        local_keys *= ny
        local_keys += np.remainder(local_indices[:, 1], ny)
        local_keys *= nz
        local_keys += np.remainder(local_indices[:, 2], nz)
        lookup = np.full(nx * ny * nz, -1, dtype=np.int32)
        lookup[local_keys] = np.arange(number_of_rows, dtype=np.int32)

        done = np.zeros(number_of_rows, dtype=bool)
        offsets = np.empty(number_of_rows + 1, dtype=np.int32)
        members = np.empty(number_of_rows, dtype=member_dtype)
        weights_array = np.empty(number_of_rows, dtype=np.complex128)
        fills_array = np.empty(number_of_rows, dtype=np.complex128)
        offsets[0] = 0
        number_of_stars = 0
        number_of_members = 0
        nsym = len(self.operations)
        shape_array = np.asarray(self.shape, dtype=np.int64)

        for representative_row, representative in enumerate(local_indices):
            if done[representative_row]:
                continue
            member_rows: list[int] = []
            member_slots: dict[int, int] = {}
            weights: list[complex] = []
            fill_factors: list[complex] = []
            for operation in self.operations:
                source = representative @ operation.matrix.T
                source_slot = np.mod(source, shape_array)
                source_key = int(
                    (source_slot[0] * ny + source_slot[1]) * nz
                    + source_slot[2]
                )
                source_row = int(lookup[source_key])
                if source_row < 0:
                    message = (
                        "a symmetry star was split between MPI ranks"
                        if distributed
                        else "symmetry-related G vector is absent from density cutoff"
                    )
                    raise ValueError(message)
                phase = np.exp(
                    -2j
                    * np.pi
                    * float(representative @ operation.translation)
                )
                member_slot = member_slots.get(source_row)
                if member_slot is None:
                    member_slot = len(member_rows)
                    member_slots[source_row] = member_slot
                    member_rows.append(source_row)
                    weights.append(0.0j)
                    # If H = G S^T, invariance gives rho(H) =
                    # rho(G) exp(+2 pi i G.t_S).
                    fill_factors.append(np.conjugate(phase))
                weights[member_slot] += phase / nsym

            star_size = len(member_rows)
            stop = number_of_members + star_size
            members[number_of_members:stop] = member_rows
            weights_array[number_of_members:stop] = weights
            fills_array[number_of_members:stop] = fill_factors
            done[members[number_of_members:stop]] = True
            number_of_members = stop
            number_of_stars += 1
            offsets[number_of_stars] = number_of_members

        if not np.all(done) or number_of_members != number_of_rows:
            raise ValueError("density G-vector stars are incomplete")
        # Shrinking an owning NumPy allocation does not copy its payload.  It
        # prevents the worst case N_G+1 offset table being retained when each
        # physical star contains several vectors.
        offsets.resize(number_of_stars + 1, refcheck=False)
        # Space-group phases and their star averages repeat heavily. Preserve
        # every complex128 value exactly once and retain only a narrow lookup
        # per G row. This is lossless (unlike complex64 metadata), removes two
        # complex vectors from the persistent symmetry plan, and improves the
        # native projector's cache footprint.
        weight_table, weight_inverse = np.unique(
            weights_array, return_inverse=True
        )
        fill_table, fill_inverse = np.unique(
            fills_array, return_inverse=True
        )

        def compact_indices(values: np.ndarray, count: int) -> np.ndarray:
            if count <= np.iinfo(np.uint8).max + 1:
                dtype = np.uint8
            elif count <= np.iinfo(np.uint16).max + 1:
                dtype = np.uint16
            else:
                dtype = np.uint32
            return np.asarray(values, dtype=dtype)

        return (
            offsets,
            members,
            np.ascontiguousarray(weight_table, dtype=np.complex128),
            compact_indices(weight_inverse, len(weight_table)),
            np.ascontiguousarray(fill_table, dtype=np.complex128),
            compact_indices(fill_inverse, len(fill_table)),
        )

    @staticmethod
    def _displacements(counts: np.ndarray) -> np.ndarray:
        if counts.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(counts[:-1]))
        )

    @property
    def mapping_bytes(self) -> int:
        return int(
            self._send_order.nbytes
            + self._send_counts.nbytes
            + self._send_displacements.nbytes
            + self._recv_counts.nbytes
            + self._recv_displacements.nbytes
            + self._packed_send.nbytes
            + self._shell_coefficients.nbytes
            + self._star_offsets.nbytes
            + self._star_members.nbytes
            + self._star_weight_table.nbytes
            + self._star_weight_indices.nbytes
            + self._star_fill_table.nbytes
            + self._star_fill_indices.nbytes
        )

    def _project_stars(self, coefficients: np.ndarray) -> None:
        # Native indexed loops avoid the two fancy-indexed NumPy temporaries
        # formerly allocated for every star.  This is particularly important
        # after pymalloc/glibc retain those small blocks over many SCF steps.
        self.workspace._native_fft.project_density_stars(
            coefficients,
            self._star_offsets,
            self._star_members,
            self._star_weight_table,
            self._star_weight_indices,
            self._star_fill_table,
            self._star_fill_indices,
        )

    def apply(self, density: np.ndarray) -> np.ndarray:
        array = np.asarray(density, dtype=float)
        expected = (
            self.shape[0],
            self.shape[1],
            self.mpi.slab(self.shape[2]).stop
            - self.mpi.slab(self.shape[2]).start,
        )
        if array.shape != expected:
            raise ValueError("density does not match this rank's FFT slab")
        if len(self.operations) <= 1:
            return array
        coefficients = self.workspace.grid_to_coefficients(array)
        if self.mpi.size == 1:
            self._project_stars(coefficients)
            projected = self.workspace.coefficients_to_grid(
                coefficients, use_scratch=True
            )
            return np.ascontiguousarray(np.real(projected))

        packed_send, shell_coefficients = self.mpi.complex_exchange_buffers(
            self._packed_size, self._shell_size
        )
        packed_send = packed_send[: self._packed_size]
        shell_coefficients = shell_coefficients[: self._shell_size]
        np.take(
            coefficients,
            self._send_order,
            out=packed_send,
        )
        self.mpi.exchange_complex_planned(
            packed_send,
            self._send_counts,
            self._send_displacements,
            self._recv_counts,
            self._recv_displacements,
            recv_buffer=shell_coefficients,
        )
        self._project_stars(shell_coefficients)
        self.mpi.exchange_complex_planned(
            shell_coefficients,
            self._recv_counts,
            self._recv_displacements,
            self._send_counts,
            self._send_displacements,
            recv_buffer=packed_send,
        )
        coefficients[self._send_order] = packed_send
        projected = self.workspace.coefficients_to_grid(
            coefficients, use_scratch=True
        )
        return np.ascontiguousarray(np.real(projected))


def fft_factors(
    operations: tuple[SymmetryOperation, ...],
    tolerance: float = 1.0e-7,
    use_all_frac: bool = False,
) -> tuple[int, int, int]:
    """Return QE-style FFT divisibility factors for fractional translations.

    Since QE 6.4, nonsymmorphic operations are retained by default and the
    real-space FFT grid is made commensurate with their translations.
    ``use_all_frac`` deliberately removes that grid constraint while keeping
    every symmetry operation; reciprocal-space phase symmetrization remains
    valid on the unconstrained grid.
    """
    if use_all_frac:
        return (1, 1, 1)
    factors = [1, 1, 1]
    for operation in operations:
        for axis, value in enumerate(operation.translation):
            reduced = float(value) % 1.0
            fraction = Fraction(reduced).limit_denominator(48)
            if abs(float(fraction) - reduced) <= tolerance:
                factors[axis] = math.lcm(factors[axis], fraction.denominator)
    return tuple(factors)


def _periodic_match(
    transformed: np.ndarray,
    labels: list[str],
    reference: np.ndarray,
    tolerance: float,
) -> bool:
    for label in set(labels):
        left = transformed[np.array([item == label for item in labels])]
        right = reference[np.array([item == label for item in labels])]
        unused = list(range(len(right)))
        for position in left:
            match = None
            for index in unused:
                delta = position - right[index]
                delta -= np.rint(delta)
                if np.linalg.norm(delta) < tolerance:
                    match = index
                    break
            if match is None:
                return False
            unused.remove(match)
    return True


def _qe_fractional_translation(
    translation: np.ndarray, tolerance: float
) -> np.ndarray | None:
    """Normalize a translation and enforce QE's crystallographic fractions."""
    reduced = np.asarray(translation, dtype=float) - np.rint(translation)
    for component in reduced:
        magnitude = abs(float(component))
        if magnitude <= tolerance:
            continue
        inverse = 1.0 / magnitude
        denominator = round(inverse)
        if (
            denominator not in {2, 3, 4, 6}
            or abs(inverse - denominator) > tolerance
        ):
            return None
    normalized = np.asarray(translation, dtype=float) - np.floor(translation)
    normalized[np.isclose(normalized, 1.0, rtol=0.0, atol=tolerance)] = 0.0
    normalized[np.isclose(normalized, 0.0, rtol=0.0, atol=tolerance)] = 0.0
    return normalized


def _spglib_space_group(
    lattice: np.ndarray,
    fractional_positions: np.ndarray,
    labels: list[str],
    tolerance: float,
) -> tuple[SymmetryOperation, ...] | None:
    """Use spglib for robust lattice/basis matching, retaining QE semantics."""
    try:
        import spglib
    except ImportError:
        return None
    species_numbers: dict[str, int] = {}
    numbers = []
    for label in labels:
        numbers.append(species_numbers.setdefault(label, len(species_numbers) + 1))
    cartesian_tolerance = max(
        tolerance * float(np.max(np.linalg.norm(lattice, axis=1))),
        np.finfo(float).eps * 100.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        symmetry = spglib.get_symmetry(
            (
                np.asarray(lattice, dtype=float),
                np.asarray(fractional_positions, dtype=float),
                np.asarray(numbers, dtype=int),
            ),
            symprec=cartesian_tolerance,
            angle_tolerance=-1.0,
        )
    if symmetry is None:
        return None
    operations: list[SymmetryOperation] = []
    for rotation, translation in zip(
        symmetry["rotations"], symmetry["translations"]
    ):
        # spglib acts on column coordinates, x' = R x + t; qepy-pw uses
        # row coordinates, x' = x S + t, hence S = R.T.
        matrix = np.asarray(rotation, dtype=int).T
        normalized = _qe_fractional_translation(translation, tolerance)
        if normalized is None:
            continue
        transformed = fractional_positions @ matrix + normalized
        if not _periodic_match(
            transformed, labels, fractional_positions, tolerance
        ):
            continue
        operations.append(SymmetryOperation(matrix.copy(), normalized.copy()))
    return tuple(operations) if operations else None


def _local_space_group(
    lattice: np.ndarray,
    fractional_positions: np.ndarray,
    labels: list[str],
    tolerance: float,
) -> tuple[SymmetryOperation, ...]:
    """Find ordinary crystallographic operations without a heavy runtime.

    The candidate integer matrices are small and geometry-independent.
    Screening all of them in vectorized NumPy is both faster than the former
    19,683-iteration Python loop and lets normal Bravais cells avoid loading
    spglib's extension and database separately in every MPI rank.
    """
    metric = lattice @ lattice.T
    scale = max(1.0, float(np.max(np.abs(metric))))
    digits = np.indices((3,) * 9, dtype=np.int8).reshape(9, -1).T - 1
    matrices = digits.reshape(-1, 3, 3)
    determinants = (
        matrices[:, 0, 0]
        * (matrices[:, 1, 1] * matrices[:, 2, 2]
           - matrices[:, 1, 2] * matrices[:, 2, 1])
        - matrices[:, 0, 1]
        * (matrices[:, 1, 0] * matrices[:, 2, 2]
           - matrices[:, 1, 2] * matrices[:, 2, 0])
        + matrices[:, 0, 2]
        * (matrices[:, 1, 0] * matrices[:, 2, 1]
           - matrices[:, 1, 1] * matrices[:, 2, 0])
    )
    matrices = matrices[np.abs(determinants) == 1]
    transformed_metrics = np.einsum(
        "aik,kl,ajl->aij", matrices, metric, matrices, optimize=True
    )
    metric_match = np.all(
        np.abs(transformed_metrics - metric[None, :, :])
        <= tolerance * scale,
        axis=(1, 2),
    )
    matrices = matrices[metric_match]

    operations: list[SymmetryOperation] = []
    first_label = labels[0]
    candidate_targets = fractional_positions[
        np.array([label == first_label for label in labels])
    ]
    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=int)
        rotated = fractional_positions @ matrix
        for target in candidate_targets:
            normalized_translation = _qe_fractional_translation(
                target - rotated[0], tolerance
            )
            if normalized_translation is None:
                continue
            if _periodic_match(
                rotated + normalized_translation,
                labels,
                fractional_positions,
                tolerance,
            ):
                operations.append(
                    SymmetryOperation(
                        matrix.copy(), normalized_translation.copy()
                    )
                )
                break
    if not operations:
        operations.append(
            SymmetryOperation(np.eye(3, dtype=int), np.zeros(3))
        )
    return tuple(operations)


def find_space_group(
    lattice: np.ndarray,
    fractional_positions: np.ndarray,
    labels: list[str],
    tolerance: float = 1.0e-7,
) -> tuple[SymmetryOperation, ...]:
    """Find space-group operations, using spglib with a local safe fallback.

    spglib supplies the crystallographic lattice and labeled-basis search.
    The result is converted to this module's row-vector convention and then
    filtered through QE's accepted fractional-translation rule.  The compact
    original enumerator remains available when spglib is not installed or
    cannot classify a deliberately pathological cell.
    """
    lattice = np.asarray(lattice, dtype=float)
    fractional_positions = np.asarray(fractional_positions, dtype=float)
    if len(fractional_positions) != len(labels) or not labels:
        raise ValueError("fractional positions and nonempty labels must match")
    local_operations = _local_space_group(
        lattice, fractional_positions, labels, tolerance
    )
    if len(local_operations) > 1:
        return local_operations
    spglib_operations = _spglib_space_group(
        lattice, fractional_positions, labels, tolerance
    )
    return (
        spglib_operations
        if spglib_operations is not None
        else local_operations
    )


def _kpoint_key(point: np.ndarray, decimals: int = 10) -> tuple[float, ...]:
    reduced = np.asarray(point) - np.floor(np.asarray(point))
    reduced[np.isclose(reduced, 1.0, atol=10.0 ** (-decimals))] = 0.0
    return tuple(np.round(reduced, decimals))


def reduce_kpoints(
    coordinates: np.ndarray,
    weights: np.ndarray,
    operations: tuple[SymmetryOperation, ...],
    time_reversal: bool = True,
    return_mapping: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a complete fractional reciprocal mesh into symmetry orbits."""
    coordinates = np.asarray(coordinates, dtype=float)
    weights = np.asarray(weights, dtype=float)
    def keys(points: np.ndarray) -> list[tuple[float, ...]]:
        reduced = np.asarray(points, dtype=float)
        reduced = reduced - np.floor(reduced)
        reduced[np.isclose(reduced, 1.0, atol=1.0e-10)] = 0.0
        return [tuple(row) for row in np.round(reduced, 10)]

    coordinate_keys = keys(coordinates)
    lookup = {key: index for index, key in enumerate(coordinate_keys)}
    reciprocal_matrices = tuple(
        np.linalg.inv(operation.matrix).T for operation in operations
    )
    negative_keys = keys(-coordinates) if time_reversal else []
    time_reversal = time_reversal and all(
        key in lookup for key in negative_keys
    )
    # Build every symmetry/time-reversal map in vectorized blocks.  The old
    # inner loop called floor/isclose/round on a three-element NumPy array for
    # each (k,symmetry,sign) tuple; on a 12^3 mesh that Python dispatch cost
    # alone was several seconds in every MPI rank before SCF timing began.
    orbit_maps: list[np.ndarray] = []
    signs = (1.0, -1.0) if time_reversal else (1.0,)
    for reciprocal_matrix in reciprocal_matrices:
        transformed = coordinates @ reciprocal_matrix
        for sign in signs:
            mapped_keys = keys(sign * transformed)
            orbit_map = np.fromiter(
                (lookup.get(key, -1) for key in mapped_keys),
                dtype=np.int32,
                count=len(coordinates),
            )
            if np.all(orbit_map >= 0):
                orbit_maps.append(orbit_map)
    assigned: set[int] = set()
    representatives: list[np.ndarray] = []
    reduced_weights: list[float] = []
    mapping = np.full(len(coordinates), -1, dtype=np.int32)
    for index, point in enumerate(coordinates):
        if index in assigned:
            continue
        orbit: set[int] = set()
        for orbit_map in orbit_maps:
            mapped = int(orbit_map[index])
            if mapped >= 0:
                orbit.add(mapped)
        if not orbit:
            orbit.add(index)
        assigned.update(orbit)
        representative_index = len(representatives)
        mapping[list(orbit)] = representative_index
        representatives.append(point.copy())
        reduced_weights.append(float(np.sum(weights[list(orbit)])))
    normalized = np.asarray(reduced_weights)
    normalized /= normalized.sum()
    if np.any(mapping < 0):
        raise ValueError("k-point reduction left points without a representative")
    if return_mapping:
        return np.asarray(representatives), normalized, mapping
    return np.asarray(representatives), normalized


def mesh_compatible_operations(
    coordinates: np.ndarray,
    operations: tuple[SymmetryOperation, ...],
) -> tuple[SymmetryOperation, ...]:
    """Keep only crystal operations that map the complete sampled mesh to itself."""
    coordinates = np.asarray(coordinates, dtype=float)
    reduced = coordinates - np.floor(coordinates)
    reduced[np.isclose(reduced, 1.0, atol=1.0e-10)] = 0.0
    keys = {tuple(row) for row in np.round(reduced, 10)}
    compatible = []
    for operation in operations:
        inverse_transpose = np.linalg.inv(operation.matrix).T
        transformed = coordinates @ inverse_transpose
        transformed -= np.floor(transformed)
        transformed[
            np.isclose(transformed, 1.0, atol=1.0e-10)
        ] = 0.0
        if all(tuple(row) in keys for row in np.round(transformed, 10)):
            compatible.append(operation)
    return tuple(compatible)


def symmetrize_density(
    density: np.ndarray,
    operations: tuple[SymmetryOperation, ...],
    mpi: "MPIContext | None" = None,
) -> np.ndarray:
    """Average a real-space density over FFT-grid-compatible operations."""
    return DensitySymmetrizer(
        density.shape, operations, mpi
    ).apply(density)


def symmetrize_forces(
    forces: np.ndarray,
    lattice: np.ndarray,
    fractional_positions: np.ndarray,
    labels: list[str],
    operations: tuple[SymmetryOperation, ...],
    tolerance: float = 1.0e-7,
) -> np.ndarray:
    """Project Cartesian atomic forces onto the crystal space group."""
    values = np.asarray(forces, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    fractional = np.asarray(fractional_positions, dtype=float)
    if values.shape != (len(fractional), 3):
        raise ValueError("forces must have shape (number of atoms, 3)")
    if len(operations) <= 1:
        return values.copy()
    inverse_lattice = np.linalg.inv(lattice)
    projected = np.zeros_like(values)
    for operation in operations:
        cartesian_rotation = (
            inverse_lattice @ operation.matrix @ lattice
        )
        transformed = (
            fractional @ operation.matrix + operation.translation
        )
        transformed -= np.floor(transformed)
        for source, (position, label) in enumerate(
            zip(transformed, labels)
        ):
            target = None
            for candidate, (reference, reference_label) in enumerate(
                zip(fractional, labels)
            ):
                if reference_label != label:
                    continue
                delta = position - reference
                delta -= np.rint(delta)
                if np.linalg.norm(delta) < tolerance:
                    target = candidate
                    break
            if target is None:
                raise ValueError("symmetry operation does not map the atomic basis")
            projected[target] += values[source] @ cartesian_rotation
    projected /= len(operations)
    return projected


def symmetrize_stress(
    stress: np.ndarray,
    lattice: np.ndarray,
    operations: tuple[SymmetryOperation, ...],
) -> np.ndarray:
    """Project a Cartesian symmetric stress tensor onto the point group."""
    tensor = np.asarray(stress, dtype=float)
    if tensor.shape != (3, 3):
        raise ValueError("stress must have shape (3, 3)")
    if len(operations) <= 1:
        return 0.5 * (tensor + tensor.T)
    inverse_lattice = np.linalg.inv(np.asarray(lattice, dtype=float))
    projected = np.zeros((3, 3), dtype=float)
    for operation in operations:
        cartesian_rotation = (
            inverse_lattice @ operation.matrix @ lattice
        )
        projected += (
            cartesian_rotation.T @ tensor @ cartesian_rotation
        )
    projected /= len(operations)
    return 0.5 * (projected + projected.T)
