"""Charge-density mixing algorithms.

``PlainBroydenMixer`` follows the algebra in QE ``PW/src/mix_rho.f90`` for
the ``plain``, ``TF``, and ``local-TF`` mixing modes. Despite the historical
name, this is a multisecant modified-Broyden/Anderson method, not simple
linear mixing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .mpi import MPIContext
from .basis import LocalPotentialWorkspace


_MIXING_MODES = {"plain", "tf", "local-tf"}


def _normalise_mixing_mode(mode: str) -> str:
    normalised = str(mode).strip().lower().replace("_", "-")
    if normalised == "default":
        normalised = "plain"
    if normalised not in _MIXING_MODES:
        raise ValueError(
            "mixing_mode must be 'plain', 'TF', or 'local-TF'"
        )
    return normalised


def _validate_mixing_controls(beta: float, ndim: int) -> None:
    if not np.isfinite(beta) or not 0.0 < beta <= 1.0:
        raise ValueError("mixing_beta must satisfy 0 < mixing_beta <= 1")
    if ndim < 1 or ndim > 25:
        raise ValueError("mixing_ndim must be between 1 and 25")


def _thomas_fermi_g2(nelec: float, volume: float) -> float:
    """Return QE's Thomas-Fermi screening wavevector squared (bohr^-2)."""
    if not np.isfinite(nelec) or nelec <= 0.0:
        raise ValueError("Thomas-Fermi mixing requires a positive nelec")
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("Thomas-Fermi mixing requires a positive volume")
    rs = (3.0 * volume / (4.0 * np.pi * nelec)) ** (1.0 / 3.0)
    return float((12.0 / np.pi) ** (2.0 / 3.0) / rs)


def _local_thomas_fermi_direction(
    residual: np.ndarray,
    density: np.ndarray,
    g2: np.ndarray,
    active: np.ndarray,
    mpi: MPIContext,
    coefficients_to_grid,
    grid_to_coefficients,
) -> np.ndarray:
    """Apply QE's local-density Thomas-Fermi approximate inverse."""
    absolute_density = np.abs(np.asarray(density, dtype=float))
    occupied = absolute_density > 1.0e-32
    local_rs = absolute_density.copy()
    local_rs[occupied] = (
        3.0 / (4.0 * np.pi * absolute_density[occupied])
    ) ** (1.0 / 3.0)
    inverse_rs_local = float(np.sum(1.0 / local_rs[occupied]))
    inverse_rs = mpi.sum_scalar(inverse_rs_local)
    points = mpi.sum_scalar(float(absolute_density.size))
    if inverse_rs <= 0.0:
        return residual.copy()
    average_rs = points / inverse_rs
    screening_g2 = (12.0 / np.pi) ** (2.0 / 3.0) / average_rs
    alpha = local_rs * (3.0 * (2.0 * np.pi / 3.0) ** (5.0 / 3.0))

    def multiply_by_alpha(vector: np.ndarray) -> np.ndarray:
        real = coefficients_to_grid(vector)
        return np.asarray(
            grid_to_coefficients(alpha * np.real(real)),
            dtype=np.complex128,
        ).copy()

    def metric(left: np.ndarray, right: np.ndarray) -> float:
        local = np.sum(
            np.conj(left[active]) * right[active] / g2[active]
        )
        return mpi.sum_scalar(float(np.real(local)))

    alpha_residual = multiply_by_alpha(residual)
    source = g2 * alpha_residual
    direction = alpha_residual * g2 / (g2 + screening_g2)
    direction[g2 <= 1.0e-14] = 0.0
    best_direction = direction.copy()
    target: float | None = None
    refreshes = 0
    vectors: list[np.ndarray] = []
    images: list[np.ndarray] = []
    while refreshes < 4:
        image = 4.0 * np.pi * direction + g2 * multiply_by_alpha(direction)
        vectors.append(direction.copy())
        images.append(image)
        size = len(vectors)
        gram = np.empty((size, size), dtype=float)
        projection = np.empty(size, dtype=float)
        for row in range(size):
            projection[row] = metric(images[row], source)
            for column in range(row + 1):
                value = metric(images[row], images[column])
                gram[row, column] = value
                gram[column, row] = value
        try:
            coefficients = np.linalg.solve(gram, projection)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(
                gram, projection, rcond=1.0e-14
            )[0]
        best_direction = np.zeros_like(residual)
        remainder = source.copy()
        for coefficient, vector, vector_image in zip(
            coefficients, vectors, images
        ):
            best_direction += coefficient * vector
            remainder -= coefficient * vector_image
        remainder_norm = metric(remainder, remainder)
        if target is None:
            target = max(1.0e-12, 1.0e-6 * remainder_norm)
        if remainder_norm <= target:
            return best_direction
        if size == 12:
            refreshes += 1
            vectors.clear()
            images.clear()
            direction = best_direction.copy()
        else:
            direction = remainder / (g2 + screening_g2)
            direction[g2 <= 1.0e-14] = 0.0
    # QE uses the best accumulated direction after the bounded inner solve.
    return best_direction


class _HistoryBuffer(Sequence[np.ndarray]):
    """Fixed-capacity contiguous history without per-iteration ``stack``."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._storage: np.ndarray | None = None
        self._count = 0

    def append(self, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=np.complex128)
        if self._storage is None:
            self._storage = np.empty(
                (self.capacity, array.size), dtype=np.complex128
            )
        if array.shape != self._storage.shape[1:]:
            raise ValueError("Broyden history vector has changed size")
        if self._count < self.capacity:
            self._storage[self._count] = array
            self._count += 1
            return
        # Preserve QE/deque's chronological ordering, and copy one row at a
        # time so an overlapping whole-array assignment cannot allocate a
        # second history-sized temporary.
        for row in range(self.capacity - 1):
            self._storage[row] = self._storage[row + 1]
        self._storage[-1] = array

    @property
    def array(self) -> np.ndarray:
        if self._storage is None:
            return np.empty((0, 0), dtype=np.complex128)
        return self._storage[: self._count]

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int | slice) -> np.ndarray:
        return self.array[index]


def _broyden_projection_and_gram(
    delta_residuals: np.ndarray,
    residual: np.ndarray,
    g2: np.ndarray,
    mpi: MPIContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Form Coulomb-metric products with one vector-sized temporary."""
    history = delta_residuals.shape[0]
    projection_local = np.empty(history, dtype=float)
    gram_local = np.empty((history, history), dtype=float)
    weighted = np.empty_like(residual)
    for row in range(history):
        np.conjugate(delta_residuals[row], out=weighted)
        weighted /= g2
        projection_local[row] = float(np.real(weighted @ residual))
        gram_local[row] = np.real(weighted @ delta_residuals.T)
    return (
        mpi.sum_array(projection_local),
        mpi.sum_array(gram_local),
    )


class PlainBroydenMixer:
    """History-based Broyden mixing in the reciprocal-space Coulomb metric."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        reciprocal: np.ndarray,
        beta: float = 0.7,
        ndim: int = 8,
        g2_cutoff: float | None = None,
        mpi: MPIContext | None = None,
        mode: str = "plain",
        nelec: float | None = None,
        volume: float | None = None,
    ) -> None:
        _validate_mixing_controls(beta, ndim)
        self.shape = shape
        self.beta = float(beta)
        self.ndim = int(ndim)
        self.mpi = mpi if mpi is not None else MPIContext()
        self.mode = _normalise_mixing_mode(mode)
        self.screening_g2 = (
            None
            if self.mode == "plain"
            else _thomas_fermi_g2(float(nelec or 0.0), float(volume or 0.0))
        )
        axes = [np.fft.fftfreq(size) * size for size in shape]
        indices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        vectors = indices @ reciprocal
        self.g2 = np.einsum("...j,...j->...", vectors, vectors)
        self.nonzero = self.g2 > 1.0e-14
        if g2_cutoff is not None:
            if g2_cutoff <= 0.0:
                raise ValueError("charge-density cutoff must be positive")
            self.nonzero &= self.g2 <= g2_cutoff + 1.0e-12
        all_active = np.flatnonzero(self.nonzero.ravel())
        self.all_active = all_active
        active_slab = self.mpi.slab(len(all_active))
        self.active = all_active[active_slab]
        self.active_g2 = self.g2.ravel()[self.active].copy()
        self.delta_inputs = _HistoryBuffer(self.ndim)
        self.delta_residuals = _HistoryBuffer(self.ndim)
        self.previous_input: np.ndarray | None = None
        self.previous_residual: np.ndarray | None = None

    def _metric(self, left: np.ndarray, right: np.ndarray) -> float:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.ndim != 1 or left_array.size != len(self.active):
            left_array = left_array.ravel()[self.active]
            right_array = right_array.ravel()[self.active]
        value = np.sum(
            np.conj(left_array) * right_array / self.active_g2
        )
        return self.mpi.sum_scalar(float(np.real(value)))

    def mix(self, density_in: np.ndarray, density_out: np.ndarray) -> np.ndarray:
        """Return the next input density and append the current secant pair."""
        normalization = np.prod(self.shape)
        current = np.fft.fftn(density_in) / normalization
        residual = np.fft.fftn(density_out) / normalization
        residual -= current
        current_active = current.ravel()[self.active].copy()
        residual_active = residual.ravel()[self.active].copy()
        saved_input = current_active.copy()
        saved_residual = residual_active.copy()
        if self.previous_input is not None:
            assert self.previous_residual is not None
            self.delta_inputs.append(
                self.previous_input - current_active
            )
            self.delta_residuals.append(
                self.previous_residual - residual_active
            )
        if self.delta_inputs:
            delta_inputs = self.delta_inputs.array
            delta_residuals = self.delta_residuals.array
            projection, gram = _broyden_projection_and_gram(
                delta_residuals,
                residual_active,
                self.active_g2,
                self.mpi,
            )
            try:
                # QE factorizes and inverts this symmetric Gram matrix.
                gamma = np.linalg.solve(gram, projection)
            except np.linalg.LinAlgError:
                # Preserve a usable SCF path for exactly dependent histories,
                # for which QE's factorization would terminate the run.
                gamma = np.linalg.lstsq(
                    gram, projection, rcond=1.0e-14
                )[0]
            current_active -= gamma @ delta_inputs
            residual_active -= gamma @ delta_residuals
        if self.mode == "tf":
            assert self.screening_g2 is not None
            residual_active *= self.active_g2 / (
                self.active_g2 + self.screening_g2
            )
        elif self.mode == "local-tf":
            full_residual = np.zeros_like(residual)
            full_residual.ravel()[self.active] = residual_active
            screened = _local_thomas_fermi_direction(
                full_residual.ravel(),
                density_in,
                self.g2.ravel(),
                self.all_active,
                self.mpi,
                lambda coefficients: np.fft.ifftn(
                    coefficients.reshape(self.shape) * normalization
                ),
                lambda grid: (
                    np.fft.fftn(grid) / normalization
                ).ravel(),
            )
            residual_active = screened[self.active]
        local_mixed = (
            current_active
            + self.beta * residual_active
        )
        packed_mixed = self.mpi.gather_flat_chunks(
            local_mixed, len(self.all_active)
        )
        mixed = current + self.beta * residual
        mixed.ravel()[self.all_active] = packed_mixed
        # Electron number is fixed: residual G=0 should be zero analytically.
        # Retaining the input coefficient prevents roundoff/history drift.
        mixed.flat[0] = current.flat[0]
        self.previous_input = saved_input
        self.previous_residual = saved_residual
        return np.real(np.fft.ifftn(mixed * normalization))


class LinearMixer:
    """Explicit fallback useful for comparisons and debugging."""

    def __init__(self, beta: float = 0.7) -> None:
        _validate_mixing_controls(beta, 1)
        self.beta = float(beta)

    def mix(self, density_in: np.ndarray, density_out: np.ndarray) -> np.ndarray:
        return (1.0 - self.beta) * density_in + self.beta * density_out


class DistributedBroydenMixer:
    """QE-style Broyden mixer over locally owned charge-density G rows."""

    def __init__(
        self,
        workspace: LocalPotentialWorkspace,
        local_g2: np.ndarray,
        beta: float = 0.7,
        ndim: int = 8,
        g2_cutoff: float | None = None,
        mode: str = "plain",
        nelec: float | None = None,
        volume: float | None = None,
    ) -> None:
        _validate_mixing_controls(beta, ndim)
        if g2_cutoff is not None and g2_cutoff <= 0.0:
            raise ValueError("charge-density cutoff must be positive")
        self.workspace = workspace
        self.mpi = workspace.mpi
        self.beta = float(beta)
        self.ndim = int(ndim)
        self.mode = _normalise_mixing_mode(mode)
        self.screening_g2 = (
            None
            if self.mode == "plain"
            else _thomas_fermi_g2(float(nelec or 0.0), float(volume or 0.0))
        )
        self.local_g2 = np.asarray(local_g2)
        active = self.local_g2 > 1.0e-14
        if g2_cutoff is not None:
            active &= self.local_g2 <= g2_cutoff + 1.0e-12
        self.active = np.flatnonzero(active)
        self.active_g2 = self.local_g2[self.active]
        self.delta_inputs = _HistoryBuffer(self.ndim)
        self.delta_residuals = _HistoryBuffer(self.ndim)
        self.previous_input: np.ndarray | None = None
        self.previous_residual: np.ndarray | None = None

    def _metric(self, left: np.ndarray, right: np.ndarray) -> float:
        local = float(
            np.real(np.sum(np.conj(left) * right / self.active_g2))
        )
        return self.mpi.sum_scalar(local)

    def mix(self, density_in: np.ndarray, density_out: np.ndarray) -> np.ndarray:
        current = self.workspace.grid_to_coefficients(density_in)
        residual = self.workspace.grid_to_coefficients(density_out)
        residual -= current
        current_active = current[self.active].copy()
        residual_active = residual[self.active].copy()
        saved_input = current_active.copy()
        saved_residual = residual_active.copy()
        if self.previous_input is not None:
            self.delta_inputs.append(self.previous_input - current_active)
            self.delta_residuals.append(
                self.previous_residual - residual_active
            )
        if self.delta_inputs:
            delta_inputs = self.delta_inputs.array
            delta_residuals = self.delta_residuals.array
            projection, gram = _broyden_projection_and_gram(
                delta_residuals,
                residual_active,
                self.active_g2,
                self.mpi,
            )
            try:
                gamma = np.linalg.solve(gram, projection)
            except np.linalg.LinAlgError:
                gamma = np.linalg.lstsq(
                    gram, projection, rcond=1.0e-14
                )[0]
            current_active -= gamma @ delta_inputs
            residual_active -= gamma @ delta_residuals
        if self.mode == "tf":
            assert self.screening_g2 is not None
            residual_active *= self.active_g2 / (
                self.active_g2 + self.screening_g2
            )
        elif self.mode == "local-tf":
            full_residual = np.zeros_like(residual)
            full_residual[self.active] = residual_active
            screened = _local_thomas_fermi_direction(
                full_residual,
                density_in,
                self.local_g2,
                self.active,
                self.mpi,
                self.workspace.coefficients_to_grid,
                self.workspace.grid_to_coefficients,
            )
            residual_active = screened[self.active]
        mixed = current + self.beta * residual
        mixed[self.active] = current_active + self.beta * residual_active
        zero = np.flatnonzero(self.local_g2 <= 1.0e-14)
        if zero.size:
            mixed[zero] = current[zero]
        self.previous_input = saved_input
        self.previous_residual = saved_residual
        transformed = self.workspace.coefficients_to_grid(
            mixed, use_scratch=True
        )
        return np.ascontiguousarray(np.real(transformed))
