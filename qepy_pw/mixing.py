"""Charge-density mixing algorithms.

``PlainBroydenMixer`` follows the algebra in QE ``PW/src/mix_rho.f90`` for
the ``plain``, ``TF``, and ``local-TF`` mixing modes. Despite the historical
name, this is a multisecant modified-Broyden/Anderson method, not simple
linear mixing.  A Pulay frequency greater than one implements periodic
Pulay: history is collected on every step, while multisecant extrapolation
is used only periodically and intervening steps use preconditioned linear
mixing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .mpi import MPIContext
from .basis import LocalPotentialWorkspace


_MIXING_MODES = {"plain", "tf", "local-tf"}
_LOCAL_TF_SUBSPACE = 12
_LOCAL_TF_MAX_REFRESHES = 4


def _normalise_mixing_mode(mode: str) -> str:
    normalised = str(mode).strip().lower().replace("_", "-")
    if normalised == "default":
        normalised = "plain"
    if normalised not in _MIXING_MODES:
        raise ValueError(
            "mixing_mode must be 'plain', 'TF', or 'local-TF'"
        )
    return normalised


def _validate_mixing_controls(
    beta: float, ndim: int, pulay_frequency: int = 1
) -> None:
    if not np.isfinite(beta) or not 0.0 < beta <= 1.0:
        raise ValueError("mixing_beta must satisfy 0 < mixing_beta <= 1")
    if ndim < 1 or ndim > 25:
        raise ValueError("mixing_ndim must be between 1 and 25")
    if pulay_frequency < 1:
        raise ValueError("mixing_pulay_frequency must be at least 1")


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
    real_workspace = absolute_density

    def multiply_by_alpha(vector: np.ndarray) -> np.ndarray:
        real = coefficients_to_grid(vector)
        np.multiply(np.real(real), alpha, out=real_workspace)
        return np.asarray(
            grid_to_coefficients(real_workspace), dtype=np.complex128
        )

    metric_weights = np.zeros_like(g2, dtype=float)
    metric_weights[active] = 1.0 / g2[active]
    metric_workspace = np.empty_like(residual)

    def metric(left: np.ndarray, right: np.ndarray) -> float:
        np.multiply(right, metric_weights, out=metric_workspace)
        local = np.vdot(left, metric_workspace)
        return mpi.sum_scalar(float(np.real(local)))

    alpha_residual = multiply_by_alpha(residual)
    source = g2 * alpha_residual
    direction = alpha_residual * g2 / (g2 + screening_g2)
    direction[g2 <= 1.0e-14] = 0.0
    best_direction = np.empty_like(residual)
    remainder = np.empty_like(residual)
    target: float | None = None
    refreshes = 0
    vectors = np.empty(
        (_LOCAL_TF_SUBSPACE, residual.size), dtype=np.complex128
    )
    images = np.empty_like(vectors)
    gram = np.empty(
        (_LOCAL_TF_SUBSPACE, _LOCAL_TF_SUBSPACE), dtype=float
    )
    projection = np.empty(_LOCAL_TF_SUBSPACE, dtype=float)
    size = 0
    while True:
        vectors[size] = direction
        # QE works in Rydberg atomic units, where e2=2. Its local-TF
        # operator is therefore 4*pi*e2 + G^2*alpha = 8*pi + G^2*alpha.
        np.multiply(direction, 8.0 * np.pi, out=images[size])
        np.multiply(
            multiply_by_alpha(direction),
            g2,
            out=metric_workspace,
        )
        images[size] += metric_workspace

        # Only the new row/column changes. QE likewise retains the previous
        # aa/bb entries instead of rebuilding the full Gram matrix at every
        # inner iteration.
        projection[size] = metric(images[size], source)
        for row in range(size + 1):
            value = metric(images[row], images[size])
            gram[row, size] = value
            gram[size, row] = value
        dimension = size + 1
        try:
            coefficients = np.linalg.solve(
                gram[:dimension, :dimension], projection[:dimension]
            )
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(
                gram[:dimension, :dimension],
                projection[:dimension],
                rcond=1.0e-14,
            )[0]
        best_direction.fill(0.0)
        remainder[...] = source
        for index, coefficient in enumerate(coefficients):
            best_direction += coefficient * vectors[index]
            remainder -= coefficient * images[index]
        remainder_norm = metric(remainder, remainder)
        if target is None:
            target = max(1.0e-12, 1.0e-6 * remainder_norm)
        if remainder_norm < target or (
            dimension >= _LOCAL_TF_SUBSPACE
            and refreshes >= _LOCAL_TF_MAX_REFRESHES
        ):
            return best_direction
        if dimension >= _LOCAL_TF_SUBSPACE:
            refreshes += 1
            direction[...] = best_direction
            size = 0
        else:
            np.divide(
                remainder, g2 + screening_g2, out=direction
            )
            direction[g2 <= 1.0e-14] = 0.0
            size += 1
    # QE uses the best accumulated direction after the bounded inner solve.
    return best_direction


class _HistoryBuffer(Sequence[np.ndarray]):
    """Fixed-capacity contiguous history without per-iteration ``stack``."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._storage: np.ndarray | None = None
        self._count = 0
        self._next = 0

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
            self._next = self._count % self.capacity
            return
        # QE overwrites one cyclic history slot. The multisecant equations do
        # not depend on chronological row ordering, so avoid shifting every
        # stored G-vector once per SCF iteration.
        self._storage[self._next] = array
        self._next = (self._next + 1) % self.capacity

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
        pulay_frequency: int = 1,
    ) -> None:
        _validate_mixing_controls(beta, ndim, pulay_frequency)
        self.shape = shape
        self.beta = float(beta)
        self.ndim = int(ndim)
        self.pulay_frequency = int(pulay_frequency)
        self.mixing_step = 0
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
        self.tf_factor = (
            self.active_g2
            / (self.active_g2 + self.screening_g2)
            if self.mode == "tf"
            else None
        )
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
        self.mixing_step += 1
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
        if (
            self.delta_inputs
            and self.mixing_step % self.pulay_frequency == 0
        ):
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
            assert self.tf_factor is not None
            residual_active *= self.tf_factor
        elif self.mode == "local-tf":
            full_residual = np.zeros_like(residual)
            full_residual.ravel()[self.active] = residual_active
            projected_input = self.mpi.gather_flat_chunks(
                current_active, len(self.all_active)
            )
            best_current = current.copy()
            best_current.ravel()[self.all_active] = projected_input
            best_density = np.fft.ifftn(best_current * normalization)
            screened = _local_thomas_fermi_direction(
                full_residual.ravel(),
                np.real(best_density),
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
        pulay_frequency: int = 1,
    ) -> None:
        _validate_mixing_controls(beta, ndim, pulay_frequency)
        if g2_cutoff is not None and g2_cutoff <= 0.0:
            raise ValueError("charge-density cutoff must be positive")
        self.workspace = workspace
        self.mpi = workspace.mpi
        self.beta = float(beta)
        self.ndim = int(ndim)
        self.pulay_frequency = int(pulay_frequency)
        self.mixing_step = 0
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
        self.tf_factor = (
            self.active_g2
            / (self.active_g2 + self.screening_g2)
            if self.mode == "tf"
            else None
        )
        self.delta_inputs = _HistoryBuffer(self.ndim)
        self.delta_residuals = _HistoryBuffer(self.ndim)
        self.previous_input: np.ndarray | None = None
        self.previous_residual: np.ndarray | None = None

    def _metric(self, left: np.ndarray, right: np.ndarray) -> float:
        local = float(
            np.real(np.sum(np.conj(left) * right / self.active_g2))
        )
        return self.mpi.sum_scalar(local)

    def mix(
        self,
        density_in: np.ndarray,
        density_out: np.ndarray,
        *,
        residual_coefficients: np.ndarray | None = None,
    ) -> np.ndarray:
        self.mixing_step += 1
        # grid_to_coefficients returns a writable compact owner (advanced
        # sparse extraction in MPI, native compact output in serial).  Copying
        # it here briefly doubled a charge-G vector immediately before the
        # already memory-intensive Broyden update.
        current = self.workspace.grid_to_coefficients(density_in)
        if residual_coefficients is None:
            residual = self.workspace.grid_to_coefficients(density_out)
            residual -= current
        else:
            residual = np.asarray(
                residual_coefficients, dtype=np.complex128
            )
            if residual.shape != current.shape:
                raise ValueError(
                    "precomputed density residual has the wrong shape"
                )
        current_active = current[self.active].copy()
        residual_active = residual[self.active].copy()
        saved_input = current_active.copy()
        saved_residual = residual_active.copy()
        if self.previous_input is not None:
            self.delta_inputs.append(self.previous_input - current_active)
            self.delta_residuals.append(
                self.previous_residual - residual_active
            )
        if (
            self.delta_inputs
            and self.mixing_step % self.pulay_frequency == 0
        ):
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
            assert self.tf_factor is not None
            residual_active *= self.tf_factor
        elif self.mode == "local-tf":
            full_residual = np.zeros_like(residual)
            full_residual[self.active] = residual_active
            current[self.active] = current_active
            best_density = self.workspace.coefficients_to_grid(
                current, use_scratch=True
            )
            screened = _local_thomas_fermi_direction(
                full_residual,
                np.real(best_density),
                self.local_g2,
                self.active,
                self.mpi,
                lambda coefficients: self.workspace.coefficients_to_grid(
                    coefficients, use_scratch=True
                ),
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
