"""Charge-density mixing algorithms.

``PlainBroydenMixer`` follows the algebra in QE ``PW/src/mix_rho.f90`` for
``mixing_mode='plain'``. Despite the historical name, this is a multisecant
modified-Broyden/Anderson method, not simple linear mixing.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .mpi import MPIContext
from .basis import LocalPotentialWorkspace


class PlainBroydenMixer:
    """History-based Broyden mixing in the reciprocal-space Coulomb metric."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        reciprocal: np.ndarray,
        beta: float = 0.7,
        ndim: int = 8,
        g2_cutoff: float | None = None,
        use_numba: bool = False,
        mpi: MPIContext | None = None,
    ) -> None:
        if not 0.0 < beta <= 1.0:
            raise ValueError("mixing_beta must satisfy 0 < mixing_beta <= 1")
        if ndim < 1:
            raise ValueError("mixing_ndim must be at least 1")
        self.shape = shape
        self.beta = float(beta)
        self.ndim = int(ndim)
        self.use_numba = bool(use_numba)
        self.mpi = mpi if mpi is not None else MPIContext()
        if self.use_numba:
            from .acceleration import numba_kernels

            self._jit = numba_kernels()
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
        self.delta_inputs: deque[np.ndarray] = deque(maxlen=self.ndim)
        self.delta_residuals: deque[np.ndarray] = deque(maxlen=self.ndim)
        self.previous_input: np.ndarray | None = None
        self.previous_residual: np.ndarray | None = None

    def _metric(self, left: np.ndarray, right: np.ndarray) -> float:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.ndim != 1 or left_array.size != len(self.active):
            left_array = left_array.ravel()[self.active]
            right_array = right_array.ravel()[self.active]
        if self.use_numba:
            local_value = float(
                self._jit.coulomb_metric_packed(
                    left_array, right_array, self.active_g2
                )
            )
            return self.mpi.sum_scalar(local_value)
        value = np.sum(
            np.conj(left_array) * right_array / self.active_g2
        )
        return self.mpi.sum_scalar(float(np.real(value)))

    def mix(self, density_in: np.ndarray, density_out: np.ndarray) -> np.ndarray:
        """Return the next input density and append the current secant pair."""
        normalization = np.prod(self.shape)
        current = np.fft.fftn(density_in) / normalization
        output = np.fft.fftn(density_out) / normalization
        residual = output - current
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
            delta_inputs = np.stack(self.delta_inputs)
            delta_residuals = np.stack(self.delta_residuals)
            weighted_conjugates = (
                delta_residuals.conj() / self.active_g2[None, :]
            )
            projection = self.mpi.sum_array(
                np.real(weighted_conjugates @ residual_active)
            )
            gram = self.mpi.sum_array(
                np.real(weighted_conjugates @ delta_residuals.T)
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
        if not 0.0 < beta <= 1.0:
            raise ValueError("mixing_beta must satisfy 0 < mixing_beta <= 1")
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
    ) -> None:
        self.workspace = workspace
        self.mpi = workspace.mpi
        self.beta = float(beta)
        self.ndim = int(ndim)
        self.local_g2 = np.asarray(local_g2)
        active = self.local_g2 > 1.0e-14
        if g2_cutoff is not None:
            active &= self.local_g2 <= g2_cutoff + 1.0e-12
        self.active = np.flatnonzero(active)
        self.active_g2 = self.local_g2[self.active]
        self.delta_inputs: deque[np.ndarray] = deque(maxlen=self.ndim)
        self.delta_residuals: deque[np.ndarray] = deque(maxlen=self.ndim)
        self.previous_input: np.ndarray | None = None
        self.previous_residual: np.ndarray | None = None

    def _metric(self, left: np.ndarray, right: np.ndarray) -> float:
        local = float(
            np.real(np.sum(np.conj(left) * right / self.active_g2))
        )
        return self.mpi.sum_scalar(local)

    def mix(self, density_in: np.ndarray, density_out: np.ndarray) -> np.ndarray:
        current = self.workspace.grid_to_coefficients(density_in)
        output = self.workspace.grid_to_coefficients(density_out)
        residual = output - current
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
            delta_inputs = np.stack(self.delta_inputs)
            delta_residuals = np.stack(self.delta_residuals)
            weighted_conjugates = (
                delta_residuals.conj() / self.active_g2[None, :]
            )
            projection = self.mpi.sum_array(
                np.real(weighted_conjugates @ residual_active)
            )
            gram = self.mpi.sum_array(
                np.real(weighted_conjugates @ delta_residuals.T)
            )
            try:
                gamma = np.linalg.solve(gram, projection)
            except np.linalg.LinAlgError:
                gamma = np.linalg.lstsq(
                    gram, projection, rcond=1.0e-14
                )[0]
            current_active -= gamma @ delta_inputs
            residual_active -= gamma @ delta_residuals
        mixed = current + self.beta * residual
        mixed[self.active] = current_active + self.beta * residual_active
        zero = np.flatnonzero(self.local_g2 <= 1.0e-14)
        if zero.size:
            mixed[zero] = current[zero]
        self.previous_input = saved_input
        self.previous_residual = saved_residual
        return np.real(self.workspace.coefficients_to_grid(mixed))
