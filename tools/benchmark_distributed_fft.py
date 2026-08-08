"""MPI microbenchmark for the persistent-stick and FFTW-MPI FFT paths.

Run, for example:

    mpiexec -n 4 python tools/benchmark_distributed_fft.py --shape 72 --bands 8

"""

from __future__ import annotations

import argparse
import time

import numpy as np

from qepy_pw.basis import FFTGridDescriptor, FFTScratchPool, LocalPotentialWorkspace
from qepy_pw.memory import current_pss_bytes
from qepy_pw.mpi import MPIContext


def _sphere_indices(shape: tuple[int, int, int], fraction: float) -> np.ndarray:
    axes = [np.rint(np.fft.fftfreq(n) * n).astype(np.int32) for n in shape]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij", sparse=True)
    radius = fraction * min(shape)
    mask = gx * gx + gy * gy + gz * gz <= radius * radius
    slots = np.column_stack(np.nonzero(mask))
    return np.column_stack(
        [axes[axis][slots[:, axis]] for axis in range(3)]
    ).astype(np.int32, copy=False)


def _stick_benchmark(
    mpi: MPIContext,
    shape: tuple[int, int, int],
    bands: int,
    iterations: int,
) -> tuple[float, float, int, int]:
    indices = _sphere_indices(shape, 0.38)
    descriptor = FFTGridDescriptor.build(
        indices, shape, mpi.size, local_rank=mpi.rank
    )
    pool = FFTScratchPool()
    workspace = LocalPotentialWorkspace(
        indices,
        shape,
        mpi=mpi,
        scratch_pool=pool,
        descriptor=descriptor,
    )
    global_rows = workspace.local_plane_wave_indices.astype(np.float64)
    band_indices = np.arange(bands, dtype=np.float64)
    coefficients = np.asfortranarray(
        np.sin(0.013 * global_rows[:, None] + 0.17 * band_indices[None, :])
        + 1j
        * np.cos(0.019 * global_rows[:, None] - 0.11 * band_indices[None, :])
    )
    local_z = mpi.slab(shape[2])
    potential = np.ones(
        (local_z.stop - local_z.start, shape[0], shape[1]), dtype=np.float64
    )
    result = np.empty_like(coefficients, order="F")
    mpi.barrier()
    warmup_started = time.perf_counter()
    workspace.apply(
        potential,
        coefficients,
        native_potential_layout=True,
        out=result,
    )
    mpi.barrier()
    warmup = time.perf_counter() - warmup_started
    started = time.perf_counter()
    for _ in range(iterations):
        workspace.apply(
            potential,
            coefficients,
            native_potential_layout=True,
            out=result,
        )
    mpi.barrier()
    elapsed = time.perf_counter() - started
    local_error = float(np.max(np.abs(result - coefficients)))
    error = mpi.max_scalar(local_error)
    pss = int(mpi.sum_scalar(current_pss_bytes()))
    return elapsed / iterations, error, pss, int(round(warmup * 1.0e9))


def _fftw_mpi_benchmark(
    mpi: MPIContext,
    shape: tuple[int, int, int],
    iterations: int,
) -> tuple[float, float, int, int]:
    from qepy_pw import _native_fft

    if not _native_fft.fftw_mpi_available():
        return float("nan"), float("nan"), 0, 0
    # Native slab storage is (z,x,y), matching the first-dimension ownership
    # required by FFTW-MPI without a Python transpose.
    dimensions = (shape[2], shape[0], shape[1])
    allocation, local_n0, local_start = _native_fft.fftw_mpi_layout(
        dimensions, mpi.comm
    )
    active = local_n0 * shape[0] * shape[1]
    buffer = np.empty(allocation, dtype=np.complex128)
    points = np.arange(active, dtype=np.float64)
    buffer[:active] = (
        np.sin(0.007 * (points + local_start * shape[0] * shape[1]))
        + 1j
        * np.cos(0.009 * (points + local_start * shape[0] * shape[1]))
    )
    reference = buffer[:active].copy()
    plan = _native_fft.NativeFFTWMPIPlan(
        buffer, dimensions, mpi.comm, 0, 64, 1
    )
    plan.execute("forward")
    plan.execute("backward")
    buffer[:active] /= np.prod(shape)
    local_error = float(np.max(np.abs(buffer[:active] - reference)))
    error = mpi.max_scalar(local_error)
    buffer[:active] = reference
    mpi.barrier()
    started = time.perf_counter()
    for _ in range(iterations):
        plan.execute("forward")
        plan.execute("backward")
        buffer[:active] /= np.prod(shape)
    mpi.barrier()
    elapsed = time.perf_counter() - started
    pss = int(mpi.sum_scalar(current_pss_bytes()))
    return elapsed / iterations, error, pss, allocation - active


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, default=72)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    mpi = MPIContext.world()
    shape = (args.shape, args.shape, args.shape)
    stick = _stick_benchmark(mpi, shape, args.bands, args.iterations)
    fftw_mpi = _fftw_mpi_benchmark(mpi, shape, args.iterations)
    if mpi.is_root:
        mib = 1024.0**2
        print(
            "stick_apply_seconds=" f"{stick[0]:.8f} "
            "max_error=" f"{stick[1]:.3e} "
            "aggregate_pss_mib=" f"{stick[2] / mib:.2f} "
            "first_apply_seconds=" f"{stick[3] / 1.0e9:.8f}"
        )
        print(
            "fftw_mpi_roundtrip_seconds=" f"{fftw_mpi[0]:.8f} "
            "max_error=" f"{fftw_mpi[1]:.3e} "
            "aggregate_pss_mib=" f"{fftw_mpi[2] / mib:.2f} "
            "extra_complex_per_rank=" f"{fftw_mpi[3]}"
        )


if __name__ == "__main__":
    main()
