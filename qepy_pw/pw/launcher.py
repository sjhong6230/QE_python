"""Minimal console launcher configuring native runtimes before NumPy import."""

from __future__ import annotations

import os
import sys


def configured_environment(source: dict[str, str]) -> dict[str, str]:
    """Return conservative allocator/thread defaults for one rank process."""
    environment = dict(source)
    # Start numerical libraries with the smallest pool. run_scf raises the
    # active limit after parsing inputs when a serial reduced-k calculation
    # has BLAS shapes large enough to benefit. This avoids constructing idle
    # workers for dense-nosym and every multi-rank calculation.
    blas_threads = 1

    # These values are read by glibc and numerical runtimes during process/
    # shared-library initialization, hence the exec before importing NumPy.
    environment.setdefault("MALLOC_ARENA_MAX", "1")
    environment.setdefault("OMP_STACKSIZE", "2M")
    environment.setdefault("GOMP_STACKSIZE", "2M")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        # run_scf enforces this same policy through threadpoolctl. Setting it
        # before import prevents OpenBLAS/MKL from first constructing a larger
        # per-rank worker pool whose stacks and control blocks remain resident.
        environment[name] = str(blas_threads)
    if "OMPI_COMM_WORLD_SIZE" in environment:
        environment.setdefault("OMPI_MCA_osc", "sm")
    environment["QEPY_RUNTIME_CONFIGURED"] = "1"
    return environment


def main() -> int:
    """Re-exec the module once so allocator controls precede native imports."""
    if sys.platform.startswith("linux") and os.environ.get(
        "QEPY_RUNTIME_CONFIGURED"
    ) != "1":
        environment = configured_environment(dict(os.environ))
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "qepy_pw", *sys.argv[1:]],
            environment,
        )
    from .cli import main as cli_main

    return cli_main()
