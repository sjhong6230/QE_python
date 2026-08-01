"""Run a full SCF variant and print compact convergence/performance data."""

from __future__ import annotations

import sys
from pathlib import Path

from qepy_pw.input import read_pw_input
from qepy_pw.mpi import MPIContext
from qepy_pw.scf import run_scf


input_path = Path(sys.argv[1] if len(sys.argv) > 1 else "Si.scf.in")
residual_scale = sys.argv[2].lower() if len(sys.argv) > 2 else "default"
pw = read_pw_input(input_path)
if residual_scale != "default":
    pw.electrons["py_davidson_residual_energy_scale"] = (
        None if residual_scale == "none" else float(residual_scale)
    )
if len(sys.argv) > 3:
    pw.electrons["py_cache_projectors"] = (
        sys.argv[3].lower() not in {"false", "no", "0", "off"}
    )
if len(sys.argv) > 4:
    pw.electrons["py_numba"] = (
        sys.argv[4].lower() not in {"false", "no", "0", "off"}
    )
mpi = MPIContext.world()
result = run_scf(pw, mpi=mpi)

if mpi.is_root:
    print(
        "SCF_VARIANT",
        f"converged={result.converged}",
        f"iterations={len(result.iterations)}",
        f"energy_ry={2.0 * result.total_energy_ha:.12f}",
        f"peak_rss_mib={result.peak_rss_bytes_per_rank / 2**20:.6f}",
        f"wall_s={result.wall_seconds:.6f}",
        f"c_bands_s={result.timings['c_bands'].wall_seconds:.6f}",
        f"h_psi_s={result.timings['h_psi'].wall_seconds:.6f}",
        f"fftw_s={result.timings['fftw'].wall_seconds:.6f}",
        f"sum_band_s={result.timings['sum_band'].wall_seconds:.6f}",
    )
