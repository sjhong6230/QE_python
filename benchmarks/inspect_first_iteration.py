"""Run one SCF step and print the numerical diagnostics relevant to Davidson."""

from __future__ import annotations

import sys
from pathlib import Path

from qepy_pw.input import read_pw_input
from qepy_pw.scf import SCFSetup, run_scf


input_path = Path(sys.argv[1] if len(sys.argv) > 1 else "Si.scf.in")
pw = read_pw_input(input_path)
pw.electrons["electron_maxstep"] = 1
if len(sys.argv) > 2:
    pw.electrons["startingwfc"] = sys.argv[2]
if len(sys.argv) > 3:
    value = sys.argv[3].lower()
    pw.electrons["py_davidson_residual_energy_scale"] = (
        None if value == "none" else float(value)
    )
if len(sys.argv) > 4:
    pw.electrons["py_cache_projectors"] = (
        sys.argv[4].lower() not in {"false", "no", "0", "off"}
    )
if len(sys.argv) > 5:
    pw.electrons["py_numba"] = (
        sys.argv[5].lower() not in {"false", "no", "0", "off"}
    )
events: list[tuple[str, object]] = []
result = run_scf(
    pw,
    progress=lambda kind, payload: events.append((kind, payload)),
)
step = result.iterations[0]
setup = next(
    payload
    for kind, payload in events
    if kind == "setup" and isinstance(payload, SCFSetup)
)

if result.density is not None:
    print(
        "FIRST_ITERATION",
        f"energy_ry={2.0 * step.total_energy_ha:.12f}",
        f"accuracy_ry={2.0 * step.estimated_accuracy_ha:.12e}",
        f"max_eigen_residual_ha={step.maximum_eigen_residual_ha:.12e}",
        f"average_davidson_iterations="
        f"{step.average_diagonalization_iterations:.6f}",
        f"average_hpsi_vectors="
        f"{step.average_hamiltonian_applications:.6f}",
        f"projector_cache_mb="
        f"{setup.projector_cache_bytes_per_rank / 1.0e6:.6f}",
        f"persistent_arrays_mb="
        f"{setup.estimated_persistent_bytes_per_rank / 1.0e6:.6f}",
        f"peak_workspace_mb="
        f"{setup.estimated_peak_workspace_bytes_per_rank / 1.0e6:.6f}",
        f"peak_rss_mib="
        f"{result.peak_rss_bytes_per_rank / 2**20:.6f}",
        f"init_us_2_wall_s="
        f"{result.timings['init_us_2'].wall_seconds:.6f}",
        f"wall_s={result.wall_seconds:.6f}",
    )
