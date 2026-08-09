"""QE-shaped text reporting for the spinless SCF driver.

Keeping presentation here prevents numerical code from accumulating print
branches and gives the CLI and regression-reference generator one formatter.
"""

from __future__ import annotations

import hashlib
import io
from typing import TYPE_CHECKING

import numpy as np

from ..constants import EV_PER_HARTREE
from .input import PWInput
from ..memory import format_bytes
from ..occupations import default_number_of_bands
from ..qe_format import (
    format_qe_closing,
    format_qe_duration,
    qe_date_and_time,
)
from ..upf import read_upf
from ..xc import canonical_xc_name
from ..version import __version__

if TYPE_CHECKING:
    from .scf import (
        ProgressKind,
        ProgressPayload,
        SCFIteration,
        SCFResult,
        SCFSetup,
    )


def _clean_zero(value: float, tolerance: float = 5.0e-12) -> float:
    """Avoid negative zero in fields where QE prints an exact zero."""
    return 0.0 if abs(value) < tolerance else float(value)


def _qe_symmetry_operations(pw: PWInput):
    """Return operations in a stable, QE-like order, with identity first."""
    identity = np.eye(3, dtype=int)
    return tuple(
        sorted(
            pw.symmetry_operations,
            key=lambda op: (
                0 if np.array_equal(op.matrix, identity) and np.allclose(op.translation, 0.0) else 1,
                0 if np.allclose(op.translation, 0.0) else 1,
                tuple(int(value) for value in op.matrix.ravel()),
                tuple(float(value) for value in op.translation),
            ),
        )
    )


def _format_symmetry(out: io.StringIO, pw: PWInput) -> None:
    operations = _qe_symmetry_operations(pw)
    fractional = sum(not np.allclose(op.translation, 0.0) for op in operations)
    has_inversion = any(np.array_equal(op.matrix, -np.eye(3, dtype=int)) for op in operations)
    if len(operations) == 1:
        print("\n     No symmetry found", file=out)
    else:
        inversion = ", with inversion," if has_inversion else " (no inversion)"
        suffix = (
            f" ({fractional:2d} have fractional translation)" if fractional else ""
        )
        print(
            f"\n     {len(operations):2d} Sym. Ops.{inversion} found{suffix}",
            file=out,
        )
    if str(pw.control.get("verbosity", "low")).lower() != "high":
        print(file=out)
        return
    print("\n                                    s                        frac. trans.", file=out)
    inverse_lattice = np.linalg.inv(pw.lattice)
    for index, operation in enumerate(operations, start=1):
        from ..point_group import operation_description

        label = operation_description(pw.lattice, operation)
        print(f"\n      isym = {index:2d}     {label:<45s}\n", file=out)
        translation = np.asarray(operation.translation, dtype=float)
        has_translation = not np.allclose(translation, 0.0)
        for row_index, row in enumerate(operation.matrix):
            prefix = f" cryst.   s({index:2d}) = (" if row_index == 0 else "                  ("
            tail = ""
            if has_translation:
                marker = "    f =(" if row_index == 0 else "       ("
                tail = f"{marker} {_clean_zero(translation[row_index]):10.7f} )"
            print(f"{prefix}{row[0]:6d}{row[1]:11d}{row[2]:11d}      ){tail}", file=out)
        cartesian = inverse_lattice @ operation.matrix @ pw.lattice
        cart_translation = translation @ pw.lattice / float(
            pw.system.get("celldm(1)", np.linalg.norm(pw.lattice[0]))
        )
        print(file=out)
        for row_index, row in enumerate(cartesian):
            prefix = f" cart.    s({index:2d}) = (" if row_index == 0 else "                  ("
            tail = ""
            if has_translation:
                marker = "    f =(" if row_index == 0 else "       ("
                tail = f"{marker} {_clean_zero(cart_translation[row_index]):10.7f} )"
            values = [_clean_zero(value) for value in row]
            print(f"{prefix}{values[0]:11.7f}{values[1]:11.7f}{values[2]:11.7f} ){tail}", file=out)
        print(file=out)

    from ..point_group import point_group_character_table

    table = point_group_character_table(pw, operations)
    print(
        f"     point group "
        f"{f'{table.schoenflies} ({table.international})':<11s}",
        file=out,
    )
    print(f"     there are{len(table.classes):3d} classes", file=out)
    print("     the character table:\n", file=out)
    print("       " + "".join(f"{item.label:<6s}" for item in table.classes), file=out)
    for name, characters in table.irreps:
        print(
            f"{name:<5s}" + "".join(
                f"{value.real:6.2f}" for value in characters
            ),
            file=out,
        )
    if any(
        abs(value.imag) > 1.0e-8
        for _name, row in table.irreps
        for value in row
    ):
        print("     imaginary part", file=out)
        for name, characters in table.irreps:
            print(
                f"{name:<5s}" + "".join(
                    f"{value.imag:6.2f}" for value in characters
                ),
                file=out,
            )
    print(
        "\n     the symmetry operations in each class and the name "
        "of the first element:\n",
        file=out,
    )
    for item in table.classes:
        indices = "".join(f"{index:5d}" for index in item.operation_indices)
        print(f"     {item.label:<5s}{indices}", file=out)
        print(f"          {item.description}", file=out)


def format_header(pw: PWInput) -> str:
    out = io.StringIO()
    cdate, ctime = qe_date_and_time()
    print(
        f"\n     Program PWSCF-PY v.{__version__} starts on "
        f"{cdate} at {ctime}",
        file=out,
    )
    print("\n     Python reference port of the scalar SCF path in Quantum ESPRESSO pw.x", file=out)
    print(f"     Reading input from {pw.source}\n", file=out)
    from ..errors import QEWarning, format_qe_warning

    for diagnostic in pw.warnings:
        print(format_qe_warning(diagnostic), end="", file=out)
    if pw.warnings:
        print(file=out)
    pseudos = {
        species.label: read_upf(pw.pseudo_dir / species.pseudo_file)
        for species in pw.species
    }
    ecutwfc = float(pw.system["ecutwfc"])
    ecutrho = float(pw.system.get("ecutrho", 4.0 * ecutwfc))
    if (
        ecutrho > 4.0 * ecutwfc
        and all(pseudo.core_density is None for pseudo in pseudos.values())
    ):
        print(
            format_qe_warning(
                QEWarning("setup", "no reason to have ecutrho>4*ecutwfc")
            ),
            end="",
            file=out,
        )
        print(file=out)
    nelec = sum(pseudos[a.label].z_valence for a in pw.atoms) - float(pw.system.get("tot_charge", 0.0))
    nbnd = int(
        pw.system.get(
            "nbnd",
            default_number_of_bands(
                nelec, pw.system.get("occupations", "fixed")
            ),
        )
    )
    ibrav = int(pw.system.get("ibrav", 0))
    alat = float(
        pw.system.get("celldm(1)", np.linalg.norm(pw.lattice[0]))
    )
    reciprocal_unit = 2.0 * np.pi / alat
    crystal_axes = pw.lattice / alat
    reciprocal_axes = pw.reciprocal / reciprocal_unit
    print(f"     bravais-lattice index     = {ibrav:12d}", file=out)
    print(f"     lattice parameter (alat)  = {alat:12.4f}  a.u.", file=out)
    print(f"     unit-cell volume          = {pw.volume:12.4f} (a.u.)^3", file=out)
    print(f"     number of atoms/cell      = {len(pw.atoms):12d}", file=out)
    print(f"     number of atomic types    = {len(pw.species):12d}", file=out)
    print(f"     number of electrons       = {nelec:12.2f}", file=out)
    print(f"     number of Kohn-Sham states= {nbnd:12d}", file=out)
    print(f"     kinetic-energy cutoff     = {float(pw.system['ecutwfc']):12.4f}  Ry", file=out)
    print(
        f"     charge-density cutoff     = "
        f"{float(pw.system.get('ecutrho', 4.0 * float(pw.system['ecutwfc']))):12.4f}  Ry",
        file=out,
    )
    print(
        f"     scf convergence threshold = "
        f"{float(pw.electrons.get('conv_thr', 1.0e-6)):12.1E}",
        file=out,
    )
    print(
        f"     mixing beta               = "
        f"{float(pw.electrons.get('mixing_beta', 0.7)):12.4f}", file=out
    )
    print(
        f"     number of iterations used = "
        f"{int(pw.electrons.get('mixing_ndim', 8)):12d}  "
        f"{str(pw.electrons.get('mixing_mode', 'plain')):<9s} mixing",
        file=out,
    )
    pulay_frequency = int(
        pw.electrons.get("mixing_pulay_frequency", 1)
    )
    if pulay_frequency > 1:
        print(
            "     periodic Pulay interval  = "
            f"{pulay_frequency:12d}",
            file=out,
        )
    resolved_xc = canonical_xc_name(pw.system.get("_resolved_xc", ""))
    if resolved_xc is None and "input_dft" in pw.system:
        resolved_xc = canonical_xc_name(pw.system["input_dft"])
    if resolved_xc is None:
        pseudo_xc = {
            canonical
            for pseudo in pseudos.values()
            if (canonical := canonical_xc_name(pseudo.functional)) is not None
        }
        resolved_xc = pseudo_xc.pop() if len(pseudo_xc) == 1 else "pz"
    xc_output = {
        "pz": ("PZ", "1   1   0   0"),
        "pw": ("PW", "1   4   0   0"),
        "pbe": ("PBE", "1   4   3   4"),
        "pbesol": ("PBEsol", "1   4  10   8"),
        "revpbe": ("revPBE", "1   4   4   4"),
        "rpbe": ("RPBE", "1   4  44   4"),
    }
    short_xc, xc_indices = xc_output[resolved_xc]
    print(f"     Exchange-correlation= {short_xc}", file=out)
    print(
        f"                           (   {xc_indices}   0   0   0)",
        file=out,
    )
    celldm = [alat] + [
        float(pw.system.get(f"celldm({index})", 0.0))
        for index in range(2, 7)
    ]
    print(file=out)
    for start in (0, 3):
        print(
            "   "
            + "".join(
                f"  celldm({index + 1})={celldm[index]:11.6f}"
                for index in range(start, start + 3)
            ),
            file=out,
        )
    print("\n     crystal axes: (cart. coord. in units of alat)", file=out)
    for index, vector in enumerate(crystal_axes, start=1):
        print(
            f"               a({index}) = ("
            f" {vector[0]:10.6f} {vector[1]:10.6f} {vector[2]:10.6f} )",
            file=out,
        )
    print("\n     reciprocal axes: (cart. coord. in units 2 pi/alat)", file=out)
    for index, vector in enumerate(reciprocal_axes, start=1):
        print(
            f"               b({index}) = ("
            f"{vector[0]:10.6f}{vector[1]:10.6f}{vector[2]:10.6f} )",
            file=out,
        )
    for index, species in enumerate(pw.species, start=1):
        pseudo = pseudos[species.label]
        pseudo_path = pw.pseudo_dir / species.pseudo_file
        print(
            f"\n     PseudoPot. # {index} for {species.label} read from file:\n"
            f"     {pseudo_path}",
            file=out,
        )
        print(
            f"     MD5 check sum: "
            f"{hashlib.md5(pseudo_path.read_bytes()).hexdigest()}",
            file=out,
        )
        print(
            f"     Pseudo is Norm-conserving, Zval = {pseudo.z_valence:5.1f}",
            file=out,
        )
        if pseudo.generated:
            print(f"     Generated using {pseudo.generated}", file=out)
        if pseudo.mesh_size or pseudo.projectors:
            print(f"     Using radial grid of {pseudo.mesh_size:d} points,  "
                  f"{len(pseudo.projectors):d} beta functions with: ", file=out)
        if pseudo.projectors:
            for projector in pseudo.projectors:
                print(f"                l({projector.index}) = {projector.angular_momentum:3d}", file=out)
    print(
        "\n     atomic species   valence    mass     pseudopotential",
        file=out,
    )
    for index, species in enumerate(pw.species, start=1):
        print(
            f"     {species.label:<16s}"
            f"{pseudos[species.label].z_valence:4.2f}"
            f"{species.mass:12.5f}     {species.label}({index:5.2f})",
            file=out,
        )
    _format_symmetry(out, pw)
    inverse_lattice = np.linalg.inv(pw.lattice)
    print("\n   Cartesian axes", file=out)
    print(
        "\n     site n.     atom                  positions (alat units)",
        file=out,
    )
    for index, atom in enumerate(pw.atoms, start=1):
        cart = atom.position / alat
        print(
            f"     {index:5d}        {atom.label:<4s}   tau({index:4d}) = ("
            f" {cart[0]:11.7f} {cart[1]:11.7f} {cart[2]:11.7f}  )",
            file=out,
        )
    print("\n   Crystallographic axes", file=out)
    print(
        "\n     site n.     atom                  positions (cryst. coord.)",
        file=out,
    )
    for index, atom in enumerate(pw.atoms, start=1):
        crystal = atom.position @ inverse_lattice
        print(
            f"     {index:5d}        {atom.label:<4s}   tau({index:4d}) = ("
            f" {crystal[0]:10.7f} {crystal[1]:10.7f} {crystal[2]:10.7f}  )",
            file=out,
        )
    if pw.full_kpoint_count != len(pw.kpoints):
        print(f"     number of k points= {len(pw.kpoints):5d}\n", file=out)
    else:
        print(f"     number of k points= {len(pw.kpoints):5d}\n", file=out)
    print("                       cart. coord. in units 2pi/alat", file=out)
    for index, point in enumerate(pw.kpoints, start=1):
        cart = (point.crystal @ pw.reciprocal) / reciprocal_unit
        print(
            f"        k({index:5d}) = ("
            f" {cart[0]:11.7f} {cart[1]:11.7f} {cart[2]:11.7f}), "
            f"wk = {2.0 * point.weight:11.7f}",
            file=out,
        )
    print("\n                       cryst. coord.", file=out)
    for index, point in enumerate(pw.kpoints, start=1):
        print(
            f"        k({index:5d}) = ("
            f" {point.crystal[0]:11.7f} {point.crystal[1]:11.7f} "
            f"{point.crystal[2]:11.7f}), wk = {2.0 * point.weight:11.7f}",
            file=out,
        )
    print("", file=out)
    return out.getvalue()


def format_setup(setup: SCFSetup) -> str:
    out = io.StringIO()
    solver_names = {
        "david": "Davidson",
        "cg": "CG",
        "paro": "ParO",
        "rmm-davidson": "RMM-DIIS/Davidson",
        "rmm-paro": "RMM-DIIS/ParO",
    }
    if setup.diagonalization == "dense":
        print("     Dense reference-solver setup", file=out)
    else:
        print(
            f"     {solver_names[setup.diagonalization]} "
            "iterative-solver setup",
            file=out,
        )
    print(f"     active k points           = {setup.kpoints:12d}", file=out)
    print(
        f"     plane waves per k point   = {setup.min_plane_waves:6d} to "
        f"{setup.max_plane_waves:<6d}",
        file=out,
    )
    print(f"     number of Kohn-Sham states= {setup.number_of_bands:12d}", file=out)
    print(
        f"     Dense grid: {setup.charge_gvectors:8d} G-vectors     "
        f"FFT dimensions: ({setup.fft_shape[0]:4d}, "
        f"{setup.fft_shape[1]:4d}, {setup.fft_shape[2]:4d})",
        file=out,
    )
    print("\n     G-vector sticks info", file=out)
    print("     --------------------", file=out)
    print(
        "     sticks:   dense  smooth     PW     "
        "G-vecs:    dense   smooth      PW",
        file=out,
    )
    print(
        f"     Sum      {setup.charge_sticks:6d}"
        f"{setup.charge_sticks:8d}{setup.wavefunction_sticks:7d}"
        f"{setup.charge_gvectors:21d}{setup.charge_gvectors:8d}"
        f"{setup.max_plane_waves:8d}",
        file=out,
    )
    print("\n     Using Slab Decomposition", file=out)
    print(
        "     FFT backend              = Cython/MPI/FFTW",
        file=out,
    )
    print(
        f"     MPI FFT processes        = {setup.mpi_processes:12d}",
        file=out,
    )
    print(
        f"     Threads per MPI process  = {setup.threads_per_process:12d}",
        file=out,
    )
    for message in setup.starting_messages:
        print(f"     {message}", file=out)
    if setup.starting_wavefunctions == "file":
        print("     Starting wfcs from file", file=out)
    elif setup.starting_wavefunctions == "random":
        print("     Starting wfcs are random", file=out)
    else:
        atomic = setup.atomic_orbitals
        missing = max(0, setup.number_of_bands - atomic)
        if setup.starting_wavefunctions == "atomic":
            line = f"     Starting wfcs are {atomic:4d} atomic wfcs"
            if missing:
                line += f" + {missing:4d} random wfcs"
        elif atomic:
            line = (
                f"     Starting wfcs are {atomic:4d} "
                "randomized atomic wfcs"
            )
            if missing:
                line += f" + {missing:4d} random wfcs"
        else:
            line = "     Starting wfcs are random"
        print(line, file=out)
    if setup.starting_potential == "file":
        print("\n     Initial potential from file", file=out)
    else:
        print("\n     Initial potential from superposition of free atoms", file=out)
    print(
        f"     starting charge {setup.starting_charge:12.4f}, "
        f"renormalised to {setup.target_charge:12.4f}",
        file=out,
    )
    if setup.diagonalization == "david":
        print(
            f"     maximum Davidson subspace = "
            f"{setup.davidson_subspace:12d}",
            file=out,
        )
        print("     full Hamiltonian matrices =      avoided", file=out)
        print(file=out)
    elif setup.diagonalization == "dense":
        print(
            f"     dense work proxy sum(N^3) = {setup.dense_work:12d}\n",
            file=out,
        )
    else:
        print("     full Hamiltonian matrices =      avoided", file=out)
        print(file=out)
    estimated_array = (
        setup.estimated_persistent_bytes_per_rank
        + setup.estimated_peak_workspace_bytes_per_rank
    )
    baseline_all_ranks = (
        setup.runtime_baseline_pss_bytes_all_ranks
        or setup.runtime_baseline_rss_bytes_per_rank * setup.mpi_processes
    )
    estimated_total = (
        baseline_all_ranks
        + (
            setup.estimated_array_bytes_all_ranks
            or estimated_array * setup.mpi_processes
        )
    )
    print(
        "     Estimated array memory/rank > "
        f"{format_bytes(estimated_array)}",
        file=out,
    )
    for label, size in setup.estimated_array_components_per_rank:
        print(
            f"       {label:<31} {format_bytes(size):>12}",
            file=out,
        )
    print(
        "     Estimated total RAM, all ranks > "
        f"{format_bytes(estimated_total)}",
        file=out,
    )
    if setup.calculation in {"nscf", "bands"}:
        print("\n     Band Structure Calculation", file=out)
    print("", file=out)
    return out.getvalue()


def _format_force_component(
    out: io.StringIO, title: str, values: np.ndarray, species_types: list[int]
) -> None:
    print(f"     {title}", file=out)
    for index, (atom_type, force) in enumerate(
        zip(species_types, values), start=1
    ):
        force_ry = 2.0 * force
        force_ry = np.asarray([_clean_zero(value, 5.0e-9) for value in force_ry])
        print(
            f"     atom {index:4d} type {atom_type:2d}   force = "
            f"{force_ry[0]:14.8f}{force_ry[1]:14.8f}{force_ry[2]:14.8f}",
            file=out,
        )


def _format_stress_component(
    out: io.StringIO, title: str, values: np.ndarray
) -> None:
    hartree_kbar = (
        10.0 * 4.3597447222071e-18 / 0.529177210903e-10**3 / 1.0e9
    )
    for row_index, row in enumerate(hartree_kbar * values):
        prefix = f"     {title}" if row_index == 0 else " " * 26
        print(
            prefix
            + "".join(
                f"{_clean_zero(value, 5.0e-3):10.2f}" for value in row
            ),
            file=out,
        )
    print(file=out)


def format_iteration(step: SCFIteration) -> str:
    out = io.StringIO()
    print(
        f"     iteration # {step.number:3d}"
        f"     ecut= {step.cutoff_ry:8.2f} Ry"
        f"     beta= {step.mixing_beta:4.2f}",
        file=out,
    )
    if step.davidson_threshold_ha > 0.0:
        solver_lines = {
            "david": "Davidson diagonalization with overlap",
            "cg": "CG style diagonalization",
            "paro": "ParO style diagonalization",
            "rmm": "RMM-DIIS diagonalization",
            "rmm-davidson": "RMM-DIIS diagonalization",
            "rmm-paro": "RMM-DIIS diagonalization",
            "dense": "Direct diagonalization of the dense Hamiltonian matrix",
        }
        print(f"     {solver_lines[step.diagonalization]}", file=out)
        print(
            f"     ethr = {2.0 * step.davidson_threshold_ha:9.2E},  "
            f"avg # of iterations = "
            f"{step.average_diagonalization_iterations:4.1f}",
            file=out,
        )
        if step.maximum_eigen_residual_ha > 0.0:
            print(
                f"     avg # of Hpsi vectors = "
                f"{step.average_hamiltonian_applications:5.1f}, "
                f"max residual = "
                f"{step.maximum_eigen_residual_ha:9.2E} Ha",
                file=out,
            )
    print(
        f"     total cpu time spent up to now is "
        f"{step.cpu_seconds:10.1f} secs",
        file=out,
    )
    print(f"     total energy              ={step.total_energy_ha * 2:17.8f} Ry", file=out)
    if np.isfinite(step.estimated_accuracy_ha):
        print(
            f"     estimated scf accuracy    <"
            f"{step.estimated_accuracy_ha * 2:17.8f} Ry\n",
            file=out,
        )
    return out.getvalue()


def format_progress(kind: ProgressKind, payload: ProgressPayload) -> str:
    """Format a typed event produced by :func:`qepy_pw.scf.run_scf`.

    The local import preserves the package's lightweight import path while
    the explicit checks turn a mismatched callback event into a useful error.
    """
    from .scf import SCFIteration, SCFSetup

    if kind == "setup" and isinstance(payload, SCFSetup):
        return format_setup(payload)
    if kind == "iteration" and isinstance(payload, SCFIteration):
        return format_iteration(payload)
    raise TypeError(
        f"progress event {kind!r} does not match {type(payload).__name__}"
    )


def _format_timing(result: SCFResult, name: str) -> str:
    entry = result.timings.get(name)
    if entry is None:
        return f"{0.0:10.2f}s CPU {0.0:9.2f}s WALL ({0:8d} calls)"
    return (
        f"{entry.cpu_seconds:10.2f}s CPU "
        f"{entry.wall_seconds:9.2f}s WALL "
        f"({entry.calls:8d} calls)"
    )


def format_footer(pw: PWInput, result: SCFResult) -> str:
    out = io.StringIO()
    calculation = str(pw.control.get("calculation", "scf")).strip().lower()
    if calculation in {"nscf", "bands"}:
        cpu_so_far = sum(
            entry.cpu_seconds
            for name in ("init_run", "electrons")
            if (entry := result.timings.get(name)) is not None
        )
        print(
            f"     total cpu time spent up to now is "
            f"{cpu_so_far:10.1f} secs\n",
            file=out,
        )
        print("     End of band structure calculation\n", file=out)
    elif result.converged:
        print("     End of self-consistent calculation\n", file=out)
    else:
        print("     convergence NOT achieved after the maximum number of iterations\n", file=out)
    alat = float(
        pw.system.get("celldm(1)", np.linalg.norm(pw.lattice[0]))
    )
    reciprocal_unit = 2.0 * np.pi / alat
    pseudos = {
        species.label: read_upf(pw.pseudo_dir / species.pseudo_file)
        for species in pw.species
    }
    nelec = sum(
        pseudos[atom.label].z_valence for atom in pw.atoms
    ) - float(pw.system.get("tot_charge", 0.0))
    occupied = int(round(nelec)) // 2
    occupations_mode = str(
        pw.system.get("occupations", "fixed")
    ).strip().lower()
    for index, (point, values) in enumerate(
        zip(pw.kpoints, result.eigenvalues_ha)
    ):
        cart = (point.crystal @ pw.reciprocal) / reciprocal_unit
        npw = (
            result.plane_waves_per_k[index]
            if index < len(result.plane_waves_per_k)
            else 0
        )
        print(
            f"          k ={cart[0]:7.4f}{cart[1]:7.4f}"
            f"{cart[2]:7.4f} ({npw:6d} PWs)   bands (ev):\n",
            file=out,
        )
        for start in range(0, len(values), 8):
            print(
                "  "
                + "".join(
                    f"{value * EV_PER_HARTREE:9.4f}"
                    for value in values[start:start + 8]
                ),
                file=out,
            )
        print("\n     occupation numbers", file=out)
        if index < len(result.occupations):
            displayed_occupations = 0.5 * result.occupations[index]
        else:
            displayed_occupations = np.zeros(len(values))
            displayed_occupations[:occupied] = 1.0
        for start in range(0, len(displayed_occupations), 8):
            print(
                "  "
                + "".join(
                    f"{occupation:9.4f}"
                    for occupation in displayed_occupations[start:start + 8]
                ),
                file=out,
            )
    if result.fermi_energy_ha is not None:
        print(
            f"\n     the Fermi energy is "
            f"{result.fermi_energy_ha * EV_PER_HARTREE:10.4f} ev",
            file=out,
        )
    else:
        highest_occupied = max(
            values[occupied - 1] for values in result.eigenvalues_ha
        )
        # For fixed occupations QE reports the fundamental band-edge pair
        # whenever the calculation includes at least one empty state.  The
        # HOMO is the largest occupied eigenvalue over all k points, whereas
        # the LUMO is the smallest first-empty eigenvalue.
        has_empty_states = all(
            len(values) > occupied for values in result.eigenvalues_ha
        )
        if has_empty_states:
            lowest_unoccupied = min(
                values[occupied] for values in result.eigenvalues_ha
            )
            print(
                "\n     highest occupied, lowest unoccupied level (ev):"
                f"{highest_occupied * EV_PER_HARTREE:11.4f}"
                f"{lowest_unoccupied * EV_PER_HARTREE:11.4f}",
                file=out,
            )
        else:
            print(
                f"\n     highest occupied level (ev): "
                f"{highest_occupied * EV_PER_HARTREE:10.4f}",
                file=out,
            )
    marker = "!" if result.converged else " "
    if calculation == "scf":
        print(f"\n{marker}    total energy              ={result.total_energy_ha * 2:17.8f} Ry", file=out)
    if result.iterations:
        print(
            f"     estimated scf accuracy    <"
            f"{2.0 * result.iterations[-1].estimated_accuracy_ha:17.8f} Ry",
            file=out,
        )
    if occupations_mode == "smearing" and result.energy_terms is not None:
        smearing_ry = 2.0 * result.energy_terms.smearing_ha
        print(
            f"     smearing contrib. (-TS)   ="
            f"{smearing_ry:17.8f} Ry",
            file=out,
        )
        print(
            f"     internal energy E=F+TS    ="
            f"{2.0 * result.total_energy_ha - smearing_ry:17.8f} Ry",
            file=out,
        )
    if result.energy_terms is not None:
        terms = result.energy_terms
        if occupations_mode == "smearing":
            print(
                "\n     The total energy is F=E-TS. E is the sum of the "
                "following terms:",
                file=out,
            )
        else:
            print(
                "\n     The total energy is the sum of the following terms:",
                file=out,
            )
        print(
            f"     one-electron contribution ="
            f"{2.0 * terms.one_electron_ha:17.8f} Ry",
            file=out,
        )
        print(
            f"     hartree contribution      ="
            f"{2.0 * terms.hartree_ha:17.8f} Ry",
            file=out,
        )
        print(
            f"     xc contribution           ="
            f"{2.0 * terms.xc_ha:17.8f} Ry",
            file=out,
        )
        print(
            f"     ewald contribution        ="
            f"{2.0 * terms.ewald_ha:17.8f} Ry",
            file=out,
        )
        if abs(terms.descf_ha) > 1.0e-12:
            print(
                f"     scf correction            ="
                f"{2.0 * terms.descf_ha:17.8f} Ry",
                file=out,
            )
    if result.forces_ha_per_bohr is not None:
        species_type_map = {
            species.label: index
            for index, species in enumerate(pw.species, start=1)
        }
        species_types = [species_type_map[atom.label] for atom in pw.atoms]
        print(
            "\n     Forces acting on atoms (cartesian axes, Ry/au):\n",
            file=out,
        )
        for index, (atom, force) in enumerate(
            zip(pw.atoms, result.forces_ha_per_bohr), start=1
        ):
            force_ry = 2.0 * force
            force_ry = np.asarray([
                _clean_zero(value, 5.0e-9) for value in force_ry
            ])
            print(
                f"     atom {index:4d} type "
                f"{species_type_map[atom.label]:2d}   force = "
                f"{force_ry[0]:14.8f}{force_ry[1]:14.8f}"
                f"{force_ry[2]:14.8f}",
                file=out,
            )
        if (
            str(pw.control.get("verbosity", "low")).lower() == "high"
            and result.force_terms is not None
        ):
            terms = result.force_terms
            _format_force_component(
                out, "The non-local contrib.  to forces",
                terms.nonlocal_ha_per_bohr, species_types,
            )
            _format_force_component(
                out, "The ionic contribution  to forces",
                terms.ionic_ha_per_bohr, species_types,
            )
            _format_force_component(
                out, "The local contribution  to forces",
                terms.local_ha_per_bohr, species_types,
            )
            _format_force_component(
                out, "The core correction contribution to forces",
                terms.core_correction_ha_per_bohr, species_types,
            )
            _format_force_component(
                out, "The SCF correction term to forces",
                terms.scf_correction_ha_per_bohr, species_types,
            )
        total_force_ry = 2.0 * float(
            np.linalg.norm(result.forces_ha_per_bohr)
        )
        scf_correction_ry = 0.0
        if result.force_terms is not None:
            scf_correction_ry = 2.0 * float(
                np.linalg.norm(result.force_terms.scf_correction_ha_per_bohr)
            )
        print(
            f"\n     Total force = {total_force_ry:12.6f}"
            f"     Total SCF correction = {scf_correction_ry:12.6f}",
            file=out,
        )
    if result.stress_ha_per_bohr3 is not None:
        hartree_kbar = (
            10.0
            * 4.3597447222071e-18
            / 0.529177210903e-10**3
            / 1.0e9
        )
        stress_ry = 2.0 * result.stress_ha_per_bohr3
        stress_kbar = hartree_kbar * result.stress_ha_per_bohr3
        pressure = float(np.trace(stress_kbar) / 3.0)
        print(
            "\n          total   stress  (Ry/bohr**3) "
            + " " * 18
            + f"(kbar)     P={pressure:12.2f}",
            file=out,
        )
        for row_ry, row_kbar in zip(stress_ry, stress_kbar):
            row_ry = np.asarray([
                _clean_zero(value, 5.0e-12) for value in row_ry
            ])
            row_kbar = np.asarray([
                _clean_zero(value, 5.0e-3) for value in row_kbar
            ])
            print(
                "     "
                + "".join(f"{value:13.8f}" for value in row_ry)
                + "    "
                + "".join(f"{value:12.2f}" for value in row_kbar),
                file=out,
            )
        print(file=out)
        if (
            str(pw.control.get("verbosity", "low")).lower() == "high"
            and result.stress_terms is not None
        ):
            terms = result.stress_terms
            _format_stress_component(out, "kinetic stress (kbar)", terms.kinetic_ha_per_bohr3)
            _format_stress_component(out, "local   stress (kbar)", terms.local_ha_per_bohr3)
            _format_stress_component(out, "nonloc. stress (kbar)", terms.nonlocal_ha_per_bohr3)
            _format_stress_component(out, "hartree stress (kbar)", terms.hartree_ha_per_bohr3)
            _format_stress_component(out, "exc-cor stress (kbar)", terms.xc_ha_per_bohr3)
            _format_stress_component(out, "corecor stress (kbar)", terms.core_correction_ha_per_bohr3)
            _format_stress_component(out, "ewald   stress (kbar)", terms.ewald_ha_per_bohr3)
    if result.converged and calculation == "scf":
        print(
            f"\n     convergence has been achieved in "
            f"{len(result.iterations):3d} iterations",
            file=out,
        )
    if result.peak_sampled_pss_bytes_all_ranks:
        print(
            "\n     Measured peak RAM, all ranks = "
            f"{format_bytes(result.peak_sampled_pss_bytes_all_ranks)} "
            "(aggregate PSS, sampled)",
            file=out,
        )
    else:
        measured_peak = (
            result.peak_rss_bytes_all_ranks
            or result.peak_rss_bytes_per_rank
        )
        print(
            "\n     Measured peak RAM, all ranks = "
            f"{format_bytes(measured_peak)} (RSS upper bound)",
            file=out,
        )
    print("\n     init_run     :"
          + _format_timing(result, "init_run"), file=out)
    print("     electrons    :"
          + _format_timing(result, "electrons"), file=out)
    if result.forces_ha_per_bohr is not None:
        print("     forces       :" + _format_timing(result, "forces"), file=out)
    if result.stress_ha_per_bohr3 is not None:
        print("     stress       :" + _format_timing(result, "stress"), file=out)
    if (
        result.forces_ha_per_bohr is not None
        or result.stress_ha_per_bohr3 is not None
    ):
        print(
            "     nonlocal deriv.:"
            + _format_timing(result, "nonlocal_derivatives"),
            file=out,
        )
    print("\n     Called by init_run:", file=out)
    for name in ("wfcinit", "potinit", "hinit0"):
        print(f"     {name:<13s}:" + _format_timing(result, name), file=out)
    print("     sym_rho:init :" + _format_timing(result, "sym_rho:init"), file=out)
    print("\n     Called by electrons:", file=out)
    for name in ("c_bands", "sum_band", "v_of_rho", "v_h", "v_xc", "mix_rho"):
        print(f"     {name:<13s}:" + _format_timing(result, name), file=out)
    print("\n     Called by sum_band:", file=out)
    for name in (
        "sum_band:wfc",
        "sum_band:rho",
        "sum_band:collect",
        "sum_band:sym",
    ):
        print(f"     {name:<16s}:" + _format_timing(result, name), file=out)
    print("\n     Called by c_bands:", file=out)
    print("     init_us_2    :" + _format_timing(result, "init_us_2"), file=out)
    print("     cegterg      :" + _format_timing(result, "cegterg"), file=out)
    for name in (
        "cdiaghg",
        "cegterg:over",
        "cegterg:upda",
        "cegterg:last",
        "g_psi",
    ):
        print(f"     {name:<13s}:" + _format_timing(result, name), file=out)
    print("\n     Called by h_psi:", file=out)
    for name in ("h_psi", "h_psi:calbec", "vloc_psi", "add_vuspsi", "fftw"):
        print(f"     {name:<13s}:" + _format_timing(result, name), file=out)
    print("     calbec       :" + _format_timing(result, "h_psi:calbec"), file=out)
    total_cpu = sum(
        result.timings.get(name).cpu_seconds
        for name in ("init_run", "electrons")
        if result.timings.get(name) is not None
    )
    print(
        "\n     PWSCF        : "
        f"{format_qe_duration(total_cpu, 'CPU')} "
        f"{format_qe_duration(result.wall_seconds, 'WALL')}\n",
        file=out,
    )
    print(format_qe_closing(success=result.converged), end="", file=out)
    return out.getvalue()


def format_output(pw: PWInput, result: SCFResult) -> str:
    return (
        format_header(pw)
        + "".join(format_iteration(step) for step in result.iterations)
        + format_footer(pw, result)
    )
