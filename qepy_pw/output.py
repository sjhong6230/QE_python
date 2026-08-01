"""QE-shaped text reporting for the Python scalar-SCF driver."""

from __future__ import annotations

from datetime import datetime
import hashlib
import io

import numpy as np

from .constants import EV_PER_HARTREE
from .input import PWInput
from .memory import format_bytes
from .occupations import default_number_of_bands
from .scf import SCFIteration, SCFResult, SCFSetup
from .upf import read_upf
from . import __version__


def format_header(pw: PWInput) -> str:
    out = io.StringIO()
    now = datetime.now()
    print(f"     Program PWSCF-PY v.{__version__} starts on {now:%d%b%Y at %H:%M:%S}", file=out)
    print("\n     Python reference port of the scalar SCF path in Quantum ESPRESSO pw.x", file=out)
    print(f"     Reading input from {pw.source}\n", file=out)
    pseudos = {s.label: read_upf(pw.pseudo_dir / s.pseudo_file) for s in pw.species}
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
    print(f"     kinetic-energy cutoff     = {float(pw.system['ecutwfc']):12.4f} Ry", file=out)
    print(
        f"     charge-density cutoff     = "
        f"{float(pw.system.get('ecutrho', 4.0 * float(pw.system['ecutwfc']))):12.4f} Ry",
        file=out,
    )
    print(
        f"     scf convergence threshold = "
        f"{float(pw.electrons.get('conv_thr', 1.0e-6)):12.1E}",
        file=out,
    )
    resolved_xc = str(pw.system.get("_resolved_xc", "")).lower()
    if not resolved_xc:
        requested = str(pw.system.get("input_dft", "")).lower()
        pseudo_functionals = " ".join(
            pseudo.functional.lower() for pseudo in pseudos.values()
        )
        resolved_xc = (
            "pbe"
            if (
                "pbe" in requested
                or "pbe" in pseudo_functionals
                or (
                    "pbx" in pseudo_functionals
                    and "pbc" in pseudo_functionals
                )
            )
            else "pz"
        )
    xc_label = (
        "SLA  PW   PBX  PBC"
        if resolved_xc == "pbe"
        else "SLA  PZ   NOGX NOGC"
    )
    print(f"     Exchange-correlation      =  {xc_label}", file=out)
    print(
        f"\n     celldm(1)= {alat:10.6f}  "
        + "  ".join(
            f"celldm({index})= "
            f"{float(pw.system.get(f'celldm({index})', 0.0)):10.6f}"
            for index in range(2, 7)
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
            f" {vector[0]:10.6f} {vector[1]:10.6f} {vector[2]:10.6f} )",
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
        print(
            f"     Using {len(pseudo.projectors):d} radial beta functions, "
            f"{pseudo.number_of_projector_channels:d} (l,m) channels",
            file=out,
        )
        if pseudo.projectors:
            print(
                "     Angular momenta: "
                + " ".join(
                    f"l({projector.index})={projector.angular_momentum}"
                    for projector in pseudo.projectors
                ),
                file=out,
            )
        print(
            "     Nonlinear core correction = "
            + ("present" if pseudo.has_nlcc else "absent"),
            file=out,
        )
        print(
            "     Atomic starting density   = "
            + ("present" if pseudo.has_atomic_density else "uniform fallback"),
            file=out,
        )
        print(
            f"     Atomic starting orbitals  = "
            f"{pseudo.number_of_atomic_orbitals:d}",
            file=out,
        )
    print(
        "\n     atomic species   valence       mass     pseudopotential",
        file=out,
    )
    for species in pw.species:
        print(
            f"     {species.label:<8s}"
            f"{pseudos[species.label].z_valence:12.2f}"
            f"{species.mass:12.5f}     {species.pseudo_file}",
            file=out,
        )
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
            f" {cart[0]:11.7f} {cart[1]:11.7f} {cart[2]:11.7f} )",
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
            f" {crystal[0]:11.7f} {crystal[1]:11.7f} {crystal[2]:11.7f} )",
            file=out,
        )
    mixing_mode = str(pw.electrons.get("mixing_mode", "plain")).lower()
    diagonalization = str(
        pw.electrons.get("diagonalization", "david")
    ).lower()
    print(
        f"\n     diagonalization           = {diagonalization}",
        file=out,
    )
    print(
        f"     charge mixing mode        = {mixing_mode}",
        file=out,
    )
    print(
        f"     mixing beta               = "
        f"{float(pw.electrons.get('mixing_beta', 0.7)):12.4f}",
        file=out,
    )
    if mixing_mode in {"plain", "default"}:
        print(
            f"     Broyden history dimension = "
            f"{int(pw.electrons.get('mixing_ndim', 8)):12d}",
            file=out,
        )
    print(f"\n     number of symmetry operations = {len(pw.symmetry_operations):5d}", file=out)
    identity = (
        len(pw.symmetry_operations) == 1
        and np.array_equal(
            pw.symmetry_operations[0].matrix, np.eye(3, dtype=int)
        )
    )
    if identity:
        print("     No symmetry found", file=out)
    if str(pw.control.get("verbosity", "low")).lower() == "high":
        for index, operation in enumerate(
            pw.symmetry_operations, start=1
        ):
            print(
                f"\n      isym = {index:3d}     crystal symmetry, "
                f"frac. trans. = ("
                f" {operation.translation[0]:9.6f}"
                f" {operation.translation[1]:9.6f}"
                f" {operation.translation[2]:9.6f} )",
                file=out,
            )
            for row_index, row in enumerate(operation.matrix):
                prefix = " cryst.   s = (" if row_index == 0 else "                ("
                print(
                    f"{prefix} {row[0]:4d} {row[1]:4d} {row[2]:4d} )",
                    file=out,
                )
    if pw.full_kpoint_count != len(pw.kpoints):
        print(
            f"     number of k points= {len(pw.kpoints):5d} "
            f"(reduced from {pw.full_kpoint_count})\n",
            file=out,
        )
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
    if setup.diagonalization == "david":
        print("     Davidson iterative-solver setup", file=out)
    else:
        print("     Dense reference-solver setup", file=out)
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
        f"     FFT backend              = {setup.fft_backend}",
        file=out,
    )
    print(
        f"     MPI FFT processes        = {setup.mpi_processes:12d}",
        file=out,
    )
    print(
        "     Numba scalar kernels    = "
        + ("enabled" if setup.numba_enabled else "disabled"),
        file=out,
    )
    if setup.fft_backend == "pyfftw":
        print(
            f"     FFTW planner/threads      = "
            f"{setup.fft_planner}/{setup.fft_threads}",
            file=out,
        )
    if setup.starting_wavefunctions == "file":
        print("     Starting wfcs read from file", file=out)
    elif setup.starting_wavefunctions == "random":
        print("     Starting wfcs are random wfcs", file=out)
    else:
        qualifier = (
            "randomized atomic wfcs"
            if setup.starting_wavefunctions == "atomic+random"
            else "atomic wfcs"
        )
        print(
            f"     Starting wfcs are {setup.atomic_orbitals:4d} {qualifier}",
            file=out,
        )
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
        print(
            "     nonlocal projector cache  = "
            + (
                f"enabled ({setup.projector_cache_bytes_per_rank / 1e6:.2f} "
                "MB/rank)"
                if setup.projector_cache_enabled
                else "disabled"
            ),
            file=out,
        )
        print(file=out)
    else:
        print(
            f"     dense work proxy sum(N^3) = {setup.dense_work:12d}\n",
            file=out,
        )
    persistent_mb = setup.estimated_persistent_bytes_per_rank / 1.0e6
    maximum_mb = (
        setup.estimated_persistent_bytes_per_rank
        + setup.estimated_peak_workspace_bytes_per_rank
    ) / 1.0e6
    print(
        f"     Estimated static dynamical RAM per process > "
        f"{persistent_mb:10.2f} MB",
        file=out,
    )
    print(
        f"\n     Estimated max dynamical RAM per process > "
        f"{maximum_mb:10.2f} MB",
        file=out,
    )
    if setup.mpi_processes > 1:
        print(
            f"\n     Estimated total dynamical RAM > "
            f"{maximum_mb * setup.mpi_processes:10.2f} MB",
            file=out,
        )
    print("", file=out)
    if setup.diagonalization == "dense" and setup.kpoints > 64:
        print(
            "     WARNING: the active mesh remains outside the default "
            "dense-solver k-point limit after symmetry handling.\n",
            file=out,
        )
    return out.getvalue()


def format_iteration(step: SCFIteration) -> str:
    out = io.StringIO()
    print(
        f"     iteration # {step.number:3d}"
        f"     ecut= {step.cutoff_ry:8.2f} Ry"
        f"     beta= {step.mixing_beta:4.2f}",
        file=out,
    )
    if step.davidson_threshold_ha > 0.0:
        print("     Davidson diagonalization with overlap", file=out)
        print(
            "\n---- Real-time Memory Report at c_bands before "
            "calling an iterative solver",
            file=out,
        )
        print(
            f"{step.memory_rss_bytes_per_rank / 2**20:14.0f} MiB "
            "given to the printing process from OS",
            file=out,
        )
        if step.memory_available_bytes:
            print(
                f"{step.memory_available_bytes / 2**20:14.0f} MiB "
                "available memory on the node where the printing "
                "process lives",
                file=out,
            )
        print(
            f"{step.memory_density_bytes_per_rank / 1e6:14.2f} MB "
            "estimated for rho,v,vnew",
            file=out,
        )
        print(
            f"{step.memory_mixing_bytes_per_rank / 1e6:14.2f} MB "
            "estimated for rho*nmix",
            file=out,
        )
        print(
            f"{step.memory_wavefunctions_bytes_per_rank / 1e6:14.2f} MB "
            "estimated for psi",
            file=out,
        )
        print(
            f"{step.memory_davidson_bytes_per_rank / 1e6:14.2f} MB "
            "estimated for Davidson subspace",
            file=out,
        )
        print(
            f"{step.memory_fft_bytes_per_rank / 1e6:14.2f} MB "
            "estimated for FFT workspace",
            file=out,
        )
        if step.memory_projector_cache_bytes_per_rank:
            print(
                f"{step.memory_projector_cache_bytes_per_rank / 1e6:14.2f} MB "
                "estimated for nonlocal projector cache",
                file=out,
            )
        print("------------------", file=out)
        print(
            f"     ethr = {2.0 * step.davidson_threshold_ha:9.2E}, "
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
    print(f"     total energy              = {step.total_energy_ha * 2:18.10f} Ry", file=out)
    if np.isfinite(step.estimated_accuracy_ha):
        print(
            f"     estimated scf accuracy    < "
            f"{step.estimated_accuracy_ha * 2:18.10e} Ry\n",
            file=out,
        )
    return out.getvalue()


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
    if result.converged:
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
            f"          k = {cart[0]:8.4f} {cart[1]:8.4f} "
            f"{cart[2]:8.4f} ({npw:6d} PWs)   bands (ev):",
            file=out,
        )
        print("    " + " ".join(f"{value * EV_PER_HARTREE:9.4f}" for value in values), file=out)
        print("\n     occupation numbers", file=out)
        if index < len(result.occupations):
            displayed_occupations = 0.5 * result.occupations[index]
        else:
            displayed_occupations = np.zeros(len(values))
            displayed_occupations[:occupied] = 1.0
        print(
            "    "
            + " ".join(
                f"{occupation:9.4f}"
                for occupation in displayed_occupations
            ),
            file=out,
        )
    if occupations_mode == "smearing" and result.fermi_energy_ha is not None:
        print(
            f"\n     the Fermi energy is "
            f"{result.fermi_energy_ha * EV_PER_HARTREE:10.4f} ev",
            file=out,
        )
    else:
        highest_occupied = max(
            values[occupied - 1] for values in result.eigenvalues_ha
        )
        print(
            f"\n     highest occupied level (ev): "
            f"{highest_occupied * EV_PER_HARTREE:10.4f}",
            file=out,
        )
    marker = "!" if result.converged else " "
    print(f"\n{marker}    total energy              = {result.total_energy_ha * 2:18.10f} Ry", file=out)
    if result.iterations:
        print(
            f"     estimated scf accuracy    < "
            f"{2.0 * result.iterations[-1].estimated_accuracy_ha:18.10f} Ry",
            file=out,
        )
    if occupations_mode == "smearing" and result.energy_terms is not None:
        smearing_ry = 2.0 * result.energy_terms.smearing_ha
        print(
            f"     smearing contrib. (-TS)   = "
            f"{smearing_ry:18.10f} Ry",
            file=out,
        )
        print(
            f"     internal energy E=F+TS    = "
            f"{2.0 * result.total_energy_ha - smearing_ry:18.10f} Ry",
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
            f"     one-electron contribution = "
            f"{2.0 * terms.one_electron_ha:18.10f} Ry",
            file=out,
        )
        print(
            f"     hartree contribution      = "
            f"{2.0 * terms.hartree_ha:18.10f} Ry",
            file=out,
        )
        print(
            f"     xc contribution           = "
            f"{2.0 * terms.xc_ha:18.10f} Ry",
            file=out,
        )
        print(
            f"     ewald contribution        = "
            f"{2.0 * terms.ewald_ha:18.10f} Ry",
            file=out,
        )
        if abs(terms.descf_ha) > 1.0e-12:
            print(
                f"     scf correction            = "
                f"{2.0 * terms.descf_ha:18.10f} Ry",
                file=out,
            )
    if result.forces_ha_per_bohr is not None:
        species_types = {
            species.label: index
            for index, species in enumerate(pw.species, start=1)
        }
        print(
            "\n     Forces acting on atoms (cartesian axes, Ry/au):\n",
            file=out,
        )
        for index, (atom, force) in enumerate(
            zip(pw.atoms, result.forces_ha_per_bohr), start=1
        ):
            force_ry = 2.0 * force
            print(
                f"     atom {index:4d} type "
                f"{species_types[atom.label]:2d}   force = "
                f"{force_ry[0]:14.8f}{force_ry[1]:14.8f}"
                f"{force_ry[2]:14.8f}",
                file=out,
            )
        total_force_ry = 2.0 * float(
            np.linalg.norm(result.forces_ha_per_bohr)
        )
        print(
            f"\n     Total force = {total_force_ry:12.6f}"
            "     Total SCF correction =     0.000000",
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
            "\n          total   stress  (Ry/bohr**3)"
            f"                   (kbar)     P= {pressure:8.2f}",
            file=out,
        )
        for row_ry, row_kbar in zip(stress_ry, stress_kbar):
            print(
                "  "
                + "".join(f"{value:13.8f}" for value in row_ry)
                + "   "
                + "".join(f"{value:11.2f}" for value in row_kbar),
                file=out,
            )
    if result.converged:
        print(
            f"\n     convergence has been achieved in "
            f"{len(result.iterations):3d} iterations",
            file=out,
        )
    print(
        "\n     peak resident memory/rank = "
        f"{format_bytes(result.peak_rss_bytes_per_rank)}",
        file=out,
    )
    if result.peak_rss_bytes_all_ranks:
        print(
            "     aggregate rank high-water marks = "
            f"{format_bytes(result.peak_rss_bytes_all_ranks)}",
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
        f"\n     PWSCF        : {total_cpu:9.2f}s CPU "
        f"{result.wall_seconds:9.2f}s WALL",
        file=out,
    )
    print("\n   JOB DONE." if result.converged else "\n   JOB FAILED.", file=out)
    return out.getvalue()


def format_output(pw: PWInput, result: SCFResult) -> str:
    return (
        format_header(pw)
        + "".join(format_iteration(step) for step in result.iterations)
        + format_footer(pw, result)
    )
