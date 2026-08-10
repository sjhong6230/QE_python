# qepy-pw documentation

This directory documents the implemented scalar, nonmagnetic SCF, NSCF, and
band-structure subset of Quantum ESPRESSO `pw.x`, plus the associated
`bands.x` and `plotband.x` post-processing path. The documentation describes
the current source tree, not the complete feature set of upstream QE.

## Reading guide

| Document | Purpose |
|---|---|
| [Input parameters](input_parameters.md) | Supported namelists, cards, defaults, units, accepted values, and explicit limitations |
| [Equations](equations.md) | Plane-wave Kohn–Sham equations, pseudopotentials, occupations, SCF mixing, forces, stress, and numerical eigensolvers |
| [Differences from QE](qe_differences.md) | Compatibility boundary and important physical, numerical, parallel, and file-format differences |
| [Installation and running](installation_and_running.md) | System libraries, pip build behavior, MPI/OpenMP launch examples, restart, and troubleshooting |
| [Validation and performance](validation_and_performance.md) | Regression tests, numerical tolerances, timing interpretation, memory accounting, and benchmarking practice |
| [Architecture](architecture.md) | Package decomposition, SCF data flow, Cython boundary, distributed FFT design, ownership, and memory lifetime |
| [QE-compatible diagnostics](qe_diagnostics.md) | Implemented QE-style error and warning conditions and their compatibility boundary |
| [Band post-processing](band_postprocessing.md) | `bands.py`, scalar irrep classification, overlap ordering, and `plotband.py` formats |
| [DOS post-processing](dos_postprocessing.md) | `dos.py` smearing and tetrahedron integration, input, and output formats |
| [Scalar field post-processing](pp_postprocessing.md) | `pp.py` extraction, intermediate files, weighted combinations, interpolation, XSF, and cube output |
| [Projected DOS](projwfc_postprocessing.md) | `projwfc.py` Löwdin atomic projections, charges, spilling, and PDOS files |

## Scope at a glance

The implemented production path is periodic scalar Kohn–Sham DFT with
norm-conserving UPF pseudopotentials. It supports LDA and selected semilocal
GGA functionals, fixed and metallic occupations, symmetry reduction,
distributed plane waves, QE-style stick/slab FFT transposes, several iterative
diagonalizers, density mixing, selected forces and stress, and QE-shaped
restart/output files.

The following major QE capabilities are outside the present scope:

- spin polarization, noncollinear magnetism, spin–orbit coupling, and
  constrained magnetization;
- ultrasoft and PAW augmentation, DFT+U, hybrid functionals, exact exchange,
  dispersion corrections, and time-dependent DFT;
- ionic relaxation, molecular dynamics, variable-cell dynamics, phonons, and
  response calculations;
- electric-field, Berry-phase polarization, and Wannier functionality;
- k-point pools, band groups, task groups, images, GPU backends, and
  ScaLAPACK-distributed subspace diagonalization.

Recognized but unimplemented QE variables are rejected with an explicit
`is not implemented in PWSCF-PY` diagnostic. They are never silently ignored.

## Conventions

Inputs retain QE conventions: energy cutoffs and broadening are entered in
Rydberg, lattice lengths may be supplied in bohr or ångström according to the
card, and printed energies follow QE's Rydberg-oriented text format. Internal
electronic-structure arrays and equations use Hartree atomic units unless a
section explicitly states otherwise.

The code uses row-vector lattice and reciprocal matrices. For a fractional
coordinate `s`, the Cartesian position is `r = s @ lattice`; reciprocal
vectors include the factor `2π`.

## Version and authority

The current package version is `0.6.14`. The source code and automated tests
are authoritative when a document and implementation disagree. In
particular, consult `qepy_pw/pw/input.py` for the parser boundary,
`qepy_pw/pw/scf.py` for the SCF path, `qepy_pw/pp` for band post-processing,
and `tests/qe_reference/manifest.json` for the numerical regression contract.
