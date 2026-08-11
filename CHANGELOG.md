# Changelog

All notable changes to this project are documented in this file.

## [1.0.2] - 2026-08-11

### Added

- Collinear `nspin=2` SCF, NSCF, force, and stress paths with QE-ordered
  spin-up/spin-down k-point blocks, spin-resolved occupations, and
  charge/magnetization density mixing.
- `starting_magnetization(i)` and `tot_magnetization` input handling, including
  fixed-occupation validation and spin-polarized atomic density/Hamiltonian
  initialization.
- PZ81/PW92 LSDA and PBE-family spin-GGA exchange-correlation functionals.
- QE-compatible `rhotot_g`/`rhodiff_g` density persistence, spin-tagged
  wavefunctions, LSDA restart/NSCF support, and magnetization reporting.
- Spin-resolved `bands.py`, `dos.py`, `projwfc.py`, and `pp.py` support,
  including magnetization fields and k-resolved LSDA projected DOS.
- Modular unit and integration coverage organized by numerical kernel, PW
  runtime, and post-processing subsystem.

### Changed

- Reduced LSDA iteration cost by reusing reciprocal-space spin densities,
  tiling exchange-correlation evaluation, and avoiding redundant occupation
  work.
- Added reproducible bcc Fe comparison workflows for bands, DOS, projected DOS,
  and spin-resolved real-space fields.

### Fixed

- Corrected spin-density Broyden mixing, LSDA saved-density MPI restarts, and
  spin-dependent LDA/GGA persistence and reporting.
- Corrected QE-compatible k-resolved PDOS layout, PP spin-component selection,
  atomic magnetization subtraction, and DOS comparison grids.
- Matched QE's `ibrav=3` and `ibrav=-3` primitive-vector conventions, fixing
  high-symmetry bcc band paths.

## [1.0.1] - 2026-08-10

### Added

- QE-compatible NSCF and bands calculation paths, including calculation-specific
  `disk_io` defaults, binary wavefunction buffering in `wfcdir`, and high-verbosity
  per-k-point progress output.
- `bands.py`, `plotband.py`, `dos.py`, `projwfc.py`, and `pp.py` post-processing
  executables with QE-style command-line input aliases, output headers, timing,
  and termination messages.
- Band symmetry and irreducible-representation analysis, overlap-based band
  reordering, two-dimensional band grids, and momentum-matrix output.
- Total and projected density-of-states workflows, including tetrahedron and
  smearing integrations, symmetry-averaged orbital projections, local orbital
  bases, and box projections.
- Scalar norm-conserving `pp.x`-style real-space post-processing for supported
  charge-density, potential, LDOS, STM, ELF, kinetic-density, reduced-gradient,
  Hessian, DORI, wavefunction, and related plot quantities.
- QE-compatible array-valued namelist parsing, including indexed assignments and
  comma-separated whole-array assignments.
- GitHub Actions coverage for supported serial Python versions and MPI FFT tests.

### Changed

- Reorganized the runtime into `pw` and `pp` subpackages following the Quantum
  ESPRESSO source-tree separation while retaining compatibility imports.
- Matched QE formatting more closely for headers, whitespace, significant digits,
  symmetry descriptions, energy/force/stress decompositions, save ordering, error
  reporting, and `JOB DONE.` output.
- Improved tetrahedron, projected-DOS, ELF, kinetic-density, and general PP grid
  performance through vectorization, batching, FFT reuse, and threaded execution.
- Updated `plotband.py` interactive parsing and PostScript generation to follow the
  QE dialogue and output dialect.

### Fixed

- Preserved symmetry-required SCF and NSCF eigenvalue degeneracies and corrected
  band irreducible representations at Gamma and general k points.
- Corrected unconverged NSCF Davidson retry behavior, projected-state symmetry
  rotations, k-resolved PDOS, momentum-matrix evaluation, and PP smearing
  normalization.
- Corrected QE lifecycle ordering for saved densities and wavefunctions and made
  `wfcdir` default consistently to `outdir`.

## [1.0.0] - 2026-08-09

- Initial public release of the scalar, nonmagnetic plane-wave SCF implementation.

[1.0.2]: https://github.com/sjhong6230/QE_python/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/sjhong6230/QE_python/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/sjhong6230/QE_python/releases/tag/v1.0.0
