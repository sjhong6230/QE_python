# Quantum ESPRESSO-compatible diagnostics

PWSCF-PY models user-facing diagnostics on Quantum ESPRESSO (QE) 7.5.  The
reference implementation is QE's `UtilXlib/error_handler.f90`; individual
conditions and messages are taken from `PW/src/input.f90`,
`PW/src/set_occupations.f90`, `PW/src/lchk_tauxk.f90`,
`Modules/read_namelists.f90`, and `Modules/read_cards.f90`.

An error is printed in the same structural form as QE's `errore` routine:

```text
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Error in routine set_cutoff (1):
     ecutwfc not set
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     stopping ...
```

A non-fatal condition uses QE's `infomsg` form:

```text
     Message from routine set_cutoff:
     ecutrho < 4*ecutwfc, are you sure?
```

## Implemented QE diagnostic paths

| Area | Conditions now checked |
|---|---|
| Namelists | unknown namelist, duplicate namelist, malformed assignment, unknown QE 7.5 variable, and valid QE variable not implemented by PWSCF-PY |
| Calculation mode | unsupported `calculation`, unknown `restart_mode`, and restart/starting-state inconsistencies |
| Cutoffs | absent/non-positive or meaningless `ecutwfc`, `ecutrho <= ecutwfc`, `ecutrho < 4*ecutwfc`, and an unnecessarily large norm-conserving `ecutrho` without NLCC |
| Occupations | unknown or unimplemented occupation mode, QE's silent reset of `degauss` for fixed occupations, missing smearing width, and unknown smearing kind |
| Electronic algorithms | removed PPCG, unknown/unimplemented diagonalization, removed potential mixing, and unknown/unimplemented mixing mode |
| Cards | duplicate cards, malformed `ATOMIC_SPECIES`, duplicate atomic labels, malformed atomic-position columns, nonexistent species, invalid automatic k-point grids and offsets, and unknown k-point options |
| Units | QE deprecation messages for omitted `ATOMIC_POSITIONS` and `CELL_PARAMETERS` units |
| Geometry | atoms that overlap directly or through a lattice translation |
| Implemented subset | incompatible scalar/LSDA/noncollinear flags, scalar-relativistic UPFs in noncollinear runs, `lspinorb` without `noncolin`, DFT+U, unsupported XC functionals, and every explicitly supplied but unused QE variable are diagnosed explicitly |

Warnings are stored as structured `(routine, message)` records in `PWInput`.
They are emitted once by the root MPI rank.  Input parsing and symmetry
reduction still occur only on that rank, so parallel runs do not duplicate a
diagnostic.

## Deliberately not claimed as implemented

QE contains many errors that can only arise inside code paths absent from this
norm-conserving SCF port.  PWSCF-PY does **not** pretend to reproduce
those internal diagnostics.  The following categories remain unimplemented:

- ionic and variable-cell dynamics (`relax`, `md`, `vc-relax`, and `vc-md`);
- constrained magnetization and noncollinear DFT+U;
- ultrasoft and PAW augmentation-specific checks;
- DFT+U, hybrid/exact-exchange, dispersion corrections, SIC, and DMFT;
- electric fields, Berry-phase polarization, gates, ESM, grand-canonical SCF,
  RISM, FCP, and QM/MM;
- GPU, image/pool/band/task-group, and ScaLAPACK-specific internal failures;
- developer assertions for QE data structures that do not exist in the Python
  implementation.

When one of these features is requested through a recognized QE 7.5 input
variable, parsing stops before the calculation and names the variable as not
implemented.  A misspelled or non-QE variable instead reports a
`read_namelists` unknown-variable error.  This distinction prevents a valid
but absent physical term from being silently ignored.

## Compatibility boundary

Message text is kept verbatim where PWSCF-PY implements the same condition.
For valid QE functionality that is absent, the message necessarily adds
`is not implemented in PWSCF-PY`; this is intentional and avoids suggesting
that QE itself lacks the feature.  Python, operating-system, UPF/XML, MPI, and
native-extension failures that have no direct QE analogue retain their native
detail but are wrapped in the same error banner.
