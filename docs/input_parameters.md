# Input parameters

`qepy-pw` reads ordinary QE `pw.x` namelists and cards. Fortran booleans,
quoted strings, indexed names such as `celldm(1)`, comma- or newline-separated
assignments, `D` exponents, and comments beginning with `!` or `#` are
accepted. Names are case-insensitive.

Only variables listed below affect the implemented scalar SCF, NSCF, and
band-structure calculations.
Supplying a valid QE 7.5 variable outside this list produces an explicit
not-implemented error; a misspelled or unknown variable produces a QE-style
bad-namelist error.

## Minimal input

```text
&CONTROL
  calculation = 'scf'
  pseudo_dir  = './pseudo'
  disk_io     = 'none'
/
&SYSTEM
  ibrav   = 2
  celldm(1) = 10.20
  nat     = 2
  ntyp    = 1
  ecutwfc = 40.0
/
&ELECTRONS
  conv_thr   = 1.0d-8
  mixing_beta = 0.7
/
ATOMIC_SPECIES
Si  28.0855  Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
Si  0.00 0.00 0.00
Si  0.25 0.25 0.25
K_POINTS automatic
4 4 4 0 0 0
```

## `&CONTROL`

| Variable | Type and default | Implemented behavior |
|---|---|---|
| `title` | string, empty | Printed and written to save metadata. |
| `calculation` | string, `'scf'` | `'scf'`, `'nscf'`, or `'bands'`. The latter two read the converged density from `<outdir>/<prefix>.save`, diagonalize the resulting fixed Kohn--Sham potential at the requested `K_POINTS`, and do not mix a new density. |
| `verbosity` | string, `'low'` | `'high'` enables additional QE-shaped structural output; other strings behave as low verbosity. |
| `restart_mode` | string, `'from_scratch'` | `'from_scratch'` or `'restart'`. Restart requires compatible saved density and wavefunctions. |
| `tstress` | logical, `.false.` | Compute and print the implemented analytic stress contributions. |
| `tprnfor` | logical, `.false.` | Compute and print implemented ionic forces. |
| `outdir` | string, `ESPRESSO_TMPDIR` or `.` | Parent of `<prefix>.save`; environment variables and `~` are expanded. |
| `prefix` | string, `'pwscf'` | Save-directory prefix. It must be a filename component, not a path. |
| `pseudo_dir` | string, `.` | UPF directory. A relative path is resolved relative to the input file. |
| `disk_io` | string, `'low'` | One of `none`, `low`, `medium`, or `high`. `none` disables persistent output; the other levels write the QE-shaped XML/HDF5 save data. `medium` is suitable for an SCF → NSCF/bands workflow. Checkpoint frequency differences between the three persistent levels are not yet distinguished. |
| `iprint` | integer, `100000` in saved metadata | Accepted for QE compatibility. It does not introduce ionic-step printing because ionic dynamics are not implemented. |

`tprfor` is not an alias for `tprnfor`; it is diagnosed as an unknown variable,
matching QE spelling requirements.

## `&SYSTEM`

### Cell and composition

| Variable | Type and default | Implemented behavior |
|---|---|---|
| `ibrav` | integer, `0` | Bravais lattice selector. Supported values are `0, 1, 2, 3, -3, 4, 5, -5, 6, 7, 8, 9, -9, 91, 10, 11, 12, -12, 13, -13, 14`. |
| `celldm(1..6)` | real | QE legacy cell parameters. `celldm(1)` is in bohr; the remaining values are ratios or cosines according to `ibrav`. |
| `A`, `B`, `C` | real, ångström | Modern cell lengths. They cannot be mixed with any `celldm(i)`. |
| `cosAB`, `cosAC`, `cosBC` | real | Modern cell-angle cosines. Each required cosine must lie strictly between `-1` and `1`. |
| `nat` | integer | Number of atomic-position rows, except that `crystal_sg` expands inequivalent sites and updates the internal count. |
| `ntyp` | integer | Number of `ATOMIC_SPECIES` rows. |
| `space_group` | integer | Enables `ATOMIC_POSITIONS crystal_sg` expansion using `spglib`/`pyxtal`. |
| `uniqueb` | logical, `.false.` | Selects the unique-axis convention used for monoclinic space groups. |
| `origin_choice` | integer, `1` | Space-group origin choice. |
| `rhombohedral` | logical, `.true.` | Selects rhombohedral rather than hexagonal axes for the relevant setting. |

For `ibrav=0`, a three-row `CELL_PARAMETERS` card is mandatory. For nonzero
`ibrav`, either `celldm(1)` or `A` must define the primary lattice length.

### Plane waves, charge, and bands

| Variable | Type and default | Implemented behavior |
|---|---|---|
| `ecutwfc` | real, required, Ry | Wavefunction kinetic-energy cutoff. It must be positive; values below `1 Ry` or above `10000 Ry` are rejected as meaningless. |
| `ecutrho` | real, `4*ecutwfc`, Ry | Charge-density cutoff. It must exceed `ecutwfc`; values below `4*ecutwfc` generate a warning. |
| `nbnd` | integer, automatic | Number of Kohn–Sham bands. If present it must be positive. The automatic value depends on electron count and occupation mode. |
| `tot_charge` | real, `0` | Total cell charge in electron-charge units; `N_e = Σ_I Z_I - tot_charge`. |
| `starting_charge(i)` | real, `0` | Species-indexed scaling used while constructing the atomic starting density. |

`nr1`, `nr2`, `nr3`, and related explicit FFT-grid controls are deliberately
not implemented. FFT dimensions are selected from the cutoff and lattice and
rounded to FFTW-friendly lengths.

### Symmetry and k-point reduction

| Variable | Type and default | Implemented behavior |
|---|---|---|
| `nosym` | logical, `.false.` | Disable crystal symmetry, irreducible-k reduction, and density/force symmetrization. |
| `nosym_evc` | logical, `.false.` | Use identity symmetry in the electronic path after completing the supplied k-point list under the Bravais star. |
| `noinv` | logical, `.false.` | Disable ordinary inversion/time-reversal k-point reduction in this scalar nonmagnetic implementation. |
| `force_symmorphic` | logical, `.false.` | Retain only operations with zero fractional translation. |
| `use_all_frac` | logical, `.false.` | Retain all compatible nonsymmorphic fractional translations in the symmetry representation. |

Symmetry operations are found from the lattice, fractional positions, and
species labels. An automatic Monkhorst–Pack mesh is reduced only by operations
that map that mesh onto itself.

### Occupations and exchange–correlation

| Variable | Type and default | Implemented behavior |
|---|---|---|
| `occupations` | string, `'fixed'` | `fixed`, `smearing`, `tetrahedra`, `tetrahedra_lin`, or `tetrahedra_opt`; hyphen/underscore variants are normalized. `from_input` is not implemented. |
| `degauss` | real, `0`, Ry | Required and positive for smearing. It is silently reset to zero for fixed occupations, following QE 7.5 behavior. |
| `smearing` | string, `'gaussian'` | Gaussian (`gaussian`, `gauss`, `0`), Methfessel–Paxton (`methfessel-paxton`, `m-p`, `mp`), Marzari–Vanderbilt cold (`marzari-vanderbilt`, `cold`, `m-v`, `mv`), or Fermi–Dirac (`fermi-dirac`, `f-d`, `fd`). |
| `input_dft` | string, pseudo-dependent | Supported canonical families: `LDA`/`PZ`/`PZ81`, `PW`/`PW92`, `PBE`, `PBEsol`, `revPBE`, and `RPBE`. If absent, the UPF functional metadata is used. |

Tetrahedron occupations require an automatic uniform k mesh. Fixed
occupations are appropriate only when the supplied number of bands can hold
the scalar spin-degenerate electron count without fractional filling.

### Accepted only to diagnose unsupported physics

`nspin`, `noncolin`, `lda_plus_u`, and `lspinorb` are parsed so that requests
for spin polarization, noncollinearity, DFT+U, or spin–orbit coupling fail with
a precise unsupported-feature error. The only working spin setting is
`nspin=1`, `noncolin=.false.`, `lspinorb=.false.`, and no Hubbard correction.

## `&ELECTRONS`

| Variable | Type and default | Implemented behavior |
|---|---|---|
| `electron_maxstep` | integer, `100` | Maximum SCF iterations. A nonconverged run returns process exit status `2`. |
| `conv_thr` | real, `1.0e-6`, Ry | SCF energy/residual convergence threshold. |
| `diago_thr_init` | real, adaptive | Initial iterative-eigensolver threshold in Ry. The SCF driver tightens it according to the density residual. |
| `diagonalization` | string, `'david'` | See the solver table below. |
| `diago_david_ndim` | integer, `2` | Maximum Davidson subspace multiplier; must be at least `2`. |
| `diago_cg_maxiter` | integer, `20` | Maximum conjugate-gradient iterations per electronic solve. |
| `diago_rmm_ndim` | integer, `4` | RMM-DIIS history dimension; must be at least `2`. |
| `diago_rmm_conv` | logical, `.false.` | Continue RMM iterations toward the requested residual threshold rather than using the bounded default trajectory. |
| `diago_gs_nblock` | integer, `16` | Block size for RMM/ParO Gram–Schmidt operations; must be positive. |
| `diago_full_acc` | logical, `.false.` | Apply the strict occupied-state accuracy policy to all requested roots. |
| `mixing_mode` | string, `'plain'` | `plain`/`default`, `TF`, or `local-TF`; underscore and hyphen spellings are normalized. Potential mixing is removed. |
| `mixing_beta` | real, `0.7` | Mixing amplitude. A negative input is reset to `0.7`; the runtime mixer requires `0 < beta <= 1`. |
| `mixing_ndim` | integer, `8` | Broyden history length, constrained to `1..25`. |
| `mixing_pulay_frequency` | integer, `1` | Apply multisecant extrapolation every `n`th step while collecting history every step. This is a qepy-pw research extension, not a standard QE input variable. |
| `startingpot` | string, `'atomic'` | `atomic` or `file`. Invalid values warn and fall back to `atomic`. Restart mode forces file potential. |
| `startingwfc` | string, `'atomic+random'` | `atomic`, `atomic+random`, `random`, or `file`. Invalid values warn and fall back to `atomic+random`. |

### Diagonalization names

| Canonical path | Accepted names | Description |
|---|---|---|
| Davidson | `david`, `davidson` | Restarted block Davidson with QE's smooth diagonal preconditioner. |
| Conjugate gradient | `cg` | Band-by-band preconditioned conjugate-gradient minimization. |
| Parallel orbital | `paro` | Simultaneous orbital update with block orthogonalization. |
| RMM-DIIS | `rmm`, `rmm-diis` | Residual minimization with bounded per-band DIIS history. |
| Hybrid RMM/Davidson | `rmm-davidson` | RMM path with Davidson fallback/restart behavior. |
| Hybrid RMM/ParO | `rmm-paro` | RMM path combined with parallel-orbital updates. |
| Dense diagnostic | `direct`, `dense` | Materialize and diagonalize the Hamiltonian. Intended only for small validation cases. |

`ppcg` is rejected with QE's removal diagnostic.

## Cards

### `ATOMIC_SPECIES`

Exactly `ntyp` rows are required:

```text
ATOMIC_SPECIES
label  mass_amu  pseudopotential.UPF
```

Species labels must be unique. The current production physics requires
norm-conserving UPF data; unsupported UPF features are rejected when read.

### `ATOMIC_POSITIONS`

Supported units are `crystal`, `angstrom`/`ang`, `bohr`, `alat`, and
`crystal_sg`. Omitting the unit is deprecated and interpreted as `alat`.

```text
ATOMIC_POSITIONS crystal
Si  0.25 0.25 0.25  1 1 1
```

The optional three `if_pos` integers are parsed. They are relevant to force
masking/output but do not activate ionic relaxation. Coordinate tokens may be
simple arithmetic expressions accepted by the parser. Overlapping periodic
images are rejected.

For `crystal_sg`, each row may contain either conventional fractional
coordinates or a Wyckoff label followed by its free coordinates. `space_group`
is mandatory, `spglib` supplies database operations, and `pyxtal` resolves
Wyckoff positions.

### `CELL_PARAMETERS`

Required for `ibrav=0`; exactly three vectors are read. Supported units are
`angstrom`/`ang`, `bohr`, and `alat`. With `alat`, `celldm(1)` or `A` must be
present. Omitting the unit is deprecated and interpreted as `alat`.

### `K_POINTS`

Supported forms are:

- `gamma`;
- `automatic`, followed by `nk1 nk2 nk3 sk1 sk2 sk3`, where mesh dimensions
  are positive and each shift is `0` or `1`;
- explicit `tpiba`, `crystal`, or an empty option, followed by a positive row
  count and `kx ky kz weight [label]` rows;
- path forms `tpiba_b` and `crystal_b`, including numeric anchors or
  Setyawan–Curtarolo labels resolved through ASE;
- contour forms `tpiba_c` and `crystal_c`, with exactly three anchor rows.

Explicit weights are normalized to sum to one. `tpiba` coordinates are in
units of `2π/alat`; `crystal` coordinates use the reciprocal lattice basis.

The implementation deliberately does not provide k-pool parallelism. All
active MPI ranks cooperate on the plane-wave rows and FFT grid for each
k-point.

### `OCCUPATIONS`

The parser recognizes the card name for diagnostics, but
`occupations='from_input'` is not implemented. Consequently, an `OCCUPATIONS`
card cannot currently define working band occupations.

## `&IONS` and `&CELL`

The namelist syntax is recognized, but no variables are implemented because
relaxation, molecular dynamics, and variable-cell dynamics are outside the
scalar SCF scope. Any supplied variable is rejected explicitly.

## Error policy

The parser distinguishes three cases:

1. malformed syntax or an unknown name: QE-style input error;
2. a valid QE variable whose physics is absent: explicit not-implemented
   error;
3. a supported variable with an invalid value: the closest implemented QE
   error or warning path.

See [QE-compatible diagnostics](qe_diagnostics.md) for the detailed list.
