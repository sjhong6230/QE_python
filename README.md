# qepy-pw

`qepy-pw` is an incremental, readable Python port of the ordinary scalar
self-consistent-field path in Quantum ESPRESSO's `pw.x`. It keeps QE namelists,
cards, UPF input, units, and recognizable text output while replacing the
computational path with NumPy/SciPy code.

The current `0.6.14` milestone is an end-to-end reference implementation for
small scalar insulating and metallic cells:

- `calculation='scf'`, `nspin=1`, fixed or smeared occupations;
- QE Bravais lattices `ibrav=0`, `\u00b13`, 4--14, and 91, with both legacy
  `celldm(i)` and modern `A/B/C/cosAB/cosAC/cosBC` cell parameters;
- `K_POINTS gamma`, `automatic`, explicit `tpiba`, and explicit `crystal`;
- plane-wave kinetic energy, matrix-free FFT Hamiltonian application, and
  restarted block Davidson diagonalization;
- periodic local ionic, Hartree, and Perdew–Zunger LDA potentials;
- spin-unpolarized PBE GGA energy and its full periodic variational potential,
  selected by `input_dft='PBE'` or PBE UPF metadata;
- Ewald ion-ion energy and QE-style plain Broyden density mixing in the
  reciprocal-space Coulomb metric;
- symmetry-projected Hellmann–Feynman forces through `tprnfor` and
  compressive-positive stress through `tstress`;
- QE-compatible Gaussian, first-order Methfessel–Paxton,
  Marzari–Vanderbilt cold, and Fermi–Dirac smearing, including the Fermi-level
  search, fractional-band density, and variational `-TS` contribution;
- scalar norm-conserving UPF 2 potentials, including multiple radial
  Kleinman–Bylander beta projectors per angular-momentum channel and full
  `PP_DIJ` coupling;
- nonlinear core correction from `PP_NLCC`, including QE-normalized radial
  transforms and valence/core exchange-correlation energy bookkeeping;
- atomic starting-density superposition from `PP_RHOATOM`, with atomic phase
  factors and electron-count renormalization;
- QE-normalized `PP_CHI` atomic starting orbitals, `atomic+random`
  trial-space construction, and QE's adaptive initial Davidson threshold;
- QE `delta_e`/`delta_escf` bookkeeping for consistent total energies before
  the input and output SCF densities coincide;
- alias-free FFT grids for density and local-potential convolutions, and QE's
  Coulomb-weighted `dr2` SCF convergence criterion;
- automatic discovery of crystallographic space-group operations, irreducible
  automatic k meshes, time-reversal reduction, and reciprocal-space density
  symmetrization including fractional-translation phases;
- QE symmetry switches `nosym`, `noinv`, `force_symmorphic`, and
  `nosym_evc`; the latter disables crystal symmetry in the electronic path
  while completing the supplied k-point set under Bravais-lattice rotations;
- QE-style `<outdir>/<prefix>.save` persistence with `data-file-schema.xml`,
  HDF5 charge density and wavefunctions, and copied UPF files;
- QE `restart_mode`, `startingpot`, and `startingwfc` controls, including
  complete and selective restart from those native XML/HDF5 files;
- QE-shaped progress, eigenvalue, total-energy, timing, and completion lines.
- QE-style lattice, reciprocal-axis, species, atom, k-point, FFT-grid,
  Davidson-threshold, occupation, energy-decomposition, and convergence
  diagnostics.

This is not yet a drop-in replacement for production `pw.x`. In particular,
ultrasoft augmentation, PAW, conjugate-gradient and other iterative
eigensolvers, relaxation, spin, SOC, hybrids,
DFT+U, and response properties are explicitly rejected or absent. A
Python-only dense diagnostic solver remains available and is intentionally
limited by `py_max_dense_pw` (default 2500 plane waves).

The CLI flushes its startup, basis diagnostics, and every SCF iteration, so
redirected output and `tee` update while a calculation is running. Automatic
k meshes are reduced under lattice-and-basis space-group operations and time
reversal. For diamond Si, `8x8x8` is reduced from 512 to 29 weighted points.
`nosym=.true.` retains the full mesh. The default Davidson path reports its
plane-wave counts and maximum projected-subspace dimension before expensive
setup. It does not allocate an `Npw`-by-`Npw` Hamiltonian.

The standard QE controls `mixing_mode='plain'`, `mixing_beta`, and
`mixing_ndim` select the history-based modified-Broyden update. Their defaults
are `plain`, `0.7`, and `8`, respectively. The implementation preserves the
zero-wavevector density coefficient exactly, so mixing cannot change the
electron count. `mixing_mode='linear'` remains available as a Python-only
diagnostic fallback. As in QE, the history is a ring of consecutive input and
residual secants, its Gram system is solved directly, and the Coulomb metric
is restricted to the spherical `ecutrho` charge-density set.

The QE-compatible default `diagonalization='david'` selects restarted block
Davidson iteration. `diago_david_ndim` controls the maximum subspace as a
multiple of the requested band count and defaults to 4. Local-potential
products are evaluated on the FFT grid, while norm-conserving nonlocal terms
are applied in their compact projector form. Converged wavefunctions seed the
next SCF iteration. `diagonalization='dense'` is retained as a Python-only
reference mode for small-system comparisons.

On the first iteration, UPF `PP_CHI` functions are Fourier transformed and
phased into the same expanded atomic trial subspace used by QE; missing or
identically zero functions are replaced by kinetic-energy-damped random
vectors. `diago_thr_init` is interpreted in Ry and defaults to `1.0e-2 Ry`.
As in QE, after the first density residual `dr2` is known, a first solve whose
estimated eigensolver error is too large is discarded and repeated without
mixing at `ethr=0.1*dr2/nelec`.

The second SCF iteration first resets the diagonalization threshold
to `1.0e-2 Ry`, then applies the usual `0.1*dr2/nelec` tightening. Thus the
first-iteration retry does not force every subsequent Davidson solve to use
its tightened threshold.

For UPFs containing `PP_RHOATOM`, the initial Hartree and XC potentials are
built from the superposition of free-atom valence densities, as in QE's
default `startingpot='atomic'` path. UPFs without that table contribute their
correct average valence charge through a uniform fallback. The FFT grid is
padded for the full plane-wave difference span; using only the wavefunction
index span causes circular aliasing in density and local-potential products.

During unconverged iterations, the reported total energy is evaluated on the
mixed density and includes QE's `deband` and `descf` corrections. This avoids
combining eigenvalues generated by the input potential with Hartree and XC
terms evaluated on an unrelated output density.

## Performance

UPF radial transforms are evaluated once per distinct, bit-identical
reciprocal-space radius and scattered back to the full FFT grid or
plane-wave basis. This preserves every numerical argument while eliminating
thousands of duplicate Simpson integrations and spherical-Bessel
evaluations. On the WSL 2x2x2 Si reference, the first three SCF iterations
fell from 6.87 s to 1.79 s, while the complete 64-iteration calculation fell
from 185.41 s to 105.29 s. The final energy and iteration trajectory are
unchanged.

Version 0.6.5 adds persistent local-Hamiltonian workspaces. Plane-wave
scatter arrays and product arrays are reused by block size, and the fixed
local potential is transformed to real space once per Hamiltonian rather
than once per `Hpsi`. Davidson now follows QE's Ritz-update ordering, reports
both subspace iterations and actual `Hpsi` vector counts, locks converged
roots, and refreshes from the current Ritz vectors. A residual safety gate
is retained because eigenvalue-only acceptance was found to destabilize the
Python Broyden trajectory. With the reproducible NumPy backend, the complete
Si calculation converged in 59 iterations and 79.05 s at
`-21.28499062 Ry`.

SciPy pocketfft is the default local FFT backend. It uses overwrite-capable
complex transforms so reciprocal coefficients, real-space wavefunctions, and
forward-transform input can reuse storage. The reproducibility comparison
backend remains available with `py_fft_backend='numpy'`.

Numerical-library threading defaults to one thread per Python process. The
`run_scf` entry point applies this limit to loaded BLAS and OpenMP runtimes,
so MPI runs do not silently inherit the host's OpenBLAS, MKL, or OpenMP
thread count. Set `py_blas_threads` explicitly to opt into threaded linear
algebra; `py_fft_threads` independently controls the optional pyFFTW/SciPy
FFT worker count and also defaults to one.

The low-memory default recomputes the factorized Kleinman--Bylander projector
bases for each k point and SCF iteration. Projector overlaps use BLAS
conjugate-transpose operations and one fused MPI reduction per `Hpsi`. Set
`py_cache_projectors=.true.` in `&electrons` to retain species-centered beta
bases and compact atom-dependent phase columns for a faster, higher-memory
run. Setup and iteration reports state the exact cache allocation per rank.

For symmetry-reduced calculations, the G-vector MPI ranks also divide the
independent scalar-density space-group operations. Each rank accumulates a
partial grid and one collective combines the result. This parallelizes the
formerly replicated `sum_band` symmetry work without requiring k-point pools.
FFT-grid index mappings are constructed once per rank and reused across SCF
iterations. The constructor verifies that every fractional translation and
rotation maps onto integer FFT-grid points instead of silently rounding an
incompatible operation.

An optional pyFFTW backend supplies aligned persistent arrays and cached
forward/backward FFTW plans. Install it and select it with:

```bash
python -m pip install -e ".[fft]"
```

```fortran
&electrons
  py_fft_backend = 'pyfftw'
  py_fft_planner = 'measure'
  py_fft_threads = 1
/
```

The alternatives for `py_fft_planner` are `estimate`, `measure`, and
`patient`. On the `60x60x60` Si grid, paired inverse/forward transforms for
blocks of 4, 8, and 12 vectors took respectively 41.3, 71.5, and 121.2 ms
with NumPy; 30.5, 61.0, and 96.0 ms with one-worker SciPy; and 28.5, 58.9,
and 82.5 ms with one-thread planned pyFFTW. SciPy gives most of the FFTW
benefit without an optional dependency and is therefore the default.

The Python-specific Davidson controls are:

- `py_davidson_maxiter` (default 100);
- `py_davidson_residual_factor` (default zero, using QE's
  consecutive-Ritz-value test; set it positive to add a residual gate);
- `py_davidson_residual_energy_scale` (default 10 Ha; also requires
  `max(residual_norm)^2 < scale*ethr` to reject false Ritz-value
  convergence; set it to zero to reproduce QE's stopping rule).

The energy-scaled residual safeguard is a Python reliability extension and
is not part of QE's ordinary Davidson stopping rule. The value 10 was selected
from the unreduced 216-k-point Si validation: it retains protection from false
Ritz convergence while reproducing QE's 28-step SCF convergence count.

Version 0.6.6 adds optional Numba compilation for the scalar/indexed work
that remains around the library kernels: FFT-grid coefficient
scatter/gather, real-potential multiplication, density accumulation, the
PZ81 exchange-correlation evaluation, and Broyden Coulomb inner products.
FFT transforms, matrix products, SVD, and projected eigensolutions remain in
NumPy/SciPy because those operations already execute in compiled FFT and
BLAS/LAPACK libraries.

Install the JIT extra with:

```bash
python -m pip install -e ".[jit]"
```

Numba is opt-in because loading LLVM and compiled kernels adds approximately
100 MiB per MPI rank in the unreduced Si benchmark. It can be enabled
explicitly in `&electrons`:

```fortran
py_numba = .true.   ! require Numba
! py_numba = .false.  ! low-memory default
```

On the small seven-iteration Si example, a warm JIT run took 0.140 s versus
0.153 s for the NumPy scalar kernels, about an 8% end-to-end reduction, with
identical iteration count and total energy to the displayed precision. The
first run took 0.515 s because LLVM compiled and cached the kernels. This
one-time cost is therefore counterproductive for very short calculations
but amortizes over longer SCF calculations. The measured dominant Davidson
kernel is LAPACK SVD, so Numba is not expected to remove the remaining
Fortran-versus-Python performance gap by itself.

Version 0.6.7 corrects two multi-k-point symmetry defects. FFT dimensions are
now derived from the largest span within each individual k-point basis, so
reciprocal-lattice gauge shifts between equivalent representatives cannot
spuriously enlarge one grid axis. Reciprocal FFT frequencies are rounded to
their mathematical integers before symmetry transformations rather than
truncated. For the 60 Ry, 8x8x8 Si case this changes the grid from the
erroneous `(36,36,40)` to QE's `(36,36,36)` and gives
`-31.06956644 Ry`, compared with QE's `-31.06958221 Ry`. Output no longer
presents the sum of all independent k-point basis sizes as though it were a
single plane-wave dimension.

NumPy/SciPy use the BLAS/LAPACK implementation supplied by the Python
environment (MKL in the reference Conda environment). The
`threadpoolctl` dependency permits explicit control through the Python-only
input variable `py_blas_threads` in `&electrons`, for example
`py_blas_threads=1`. The default leaves the vendor library's thread policy
unchanged because changing reduction order can select different vectors
inside degenerate subspaces and perturb a sensitive nonlinear SCF
trajectory. Benchmark thread counts for the target system rather than
assuming that more threads are faster.

## Install and run

Use a virtual environment. On Ubuntu/WSL, install the matching `venv` package
first if `python3 -m venv` reports that `ensurepip` is unavailable:

```bash
sudo apt update
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m qepy_pw -in examples/h2.scf.in
```

For MPI-parallel FFT/G-vector execution:

```bash
python -m pip install -e ".[mpi]"
mpiexec -n 4 pw.py -in examples/si-symmetry.scf.in
```

MPI follows QE's Z-stick slab decomposition. Complete reciprocal sticks are
indexed by `(Gx,Gy)`, balanced by their plane-wave populations, and retain the
full `Gz` axis on their owning rank. After the one-dimensional Z FFT, one
`MPI_Alltoallv` transpose produces real-space Z-plane slabs for the
two-dimensional XY transform. The forward transform reverses these steps.
Local-potential multiplication and density accumulation operate on the
real-space Z slab without assembling a complete wavefunction grid. Only rank
zero writes output.

Davidson wavefunctions, applied wavefunctions, residuals, corrections, and
subspace-basis rows are distributed over those G-vector owners. Kinetic and
local-potential operations therefore return rank-local rows. Norm-conserving
projector overlaps, vector overlaps, residual norms, and Rayleigh--Ritz
matrices are summed collectively; only the small band/subspace matrices are
replicated. Small FFT grids may nevertheless remain latency-bound, but each
Hamiltonian application now needs two distributed transposes rather than four.

## XML/HDF5 save output

Command-line calculations write a QE-shaped save directory at
`<outdir>/<prefix>.save`. The default prefix is `pwscf`; the default output
directory is `ESPRESSO_TMPDIR` when that environment variable is set and the
current directory otherwise. `disk_io='none'` disables all save writes.

`data-file-schema.xml` follows QE's QEXSD organization and contains the input,
structure, basis and k-point metadata, convergence state, energies, bands,
occupations, optional forces/stress, and timing information.
`charge-density.hdf5` follows QE's `MillerIndices`/`rhotot_g` organization.
Each irreducible k point is written to a QE-layout `wfcN.hdf5` file containing
its physical Cartesian k vector, reciprocal-basis metadata, ordered Miller
indices, and the complex `evc` band-by-plane-wave coefficient matrix. In MPI
runs, distributed non-contiguous plane-wave rows are gathered only to rank
zero after the final SCF iteration.
Referenced UPF pseudopotentials are copied into the save directory. No
qepy-specific results container is added. Library callers can use
`write_qe_save(pw, result)` explicitly; filesystem writes remain root-only in
MPI command-line runs.

`restart_mode='restart'` requires and loads both the saved density and every
saved k-point wavefunction. With `restart_mode='from_scratch'`,
`startingpot='file'` and `startingwfc='file'` can be selected independently;
their QE defaults remain `atomic` and `atomic+random`. Before file data are
accepted, the lattice, FFT grid, band count, k points, and plane-wave Miller
indices are checked against the current calculation. MPI density loading
scatters root's grid by Z slab, while each rank reads only its owned
wavefunction rows.

## Memory reporting

The setup output reports an analytical per-rank estimate for persistent SCF
arrays, the largest Davidson/FFT workspace, and their sum. This estimate
counts NumPy array payloads and intentionally excludes the Python interpreter,
loaded shared libraries, BLAS/FFT/MPI internal workspaces, Numba code, and
allocator fragmentation. At completion, the code also reports the measured
peak resident set size of the largest rank and the sum of all rank high-water
marks.

FFT storage is bounded to the currently active Davidson block size rather than
retaining one full grid for every block size encountered. All sequential
k-point workspaces share one process-local distributed-FFT scratch pool, and
both serial and distributed paths multiply inverse-FFT wavefunctions in
place. MPI exchanges pack directly into one send buffer, collective sums
operate in place, and Miller, slot, and stick metadata use 32-bit indices.
The pyFFTW backend likewise retains only its current block-size plan and
buffers; changing block size replaces the previous cache.

Each SCF iteration emits a QE-shaped real-time memory block with process RSS,
node-available memory, and separate estimates for density/potential grids,
Broyden history, wavefunctions, Davidson storage, and FFT workspaces. Broyden
secants are stored only for reciprocal vectors inside the spherical
charge-density cutoff instead of as full FFT grids.

The footer follows QE's cumulative timing hierarchy. It reports `init_run`,
`electrons`, `c_bands`, `sum_band`, `v_of_rho`, `mix_rho`, `cegterg`,
`cdiaghg`, Davidson overlap/update/restart work, `h_psi`, projector work,
`vloc_psi`, and FFT transforms with CPU time, wall time, and call counts.
Occupied wavefunctions are transformed as one block during `sum_band`, which
reduces repeated FFT setup, MPI transpose, and allocation overhead.

If Ubuntu provides a version-specific Python package, such as Python 3.14,
the required package may be named `python3.14-venv`. The quoted `pw.py`
console entry point is also installed, so this is equivalent:

```bash
pw.py -in examples/h2.scf.in
```

On Windows PowerShell with the Anaconda interpreter used during development:

```powershell
C:\Users\sjhon\anaconda3\python.exe -m pip install -e .
C:\Users\sjhon\anaconda3\python.exe -m qepy_pw -in examples\h2.scf.in
```

The hydrogen example uses a deliberately simple pure-Coulomb UPF. The scalar
norm-conserving silicon example uses an LDA UPF distributed with the checked-out
QE source:

```bash
python -m qepy_pw -in examples/si-nc.scf.in
```

`examples/si-symmetry.scf.in` exercises space-group reduction and density
symmetrization.

The Fe example exercises nonlocal projectors and nonlinear core correction
together in a scalar, deliberately nonmagnetic calculation:

```bash
python -m qepy_pw -in examples/fe-nlcc.scf.in
```
## Port map

| QE responsibility | Python module |
|---|---|
| `read_input`, `input.f90` | `qepy_pw.input` |
| `recips`, `ggen`, `gk_sort` | `qepy_pw.basis` |
| `vloc_mod`, `beta_mod`, `init_us_1/2`, `setlocal` | `qepy_pw.upf`, `qepy_pw.scf` |
| `set_rhoc`, `rhoc_mod`, `v_xc`, `v_of_rho` | `qepy_pw.upf`, `qepy_pw.scf`, `qepy_pw.xc` |
| `h_psi`, `c_bands`, `cegterg`, `g_psi` | `qepy_pw.basis`, `qepy_pw.diagonalization`, `qepy_pw.scf` |
| `electrons_scf`, `mix_rho` | `qepy_pw.scf`, `qepy_pw.mixing` |
| `ewald` | `qepy_pw.ewald` |
| `printout`, `punch` (text subset) | `qepy_pw.output` |

See `PORTING.md` for the staged coverage contract and numerical-validation
criteria.
