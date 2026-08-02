# Porting contract and roadmap

## 1. Interpretation of “ordinary DFT without spin”

The target is the Born–Oppenheimer Kohn–Sham plane-wave/pseudopotential
problem for a periodic, scalar, nonmagnetic system. “Ordinary” excludes
hybrids, Hubbard corrections, external fields, Berry phases, van der Waals
add-ons, constrained magnetization, time dependence, response functions, and
ionic dynamics. The eventual baseline includes SCF and NSCF electronic
structure, norm-conserving pseudopotentials, LDA/GGA, insulating and metallic
occupations, general Bravais cells, k-point sampling, forces, and stress.

## 2. Compatibility boundary

Input compatibility means existing QE `&CONTROL`, `&SYSTEM`, and `&ELECTRONS`
namelists and `ATOMIC_SPECIES`, `ATOMIC_POSITIONS`, `CELL_PARAMETERS`, and
`K_POINTS` cards remain the public interface. Output compatibility means the
human-readable stream retains QE's important labels and units so established
parsers can recognize iteration energies, eigenvalues, final `! total energy`,
timing, and `JOB DONE.`. Byte-for-byte output is not a goal; QE's XML restart
schema is a later milestone.

Unsupported input must raise `UnsupportedFeatureError`. It must never be
ignored when doing so changes the Hamiltonian.

## 3. Milestones

### 0.1 — executable scalar SCF spine (implemented)

Local/projector-free UPF 2, PZ81 LDA, fixed occupations, general explicit
cells plus `ibrav=1/2`, unreduced k grids, dense diagonalization, Ewald energy,
linear mixing, and text output.

### 0.2 — scalar norm-conserving nonlocal pseudopotentials (implemented)

UPF radial beta projectors, QE-ordered real spherical harmonics,
Kleinman–Bylander `V_NL`, multiple projectors per angular channel, and full
`PP_DIJ` coupling are implemented. Radial transforms use QE's `PP_RAB`
Simpson weights and reciprocal normalization. A scalar Si LDA UPF is exercised
end-to-end by `examples/si-nc.scf.in`.

Atomic-density initialization remains a separate ordinary-DFT task.

### 0.3 — nonlinear core correction (implemented)

`PP_NLCC` pseudo-core densities are parsed and transformed using QE's radial
normalization and `PP_RAB` quadrature, superposed with atomic phase factors,
and represented on the periodic FFT grid. PZ81 LDA is evaluated on
`rho_valence + rho_core`; only `rho_valence` enters the band-energy
double-counting integral. The frozen core does not enter Hartree energy,
electron counting, or density mixing. `examples/fe-nlcc.scf.in` validates a
real scalar norm-conserving Fe UPF containing both nonlocal projectors and
NLCC.

### 0.4 — space-group and k-point symmetry (implemented)

Integer lattice automorphisms are filtered against the labeled atomic basis,
including fractional translations. Automatic meshes are reduced into
reciprocal-space orbits with time reversal and normalized star weights.
The valence density is projected onto the space group in reciprocal space,
including nonsymmorphic phase factors. Diamond Si has 48 operations and its
Gamma-centered `8x8x8` mesh reduces from 512 to 29 points. `nosym` and
`noinv` are honored.

The internal integer search currently covers rotation matrices with entries
in `{-1,0,1}`, encompassing the common primitive cells targeted by this
reference port. A later spglib-backed path may be useful for unusual
user-defined bases requiring larger integer representations.

### 0.5 — plain Broyden density mixing (implemented)

QE's `mixing_mode='plain'` history-based modified-Broyden algorithm is
implemented in the reciprocal-space Coulomb metric. The mixed object is the
valence charge density, with the `G=0` coefficient held fixed to preserve the
electron count. `mixing_beta` controls the residual step and `mixing_ndim`
limits the stored secant history; their QE defaults of 0.7 and 8 are retained.
A rank-revealing least-squares solution replaces a brittle explicit inverse
when residual-difference histories are linearly dependent. A Python-only
`mixing_mode='linear'` fallback is retained for diagnosis and comparison.

Kerker/Thomas–Fermi screening and the other QE mixing modes remain later
work.

### 0.6 — matrix-free Davidson diagonalization (implemented)

The scalar norm-conserving Hamiltonian is applied without constructing an
`Npw × Npw` array: kinetic energy is diagonal, the local potential uses FFT
products, and nonlocal terms retain the compact `B D B†` projector form.
A restarted block Davidson solver computes the lowest bands using Rayleigh–
Ritz projection, QE's smooth `g_psi` diagonal preconditioner, rank-revealing
orthogonalization, and the standard `diago_david_ndim=4` subspace bound.
Wavefunctions are reused across SCF iterations and the eigensolver threshold
is tightened with the density residual. `diagonalization='david'` is the
default; a guarded Python-only `dense` mode provides numerical comparisons.

Conjugate-gradient and other iterative diagonalizers remain future work.

### 0.6.1 — atomic initialization and density-grid correction (implemented)

`PP_RHOATOM` radial valence charges are parsed, Fourier transformed with QE's
`rhoat_mod` normalization, phased over atomic positions, and renormalized to
the requested electron count. They now initialize the first Hartree and XC
potentials instead of a homogeneous electron density. Species without an
atomic table retain a uniform average-charge fallback.

The FFT grid now spans twice the complete wavefunction-index range, preventing
circular aliasing of density Fourier components and local-potential
convolutions. SCF convergence uses QE's Coulomb-weighted `rho_ddot` (`dr2`)
metric in Ry rather than a real-space L1 residual combined with an energy
difference.

### 0.6.2 — QE-exact plain-mixing history (implemented)

Plain Broyden mixing now stores consecutive input-density and residual
secants in the same bounded ring representation as `PW/src/mix_rho.f90`.
The symmetric Gram system is solved directly, with least squares retained
only as a recovery path for exact rank deficiency. Inner products and `dr2`
are restricted to the spherical `ecutrho` reciprocal set rather than all
corners of the rectangular FFT box.

### 0.6.3 - atomic wavefunctions and SCF energy consistency (implemented)

UPF `PP_CHI` pseudo-wavefunctions are transformed with QE's
`4*pi/sqrt(Omega)` radial normalization and combined with real spherical
harmonics, the `i^l` angular phase, and atom-dependent Bloch phases. The
resulting `atomic+random` trial space may contain more vectors than requested
bands and is Rayleigh-Ritz reduced by Davidson. Missing or unusable atomic
functions receive QE-like kinetic-energy-damped random replacements. The
first eigensolver threshold defaults to `1.0e-2 Ry` and subsequently tightens
from the preceding density residual.

Unconverged total energies now follow `PW/src/electrons.f90`: band
eigenvalues are corrected by `delta_e` against the input Hartree-XC
potential, Hartree and XC functionals are evaluated on the mixed density,
and `delta_escf` corrects the difference between mixed and output densities.
This removes the large nonvariational first-iteration offset caused by
combining quantities evaluated at three different densities.

### 0.6.4 - radial-shell optimization and output parity (implemented)

Local-potential, atomic-density, NLCC, beta-projector, and atomic-orbital
radial transforms now operate on distinct bit-identical `|G|` or `|G+k|`
shells and scatter the results to the complete basis. This removes redundant
UPF quadrature without rounding reciprocal arguments or changing the
validated SCF trajectory. `threadpoolctl` provides the opt-in
`py_blas_threads` control for the external BLAS/LAPACK runtime.

Text output now includes the Bravais index, lattice parameter, direct and
reciprocal axes, Kohn-Sham state count, convergence threshold,
pseudopotential checksum and angular channels, species and atomic-position
tables, Cartesian and crystal k points, FFT dimensions, atomic starting-wfc
count, per-iteration `ethr` and average Davidson work, occupation numbers,
highest occupied level, final SCF accuracy, total-energy decomposition, and
the explicit convergence iteration count.

### 0.6.5 - persistent Hpsi workspaces and planned FFTs (implemented)

The local Hamiltonian now reuses reciprocal scatter and real-space product
buffers for each block size and transforms the fixed local potential only
once per Hamiltonian. An optional pyFFTW backend uses aligned storage and
persistent measured/estimated/patient forward and inverse plans. SciPy
pocketfft remains the default because backend-level roundoff can alter a
nonlinear SCF trajectory even when every individual `Hpsi` agrees to
numerical precision.

Davidson calculations construct one species-centered nonlocal
Kleinman--Bylander beta basis for the current k point and release it after the
solve, matching QE's `init_us_2` lifetime. Only each MPI rank's local
plane-wave rows are retained. All atom/projector overlaps are fused into one
collective reduction per Hamiltonian application, while BLAS
conjugate-transpose flags avoid temporary copies of complete beta matrices.
The former all-k `py_cache_projectors` mode was removed.

The k-point bases now retain int32 mappings into one QE-ordered global G
catalog rather than replicated Miller, Cartesian, and kinetic arrays. Serial
FFT workspaces retain compact linear slots and omit distributed stick-owner
maps. A shared density-grid descriptor supplies G2, cutoff-sphere indices,
Cartesian G vectors, and FFT slots to every potential, mixing-error, force,
stress, and PBE consumer. The effective real-space potential is assembled
once per SCF iteration and shared across the complete k loop.

NLCC transforms now reproduce `rhoc_mod`: `rho_core(G)` is tabulated at
`dq=0.01` on QE's shortened species mesh, interpolated with the four-point
cubic formula, and differentiated by the analytic derivative of that same
interpolant for stress.

The distributed SciPy FFT path shares one grow-only scratch pool across the
sequential k-point workspaces and overwrites inverse-transformed
wavefunctions with their local-potential products. Davidson basis and
Hamiltonian-basis matrices are preallocated to `diago_david_ndim*nbnd`;
density accumulation traverses all occupied bands in one Numba kernel; and
immutable basis, FFT-slot, and stick maps use 32-bit indices.

Davidson iteration is organized around QE's initial Ritz estimate followed
by correction, expansion, and consecutive-eigenvalue testing, so the printed
iteration count has the same meaning as `cegterg`. Converged roots are
excluded from correction blocks, restarts retain current Ritz vectors, and
the output reports actual Hamiltonian-vector applications and the maximum
eigen-residual. The stable path retains rank-revealing orthonormal corrections
instead of QE's nonorthogonal correction space. Restarts carry both Ritz
vectors and their rotated `Hpsi` images, and reduced Hamiltonian/overlap
matrices are updated only for newly added columns. QE's eigenvalue-change
acceptance is the default; residual and energy-scaled safety gates remain
available through Python-specific controls.

### 0.6.6 - optional Numba scalar kernels (implemented)

The residual Python-side array work around FFT and BLAS/LAPACK calls now has
a lazily imported Numba path. Cached, no-GIL kernels implement plane-wave
scatter/gather, local-potential multiplication, density accumulation, PZ81
LDA evaluation, and the reciprocal-space Coulomb metric used by plain
Broyden mixing. The path is opt-in with `py_numba=.true.` because the LLVM/JIT
runtime costs approximately 100 MiB per MPI rank in the unreduced Si case.
The default NumPy/SciPy path preserves the lower-memory dependency-free mode.

Profiling prevents inappropriate JIT substitution: FFTs remain in the FFT
backend, projector products remain in BLAS, and Davidson orthonormalization
and subspace diagonalization remain in LAPACK. On the included small Si
case, the cached JIT path reduced SCF time by about eight percent while the
first invocation was slower because of LLVM compilation. Larger calculations
are expected to amortize compilation more effectively, but their dominant
SVD/FFT cost bounds the attainable Numba speedup.

### 0.6.7 - symmetry-grid and reciprocal-index corrections (implemented)

Multi-k-point FFT sizing is invariant under independent reciprocal-lattice
gauge shifts of k-point representatives: the grid uses the maximum
within-basis `G-G'` span rather than a global span across unrelated bases.
This restores the cubic 60 Ry Si grid from `(36,36,40)` to `(36,36,36)`.

The reciprocal-space density symmetrizer now rounds `fftfreq(n)*n` before
integer conversion. Direct truncation corrupted indices on non-power-of-two
grids because mathematically integral frequencies can lie infinitesimally
below their integer in floating-point arithmetic. The corrected projector
leaves the Si atomic density and ionic potential invariant at approximately
machine precision. The 8x8x8 Si energy is `-31.06956644 Ry`, within
`1.58e-5 Ry` of the corresponding QE result.

### 0.6.8 - factorized projectors and bounded solver storage (implemented)

The unreduced 216-k-point Si path now combines species/phase-factorized
projector caching, fused nonlocal reductions, shared distributed-FFT scratch,
32-bit reciprocal metadata, preallocated Davidson subspaces, and fused
occupied-band density accumulation. The tuned energy-residual safeguard
retains the Python solver's false-convergence protection while matching QE's
28-iteration SCF count.

### 0.6.9 - spin-unpolarized PBE GGA (implemented)

PBE exchange and correlation use the QE/XClib constants and density
thresholds, PW92 local correlation, and a periodic spectral evaluation of the
full variational GGA potential. Both valence and nonlinear-core density enter
the PBE energy and gradient, while only the valence density is mixed and
included in band-energy double counting. `input_dft='PBE'` and PBE UPF
metadata select this path; PZ81 remains the default for unknown metadata.

### 0.6.11 - symmetry-projected Hellmann–Feynman derivatives (implemented)

`tprnfor` evaluates ion-ion Ewald, local pseudopotential, nonlocal projector,
and nonlinear-core force terms from the converged density and occupied
states.
`tstress` evaluates the compressive-positive cell derivative of the complete
fixed-state Hellmann–Feynman energy, including kinetic, local, nonlocal,
Hartree, PZ/PBE XC, NLCC, and Ewald terms. Atomic permutations and Cartesian
vector/tensor rotations are projected over the crystal space group, so the
ordinary irreducible automatic k mesh remains active. The misspelled
`tprfor` variable is rejected with guidance to use QE's `tprnfor` spelling.
Space-group discovery also follows QE's `find_sym` restriction for
nonsymmorphic translations: every nonzero fractional component must be
`1/n`, up to sign and a lattice vector, for `n = 2, 3, 4, 6`. This prevents
slightly displaced structures from acquiring spurious symmetry operations.

### 0.6.12 - metallic occupations and smearing (implemented)

`occupations='smearing'` supports QE's `gaussian`, `mp`/
`methfessel-paxton`, `mv`/`cold`, and `fd`/`fermi-dirac` spellings. The
implementation uses QE's `wgauss` and `w1gauss` definitions, solves the
weighted electron-number constraint for the Fermi energy, and propagates
fractional occupations through the density, band energy, nonlocal
Hellmann–Feynman force, and fixed-state stress paths. The variational
smearing contribution is included in the reported free energy, with the
corresponding internal energy reported separately. `degauss` retains QE's Ry
input units.

### 0.7 — ordinary occupation and calculation coverage

Add tetrahedra occupations, NSCF, and bands calculations.

### 0.6.13 - QE-layout HDF5 wavefunctions (implemented)

Successful command-line runs write one `wfcN.hdf5` file per irreducible k
point alongside the existing QEXSD XML and reciprocal charge density. Each
file records QE's scalar-wavefunction attributes, ordered Miller indices with
reciprocal vectors, and the complex band-major `evc` dataset. Distributed
plane-wave rows are gathered directly to rank zero only after the final SCF
iteration; `disk_io='none'` avoids both collection and persistence.

### 0.6.14 - descriptor-driven MPI hot paths (implemented)

Wavefunction and charge grids now each build one QE-style stick/slab
descriptor shared by every lazy current-k workspace. It caches ownership,
transpose counts/displacements, and flattened slab indices. Distributed FFT
transposes consequently use one native `MPI_Alltoallv` without a preceding
Python-object count exchange; reciprocal-to-real sends are already contiguous,
and the reverse indexed pack is vectorized or compiled by the optional Numba
path. Grow-only exchange and aligned FFT scratch buffers remove steady-state
allocation.

The optional pyFFTW backend now plans the distributed one-dimensional stick
and two-dimensional slab transforms directly on those reusable buffers, so
the inverse FFT, local-potential product, and forward FFT remain in place.
Independent batch dimensions are flattened only in the plan view, preserving
the communication layout and numerical normalization.

Davidson combines projection with the raw Gram matrix in one collective and
forms the projected Gram matrix by a Schur complement, with explicit
reprojection retained for cancellation-dominated corrections. New projected
Hamiltonian and overlap rows are likewise reduced together. The common
expansion path therefore uses three small-matrix reductions instead of six.

### 0.8 — ground-state derivatives and persistence

XML/HDF5 density and wavefunction save output and `disk_io='none'` are
implemented. Add restart from the saved state, the remaining `disk_io`
storage levels, structural relaxation, and deterministic regression fixtures.

## 4. Validation ladder

1. Algebraic unit tests: lattice conventions, FFT normalization, Hermiticity,
   charge conservation, PZ81 limiting branches, UPF transforms, Ewald
   translation invariance.
2. Manufactured Hamiltonians: free electron and one-Fourier-mode local
   potential compared with analytic eigenvalues.
3. Cross-code tests: total energy, bands, density norms, forces, and stress
   against the checked-in QE test suite at identical cutoffs and k points.
4. Convergence studies: basis cutoff, FFT grid, k mesh, SCF threshold, and
   finite-size behavior. Agreement is required in converged observables, not
   in iteration-by-iteration trajectories.
5. Performance gates: Davidson memory is O(Npw × Nband ×
   `diago_david_ndim`) and local Hamiltonian application is O(Nband × Ngrid
   log Ngrid). Distributed FFTs, band/k-point parallelism, and large-system
   benchmarks remain prerequisites for production-sized claims.

## 5. Scholarly and source basis

The architecture follows the `pw.x` sources checked out under
`quantum-espresso/PW/src`, especially `pwscf.f90`, `run_pwscf.f90`,
`electrons.f90`, `electrons_scf.f90`, `h_psi.f90`, and the UPF library.
The physical formulation follows Kohn and Sham (1965), Ceperley and Alder
(1980), Perdew and Zunger (1981), Kleinman and Bylander (1982), and the
Quantum ESPRESSO descriptions by Giannozzi and collaborators (2009, 2017,
2020). Nonlinear core correction follows Louie, Froyen, and Cohen,
Phys. Rev. B 26, 1738 (1982). Numerical milestones should be read alongside
Payne et al. (1992) for plane-wave methods and Pulay (1980) and Johnson (1988)
for density mixing.
