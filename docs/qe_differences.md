# Differences from Quantum ESPRESSO

`qepy-pw` follows the scalar SCF design and many numerical conventions of
Quantum ESPRESSO `pw.x`, but it is not a drop-in replacement. This document
states the compatibility boundary explicitly so that numerical agreement is
not confused with feature equivalence.

## 1. Purpose and implementation language

Quantum ESPRESSO is a mature Fortran suite for production electronic-structure
and materials simulation. `qepy-pw` is a Python/Cython research port of a
restricted `pw.x` path. Its principal goals are readable physical logic,
controlled numerical comparison, and investigation of time and memory behavior
without giving up compiled FFT, MPI, and linear-algebra kernels.

Physical orchestration and formula selection remain in Python. Cython owns
large elementwise loops, FFTW plans, MPI transpose calls, packing, density
accumulation, and selected BLAS/LAPACK bridges. This boundary makes the
equations inspectable but leaves a larger per-process runtime baseline than a
Fortran executable.

## 2. Physics scope

| Area | `qepy-pw` | Upstream QE |
|---|---|---|
| Ground state | Scalar periodic SCF | Scalar, spin, noncollinear, many additional modes |
| Pseudopotentials | Norm-conserving UPF subset, including separable nonlocal projectors and NLCC | Norm-conserving, ultrasoft, PAW, broader formats and features |
| XC | PZ81, PW92, PBE, PBEsol, revPBE, RPBE | Broad LibXC/internal functional coverage, hybrids, meta-GGA, etc. |
| Occupations | Fixed, major smearing schemes, linear/optimized tetrahedra | Broader occupation and constraint machinery |
| Forces/stress | Implemented scalar norm-conserving analytic terms | Full production coverage for supported QE physics |
| Ionic/cell motion | Not implemented | Relaxation, MD, variable-cell dynamics |
| Magnetism | Not implemented | LSDA, noncollinear, spin–orbit, constraints |
| Correlated methods | Not implemented | DFT+U, hybrids, exact exchange, and interfaces to further methods |
| Electric/Berry/Wannier | Not implemented | Electric fields, polarization, Berry phase, Wannier interfaces |

An unsupported physical variable causes an error before the SCF loop. It is
not accepted and ignored.

## 3. Input compatibility

The parser accepts QE-style Fortran namelists and the supported cards, but only
the variables listed in [Input parameters](input_parameters.md) are active.
Important distinctions are:

- `calculation='scf'`, `'nscf'`, and `'bands'` are available; the fixed-potential modes require a compatible density saved by a preceding SCF calculation;
- `&IONS` and `&CELL` have no implemented variables;
- explicit FFT dimensions such as `nr1`, `nr2`, and `nr3` are not available;
- `occupations='from_input'` and a working `OCCUPATIONS` card are absent;
- `mixing_pulay_frequency` is a qepy-pw research extension;
- `disk_io` accepts QE's `none`, `low`, `medium`, and `high` values, while the persistent levels currently share one end-of-run save policy;
- QE-recognized variables outside the implemented subset fail explicitly.

The error/warning presentation deliberately resembles QE 7.5 where the same
condition is implemented. See [QE-compatible diagnostics](qe_diagnostics.md).

## 4. Pseudopotential behavior

The code reads norm-conserving UPF radial data, local potentials, nonlocal
projectors, atomic starting orbitals/densities, functional metadata, and
nonlinear core corrections used by the scalar path. It does not implement
ultrasoft augmentation charges, generalized overlap matrices, or PAW onsite
terms. A UPF that requires absent physics must not be expected to work merely
because its XML can be parsed.

If `input_dft` is absent, the functional is resolved from UPF metadata. As in
QE, users should avoid combining incompatible functional metadata and
pseudopotential generation assumptions.

## 5. Plane-wave and FFT decomposition

Both codes organize reciprocal vectors into complete `(Gx,Gy)` sticks, balance
sticks across FFT ranks, and transpose between reciprocal sticks and real-space
planes/slabs. The present one-dimensional decomposition uses:

- one persistent owner per complete z stick;
- G-vector count as the primary load balance and stick count as the tie
  breaker, following FFTXlib's `sticks_sort_new`/`sticks_dist_new` logic;
- exact-count `MPI_Alltoallv` for stick-to-slab and slab-to-stick transposes;
- a native `(z,x,y)` real-slab layout for contiguous two-dimensional FFTs;
- one-band real-space streaming to bound peak memory;
- direct reverse reception into the original stick workspace.

QE FFTXlib commonly pads each peer block to a common `sendsize` and calls
fixed-count `MPI_Alltoall`. That can be faster on some MPI implementations but
stores and transforms padding. `qepy-pw` currently keeps exact payloads because
the padded variant did not satisfy the project's joint time-and-memory
criterion.

FFTW-MPI is detected and linked when ABI-compatible system headers and
libraries are available. A dense distributed plan is exposed and tested, but
it is not the Hψ backend. Standard FFTW-MPI reciprocal output has slab
ownership; returning it to the persistent sparse stick owners would add a
second redistribution beyond FFTW-MPI's internal transpose.

## 6. Parallel hierarchy

The implemented hierarchy is plane-wave/FFT MPI plus rank-local OpenMP/FFTW
threads. It deliberately has no:

- k-point pools;
- band groups;
- FFT task groups;
- image parallelism;
- ScaLAPACK process grid;
- GPU backend.

All MPI ranks cooperate on each active k-point. Consequently, comparisons with
QE must use QE without pool parallelization and should match total physical
cores, affinity, FFT grid, BLAS threading, and MPI implementation.

MPI is initialized at `MPI_THREAD_FUNNELED`; worker threads do not call MPI.
Large wavefunction rows and FFT grids are distributed, while Python interpreter
state and several rank-local workspaces are replicated.

## 7. Diagonalization

The iterative solvers reproduce central QE algorithms and preconditioners but
are independent implementations:

- restarted block Davidson with a QE-style smooth diagonal preconditioner;
- band conjugate gradient;
- parallel-orbital block iteration;
- bounded RMM-DIIS and RMM/Davidson or RMM/ParO hybrids;
- dense Hermitian diagonalization for small diagnostic problems.

Operation ordering, orthogonalization thresholds, LAPACK provider, and
floating-point reduction trees may differ. Therefore band energies and SCF
trajectories should agree within tolerances, but bitwise identity is not a
contract.

ScaLAPACK is not presently used. Reduced subspace matrices are replicated and
solved locally after MPI reduction; this is appropriate for the targeted band
counts but differs from large-band production QE configurations.

## 8. Density mixing

`mixing_mode='plain'` is a reciprocal-space Coulomb-metric modified-Broyden /
Anderson update modeled on QE `mix_rho`, not simple linear mixing. TF and
local-TF preconditioners are implemented. Singular history Gram matrices fall
back to a least-squares solution so an exactly dependent history need not
terminate the run; this is a deliberate robustness difference from a failing
factorization.

Periodic Pulay (`mixing_pulay_frequency>1`) is experimental and is not enabled
automatically. Its default of one follows the ordinary every-step multisecant
trajectory.

## 9. Symmetry and reproducibility

Space-group operations are found through the Python symmetry implementation
and `spglib`, then filtered for FFT-mesh compatibility. Scalar reciprocal
density stars include nonsymmorphic phase factors. Symmetry metadata are
stored compactly and the projector operates without one full density per
operation.

MPI and thread counts can change the order of reductions and FFT execution.
The code is designed to produce the same physical result within floating-point
tolerance, not necessarily identical last bits. Symmetry can make such
last-bit differences more visible because star averaging and k-point weights
introduce additional sums. A meaningful reproducibility test compares energy,
eigenvalues, density norms, force, and stress tolerances rather than raw byte
identity.

## 10. Memory behavior

QE's compiled-process baseline is substantially smaller. Every qepy-pw MPI
rank contains a CPython interpreter, extension-module state, NumPy/SciPy,
mpi4py, allocator arenas, dynamic libraries, and rank-private numerical
runtimes. Increasing the physical calculation size at fixed rank count mainly
increases explicit arrays and their bounded workspaces; increasing ranks also
replicates the Python/runtime baseline.

The code offsets this disadvantage through:

- distributed plane-wave rows and FFT slabs;
- node-shared read-only arrays where safe;
- compact unsigned index arrays;
- lazy imports and lazy k-point workspaces;
- reusable grow-only FFT/MPI buffers;
- batch-size-one real-space orbital processing;
- output arrays supplied by callers to avoid result temporaries;
- fused kinetic/local-potential and density kernels;
- allocator trimming at known lifetime boundaries.

The printed estimated array memory is based on owned numerical arrays. The
estimated total adds a runtime/allocator model, while measured peak aggregate
PSS is sampled from the operating system. These quantities are related but not
identical; details are in [Validation and performance](validation_and_performance.md).

## 11. File I/O and restart

When `disk_io!='none'`, qepy-pw writes a QE-shaped `<prefix>.save` directory
containing XML/HDF5 density and wavefunction data plus pseudopotential copies.
This is intended for the implemented restart path and interoperability tests,
not as a guarantee that every QE postprocessor accepts every file.

QE has several disk-I/O levels controlling wavefunction retention and
frequency. qepy-pw currently keeps active wavefunctions in memory and treats
all non-`none` values as one save policy. `disk_io='none'` skips persistent
output but does not change the mathematical SCF state.

## 12. Output and diagnostics

Text output is QE-shaped so familiar quantities and timings can be compared.
It is not a byte-for-byte QE transcript. Timing labels map to analogous work,
but Python/Cython boundaries and fused kernels can assign time differently.
For example, `fftw` includes the fused distributed FFT region, and some kinetic
or packing work may be included in a caller's timer rather than a standalone
QE timer.

Error banners and messages are copied or closely modeled only when the same
condition exists. Errors for absent Python-port features explicitly name
PWSCF-PY rather than implying upstream QE lacks the feature.

## 13. Numerical validation contract

The project validates selected official QE inputs but stores qepy-pw-generated
reference output. Current automated tolerances include `5e-8 Ry` for total
energy, `1e-4 eV` for bands/Fermi levels, `1e-4 Ry/bohr` for total force, and
`0.01 kbar` for pressure. Passing these tests demonstrates stability of the
implemented port; it is not a general certification over all materials,
pseudopotentials, cutoffs, or QE features.
