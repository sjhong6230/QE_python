# Validation and performance

This document separates three questions that are often conflated:

1. does the implementation satisfy its own regression contract;
2. does it reproduce the supported Quantum ESPRESSO physics within numerical
   tolerances;
3. how quickly and with how much memory does it run on a controlled machine?

A single wall-time number cannot answer all three.

## 1. Automated test layers

### 1.1 Unit and algorithm tests

The test suite directly exercises:

- QE-style diagnostics and parser errors;
- LDA/GGA XC values and memory-bounded block evaluation;
- Broyden, TF, local-TF, and periodic-Pulay behavior;
- persistent FFT-stick ownership;
- FFTW-MPI layout and collective forward/backward correctness;
- selected save/restart and reference-output behavior.

Run all tests with:

```bash
python -m pytest -q
```

At the time this document was written, the current tree passes 67 tests. The
number is informational; the exit status and current collected test count are
authoritative.

### 1.2 QE-derived regression cases

`tests/qe_reference/upstream` contains 24 supported inputs selected from the
official QEF/q-e test suite at commit
`4a218b993489604a844db92fe85c747bd09b2442`. The fixtures cover:

- several `ibrav` conventions and explicit cell vectors;
- gamma, automatic, and explicit crystal k points;
- symmetry, nonsymmorphic controls, and automatic k-point reduction;
- Davidson, CG, ParO, RMM/Davidson, and RMM/ParO paths;
- plain, TF, and local-TF mixing;
- PZ, revPBE, and PBEsol examples;
- fixed, Fermi–Dirac, Methfessel–Paxton, cold-smearing, and optimized
  tetrahedron occupations;
- selected force, stress, band, and Fermi-level output.

`tests/qe_reference/reference` stores qepy-pw-generated regression output, not
fresh upstream-QE output. These tests detect changes in the port. Direct QE
comparison remains a separate validation activity.

Regenerate references only after reviewing the numerical change:

```bash
python -m tests.qe_reference.check
```

Never treat reference regeneration as a way to make an unexplained failure
disappear.

## 2. Regression tolerances

The manifest defines the following absolute tolerances:

| Quantity | Absolute tolerance |
|---|---:|
| Total energy | `5e-8 Ry` |
| SCF iteration count | exact |
| Total force | `1e-4 Ry/bohr` |
| Pressure | `0.01 kbar` |
| Fermi energy | `1e-4 eV` |
| HOMO or HOMO/LUMO | `1e-4 eV` |
| Band energies | `1e-4 eV` |
| Average diagonalization iterations | `0.05` |
| Number of k points | exact |

These tolerances are a regression policy, not universal error bars. A
scientific calculation must separately converge `ecutwfc`, `ecutrho`, k mesh,
number of bands, smearing, eigensolver threshold, and SCF threshold.

## 3. Collective FFT validation

The MPI-specific tests should also be run under the intended launcher:

```bash
mpiexec -n 4 python -m pytest -q tests/integration/test_fft_parallel.py \
  tests/integration/test_fft_pencil.py
```

They verify that every reciprocal stick has exactly one persistent owner and
that a native FFTW-MPI forward/backward pair recovers the distributed input to
floating-point tolerance.

The scalable engine adds:

- `tests/unit/core/test_fft_engine.py`: process-grid and memory-plan policy,
  Gamma half-G reconstruction, two-real-band packing, and Gamma metric;
- `tests/integration/test_fft_pencil.py`: Z/Y/X pencil round trips, a
  nonconstant local-potential convolution, and independent task groups;
- a slab test that proves distributed band tiling preserves Hpsi while
  retaining less scratch than the full-band block.

The dedicated microbenchmark reports the sparse slab local-potential path, a
dense FFTW-MPI round trip, and a dense pencil local-potential path:

```bash
export QEPY_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mpiexec -n 4 python tools/benchmark_distributed_fft.py \
  --shape 72 --bands 8 --iterations 100 --mode slab \
  --slab-band-tile 1
```

It reports:

- seconds per stick-owned local-potential application;
- maximum identity-potential roundtrip error;
- aggregate proportional set size (PSS);
- first-application time, including initial plan creation;
- dense FFTW-MPI roundtrip time and error;
- FFTW-MPI allocation padding beyond the active slab.
- pencil local-potential time, process grid, and planned tile scratch.

The two timings do not represent identical work. The stick benchmark includes
sparse packing, two distributed transposes, local-potential multiplication,
and inverse/forward transforms. The FFTW-MPI number is a dense three-dimensional
roundtrip. It is useful for architectural decisions, not as a claim that one
line is a drop-in replacement for the other.

## 4. Fair comparison with QE

For a meaningful QE comparison, hold the following fixed:

- exactly the same input structure and UPF files;
- same QE physical features, including `nosym`, `noinv`, smearing, and
  functional;
- same `ecutwfc`, `ecutrho`, k mesh, number of bands, and convergence
  thresholds;
- same number of physical cores and nodes;
- same MPI implementation and process binding when possible;
- no k-pool parallelization in QE, because qepy-pw does not implement it;
- same effective BLAS and FFT thread counts;
- same disk policy and filesystem class;
- an otherwise idle system and a consistent CPU power policy.

Compare both final observables and work performed. A faster run that used fewer
Davidson iterations because of a different starting state is not a pure kernel
comparison. Record at least:

- irreducible k-point count and plane waves per k point;
- FFT dimensions and stick counts;
- SCF iterations;
- average eigensolver iterations and Hψ applications;
- `h_psi`, `vloc_psi`, `fftw`, `calbec`, `sum_band`, `v_xc`, and total wall
  time;
- array estimate and peak aggregate PSS.

Run more than once. The first run may include page faults, dynamic-loader
work, filesystem cache misses, FFT plan creation, and CPU frequency ramp-up.
Use a median and preserve individual samples rather than reporting only the
best observation.

## 5. Interpreting timing output

The QE-shaped timer hierarchy includes these major categories:

| Timer | Principal work |
|---|---|
| `init_run` | Input-dependent setup, basis/grid construction, pseudopotentials, symmetry, starting state |
| `electrons` | Entire SCF electronic loop |
| `c_bands` | Kohn–Sham eigensolves over active k points |
| `cegterg` | Iterative eigensolver body; historical QE label also appears for non-Davidson paths in some summaries |
| `h_psi` | Hamiltonian applications |
| `vloc_psi` | Local-potential pseudospectral application |
| `fftw` | Native FFT region, including distributed FFT phases used by the fused kernel |
| `calbec` / `h_psi:calbec` | Nonlocal-projector overlaps |
| `sum_band` | Density accumulation from occupied states |
| `v_of_rho` | Hartree/XC potential construction |
| `v_xc` | Exchange–correlation evaluation, including GGA gradient work |
| `mix_rho` | Reciprocal density mixing and history update |
| `sym_rho` | Scalar density star projection |

Timers are nested and may overlap conceptually; do not sum every printed line.
Fused Cython kernels also move work across the labels used in upstream QE. Use
the hierarchy to identify dominant regions, not as an exact source-level
profile equivalence.

### CPU time versus wall time

On a healthy dedicated run, wall scaling should be consistent with the chosen
rank/thread layout. If wall time is much larger than the corresponding process
CPU time, investigate host contention, WSL scheduling, affinity, filesystem
latency, and oversubscription before changing numerical kernels.

## 6. Memory reports

The normal output deliberately presents three top-level quantities.

### 6.1 Estimated array memory per rank

```text
Estimated array memory/rank > ...
```

This is a source-level high-water estimate consisting of persistent arrays and
the largest expected numerical workspace. The component breakdown includes:

- density/potential grids;
- mixing history;
- wavefunctions;
- basis and reciprocal metadata;
- symmetry metadata;
- active eigensolver workspace;
- FFT/MPI workspace;
- nonlocal-projector workspace;
- GGA XC workspace where applicable.

It is marked with `>` because alignment, allocator size classes, FFTW internal
plan data, MPI internals, and library-private scratch cannot be reconstructed
exactly from NumPy `nbytes`.

### 6.2 Estimated total RAM for all ranks

```text
Estimated total RAM, all ranks > ...
```

This combines the aggregate explicit-array estimate with a measured or modeled
runtime baseline. The baseline includes Python, imported extension modules,
shared libraries, allocator state, MPI, BLAS, and FFT runtimes. It is a
pre-calculation capacity estimate, not an operating-system high-water counter.

### 6.3 Measured peak RAM for all ranks

```text
Measured peak RAM, all ranks = ... (aggregate PSS, sampled)
```

On Linux, each rank reads `/proc/self/smaps_rollup`. PSS divides each shared
page by the number of processes mapping it, so summing PSS across colocated
ranks avoids the most severe shared-library double counting of summed RSS.

The value is sampled at SCF checkpoints. It can miss a very short allocation
between samples and is therefore not identical to `VmHWM`. On platforms
without PSS, the output falls back to an available resident-memory measure and
labels it accordingly.

## 7. Memory-scaling analysis

For fixed rank count and a sequence of larger calculations, a useful empirical
model is

$$
M_{\mathrm{peak}}\approx M_0+\alpha M_{\mathrm{arrays}},
$$

where $M_0$ is the interpreter/runtime baseline and $\alpha-1$ represents
unaccounted allocator/library scratch and high-water overlap. For a rank sweep,
a more informative model is

$$
M_{\mathrm{node}}\approx M_{\mathrm{shared}}
+R M_{\mathrm{rank\ baseline}}
+\alpha\sum_{r=1}^{R}M_{\mathrm{arrays},r}.
$$

Fit only comparable calculations. Mixing LDA and GGA cases changes the
full-grid workspace; changing symmetry changes k-point and star metadata;
changing `disk_io` changes serialization lifetimes. These are explanatory
variables, not noise.

For a regression decision, prefer an optimization that reduces one of wall
time or peak PSS without increasing the other. A memory/time tradeoff should
be explicit and disabled by default unless the user selects it knowingly.

## 8. Performance methodology

A reproducible benchmark record should include:

```text
date and git revision
CPU model, physical cores, sockets, NUMA topology
RAM and WSL/native-Linux status
Python, NumPy, SciPy, Cython, mpi4py versions
MPI vendor and version
FFTW and BLAS provider
rank × thread layout and binding
input and pseudopotential checksums
individual run times, median, and peak PSS
```

Useful inspection commands include:

```bash
python -c "import numpy as np; np.show_config()"
python -c "from mpi4py import MPI; print(MPI.get_vendor()); print(MPI.Get_library_version())"
lscpu
ldd qepy_pw/_native_fft*.so
```

## 9. Correctness gates for an optimization

Before retaining a performance change:

1. run the full serial test suite;
2. run the FFT collective test under multiple ranks;
3. verify identity-potential and FFT roundtrip errors;
4. compare SCF energy, bands, density normalization, forces, and stress;
5. compare SCF and eigensolver work counts;
6. measure repeated wall time under controlled load;
7. compare estimated arrays and peak aggregate PSS;
8. reject or gate the change if its gain depends on a narrow case and regresses
   the supported serial/symmetry/nosym paths.

This procedure is particularly important for FFT planner flags, persistent MPI
collectives, padding, and batching. Their microbenchmark behavior can reverse
once plan creation, SCF block-size variation, allocator high water, and MPI
contention are included.
