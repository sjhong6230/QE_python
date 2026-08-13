# FFT benchmark results (2026-08-12)

Host: the same WSL instance for all runs. Numerical stack: Open MPI, FFTW3,
one OpenMP thread per MPI rank. Timings are single runs and should be replaced
by repeated, pinned-node medians before publication.

## QE 7.5 versus qepy-pw SCF

Input: `Si.gamma.in`, 4 MPI ranks, Gamma point, 160/640 Ry cutoffs, 32 bands.

| Measurement | QE 7.5 | qepy-pw | Port / QE |
|---|---:|---:|---:|
| External elapsed time | 1.13 s | 2.93 s | 2.59x |
| Internal PWSCF wall time | 0.87 s | 2.09 s | 2.40x |
| Local-potential wall time | 0.39 s | 0.75 s | 1.92x |
| FFT-related wall timer | 0.36 s | 0.79 s | 2.19x |
| SCF iterations | 8 | 7 | -- |
| Final total energy | -14.57882210 Ry | -14.57882210 Ry | equal at print precision |
| Reciprocal rows at Gamma | 4593 | 9185 | 2.00x |

The FFT rows are not perfectly congruent counters. QE reports local FFT work
under `fftw` and MPI work separately, while qepy-pw's `fftw` scope includes
its fused sparse packing, local transforms, collectives, potential multiply,
and gather. The decisive structural observation is independent of timer
scope: QE stores the Gamma half-space (4593 rows), while the current production
port stores the complete +/-G basis (9185 rows). This explains much of the
measured local-potential gap and is the reason `GammaHalfSpectrum` was added.

Raw outputs: `qe.out`, `qe.time`, `qepy.out`, and `qepy.time` in this directory.

## Distributed FFT microbenchmark

Input: 72^3 grid, 8 bands, 4 MPI ranks, one thread per rank. Identity potential
gives an end-to-end accuracy check. These are separate process runs so PSS is
not contaminated by another backend's retained plans.

### Sparse production slab

| Band tile | Apply time | Aggregate PSS | Max error |
|---:|---:|---:|---:|
| Full block (8) | 0.02701 s | 215.28 MiB | 1.25e-15 |
| 2 | 0.02027 s | 183.98 MiB | 1.25e-15 |
| 1 | 0.01866 s | 176.23 MiB | 1.25e-15 |

For this small four-rank case, tile 1 reduced aggregate PSS by 18.1% and time
by 30.9% relative to the full-band block. The speed result is workload- and
MPI-dependent; the robust property is that FFT scratch now scales with the
tile instead of total band count.

### Dense pencil engine

| Task groups | FFT ranks/group | Process grid | Apply time | Aggregate PSS | Planned tile scratch/rank | Max error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 2x2 | 0.10212 s | 334.25 MiB | 14.24 MiB | 1.59e-15 |
| 2 | 2 | 1x2 | 0.09097 s | 394.10 MiB | 28.48 MiB | 1.59e-15 |

Two task groups improved throughput by 10.9% here, but increased aggregate PSS
by 17.9% because each group uses fewer spatial ranks and therefore owns larger
pencils. This directly demonstrates why task-group count must be constrained
by a memory model rather than maximized blindly.

The pencil and sparse slab times are not a backend speed comparison. The
pencil test transforms every grid point; the slab Hpsi test transforms only
active plane-wave sticks and uses a fused native kernel. Pencil's current role
is correctness-validated scaling beyond the `MPI ranks <= Nz` slab limit.

## Production SCF slab versus pencil

The production sparse plane-wave boundary was then connected to the pencil
engine and rerun with the same `Si.gamma.in` input and four ranks. Both paths
converged in seven iterations to `-14.57882210 Ry`, with identical printed
bands and energy components.

| Decomposition | PWSCF wall | FFT wall | Vloc wall | Aggregate peak PSS |
|---|---:|---:|---:|---:|
| Sparse slab | 1.78 s | 0.63 s | 0.60 s | 216.43 MiB |
| Dense pencil | 3.53 s | 2.42 s | 2.18 s | 225.31 MiB |

The pencil result is a scaling implementation, not a claim of a crossover at
four ranks: its dense Z/Y/X plan arrays and two MPI transposes are expected to
lose to the fused sparse slab kernel at this modest process count. The result
establishes end-to-end SCF equivalence and removes the slab's `ranks <= Nz`
topological limit. The next performance work is native packed pencil
transposes and avoiding dense inactive-grid traffic.

## Completed production Gamma and non-Gamma comparison

These measurements supersede the earlier one-shot Python timing above. Each
entry is the median of three sequential runs on the same WSL host, with four
MPI ranks and one FFT/OpenMP/BLAS thread per rank. QE is the locally installed
7.5 build. The Python Gamma runs force `QEPY_GAMMA_MODE=half`; both Python
cases use the sparse slab backend because four ranks are well below `Nz`.

### Gamma point (`Si.gamma.in`)

| Measurement | QE 7.5 | qepy-pw half-G | Port / QE |
|---|---:|---:|---:|
| External elapsed | 1.13 s | 2.40 s | 2.12x |
| Internal PWSCF wall | 0.83 s | 1.64 s | 1.98x |
| Local potential wall | 0.35 s | 0.29 s | 0.83x |
| FFT wall timer | 0.34 s | 0.33 s | 0.97x |
| SCF iterations | 8 | 7 | -- |
| Final energy | -14.57882210 Ry | -14.57882210 Ry | equal at print precision |
| Stored solver rows | 4593 | 4593 | equal |

The hot Gamma FFT/local-potential region is now at parity with QE and is
slightly faster in this run. The remaining end-to-end gap is dominated by
Python process/import/setup work and by orchestration outside the FFT timer,
not by doubled `+/-G` storage. The port reports a median sampled aggregate PSS
of 220.62 MiB across four Python ranks; QE does not print a directly
comparable aggregate PSS counter.

### Non-Gamma point (`Si.kpoint.in`)

| Measurement | QE 7.5 | qepy-pw full complex | Port / QE |
|---|---:|---:|---:|
| External elapsed | 1.71 s | 2.42 s | 1.42x |
| Internal PWSCF wall | 1.46 s | 1.74 s | 1.19x |
| Local potential wall | 0.63 s | 0.62 s | 0.98x |
| FFT wall timer | 0.67 s | 0.65 s | 0.97x |
| SCF iterations | 8 | 8 | equal |
| Final energy | -14.89405563 Ry | -14.89405563 Ry | equal at print precision |

The non-Gamma result independently checks that the general complex path did
not regress: both FFT and local-potential medians are within three percent of
QE, and total internal PWSCF time is within 19 percent. The Python aggregate
sampled PSS median is 214.99 MiB.

QE's `fftw` row counts individual one-dimensional transforms, whereas the
Python timer encloses fused sparse packing, transforms, collectives,
potential multiplication, and gathering. The near-equal wall totals are
therefore more meaningful than comparing their call counts.

## Heavy automatic-grid comparison

The heavier `Si.heavy.auto.in` calculation uses a `2x2x2` automatic k-point
grid, which reduces to three irreducible k points for diamond Si, 320 Ry
wavefunction and 1280 Ry density cutoffs, a `96x96x96` dense FFT grid,
approximately 25,500 plane waves per k point, and 128 bands. Both programs
used four MPI ranks and one numerical-library/FFT thread per rank. The port
also used `QEPY_FFT_MEMORY_LIMIT_MIB=128`. Entries below are medians of three
sequential runs.

| Measurement | QE 7.5 | qepy-pw | Port / QE |
|---|---:|---:|---:|
| External elapsed | 125.85 s | 45.28 s | 0.36x |
| Internal PWSCF wall | 125.58 s | 44.56 s | 0.35x |
| `c_bands` wall | 111.07 s | 41.74 s | 0.38x |
| Local potential wall | 41.04 s | 32.31 s | 0.79x |
| FFT wall timer | 39.22 s | 32.30 s | 0.82x |
| SCF iterations | 6 | 6 | equal |
| Final energy | -15.66126555 Ry | -15.66126555 Ry | equal at print precision |
| GNU-time maximum RSS | 191.35 MiB | 268.24 MiB | 1.40x |

The port's median sampled aggregate PSS was 508.88 MiB across four ranks; its
array estimate was greater than 167.25 MiB per rank. The 128 MiB setting is a
scratch budget and must not be interpreted as a cap on total resident memory.
GNU `time -v` reports a job/process-tree maximum RSS field rather than an
aggregate PSS, so the RSS row describes the observed launcher measurement and
must not be multiplied by the rank count.

The FFT/local-potential result remains favorable at this scale: the port is
about 18-21% faster in those timed regions. Most of the much larger total-time
difference is outside FFT. This local QE executable selected serial subspace
diagonalization and links Debian's reference `libblas`, while the NumPy wheel
used by the port contains OpenBLAS. Consistently, QE spent a median 24.16 s in
`cegterg:over`, 12.84 s in `cegterg:upda`, and 23.32 s in `cegterg:last`,
versus 2.65 s, 3.19 s, and 0.19 s in the corresponding port timers. The 2.8x
end-to-end advantage is therefore a result for these concrete binaries, not
an intrinsic Python-versus-Fortran or FFT-only speed ratio.

A single four-rank QE run with `-ndiag 4` enabled its 2x2 ScaLAPACK subgroup
and reduced PWSCF wall time only from the 125.58 s baseline median to 120.51 s;
FFT remained 39.38 s. A separate six-rank topology check gave:

| Six-rank layout | PWSCF wall | FFT wall | Aggregate sampled PSS |
|---|---:|---:|---:|
| QE, three k pools (`-nk 3`, two spatial ranks/pool) | 131.78 s | 47.10 s | not reported |
| qepy-pw, six spatial ranks | 37.98 s | 26.98 s | 567.46 MiB |

For the port, increasing four to six spatial ranks improved PWSCF wall time
from 44.56 s to 37.98 s (1.17x speedup for 1.5x ranks) while aggregate PSS
rose by 11.5%. QE's three-pool run was slower on this small node because each
k point then had only two spatial FFT ranks; the reduced per-k spatial
parallelism outweighed concurrent k-point execution. This is direct evidence
that k-pool/task-group counts must be chosen jointly with FFT decomposition
and memory replication rather than maximized independently.

## Fair external-FFTW3/OpenBLAS rebuild comparison (2026-08-13)

QE 7.5 was rebuilt with CMake in Release mode against the same system FFTW3
and Open MPI used by the port, plus pthread OpenBLAS and Open MPI ScaLAPACK.
`ldd` resolved `libfftw3.so.3`, `libopenblas.so.0`, and
`libscalapack-openmpi.so.2.2`. Runs used four core-bound MPI ranks and one
OpenMP/BLAS/FFT thread per rank. Each number is the median of three sequential
runs after a separate smoke run. The port used the sparse slab path; the Gamma
case explicitly selected half-G storage. GNU-time maximum RSS is the largest
reported launcher/process-tree resident set, not aggregate rank memory.

### Small Gamma and non-Gamma cases

| Case and measurement | rebuilt QE 7.5 | qepy-pw | Port / QE |
|---|---:|---:|---:|
| Gamma external elapsed | 1.12 s | 2.43 s | 2.17x |
| Gamma PWSCF wall | 0.85 s | 1.68 s | 1.98x |
| Gamma `c_bands` wall | 0.42 s | 0.72 s | 1.71x |
| Gamma local-potential wall | 0.35 s | 0.32 s | 0.91x |
| Gamma FFT wall timer | 0.29 s | 0.37 s | 1.28x |
| Gamma iterations | 8 | 7 | -- |
| Gamma final energy | -14.57882210 Ry | -14.57882210 Ry | equal at print precision |
| Non-Gamma external elapsed | 1.44 s | 2.60 s | 1.81x |
| Non-Gamma PWSCF wall | 1.16 s | 1.88 s | 1.62x |
| Non-Gamma `c_bands` wall | 0.62 s | 1.07 s | 1.73x |
| Non-Gamma local-potential wall | 0.50 s | 0.73 s | 1.46x |
| Non-Gamma FFT wall timer | 0.50 s | 0.76 s | 1.52x |
| Non-Gamma iterations | 8 | 8 | equal |
| Non-Gamma final energy | -14.89405563 Ry | -14.89405563 Ry | equal at print precision |

Interpreter/import and MPI startup dominate much of the small-case external
gap. The Gamma local-potential region is already at parity, but the rebuilt QE
is faster in the non-Gamma hot path. The FFT timer scopes and call granularity
remain different, so end-to-end PWSCF and local-potential totals carry more
weight than the timer labels alone.

### Heavy automatic-grid case

The heavy input has a 96^3 dense FFT grid, 128 bands, three irreducible points
from a 2x2x2 automatic k grid, and six SCF iterations in both programs. Strict
algorithmic parity used QE `-ndiag 1`, since the port does not yet implement a
distributed subspace eigensolver.

| Measurement | rebuilt QE, `-ndiag 1` | qepy-pw | Port / QE |
|---|---:|---:|---:|
| External elapsed | 64.98 s | 54.24 s | 0.83x |
| PWSCF wall | 64.69 s | 53.40 s | 0.83x |
| `c_bands` wall | 49.38 s | 50.16 s | 1.02x |
| `sum_band` wall | 10.73 s | 0.58 s | 0.05x |
| Local-potential wall | 45.16 s | 39.73 s | 0.88x |
| FFT wall timer | 39.47 s | 39.78 s | 1.01x |
| Final energy | -15.66126555 Ry | -15.66126555 Ry | equal at print precision |
| GNU-time maximum RSS | 202.28 MiB | 266.57 MiB | 1.32x |

The corrected result is qualitatively different from the reference-BLAS QE
comparison. Rebuilding QE reduced its median PWSCF wall from 125.58 s to
64.69 s and `c_bands` from 111.07 s to 49.38 s. Its FFT time remained near
39 s, showing that the prior 2.8x end-to-end port advantage was chiefly a BLAS
and Davidson-build artifact, not an FFT result. With comparable libraries,
`c_bands` and FFT are at parity. The port retains a 17% PWSCF advantage mainly
because its occupied-band density construction makes `sum_band` roughly
10.15 s faster; it also has a 12% lower local-potential total in this run.

As a practical QE check, `-ndiag 4` selected a 2x2 ScaLAPACK subgroup. Its
three-run medians were 65.11 s external and 64.84 s PWSCF, statistically
indistinguishable from `-ndiag 1` at this 128-band size. ScaLAPACK therefore
adds no useful speedup on this four-rank, single-node case, although it remains
important for substantially larger band subspaces.

Both FFT paths load the same system FFTW3 3.3.10 library. Total-time BLAS
parity is close but not binary-identical: QE uses Ubuntu pthread OpenBLAS
0.3.32 with LP64 integers, whereas the NumPy wheel contains OpenBLAS 0.3.34
with ILP64 integers. A publication-grade eigensolver comparison should rebuild
NumPy/SciPy against the same OpenBLAS binary or treat total Davidson time as a
separate variable. The FFT-focused conclusion is stronger because the FFTW3
shared object, MPI runtime, rank count, binding, grid, and thread count match.

## Four-band, 8x8x8 automatic-grid comparison (2026-08-13)

To test the dependence on empty bands, `nbnd` was reduced from 128 to the four
occupied Si bands and the automatic grid was raised from `2x2x2` to `8x8x8`.
Both programs generated 29 irreducible k points, retained the 96^3 dense FFT
grid, and converged in seven iterations. The execution policy and libraries
match the fair rebuilt-QE comparison above. After separate warmups, three
measurement pairs were run in alternating QE/Python order.

| Measurement | rebuilt QE, `-ndiag 1` | qepy-pw | Port / QE |
|---|---:|---:|---:|
| External elapsed | 28.26 s | 31.49 s | 1.11x |
| PWSCF wall | 28.00 s | 30.65 s | 1.09x |
| `init_run` wall | 2.29 s | 3.11 s | 1.36x |
| `electrons` wall | 25.65 s | 27.54 s | 1.07x |
| `c_bands` wall | 20.53 s | 22.05 s | 1.07x |
| `cegterg` wall | 19.78 s | 18.96 s | 0.96x |
| Hpsi wall | 20.84 s | 18.75 s | 0.90x |
| Local-potential wall | 20.31 s | 18.19 s | 0.90x |
| FFT wall timer | 17.43 s | 22.52 s | 1.29x |
| `sum_band` wall | 4.79 s | 4.75 s | 0.99x |
| Final energy | -15.85334687 Ry | -15.85334688 Ry | 1e-8 Ry difference |
| GNU-time maximum RSS | 109.58 MiB | 203.70 MiB | 1.86x |

Reducing `nbnd` removes the port's large occupied-band density advantage:
`sum_band` is now equal instead of 0.58 versus 10.73 s. The port still has a
10% faster Hpsi/local-potential total and a 4% faster Davidson kernel, but
Python-level per-k orchestration and initialization make `c_bands` 7% slower
and PWSCF 9% slower overall. The QE and port FFT timer scopes are not
congruent; in particular, the port timer includes fused packing, collectives,
potential multiplication, and gathering. The local-potential total is the
more useful hot-path comparison here.

The result confirms the band-count hypothesis: with 128 bands and only four
occupied bands, the port was 17% faster overall; with no empty bands and many
k points, rebuilt QE is 9-11% faster while the port's actual Hpsi/local-
potential kernel remains faster. The port also consumes about 1.86 times the
reported per-process-tree RSS in this narrow-band case, so QE is preferable
under the tested memory constraint unless future work removes the Python
per-rank and per-k overhead.

## Non-FFT k-point-path optimization (2026-08-13)

The Python iterative path previously materialized every complete plane-wave
basis and rebuilt rank-local `G+k`, kinetic, and packed nonlocal-projector
arrays on every `c_bands` call. It now materializes only the FFT-owned rows
and retains this SCF-invariant data in a bounded, per-rank cache. Dense
diagonalization keeps the full-basis path. Nonlocal force and stress paths
also obtain only their rank-local basis rows. Four new timing scopes separate
basis preparation, Hamiltonian-object construction, wavefunction saving, and
cleanup from the Davidson kernel.

The cache is deliberately bounded rather than made unconditionally resident.
QE 7.5 retains `igk_k(npwx,nks)` and `ngk(nks)` for every local k point, but
uses one `g2kin(npwx)` kinetic buffer and one `vkb(npwx,nkb)` projector buffer.
Inside every `c_bands` k loop it calls `g2_kin(ik)` and `init_us_2(...)`,
overwriting those current-k buffers. The Python cache is therefore more
aggressive than QE's memory policy. `QEPY_STATIC_CACHE_LIMIT_MIB=32` was used
explicitly for the cached measurements below. Following the memory ablation,
the runtime uses a memory-first default of `0`; positive values opt into reuse,
and negative/non-finite values are rejected. The run footer reports occupancy.

### Cache ablation

Three alternating cache-off/cache-on pairs used the four-band 8x8x8 input,
four core-bound MPI ranks, one OpenMP/BLAS thread per rank, and the same Python
executable. All six runs converged in seven iterations to -15.85334688 Ry.
The table contains three-run medians.

| Measurement | cache disabled | 32 MiB/rank cache | Cached / uncached |
|---|---:|---:|---:|
| External elapsed | 34.55 s | 33.91 s | 0.981x |
| PWSCF wall | 33.63 s | 32.99 s | 0.981x |
| `c_bands` wall | 24.26 s | 23.52 s | 0.969x |
| `init_us_2` wall/calls | 0.47 s / 232 | 0.22 s / 29 | 0.468x / 0.125x |
| basis preparation wall/calls | 0.10 s / 232 | 0.01 s / 29 | 0.100x / 0.125x |
| GNU-time maximum RSS | 200.17 MiB | 200.25 MiB | 1.000x |

The cache admitted all 29 irreducible k points and retained 29.78 MiB on the
reporting rank, just below the 32 MiB cap. Retaining these objects barely
changes *peak* RSS because uncached construction already reaches nearly the
same transient peak; it does change lifetime and can overlap other large
workspaces in a production calculation. An unlimited default would therefore
be unsafe for the intended large-grid, many-k-point, memory-constrained use
case, and it would provide no further speedup here because every k point is
already cached.

### Rebuilt-QE comparison after optimization

Machine throughput drifted substantially relative to the earlier 28-31 s
series, so old and new absolute timings are not subtracted. Instead, three
new adjacent QE/Python pairs were run with alternating order. QE used the
external-FFTW/OpenBLAS/ScaLAPACK build and `-ndiag 1`; Python used the 32 MiB
static cache. Both used four core-bound MPI ranks and one thread per rank.

| Measurement | rebuilt QE, `-ndiag 1` | optimized qepy-pw | Port / QE |
|---|---:|---:|---:|
| External elapsed | 36.61 s | 35.69 s | 0.975x |
| PWSCF wall | 36.28 s | 34.67 s | 0.956x |
| `init_run` wall | 2.77 s | 3.55 s | 1.282x |
| `electrons` wall | 33.41 s | 31.10 s | 0.931x |
| `c_bands` wall | 27.87 s | 24.84 s | 0.891x |
| `cegterg` wall | 26.75 s | 22.36 s | 0.836x |
| Hpsi wall | 27.80 s | 21.84 s | 0.786x |
| Local-potential wall | 26.99 s | 21.14 s | 0.783x |
| FFT wall timer | 23.73 s | 26.11 s | 1.100x |
| `sum_band` wall | 5.47 s | 5.48 s | 1.002x |
| Final energy | -15.85334687 Ry | -15.85334688 Ry | 1e-8 Ry difference |
| GNU-time maximum RSS | 109.57 MiB | 200.21 MiB | 1.827x |

The optimized port wins all three adjacent external-time pairs (ratios 0.975,
0.945, and 0.934) and reduces the median `c_bands` wall below QE by 10.9% in
this series. The result does not imply a 10.9% gain from the cache alone: the
controlled cache ablation attributes about 3.1% of `c_bands` and 1.9% of
end-to-end time to reuse, while run-to-run kernel throughput accounts for the
remainder. The FFT rows remain scope-incommensurate, as discussed above.
Python still requires about 1.83 times QE's maximum RSS, so memory—not speed—
is now the principal disadvantage in this four-band case.

Regression validation after the change completed with 229 passed and two
expected skips across the unit, QE-reference, slab-FFT, and pencil-FFT suites.
Launching the slab/pencil integration files under four MPI ranks additionally
produced 10 passing tests on each rank.
