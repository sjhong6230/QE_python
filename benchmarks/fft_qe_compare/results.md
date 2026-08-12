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
