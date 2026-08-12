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
is correctness-validated scaling beyond the `MPI ranks <= Nz` slab limit; its
next optimization is native transpose packing and sparse Z-pencil boundaries.
