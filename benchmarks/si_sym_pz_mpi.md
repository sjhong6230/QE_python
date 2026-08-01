# Si 160 Ry MPI comparison

Input: `/home/sjhong6230/QE_porting_test/Si_sym_PZ/*/Si.scf.in`.
BLAS/OpenMP threads were limited to one for the optimized Python reruns.

## Existing QE 7.5 outputs

| MPI ranks | SCF iterations | Wall (s) | QE max estimate/rank (MB) | Largest reported RSS (MiB) |
|---:|---:|---:|---:|---:|
| 1 | 14 | 10.93 | 58.91 | 110 |
| 2 | 13 | 12.09 | 30.47 | 73 |
| 3 | 10 | 7.07 | 20.99 | 61 |
| 4 | 10 | 38.74 | 16.25 | 54 |

## Existing Python outputs before this change

| MPI ranks | SCF iterations | Wall (s) | Array estimate/rank (MiB) | Peak RSS/rank (MiB) |
|---:|---:|---:|---:|---:|
| 1 | 14 | 83.25 | 192.94 | 551.28 |
| 2 | 14 | 67.53 | 184.09 | 419.38 |
| 3 | 14 | 57.48 | 154.77 | 413.66 |
| 4 | 14 | 90.24 | 140.12 | 411.80 |

## Optimized Python reruns

| MPI ranks | SCF iterations | External wall (s) | Internal wall (s) | Array estimate/rank (MB) | Peak RSS/rank (MiB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 14 | 86.48 | 83.86 | 160.85 | 528.76 |
| 2 | 14 | 71.55 | 68.96 | 151.57 | 412.48 |
| 3 | 14 | 66.16 | 63.31 | 120.83 | 413.48 |
| 4 | 14 | 96.86 | 94.05 | 105.46 | 411.84 |

All optimized Python runs converged to `-31.1796883066 Ry` with the same
14-iteration trajectory.

## Optimized Python routine wall times

| MPI ranks | `c_bands` | `sum_band` | `h_psi` | `vloc_psi` | `fftw` |
|---:|---:|---:|---:|---:|---:|
| 1 | 44.26 | 27.73 | 38.72 | 36.26 | 36.87 |
| 2 | 29.95 | 26.89 | 26.06 | 23.52 | 24.29 |
| 3 | 24.29 | 26.73 | 20.84 | 18.23 | 18.76 |
| 4 | 52.16 | 28.84 | 41.23 | 36.02 | 38.73 |

Times are cumulative seconds. Nested rows intentionally overlap.

## Interpretation

- G-vector-distributed Hamiltonian work scales through three ranks:
  `h_psi` falls from 38.72 to 20.84 seconds.
- `sum_band` remains nearly constant because every rank processes every
  k-point and occupied band. K-point pools are the next coarse-grained
  parallelization target.
- Four-rank regression occurs in both implementations: QE rises from 7.07
  seconds at three ranks to 38.74 seconds at four, while Python rises from
  63.31 to 94.05 seconds. It is therefore not solely a Python implementation
  defect. The subsequent QE-style Z-stick backend reduces Python's local
  potential path from four `Alltoallv` transposes to two.
- Python RSS includes roughly hundreds of MiB of interpreter, SciPy, BLAS,
  MPI, and Numba runtime state that QE does not carry. The analytical array
  estimate is the relevant measure for physics-array optimization.
- Packing Broyden history to the 72,017 active charge G-vectors instead of the
  complete 216,000-point FFT grid reduces its saturated payload from
  approximately 62.21 MB to 20.74 MB per rank.

## SciPy overwrite FFT follow-up

Replacing NumPy transforms with overwrite-capable SciPy pocketfft reduced the
one-rank `vloc_psi` time from 36.26 to 26.07 seconds, `h_psi` from 38.72 to
28.37 seconds, and internal wall time from 83.86 to 73.05 seconds. Peak RSS
fell from 528.76 to 508.25 MiB. At three ranks, `vloc_psi` fell from 18.23 to
16.00 seconds and `h_psi` from 20.84 to 18.67 seconds. All runs retained the
same 14 iterations and `-31.1796883066 Ry` final energy.

The first two-rank follow-up was affected by severe WSL scheduling variation:
CPU time improved while wall time regressed. It is not used as evidence for
or against the FFT kernel change.

## QE-style Z-stick follow-up

Inspection of `FFTXlib/src/stick_base.f90`, `fft_ggen.f90`, and
`fft_parallel_2d.f90` confirms that QE indexes Z-sticks by `(Gx,Gy)`, performs
the one-dimensional transform along `Gz`, transposes sticks to Z-plane slabs,
and then applies the two-dimensional XY transform. The Python backend now
uses the same orientation.

Relative to the dense SciPy path, one-rank `vloc_psi` CPU time fell from
13.18 to 10.60 seconds and three-rank time fell from 8.11 to 5.16 seconds.
The three-rank FFT timer fell from 8.28 to 4.92 CPU seconds. One-rank peak RSS
fell from 508.25 to 417.99 MiB. Wall times during these follow-up runs were
dominated by unrelated WSL scheduling contention; the routine CPU counters
are the stable comparison. Energies and the 14-iteration trajectory were
unchanged.

## Strictly single-threaded MPI follow-up

The current Z-stick implementation was rerun with every numerical library
limited to one thread per MPI rank:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
MKL_DOMAIN_NUM_THREADS=1
BLIS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
NUMBA_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
OMP_DYNAMIC=FALSE
MKL_DYNAMIC=FALSE
```

`threadpoolctl` confirmed one OpenBLAS, one MKL, and one Intel OpenMP thread
on every rank. Without these environment settings, the WSL DFT environment
advertised eight threads in each of those pools even though
`py_fft_threads=1`.

| MPI ranks | Iterations | Final energy (Ry) | External wall (s) | `PWSCF` wall (s) | Peak reported RSS/rank (MiB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 14 | -31.1796883066 | 46.83 | 44.79 | 409 |
| 2 | 14 | -31.1796883066 | 38.47 | 36.96 | 386 |
| 3 | 14 | -31.1796883066 | 38.93 | 37.23 | 381 |
| 4 | 14 | -31.1796883066 | 47.72 | 45.79 | 382 |

The matching routine wall times were:

| MPI ranks | `c_bands` | `sum_band` | `h_psi` | `vloc_psi` | `fftw` |
|---:|---:|---:|---:|---:|---:|
| 1 | 18.92 | 17.75 | 15.23 | 13.56 | 12.65 |
| 2 | 13.16 | 16.21 | 10.24 | 8.43 | 7.96 |
| 3 | 10.57 | 17.84 | 8.08 | 6.65 | 6.15 |
| 4 | 12.71 | 22.09 | 9.44 | 7.62 | 7.49 |

The one-rank CPU and wall counters agree to within 0.3 seconds, as expected
for a genuinely single-threaded numerical run. The distributed `h_psi`
kernel scales by 1.49x at two ranks and 1.88x at three ranks, but
`sum_band` does not scale because all ranks still process all irreducible
k-points and occupied bands. It therefore becomes the limiting section.
Four ranks regress in both `h_psi` and `sum_band`.

An unrelated pre-existing four-rank Python calculation was active in WSL
during this follow-up. The absolute wall times, particularly for the
higher-rank cases, consequently include host contention and should not be
treated as isolated scaling measurements. The exact rank-independent energy,
iteration trajectory, and per-routine trend are nevertheless valid. Use
`benchmarks/run_single_thread_wsl.sh` on an otherwise idle WSL instance for a
clean wall-time repetition.

## Davidson accuracy, density comparison, and `sum_band`

The remaining default-run energy offset was traced with QE's saved
`charge-density.dat`. With the former loose implicit first Davidson threshold,
Python converged to `-31.1796883066 Ry`; the largest reciprocal-density
difference from QE was `6.47e-5`. Ewald agreed to printed precision, while
Hartree and XC differed.

With `diago_thr_init=1e-10 Ry`, Python converged to
`-31.1795722532 Ry`, compared with QE's `-31.17957226 Ry`. The energy
difference is approximately `6.8e-9 Ry`, below `conv_thr=1e-8`, and the
largest reciprocal-density difference falls to `6.55e-8`. The displaced
energy was therefore caused by the loose first Python Davidson solve feeding
a different Broyden trajectory, not by Ewald, symmetry, the FFT grid, or
`sum_band`.

The strict-threshold diagnostic used
`min(1e-2, 0.1*conv_thr/nelec) Ry`; the later requested production schedule
is documented below.

Profiling the strict-threshold run split `sum_band` into wavefunction FFT,
density accumulation, slab collection, and symmetry:

| MPI ranks | `sum_band` | wavefunction FFT | density accumulation | slab collection | symmetry |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.81 | 1.26 | 0.73 | 0.00 | 8.81 |
| 3 | 12.85 | 0.63 | 0.35 | 0.01 | 11.86 |

Thus the G-distributed density construction scales without k-point pools.
The replicated full-grid symmetry projection was the bottleneck. Replacing
the repeated reciprocal mappings, complex phases, and FFT pair with an
equivalent grid-space projection reduced the one-rank symmetry time from
8.81 to 4.21 seconds and `sum_band` from 10.81 to 6.20 seconds. Total
internal wall time fell from 65.54 to 59.70 seconds. A QE-style distributed
reciprocal-shell symmetry projection remained the next MPI optimization.

The following optimization distributes the independent space-group
operations over the existing plane-wave communicator and combines the
partial density grids with one `Allreduce`. It requires no k-point pools.
On three ranks, symmetry fell to 1.51 seconds and the complete `sum_band`
section to 2.32 seconds. The complete calculation took 32.33 seconds
internally and 34.98 seconds externally, retained nine SCF iterations, and
gave the identical `-31.1795722532 Ry` energy. Relative to the optimized
one-rank run, this is a 1.85x internal speedup.

## Direct QE/Python residual timing gap

The accuracy-aware first Davidson solve originally left its
`1.25e-10 Ry` threshold active for every SCF step. QE instead resets `ethr`
to `1e-2 Ry` at iteration two before applying `0.1*dr2/nelec`. Restoring
that reset reduced the three-rank Python `Hpsi` calls from 1,357 to 682 and
`c_bands` from 20.56 to 15.63 seconds. The energy remained
`-31.1795722524 Ry`, within `conv_thr` of QE, and the complete time fell
from 32.33 to 30.65 seconds.

The remaining direct three-rank comparison is:

| Section | QE wall (s) | Python wall (s) | Principal difference |
|---|---:|---:|---|
| initialization | 0.43 | 7.21 | Python UPF tables, potentials, and starting orbitals |
| `c_bands` | 5.47 | 15.63 | 682 versus 415 `Hpsi` calls, plus slower calls |
| `sum_band` | 0.89 | 4.65 | replicated full-grid density plus distributed operation symmetry |
| electrons | 6.59 | 23.43 | accumulated rows above |
| total | 7.07 | 30.65 | 4.34x |

For `Hpsi`, Python takes 12.06 seconds for 682 calls (17.7 ms/call);
QE takes 4.79 seconds for 415 calls (11.5 ms/call). Thus approximately a
1.64x work-count excess and 1.53x per-call cost combine. The strict first
Python solve is the largest work-count source: it averages 24.1 Davidson
iterations, while QE's loose first solve averages 2.0. Matching QE's energy
without that expensive first solve requires further correction of the
Python Davidson/starting-subspace behavior rather than looser final SCF
convergence.

## Cached UPF tables and requested QE threshold schedule

Projector and atomic-wavefunction interpolation tables are now cached per
pseudopotential and volume instead of being rebuilt for every k-point. On
three ranks this reduced initialization from 7.21 to 2.86 seconds:

| Initialization section | Before (s) | Cached (s) |
|---|---:|---:|
| `wfcinit` | 2.74 | 0.61 |
| `potinit` | 0.50 | 0.31 |
| `hinit0` | 2.35 | 0.57 |
| complete `init_run` | 7.21 | 2.86 |

The Davidson schedule now starts the first iteration at `1e-2 Ry`. From
iteration two onward it resets to `1e-2 Ry` and applies
`0.1*dr2/nelec`, as requested. This reduced `Hpsi` calls from 682 to 536,
`c_bands` from 15.63 to 11.53 seconds, and total internal time from 30.65
to 21.99 seconds. External time was 23.83 seconds.

The loose first solve again converged to `-31.1796883066 Ry`, rather than
QE's `-31.17957226 Ry`. Python's first residual does not satisfy QE's
same-iteration `dr2 < ethr*nelec` retry condition, so no retry occurs. This
confirms that the remaining energy problem lies in the Python first
Davidson/starting-density trajectory, not the later
`0.1*dr2/nelec` update.

## Symmetry audit and reusable mapping plan

The scalar real-grid projection was checked against QE's reciprocal-space
group action, including inverse rotations and nonsymmorphic translation
phases. A random-density regression agrees to approximately `3e-15`; the
diamond structure-factor regression and projection-idempotence tests also
pass. No rotation or phase-sign error was found. An explicit compatibility
check now rejects operations that do not map the selected FFT grid onto
integer grid points.

Previously every SCF iteration rebuilt three transformed-index arrays for
each local symmetry operation. The code now constructs one MPI-local flat
mapping per assigned operation during initialization and reuses a persistent
permutation workspace. On three ranks:

| Metric | Rebuilt mappings | Reusable plan |
|---|---:|---:|
| symmetry setup | included repeatedly | 0.40 s once |
| cumulative symmetry | 3.15 s | 0.79 s |
| `sum_band` | 4.76 s | 2.21 s |
| internal total | 21.99 s | 17.94 s |
| external total | 23.83 s | 19.70 s |

The local cached maps add about 13.8 MB per rank for the 60-cubed diamond-Si
grid and 48 operations divided over three ranks. Peak RSS increased from
about 425 MiB to 440 MiB. The final energy and 14-iteration trajectory were
unchanged under the requested loose-first-Davidson schedule.
