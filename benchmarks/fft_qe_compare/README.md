# QE/port FFT benchmark

This case deliberately combines a relatively large 160 Ry wavefunction grid,
32 bands, and Gamma sampling. Run both programs with the same MPI/thread
layout and numerical libraries. For example:

```bash
export OMP_NUM_THREADS=1 QEPY_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mpiexec -n 4 /path/to/qe-7.5/bin/pw.x -in Si.gamma.in > qe.out
mpiexec -n 4 pw.py -in Si.gamma.in > qepy.out
```

Compare total wall time and the timing rows headed by `cft3s`/`fft_scatter`
in QE and `fftw` in qepy-pw. These counters are not identical: QE's FFT
timers split local transforms and MPI scatter, whereas qepy-pw's `fftw` timer
encloses the fused sparse pack, transforms, collectives, potential multiply,
and gather. The input is therefore an end-to-end FFT-dominated comparison,
not a claim that the individual timer labels measure the same code region.

`Si.kpoint.in` is the matching non-Gamma case: it uses the same cell,
cutoffs, bands, MPI layout, and pseudopotential, but evaluates the single
crystal point `(0.125, 0.125, 0.125)` with symmetry disabled. For a direct
Gamma calculations use the half-G implementation by default. Set
`QEPY_GAMMA_MODE=half` only when an explicit assertion of that path is useful
for a benchmark script.
