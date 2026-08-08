# Installation and hybrid-runtime notes

## pip installation

The package builds a mandatory Cython extension linked against MPI, OpenMP,
and FFTW.  The build prefers a coherent system numerical stack in this order:

1. the MPI implementation already used by `mpi4py` and its C wrapper;
2. system `fftw3`/`fftw3_omp` and, when present, `fftw3_mpi`;
3. explicitly installed Intel oneMKL or system OpenBLAS/LAPACKE;
4. pyFFTW's bundled FFTW and NumPy's BLAS/LAPACK as portable fallbacks.

The extension never deliberately preloads pyFFTW's FFTW into a system-FFTW
build, because mixing two FFTW copies or two MPI ABIs in one rank is unsafe.
Reference Netlib BLAS/LAPACK is not selected over NumPy's optimized BLAS.

```bash
python -m pip install --upgrade pip
python -m pip install .
```

Use `python -m pip install -e .` for an editable source checkout. Build-time
Python dependencies are declared in `pyproject.toml`; the MPI implementation
and compiler toolchain are system dependencies.

Set `MPICC` (or `MPI4PY_BUILD_MPICC`) to override compiler detection.  For
Intel MPI, the build recognizes the current C wrapper `mpiicx` and the classic
`mpiicc`; if only C++ wrappers exist it can use `mpiicpx` or `mpiicpc` in C
mode.  Automatic selection checks `mpi4py.MPI.get_vendor()` first, so an Intel
wrapper is not accidentally mixed with an Open MPI or MPICH `mpi4py` build.

On Debian/Ubuntu, an Open-MPI-compatible FFTW-MPI development installation is:

```bash
sudo apt install libopenmpi-dev libfftw3-dev libfftw3-mpi-dev
```

## MPI plus OpenMP execution

The hybrid layout is described by:

```text
total cores = MPI ranks * OpenMP threads per rank
```

Set the thread count before starting MPI:

```bash
export QEPY_NUM_THREADS=2
export OMP_NUM_THREADS=2
mpiexec -n 4 --map-by slot:PE=2 --bind-to core pw.py -in <input.in>
```

`QEPY_NUM_THREADS` takes precedence over `OMP_NUM_THREADS`. OpenMP and FFTW
workers do not call MPI; MPI is initialized with `MPI_THREAD_FUNNELED`.
Multi-rank BLAS/LAPACK execution is restricted to one thread per rank to avoid
nested thread teams, while the native FFT and plane-wave kernels retain the
requested rank-local thread count.

Open MPI uses `--map-by ...:PE=N` to assign `N` processing elements to each
rank. For Slurm, the corresponding resource request normally uses
`--ntasks=<ranks>` and `--cpus-per-task=<threads>`, followed by the site's
recommended `srun` binding options.

## Periodic Pulay research control

`mixing_pulay_frequency` in `&electrons` selects how often Pulay/Broyden
extrapolation is applied. Its default value, `1`, is the conventional QE
trajectory. A value `n > 1` stores history on every iteration, applies linear
mixing between extrapolations, and performs Pulay extrapolation every `n`-th
iteration. This is deliberately not selected automatically: the Si benchmarks
used during implementation improved at 40 Ry with `n = 3` but regressed at
200 Ry, so the robust default remains `1`.

## Memory model

Increasing MPI ranks has two opposing effects:

- distributed reciprocal-space and FFT data are divided among more ranks;
- Python runtime state, rank-local solver state, communication buffers, and
  other private allocations are replicated for every rank.

Increasing OpenMP threads keeps the principal rank arrays shared, which often
reduces memory at a fixed total core count. However, thread stacks, OpenMP
reductions, FFTW plans, and temporary thread-local buffers add a smaller
thread-dependent cost.

Consequently, `R` ranks with `T` threads usually use less memory than `R*T`
single-threaded ranks, but they may not be faster. Small band matrices and
communication-heavy FFT grids can favor more MPI ranks; replicated Python
overhead and memory pressure can favor fewer ranks with two to four threads.
NUMA placement can change either result.

Use the program's estimated per-rank and aggregate memory values before the
SCF loop and its measured aggregate PSS after the calculation. PSS apportions
shared pages among processes and is more informative than summing RSS for a
multi-rank Python job.

For a fixed node allocation:

1. establish a one-thread-per-rank baseline;
2. halve the ranks and use two threads per rank;
3. optionally repeat with four threads per rank;
4. keep the total core count fixed and compare both wall time and peak PSS.

Avoid oversubscription. It increases thread stacks and runtime overhead and
usually makes both performance and memory behavior less predictable.
