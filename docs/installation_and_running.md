# Installation and running

## 1. Requirements

The package requires Python 3.10 or newer and builds one mandatory Cython
extension. A working installation needs:

- a C compiler with OpenMP support;
- an MPI implementation and ABI-matched compiler wrapper;
- `mpi4py` built for that MPI implementation;
- FFTW double-precision and threaded libraries, from the system when possible;
- BLAS/LAPACK through Intel oneMKL, OpenBLAS/LAPACKE, or NumPy/SciPy's runtime;
- Python build dependencies declared in `pyproject.toml`.

FFTW-MPI is optional. If its header and library are found beside system FFTW,
the extension builds collective dense-FFT support. The production Hψ FFT path
does not require FFTW-MPI.

## 2. Ubuntu and WSL system packages

For Open MPI and system FFTW/FFTW-MPI:

```bash
sudo apt update
sudo apt install \
  build-essential \
  openmpi-bin libopenmpi-dev \
  libfftw3-dev libfftw3-mpi-dev \
  libopenblas-dev liblapacke-dev
```

The OpenBLAS/LAPACKE packages are optional when NumPy/SciPy already provide an
optimized numerical runtime. Avoid intentionally linking the slow reference
BLAS over an accelerated NumPy installation.

On WSL, keep the project in the Linux filesystem for the lowest metadata I/O
overhead when possible. A checkout under `/mnt/c` works, but builds and tests
that create many small files may be slower.

## 3. Python environment and pip installation

Create or activate an environment whose `mpi4py` matches the intended system
MPI. Then install:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

For development:

```bash
python -m pip install -e .
```

PEP 517 build isolation is supported. When debugging compiler or library
detection, an editable non-isolated build makes the selected environment more
obvious:

```bash
python -m pip install -e . --no-build-isolation -v
```

The build output reports the selected MPI vendor/compiler, FFTW provider,
FFTW-MPI availability, and accelerated BLAS provider.

## 4. Numerical-library selection

The build follows this order:

1. use the MPI vendor already loaded by `mpi4py`;
2. prefer coherent system `fftw3`, `fftw3_omp`, and optionally `fftw3_mpi`;
3. otherwise use pyFFTW's bundled FFTW libraries;
4. link system Intel oneMKL if found through `MKLROOT` or standard oneAPI
   locations;
5. otherwise link system OpenBLAS and LAPACKE when found;
6. otherwise resolve compatible NumPy/SciPy BLAS/LAPACK symbols at runtime.

Useful overrides are:

```bash
export MPICC=/path/to/mpicc
export FFTW_ROOT=/path/to/fftw
export MKLROOT=/opt/intel/oneapi/mkl/latest
python -m pip install -e . --no-build-isolation
```

`FFTW3_ROOT` is accepted as an alias for `FFTW_ROOT`.

### Intel MPI wrappers

If `mpi4py.MPI.get_vendor()` reports Intel MPI, the build searches for current
and classic wrappers in this order:

```text
mpiicx, mpiicc, mpiicpx, mpiicpc
```

The C++ wrappers are used in C mode only when no C wrapper is available.
`MPICC` or `MPI4PY_BUILD_MPICC` always takes precedence.

Do not combine an Open MPI `mpi4py` with Intel MPI's wrapper, or an MPICH
`mpi4py` with Open MPI FFTW-MPI. MPI ABI mismatches can fail at import, hang in
a collective, or corrupt communicator handles.

## 5. Confirming the build

Check the executable and Python package:

```bash
pw.py --help
python -c "import qepy_pw; print(qepy_pw.__version__)"
```

On Linux, inspect native linkage:

```bash
python -c "import qepy_pw._native_fft as m; print(m.fftw_mpi_available())"
ldd qepy_pw/_native_fft*.so
```

A system FFTW-MPI build should show `libfftw3_mpi`, `libfftw3_omp`, `libfftw3`,
and the same `libmpi` family used by `mpi4py`.

## 6. Serial execution

```bash
pw.py -in silicon.in
```

Standard input is also accepted according to the CLI parser, but an explicit
file is preferable because relative `pseudo_dir` resolution is then anchored
to the input location.

The process exit status is:

| Status | Meaning |
|---:|---|
| `0` | SCF converged and output completed. |
| `1` | Input, unsupported-feature, file, or wrapped runtime error. |
| `2` | SCF reached `electron_maxstep` without convergence. |

## 7. MPI execution

For four Open MPI ranks:

```bash
mpiexec -n 4 --bind-to core pw.py -in silicon.in
```

All ranks participate in each k-point's plane-wave and FFT work. There is no
k-pool option to configure.

For multi-node execution, use the scheduler's launcher and placement policy.
The MPI library used at runtime must match the one used to build `mpi4py` and
the extension.

## 8. Hybrid MPI and threads

The total physical-core request is

```text
physical cores = MPI ranks × threads per rank
```

`QEPY_NUM_THREADS` takes precedence over `OMP_NUM_THREADS`; the default is one
thread per rank. An Open MPI example using four ranks and two threads per rank
is:

```bash
export QEPY_NUM_THREADS=2
export OMP_NUM_THREADS=2
mpiexec -n 4 --map-by slot:PE=2 --bind-to core pw.py -in silicon.in
```

This requires eight physical cores. On Slurm, the resource request is commonly
expressed as:

```bash
srun --ntasks=4 --cpus-per-task=2 --cpu-bind=cores pw.py -in silicon.in
```

Exact Slurm/MPI flags are site-specific.

The launcher configures conservative per-rank defaults before importing NumPy:

- `MALLOC_ARENA_MAX=1`;
- `OMP_STACKSIZE=2M` and `GOMP_STACKSIZE=2M`;
- BLAS, MKL, BLIS, Accelerate, and NumExpr pools initially limited to one
  worker;
- Open MPI shared-memory OSC selected for node-local shared windows.

The SCF driver may allow more BLAS threads for selected serial large-basis
shapes. Multi-rank BLAS remains one-threaded to avoid nested teams and rank
oversubscription. FFT and native plane-wave loops use the requested hybrid
thread count only when their work threshold justifies it.

## 9. Choosing ranks and threads

At fixed core count, more MPI ranks generally improve plane-wave distribution
but replicate more Python/runtime memory. Fewer ranks with two or four threads
share the rank's main arrays but may lose performance when small FFTs,
collectives, or short band matrices dominate.

A controlled tuning sequence is:

1. benchmark `cores × 1 thread`;
2. benchmark half as many ranks with two threads;
3. optionally benchmark one quarter as many ranks with four threads;
4. keep input, physical cores, affinity, and background load fixed;
5. compare total wall time, `fftw`, `vloc_psi`, diagonalization time, and peak
   aggregate PSS.

Do not assume `1 rank × 4 threads` must equal `4 ranks × 1 thread`: the two
layouts use different MPI message counts, local FFT sizes, Python baselines,
BLAS shapes, and memory locality.

## 10. Persistent output

With the default `disk_io='low'`, the program writes:

```text
<outdir>/<prefix>.save/
```

including QE-shaped XML/HDF5 density and wavefunction data needed by the
implemented restart path. To suppress persistent output:

```text
&CONTROL
  disk_io = 'none'
/
```

As in QE, `outdir` and `wfcdir` are created when the calculation is initialized.
`wfcdir` defaults to `outdir`. With `disk_io='medium'`, a calculation having
more than one local k point uses the processor working file
`<wfcdir>/<prefix>.wfc[rank]`; `disk_io='high'` uses it even for one k point.
These direct-record binary files are distinct from the portable collected
HDF5 wavefunctions in `<prefix>.save`. `low` and `none` retain active
wavefunctions in memory, while `none` also suppresses the final save.

For restart:

```text
&CONTROL
  restart_mode = 'restart'
  outdir = './tmp'
  prefix = 'silicon'
/
&ELECTRONS
  startingpot = 'file'
  startingwfc = 'file'
/
```

The saved cell, cutoffs, density, k points, band layout, and wavefunctions must
be compatible with the new input.

## 11. Running tests

```bash
python -m pytest -q
```

To test collective FFT behavior explicitly:

```bash
mpiexec -n 4 python -m pytest -q tests/test_fft_parallel.py
```

To run the distributed FFT microbenchmark:

```bash
export QEPY_NUM_THREADS=1
export OMP_NUM_THREADS=1
mpiexec -n 4 python tools/benchmark_distributed_fft.py \
  --shape 72 --bands 8 --iterations 100
```

## 12. Troubleshooting

### Compiler wrapper not found

Confirm that the development package is installed and set `MPICC` explicitly:

```bash
which mpicc
mpicc --showme
export MPICC=$(command -v mpicc)
```

### `mpi4py` and launcher disagree

```bash
python -c "from mpi4py import MPI; print(MPI.get_vendor(), MPI.Get_library_version())"
mpiexec --version
```

Reinstall `mpi4py` and qepy-pw using the same MPI toolchain if these identify
different implementations.

### FFTW-MPI not detected

Verify both header and library:

```bash
test -f /usr/include/fftw3-mpi.h && echo header-ok
ldconfig -p | grep fftw3_mpi
```

Then rebuild without a cached wheel. FFTW-MPI absence is not fatal for normal
Hψ calculations.

### Poor wall time under WSL

Check for unrelated host/WSL calculations, CPU oversubscription, power-policy
throttling, and missing core binding. Compare CPU time with wall time: a large
wall/CPU discrepancy indicates scheduling or contention rather than an FFT
kernel regression.

### High memory with many ranks

Every rank is a Python process. Reduce ranks and try two threads per rank while
keeping total physical cores fixed. Compare aggregate PSS, not summed RSS,
because shared libraries and shared windows otherwise appear multiple times.
