# qepy-pw

`qepy-pw` is a Python implementation of the scalar, nonmagnetic SCF path of
Quantum ESPRESSO `pw.x`. It reads QE-style input files and norm-conserving UPF
pseudopotentials and produces QE-shaped text output.

The project is intended for development and numerical comparison. It is not a
drop-in replacement for production Quantum ESPRESSO.

## Installation

Python 3.10 or newer is required. The native extension also requires an MPI
implementation with an `mpicc` compiler wrapper and OpenMP support. On Ubuntu
or WSL, install the system toolchain first:

```bash
sudo apt update
sudo apt install build-essential openmpi-bin libopenmpi-dev
```

Create or activate a Python environment, then install with pip:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

For development, use an editable installation:

```bash
python -m pip install -e .
```

The build installs the `pw.py` command and compiles the Cython/MPI/FFTW
extension. Run one calculation with:

```bash
pw.py -in <input.in>
```

## MPI and OpenMP

`QEPY_NUM_THREADS` sets the OpenMP/FFTW thread count for each MPI rank. If it
is not set, `OMP_NUM_THREADS` is used; the default is one thread per rank.

For Open MPI, a four-rank, two-thread-per-rank run is:

```bash
export QEPY_NUM_THREADS=2
export OMP_NUM_THREADS=2
mpiexec -n 4 --map-by slot:PE=2 --bind-to core pw.py -in <input.in>
```

This run requires eight CPU cores. Do not launch more ranks times threads than
the allocated core count. With a batch scheduler, request one task per MPI
rank and `QEPY_NUM_THREADS` CPUs per task. Binding options differ between MPI
implementations and schedulers.

## MPI, OpenMP, and memory

Each MPI rank is a separate Python process. Rank-private interpreter state,
solver workspaces, FFT communication buffers, and some metadata are therefore
replicated. Distributed plane-wave and FFT arrays become smaller as the number
of ranks increases, but the replicated process overhead does not.

OpenMP threads share the rank's main arrays. For a fixed total core count,
using fewer MPI ranks and more threads per rank will usually consume less
memory than using one MPI rank per core. Thread stacks, FFTW workspaces, and
some thread-local scratch still grow with the thread count, so threading is
not memory-free.

A useful first estimate for node memory is:

```text
node memory ~= ranks per node * private memory per rank
              + shared data
              + thread-local scratch
```

Start with one thread per rank for the simplest performance behavior. If
rank-replicated memory is limiting, try two to four threads per rank while
reducing the rank count so that `ranks * threads` remains equal to the
allocated cores. Compare both elapsed time and the memory report printed by
the program; the best balance depends on the FFT grid, number of bands, and
node NUMA layout.

## Tests

The test suite contains only supported cases selected from the official
Quantum ESPRESSO test suite. Run it with:

```bash
python -m pytest -q
```

The official QE inputs and pseudopotentials, together with Python-generated
output references, are stored in `tests/qe_reference`.

See [PORTING.md](PORTING.md) for the compact runtime and memory notes.
