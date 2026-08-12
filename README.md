_# qepy-pw

`qepy-pw` is a Python implementation of the scalar and collinear-LSDA SCF,
NSCF, and band-structure paths of Quantum ESPRESSO `pw.x`, together with the
`bands.x`, `plotband.x`, `dos.x`, scalar `projwfc.x`, and scalar norm-conserving
`pp.x` post-processing workflows. It reads QE-style input
files and norm-conserving UPF pseudopotentials and produces QE-shaped output.

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

The build installs `pw.py`, `bands.py`, `plotband.py`, `dos.py`, `projwfc.py`,
and `pp.py` and compiles the
Cython/MPI/FFTW extension. Run one calculation with:

```bash
pw.py -in <input.in>
```

For a band calculation saved with `disk_io='medium'`, post-process and plot it
with the usual QE-shaped inputs:

```bash
bands.py -i bands.pp.in
plotband.py -i plotband.in
```

An automatic-grid NSCF calculation can be integrated into a total DOS with:

```bash
dos.py -i dos.in
```

For symmetry-averaged Löwdin charges and orbital-projected DOS from saved wavefunctions:

```bash
projwfc.py -i projwfc.in
```

To extract a real-space quantity and render it in XSF or cube form, use a
QE-style `&INPUTPP` plus `&PLOT` input:

```bash
pp.py -in pp.in
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

`QEPY_FFT_MEMORY_LIMIT_MIB` sets a hard per-rank budget for distributed
wavefunction FFT execution scratch. The runtime estimates the selected slab
or pencil layouts and transpose buffers, selects an admissible band tile, and
takes the minimum choice across the communicator so MPI counts agree. The
limit does not include eigensolver, projector, density, or interpreter memory.

```bash
export QEPY_FFT_MEMORY_LIMIT_MIB=256
```

`QEPY_FFT_DECOMPOSITION` accepts `auto` (the default), `slab`, or `pencil`.
`auto` retains the lower-communication sparse Z-stick slab while the rank
count fits the FFT Z dimension, and selects a two-dimensional pencil grid
when a slab can no longer assign a nonempty Z plane to every rank. `pencil`
can be forced for scaling studies. It keeps the SCF density, XC, symmetry,
and mixing interfaces in their conventional Z-slab layout, but holds the
effective potential in X pencils throughout each k-point loop.

```bash
export QEPY_FFT_DECOMPOSITION=pencil
```

The pencil execution engine also supports independent FFT task-group
communicators and band slices. Production SCF currently uses one task group;
multi-group execution is benchmarked separately because duplicating each
group's spatial grid changes both solver ownership and the memory model.

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

## Documentation

The documentation index is [docs/README.md](docs/README.md). It links the
input reference, implemented equations, differences from QE, installation and
execution guide, validation/performance methodology, architecture, and
QE-compatible diagnostics.

See [PORTING.md](PORTING.md) for the compact runtime and memory notes.
Release history is recorded in [CHANGELOG.md](CHANGELOG.md).

## Generative AI

Generative AI was used for the production of the code.
