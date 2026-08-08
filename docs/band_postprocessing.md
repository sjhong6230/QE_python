# Band post-processing

The post-processing programs live under `qepy_pw/pp`, separately from the
`pw.x` driver under `qepy_pw/pw`. Shared numerical, symmetry, MPI, and UPF
modules remain at package root.

## `bands.py`

`bands.py` reads `&BANDS` and the `<outdir>/<prefix>.save` directory produced
by a preceding `calculation='bands'` run. It writes the QE `filband` format,
`filband.gnu`, and, with the default `lsym=.true.`, `filband.rap`.

```text
&BANDS
  prefix = 'silicon'
  outdir = './tmp'
  filband = 'silicon.bands'
  lsym = .true.
/
```

For scalar wavefunctions, irrep classes are obtained from numerical
little-group representation matrices in degenerate eigenspaces. The positive
integer classes are written using QE's `&plot_rap` format. With
`lsym=.false., no_overlap=.false.`, adjacent bands are instead connected by a
maximum-weight one-to-one assignment of wavefunction overlaps.

With `lp=.true.`, `bands.py` also writes QE's `&p_mat`/`filp` format. For the
implemented scalar norm-conserving Hamiltonian, this is the squared
conduction--valence matrix element of the velocity times electron mass,

```text
m v = p + i [V_nl, r].
```

The plane-wave `k+G` contribution and both analytic q derivatives of the
separable nonlocal projectors are included. Projector contractions are kept
in low-rank form rather than materializing a dense plane-wave operator.

With `plot_2d=.true.`, the saved k points must be ordered as a rectangular
grid `k(i,j) = k0 + i*dkx + j*dky`, with the second index varying fastest.
One `filband.#` file per band is written with `kx_distance`, `ky_distance`,
and energy in eV, matching QE's `punch_band_2d` convention. As in QE, this
mode bypasses symmetry classification, overlap ordering, and `lp` output.

The noncollinear `lsigma` and LSDA `spin_component` paths remain explicitly
unsupported.

## `plotband.py`

The input retains QE's interactive six-line format:

```text
silicon.bands
-10.0 10.0
silicon.plot
silicon.ps
0.0
1.0 0.0
```

The program writes a blank-line-separated gnuplot/xmgr data file and a
dependency-free PostScript plot. Energies in the data file are shifted by the
specified Fermi energy.
