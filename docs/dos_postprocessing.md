# DOS post-processing

`dos.py` implements scalar, LSDA, and noncollinear total-density-of-states paths of
Quantum ESPRESSO's `dos.x`. It reads eigenvalues and k-point metadata from
`<outdir>/<prefix>.save/data-file-schema.xml` and accepts the QE `&DOS`
namelist variables `prefix`, `outdir`, `bz_sum`, `ngauss`, `degauss`, `Emin`,
`Emax`, `DeltaE`, and `fildos`.

```text
&DOS
 prefix='silicon', outdir='./tmp',
 bz_sum='tetrahedra_opt',
 Emin=-10.0, Emax=15.0, DeltaE=0.01,
 fildos='silicon.dos'
/
```

Run it as `dos.py -i dos.in`. The output columns are energy in eV, DOS in
states/eV, and integrated states. Scalar calculations include twofold spin
degeneracy, LSDA writes separate up/down channels, and each noncollinear
spinor band has unit capacity. `bz_sum='smearing'` supports Gaussian (`ngauss=0`), first-order
Methfessel-Paxton (`1`), Marzari-Vanderbilt cold smearing (`-1`), and
Fermi-Dirac (`-99`). `tetrahedra`, `tetrahedra_lin`, and `tetrahedra_opt`
require an automatic k-point mesh saved by `pw.py`; an explicitly supplied
`degauss` selects smearing, following QE.

Projected states belong to the separate QE `projwfc.x` workflow.
