# Scalar field post-processing

`pp.py` implements the scalar, nonmagnetic, norm-conserving portion of Quantum
ESPRESSO `pp.x`. It reads `<outdir>/<prefix>.save`, extracts a quantity onto
the dense FFT grid, writes QE's formatted `filplot` intermediate file, and can
then combine and render one or more intermediate files through `&PLOT`.

```text
&INPUTPP
 prefix='silicon', outdir='./tmp',
 plot_num=0, filplot='silicon.rho'
/
&PLOT
 nfile=1, filepp(1)='silicon.rho', weight(1)=1.0,
 iflag=3, output_format=5, fileout='silicon.xsf'
/
```

Run the workflow with `pp.py -in pp.in`. As with the other executables, `-i`,
`-inp`, `-input`, and `--input` are accepted aliases. Either stage can also be
used alone: set `plot_num=-1` to render existing intermediate files, or omit
`&PLOT` to perform extraction only.

## Extracted quantities

The following `plot_num` values are implemented:

| Value | Quantity |
|---:|---|
| 0 | total valence charge density |
| 1 | local ionic + Hartree + exchange-correlation potential, in Ry |
| 2 | local ionic potential, in Ry |
| 3 | energy-resolved local density of states |
| 4 | electronic entropy density from QE's `w1gauss` kernel |
| 5 | STM-like integrated density in the bias window |
| 7 | selected Kohn-Sham orbital density, optionally signed at Gamma |
| 8 | electron localization function (ELF) |
| 9 | self-consistent density minus the superposition of atomic densities |
| 10 | integrated local density of states from `emin` to `emax` |
| 11 | local ionic + Hartree potential, in Ry |
| 19 | reduced density gradient |
| 20 | `sign(lambda_2) rho`, using the middle density-Hessian eigenvalue |
| 22 | positive kinetic-energy density |
| 23 | band-window charge density |
| 123 | density-overlap regions indicator (DORI) |

LDOS energies, `emin`, `emax`, `delta_e`, and `degauss_ldos` are expressed in
eV, following QE. `sample_bias` is entered in Ry. Wavefunction-derived fields
require the collected `wfcN.hdf5` files produced by `disk_io!='none'`.

The implementation reads both qepy-pw's native complex HDF5 datasets and QE's
interleaved real/imaginary HDF5 layout. Reciprocal-space ionic and atomic
fields are restricted to the saved charge-density cutoff. For the tested
norm-conserving Si reference, QE 7.5 and `pp.py` give identical FFT-grid arrays
for charge density, total potential, local ionic potential, and a selected
orbital density after parsing their formatted intermediate files.

## Plotting and file formats

`&PLOT` supports weighted `filepp(i)` combinations and Fourier interpolation.
The implemented combinations are:

- `iflag=0` or `1`: spherical average or line plot;
- `iflag=2`: two-dimensional formats 0 (gnuplot matrix), 2 (`plotrho.x`),
  3 (XSF), and 7 (`x y f(x,y)`);
- `iflag=3`: format 3 interpolated XSF, format 5 FFT-grid XSF, and format 6
  Gaussian cube;
- `iflag=4`: polar values on a sphere in format 0.

XSF and cube coordinates and field ordering follow QE 7.5. The formatted
scientific notation uses Fortran's unscaled `Ew.d` convention where QE does,
including the `0.xxxxxE+yy` mantissa form.

## Explicit limitations

Spin density and magnetization (`plot_num=6`, 13, and 18), PAW or ultrasoft
all-electron reconstruction (17, 21, and 24), electric-field and polarization
paths (12 and 14--16), and DFT+U projectors (25) are rejected explicitly. The
same applies to spin-polarized, noncollinear, spinor, ultrasoft, and PAW save
directories. B-spline interpolation and the optional constant-current STM
surface transform are not implemented; Fourier interpolation and ordinary STM
density extraction remain available.
