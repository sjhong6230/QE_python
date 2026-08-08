# Projected DOS and Löwdin populations

`projwfc.py` implements the scalar, nonmagnetic, norm-conserving core of
Quantum ESPRESSO's `projwfc.x`. It reconstructs the pseudo-atomic orbitals
from each UPF `PP_CHI`, evaluates them in the saved plane-wave basis, and
forms the Hermitian overlap matrix

```text
S_ij(k) = <phi_i(k)|phi_j(k)>.
```

The projection basis is the symmetric Löwdin basis
`phi_tilde = phi S^(-1/2)`. Squared overlaps with the saved Kohn–Sham states
produce orbital weights, occupations produce Löwdin charges, and their
unrepresented fraction gives the spilling parameter.

With the default `lsym=.true.`, the complete complex orbital projection
density is transformed by every saved crystal symmetry operation. The code
maps symmetry-equivalent atoms and applies the real-spherical-harmonic
representation matrix for each angular momentum before averaging. Thus
directional components such as `px/py/pz` and the five d components obey the
point-group constraints even when only irreducible k points were calculated.
The transformation preserves the sum over atoms and m components.

```text
&PROJWFC
 prefix='silicon', outdir='./tmp',
 filpdos='silicon', filproj='silicon.proj',
 Emin=-10.0, Emax=15.0, DeltaE=0.01,
 degauss=0.01, ngauss=0,
 lwrite_overlaps=.true.
/
```

The program writes `atomic_proj.xml`, `filpdos.pdos_tot`, and one
`filpdos.pdos_atm#N(X)_wfc#M(l)` file for every radial atomic wavefunction.
The real-harmonic order follows QE: `pz, px, py` for p orbitals and
`dz2, dzx, dzy, dx2-y2, dxy` for d orbitals. `kresolveddos=.true.` prepends
the k-point index and uses unit k-point weights.

Supported broadening modes are Gaussian, first-order Methfessel–Paxton,
Marzari–Vanderbilt cold smearing, and Fermi–Dirac. PAW, ultrasoft,
spin-polarized/noncollinear projection, local occupation-axis rotation,
real-space boxes, and projected tetrahedron integration are not yet ported.
For tetrahedron-generated NSCF data, specify `degauss` to request the supported
smearing path. Set `lsym=.false., kresolveddos=.true.` to retain unsymmetrized
per-k-point orbital projections.
