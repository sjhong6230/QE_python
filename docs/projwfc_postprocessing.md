# Projected DOS and Löwdin populations

`projwfc.py` implements scalar, LSDA, and noncollinear norm-conserving paths of
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

For `noncolin=.true., lspinorb=.true.`, the projection basis follows the UPF
`PP_RELWFC` channels and is labelled by `j,m_j`; symmetry averaging uses the
spin-j representation and includes Kramers partners when `domag=.false.`.
For `lspinorb=.false.`, j partners are averaged with QE's `average_pp` weights
and the basis transforms as `D^l tensor D^(1/2)` with explicit `s_z` labels.

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

`diag_basis=.true.` constructs the occupation density matrix independently
for every atom and radial `(n,l)` shell after Löwdin orthogonalization and
crystal-symmetry expansion. Its Hermitian eigenvectors define the local
orbital components used for charges and PDOS; their labels are consequently
component numbers rather than global `px`, `dxy`, and similar directions.

`lwrite_overlaps=.true.` writes the complete complex pre-orthogonalization
matrix `S_ij(k)=<phi_i(k)|phi_j(k)>` for every k point into
`atomic_proj.xml`. This is distinct from the identity overlap of the final
Löwdin basis and is useful for diagnosing nearly linearly dependent atomic
trial orbitals.

With `tdosinboxes=.true.`, atomic projection is replaced by real-space box
LDOS. `irmin(i,n)` and `irmax(i,n)` are inclusive, one-based FFT-grid bounds;
zero `irmax` means the last grid point and reversed bounds wrap periodically.
The result is written to `filpdos.ldos_boxes`. `kresolveddos` produces one
block per k point, `filproj` records every state’s integrated box weights,
and `plotboxes=.true.` writes `box#N.xsf` indicator grids.

Supported broadening modes are Gaussian, first-order Methfessel–Paxton,
Marzari–Vanderbilt cold smearing, and Fermi–Dirac. PAW, ultrasoft, and
projected tetrahedron integration are not yet ported. `diag_basis=.true.` is
also unavailable for relativistic `j,m_j` projectors, matching QE's current
noncollinear restriction.
For tetrahedron-generated NSCF data, specify `degauss` to request the supported
smearing path. Set `lsym=.false., kresolveddos=.true.` to retain unsymmetrized
per-k-point orbital projections.
