# Equations and numerical methods

This document summarizes the equations represented by the current scalar SCF
implementation. Internal electronic-structure quantities use Hartree atomic
units unless `Ry` is written explicitly.

## 1. Notation and Fourier convention

Let the primitive-cell volume be

$$
\Omega = |\det(\mathbf a_1,\mathbf a_2,\mathbf a_3)|,
$$

with reciprocal vectors satisfying

$$
\mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij}.
$$

The code stores lattice vectors as rows. A fractional row vector
$\mathbf s$ maps to $\mathbf r=\mathbf s A$, and an integer reciprocal index
$\mathbf m$ maps to $\mathbf G=\mathbf m B$.

For a cell-periodic scalar field, the convention is

$$
f(\mathbf r)=\sum_{\mathbf G}f_{\mathbf G}e^{i\mathbf G\cdot\mathbf r},
\qquad
f_{\mathbf G}=\frac{1}{\Omega}\int_\Omega
f(\mathbf r)e^{-i\mathbf G\cdot\mathbf r}\,d\mathbf r.
$$

The discrete FFT follows the same sign structure, with explicit division by
the total grid size where required. FFTW itself is unnormalized.

## 2. Bloch orbitals and plane-wave cutoff

A scalar Kohn–Sham Bloch orbital is expanded as

$$
\psi_{n\mathbf k}(\mathbf r)
=\frac{1}{\sqrt\Omega}
\sum_{\mathbf G\in\mathcal G_{\mathbf k}}
c_{n\mathbf k}(\mathbf G)
e^{i(\mathbf k+\mathbf G)\cdot\mathbf r}.
$$

The wavefunction basis is selected by

$$
\frac{1}{2}|\mathbf k+\mathbf G|^2
\le E_{\mathrm{cut}}^{\mathrm{wfc}},
$$

where the input `ecutwfc` is in Rydberg and is converted internally to
Hartree. The charge-density reciprocal set is similarly bounded by
$E_{\mathrm{cut}}^{\rho}$ from `ecutrho`.

The orthonormality condition is

$$
\langle\psi_{m\mathbf k}|\psi_{n\mathbf k}\rangle
=\sum_{\mathbf G}c^*_{m\mathbf k}(\mathbf G)
c_{n\mathbf k}(\mathbf G)=\delta_{mn}.
$$

Under plane-wave MPI decomposition, each rank holds a disjoint subset of the
$\mathbf G$ rows. Scalar products therefore require an MPI sum over ranks.

## 3. Scalar Kohn–Sham problem

The implemented generalized eigenproblem reduces to an ordinary Hermitian
problem because norm-conserving pseudopotentials have the identity overlap
operator:

$$
\hat H_{\mathrm{KS}}[n]\psi_{n\mathbf k}
=\varepsilon_{n\mathbf k}\psi_{n\mathbf k}.
$$

The Hamiltonian is

$$
\hat H_{\mathrm{KS}}
=-\frac{1}{2}\nabla^2
+\hat V_{\mathrm{loc}}^{\mathrm{ion}}
+\hat V_{\mathrm{NL}}
+V_{\mathrm H}[n]
+V_{\mathrm{xc}}[n+n_c],
$$

where $n_c$ is the nonlinear core-correction density when present in the UPF.
Ultrasoft augmentation and PAW overlap terms are not implemented.

### 3.1 Kinetic operator

The kinetic operator is diagonal in the plane-wave basis:

$$
T_{\mathbf G\mathbf G'}(\mathbf k)
=\frac{1}{2}|\mathbf k+\mathbf G|^2
\delta_{\mathbf G\mathbf G'}.
$$

The diagonal is fused with the local-potential result when possible, avoiding
a second full wavefunction temporary.

### 3.2 Local ionic and effective potential

For atoms $I$ at positions $\mathbf R_I$, the ionic local potential is

$$
V_{\mathrm{loc}}^{\mathrm{ion}}(\mathbf G)
=\sum_I e^{-i\mathbf G\cdot\mathbf R_I}
v_{\mathrm{loc}}^{s(I)}(G),
$$

with the radial transform read or constructed from the UPF. The effective
local potential is

$$
V_{\mathrm{eff}}(\mathbf r)
=V_{\mathrm{loc}}^{\mathrm{ion}}(\mathbf r)
+V_{\mathrm H}(\mathbf r)+V_{\mathrm{xc}}(\mathbf r).
$$

Its action is evaluated pseudospectrally:

$$
\{c_{n\mathbf k}(\mathbf G)\}
\xrightarrow{\mathrm{inverse\ FFT}}
\psi_{n\mathbf k}(\mathbf r)
\xrightarrow{\times V_{\mathrm{eff}}(\mathbf r)}
V_{\mathrm{eff}}\psi_{n\mathbf k}
\xrightarrow{\mathrm{forward\ FFT}}
\{(V_{\mathrm{eff}}\psi)_{\mathbf G}\}.
$$

This avoids materializing the dense convolution matrix
$V_{\mathbf G-\mathbf G'}$.

### 3.3 Kleinman–Bylander nonlocal pseudopotential

The separable norm-conserving nonlocal operator has the form

$$
\hat V_{\mathrm{NL}}
=\sum_I\sum_{ij}
|\beta_i^I\rangle D_{ij}^I\langle\beta_j^I|.
$$

For a block of orbitals $C$, application is factorized as

$$
V_{\mathrm{NL}}C
=B D(B^\dagger C),
$$

where $B$ contains atom-centered projector columns, including the phase
$e^{-i(\mathbf k+\mathbf G)\cdot\mathbf R_I}$. Projector overlaps are formed
with BLAS matrix products and MPI reductions. Dense nonlocal matrices are
formed only by the explicitly requested diagnostic dense solver.

## 4. Density and electron count

With scalar spin degeneracy included in $f_{n\mathbf k}$, the valence density is

$$
n(\mathbf r)=
\sum_{\mathbf k}w_{\mathbf k}
\sum_n f_{n\mathbf k}
|\psi_{n\mathbf k}(\mathbf r)|^2,
$$

where $\sum_{\mathbf k}w_{\mathbf k}=1$. For occupied scalar states,
$f_{n\mathbf k}\to2$. The electron-number constraint is

$$
\int_\Omega n(\mathbf r)\,d\mathbf r=N_e,
\qquad
N_e=\sum_I Z_I-\mathrm{tot\_charge}.
$$

Density accumulation transforms one bounded orbital batch at a time. This is
an important memory constraint: the code does not retain a full real-space
grid for every band.

## 5. Hartree electrostatics

For $\mathbf G\ne0$,

$$
V_{\mathrm H}(\mathbf G)
=\frac{4\pi}{G^2}n(\mathbf G),
$$

and the periodic $\mathbf G=0$ component is set by the chosen neutralizing
background convention. The Hartree energy is

$$
E_{\mathrm H}
=\frac{1}{2}\int_\Omega n(\mathbf r)V_{\mathrm H}(\mathbf r)\,d\mathbf r
=2\pi\Omega\sum_{\mathbf G\ne0}
\frac{|n(\mathbf G)|^2}{G^2}.
$$

The implementation forms Hartree coefficients directly on the compact
charge-density reciprocal basis.

## 6. Exchange and correlation

### 6.1 LDA

For an unpolarized local-density approximation,

$$
E_{\mathrm{xc}}^{\mathrm{LDA}}[n]
=\int_\Omega n(\mathbf r)\,
\varepsilon_{\mathrm{xc}}(n(\mathbf r))\,d\mathbf r,
$$

and

$$
V_{\mathrm{xc}}(n)
=\frac{d}{dn}\left[n\varepsilon_{\mathrm{xc}}(n)\right]
=\varepsilon_{\mathrm{xc}}(n)
+n\frac{d\varepsilon_{\mathrm{xc}}}{dn}.
$$

Slater exchange is combined with either the Perdew–Zunger 1981
parameterization of Ceperley–Alder data (`PZ`, `PZ81`) or the
Perdew–Wang 1992 correlation parameterization (`PW`, `PW92`). Density floors
and the high-/low-density analytic branches follow the implemented QE scalar
path.

### 6.2 Semilocal GGA

The GGA energy is represented as

$$
E_{\mathrm{xc}}^{\mathrm{GGA}}[n]
=\int_\Omega F(n,\sigma)\,d\mathbf r,
\qquad
\sigma=|\nabla n|^2.
$$

Functional differentiation gives

$$
V_{\mathrm{xc}}(\mathbf r)
=\frac{\partial F}{\partial n}
-2\nabla\cdot\left(
\frac{\partial F}{\partial\sigma}\nabla n
\right).
$$

Gradients and divergences are evaluated spectrally:

$$
\nabla n(\mathbf r)
=\sum_{\mathbf G}i\mathbf G,n_{\mathbf G}
e^{i\mathbf G\cdot\mathbf r}.
$$

The supported GGA exchange variants are PBE, PBEsol, revPBE, and RPBE, with
PW92 LDA correlation and the corresponding PBE-family gradient correction.
The physical formula remains in Python; large blockwise array traversals use
NumPy and selected native kernels to control temporaries.

When a nonlinear core correction exists, the XC functional is evaluated at
$n+n_c$, but the valence-potential energy term is contracted with $n$ in the
QE total-energy decomposition.

## 7. Occupations

### 7.0 Collinear LSDA representation

For `nspin=2`, the fundamental real-space variables are the two collinear
spin densities $n_\uparrow(\mathbf r)$ and $n_\downarrow(\mathbf r)$. The
Hartree problem depends only on

$$
n=n_\uparrow+n_\downarrow,
$$

whereas the local spin-density exchange-correlation functional is
$E_{\mathrm{xc}}[n_\uparrow,n_\downarrow]$ and produces two potentials
$V_{\mathrm{xc}}^\sigma=\delta E_{\mathrm{xc}}/\delta n_\sigma$. The mixer
uses the equivalent charge/magnetization basis

$$
n=n_\uparrow+n_\downarrow,
\qquad m=n_\uparrow-n_\downarrow.
$$

The Coulomb-preconditioned Broyden history acts on $n$; the short-ranged
magnetic channel $m$, including its $G=0$ component, is mixed without the
$1/G^2$ singular metric. Crystal symmetry operations act separately on
$n_\uparrow$ and $n_\downarrow$. This is the collinear analogue of QE's
`sym_rho` path and does not introduce a spin-flip operation.

QE represents spin-resolved Bloch problems by duplicating the spatial
k-point list. For $N_k$ spatial points, indices $1,\ldots,N_k$ carry spin up
and $N_k+1,\ldots,2N_k$ carry spin down. The atomic starting orbitals are
scalar and common to both blocks; `starting_magnetization` first polarizes the
atomic starting density and hence the two initial effective Hamiltonians,
which independently rotate the two orbital trial subspaces.

For a PBE-family GGA, exchange obeys the exact spin-scaling construction

$$
E_x[n_\uparrow,n_\downarrow]
=\frac12 E_x[2n_\uparrow]+\frac12 E_x[2n_\downarrow].
$$

The PBE correlation correction is evaluated from $n$, $\zeta=m/n$, and
$|\nabla n|^2$. Consequently its gradient derivative contributes the same
total-density flux to both spin potentials, while the exchange flux remains
spin diagonal. In compact form,

$$
\mathbf q_\sigma
=c_{x\sigma}\nabla n_\sigma+c_c\nabla n,
\qquad
V_{\mathrm{xc}}^\sigma
=V_{\mathrm{xc,local}}^\sigma-\nabla\cdot\mathbf q_\sigma.
$$

### 7.1 Fixed occupations

For a scalar nonmagnetic insulator, states are filled in energy order with a
maximum occupation of two. Degenerate frontier states are averaged within the
implemented tolerance.

For fixed LSDA occupations, each spin band has maximum occupation one. An
explicit total magnetization $M$ fixes

$$
N_\uparrow=\frac{N_e+M}{2},\qquad
N_\downarrow=\frac{N_e-M}{2}.
$$

### 7.2 Smearing

For a broadening width $\sigma$ (`degauss`, converted from Ry to Ha),

$$
f_{n\mathbf k}=2W\!\left(
\frac{\varepsilon_F-\varepsilon_{n\mathbf k}}{\sigma}
\right),
$$

and the Fermi energy solves

$$
2\sum_{\mathbf k n}w_{\mathbf k}
W\!\left(
\frac{\varepsilon_F-\varepsilon_{n\mathbf k}}{\sigma}
\right)=N_e.
$$

For Fermi–Dirac smearing,

$$
W(x)=\frac{1}{1+e^{-x}}.
$$

For Gaussian and Methfessel–Paxton smearing, $W$ is constructed from the
complementary error function and Hermite-polynomial corrections. The
Marzari–Vanderbilt cold-smearing expression uses the shifted Gaussian form.
The code also evaluates QE's variational entropy/free-energy correction
`demet`.

### 7.3 Tetrahedra

Linear and optimized tetrahedron modes partition each uniform reciprocal mesh
cell into tetrahedra, sort the four vertex energies, and integrate the
piecewise polynomial occupied volume up to $\varepsilon_F$. The optimized mode
uses the corresponding improved tetrahedral weights. This path requires a
uniform automatic k-point grid.

## 8. Total energy

The reported Kohn–Sham total energy can be written as

$$
E_{\mathrm{tot}}
=\sum_{n\mathbf k}w_{\mathbf k}f_{n\mathbf k}
\varepsilon_{n\mathbf k}
-E_{\mathrm H}[n]
-\int_\Omega n(\mathbf r)V_{\mathrm{xc}}(\mathbf r)\,d\mathbf r
+E_{\mathrm{xc}}[n+n_c]
+E_{\mathrm{II}}
+E_{\mathrm{smear}},
$$

with the usual rearrangement into one-electron, Hartree, XC, Ewald, and SCF
correction terms in the printed QE-style decomposition. The nonlocal energy is
already included in the band sum through the Hamiltonian eigenvalues.

## 9. Ewald ion–ion energy

For ionic charges $Z_I$ in a periodic cell, the Ewald decomposition is

$$
E_{\mathrm{II}}
=\frac{1}{2}\sum_{I,J}\sum_{\mathbf R}'
\frac{Z_IZ_J\operatorname{erfc}(\sqrt\eta
|\mathbf R+\mathbf R_I-\mathbf R_J|)}
{|\mathbf R+\mathbf R_I-\mathbf R_J|}
$$

$$
\quad+
\frac{2\pi}{\Omega}
\sum_{\mathbf G\ne0}
\frac{e^{-G^2/(4\eta)}}{G^2}
\left|\sum_I Z_Ie^{-i\mathbf G\cdot\mathbf R_I}\right|^2
-\sqrt{\frac{\eta}{\pi}}\sum_I Z_I^2
-\frac{\pi}{2\eta\Omega}\left(\sum_I Z_I\right)^2.
$$

The prime omits the $I=J,\mathbf R=0$ term. The implementation selects a
converged splitting parameter and evaluates analytic Ewald forces and stress.

## 10. SCF fixed-point problem

Given an input density $n^{(m)}$, one SCF step is

$$
n^{(m)}
\longrightarrow V_{\mathrm{eff}}[n^{(m)}]
\longrightarrow \{\psi_{n\mathbf k},\varepsilon_{n\mathbf k}\}
\longrightarrow n_{\mathrm{out}}^{(m)}.
$$

Define the residual

$$
R^{(m)}=n_{\mathrm{out}}^{(m)}-n^{(m)}.
$$

The zero reciprocal component is fixed to preserve the electron number.

### 10.1 Linear and Thomas–Fermi preconditioning

Linear mixing is

$$
n^{(m+1)}=n^{(m)}+\beta R^{(m)}.
$$

For Thomas–Fermi (`TF`) mixing, nonzero reciprocal components are screened by

$$
P_{\mathrm{TF}}(G)=\frac{G^2}{G^2+k_{\mathrm{TF}}^2},
$$

where

$$
r_s=\left(\frac{3\Omega}{4\pi N_e}\right)^{1/3},
\qquad
k_{\mathrm{TF}}^2=\frac{(12/\pi)^{2/3}}{r_s}.
$$

The local-TF mode uses a density-dependent $r_s(\mathbf r)$ and solves a
bounded approximate-inverse problem in a small subspace.

### 10.2 Modified Broyden/Anderson mixing

Although QE names this mode `plain`, the implemented default is a multisecant
modified-Broyden/Anderson method. With history differences

$$
\Delta n_i=n^{(i-1)}-n^{(i)},
\qquad
\Delta R_i=R^{(i-1)}-R^{(i)},
$$

the coefficients solve

$$
\sum_j\langle\Delta R_i,\Delta R_j\rangle_C\gamma_j
=\langle\Delta R_i,R^{(m)}\rangle_C,
$$

in the reciprocal-space Coulomb metric

$$
\langle a,b\rangle_C
=\sum_{\mathbf G\ne0}\frac{a^*_{\mathbf G}b_{\mathbf G}}{G^2}.
$$

The projected input and residual are

$$
\widetilde n=n^{(m)}-\sum_i\gamma_i\Delta n_i,
\qquad
\widetilde R=R^{(m)}-\sum_i\gamma_i\Delta R_i,
$$

followed by $n^{(m+1)}=\widetilde n+\beta P\widetilde R$. Periodic Pulay with
frequency $p>1$ collects history every step but performs this multisecant
projection only when $m\bmod p=0$.

## 11. Iterative diagonalization

For a trial vector $\psi_n$, define the Rayleigh quotient and residual

$$
\varepsilon_n=\langle\psi_n|H|\psi_n\rangle,
\qquad
r_n=H\psi_n-\varepsilon_n\psi_n.
$$

### 11.1 Davidson

The restarted block Davidson method builds an orthonormal basis $V$, forms

$$
H_s=V^\dagger HV,
$$

solves the small Hermitian Ritz problem, and expands the subspace with
preconditioned residuals. QE's smooth diagonal preconditioner is evaluated in
Rydberg units:

$$
x_{Gn}=2\left(H_{GG}-\varepsilon_n\right),
$$

$$
d_{Gn}=\frac{1}{2}\left[1+x_{Gn}
+\sqrt{1+(x_{Gn}-1)^2}\right],
\qquad
\delta\psi_{Gn}=\frac{2r_{Gn}}{d_{Gn}}.
$$

Occupied and empty roots may use different tolerances. The subspace is
restarted before its configured multiple of the target band count is exceeded.

### 11.2 Conjugate gradient and parallel orbital

The conjugate-gradient solver minimizes each band Rayleigh quotient in a
preconditioned direction while enforcing orthogonality to lower bands. The
parallel-orbital solver follows QE's `paro_k_new`. For each active orbital it
solves the projected correction equation

$$
(H-\varepsilon_i)\delta\psi_i
=P_c(\varepsilon_i-H)\psi_i,
\qquad
P_c=1-\sum_{j=1}^{N}|\psi_j\rangle\langle\psi_j|,
$$

using the `bpcg_k` block preconditioned conjugate-gradient trajectory. Each
inner correction solve is limited to five Hamiltonian applications. Corrected
orbitals, converged roots, and extra Ritz vectors are combined in a space of
dimension at most

$$
N+\max\!\left(\operatorname{nint}(N/2),4\right).
$$

Only the first `ntrust` roots participate in the eigenvalue-change convergence
test; roots outside that range are explicitly untrusted, as in QE.

### 11.3 RMM-DIIS

Residual-minimization DIIS stores a bounded history $\{\phi_i,H\phi_i\}$ per
band. It minimizes the residual norm through a small generalized eigenproblem
constructed from

$$
R_i=H\phi_i-e_i\phi_i,
\qquad
(R^\dagger R)c=\lambda(\Phi^\dagger\Phi)c.
$$

The large residual-history matrix is not materialized; its Gram matrix is
assembled from $H\Phi$ and $\Phi$ products. The kinetic preconditioner follows
the Kresse–Furthmüller rational form. With

$$
x=\frac{G^2}{1.5\langle E_{\mathrm{kin}}\rangle},
$$

the implemented diagonal factor is

$$
K(G)=-\frac{4}{3\langle E_{\mathrm{kin}}\rangle}
\frac{27+18x+12x^2+8x^3}
{27+18x+12x^2+8x^3+16x^4}.
$$

Hybrid `rmm-davidson` and `rmm-paro` paths combine the bounded RMM trajectory
with the respective robust subspace update.

## 12. Distributed FFT decomposition

Each active reciprocal `(Gx,Gy)` pair defines a complete $z$ stick. A stick is
owned by exactly one MPI rank throughout the calculation. The owner performs
all one-dimensional $z$ FFTs for that stick. To multiply by a real-space
potential, data are transposed to real-space $z$ slabs:

$$
\text{owned reciprocal sticks}
\xrightarrow{\mathrm{FFT}_z^{-1}}
\xrightarrow{\mathrm{MPI\ Alltoallv}}
\text{owned real-space }z\text{ slabs}
\xrightarrow{\mathrm{FFT}_{xy}^{-1}}.
$$

The reverse sequence returns the coefficients to the original stick owners.
Only active payload is transferred; unlike QE's padded fixed-count
`MPI_Alltoall` implementation, the current kernel uses exact-count
`MPI_Alltoallv`. One real-space band is processed at a time to bound memory.

FFTW-MPI dense slab transforms are available in the native extension for
validation and future dense-slab consumers. They are not used for Hψ because
their reciprocal slab ownership would require an additional redistribution
back to persistent stick owners.

## 13. Symmetry projection

For a scalar density and space-group operation $\{S|\boldsymbol\tau\}$,

$$
n'(\mathbf r)=n\!\left(S^{-1}(\mathbf r-\boldsymbol\tau)\right).
$$

In reciprocal space,

$$
n'_{\mathbf G}
=e^{-i\mathbf G\cdot\boldsymbol\tau}
n_{S^{-1}\mathbf G}.
$$

Density symmetrization averages this action over the compatible operations.
Reciprocal stars and phase factors are precomputed in compact form and applied
without retaining one full density copy per symmetry operation.

## 14. Forces and stress

Implemented forces are analytic Hellmann–Feynman derivatives of the local
pseudopotential, nonlocal projector, nonlinear-core-correction, and Ewald
terms. Schematically,

$$
\mathbf F_I=-\frac{\partial E_{\mathrm{tot}}}{\partial\mathbf R_I}.
$$

For example, differentiation of an atom phase gives

$$
\frac{\partial}{\partial\mathbf R_I}
e^{-i\mathbf G\cdot\mathbf R_I}
=-i\mathbf G e^{-i\mathbf G\cdot\mathbf R_I}.
$$

The stress is the symmetric strain derivative

$$
\sigma_{\alpha\beta}
=-\frac{1}{\Omega}
\frac{\partial E_{\mathrm{tot}}}{\partial\epsilon_{\alpha\beta}},
$$

reported in QE's compressive-positive convention. The implemented terms
include kinetic, local, nonlocal, Hartree, XC/GGA, nonlinear core, and Ewald
contributions for the supported norm-conserving scalar path.

## 15. Principal references

- P. Hohenberg and W. Kohn, *Phys. Rev.* **136**, B864 (1964).
- W. Kohn and L. J. Sham, *Phys. Rev.* **140**, A1133 (1965).
- L. Kleinman and D. M. Bylander, *Phys. Rev. Lett.* **48**, 1425 (1982).
- J. P. Perdew and A. Zunger, *Phys. Rev. B* **23**, 5048 (1981).
- J. P. Perdew and Y. Wang, *Phys. Rev. B* **45**, 13244 (1992).
- J. P. Perdew, K. Burke, and M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865
  (1996).
- P. E. Blöchl, O. Jepsen, and O. K. Andersen, *Phys. Rev. B* **49**, 16223
  (1994).
- E. R. Davidson, *J. Comput. Phys.* **17**, 87 (1975).
- G. Kresse and J. Furthmüller, *Phys. Rev. B* **54**, 11169 (1996).
- D. D. Johnson, *Phys. Rev. B* **38**, 12807 (1988).
