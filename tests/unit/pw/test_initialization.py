from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qepy_pw.basis import LocalPotentialWorkspace, PlaneWaveBasis
from qepy_pw.errors import QEInputError
from qepy_pw.input import read_pw_input
from qepy_pw.pw.scf import (
    ReciprocalGrid,
    _QERandom,
    _atomic_starting_orbitals,
    _atomic_starting_density,
    _scf_force_correction,
    _static_cache_limit_bytes,
    _starting_charge_scales,
    _starting_magnetizations,
)
from qepy_pw.mpi import MPIContext
from qepy_pw.upf import read_upf


PSEUDO_DIR = Path(__file__).resolve().parents[2] / "qe_reference" / "upstream" / "pseudo"


class _UsableLocalTrialsMPI:
    def sum_array(self, value: np.ndarray) -> np.ndarray:
        return np.ones_like(value)


class _SyntheticAtomicOrbitals:
    def atomic_orbital_basis(
        self, vectors: np.ndarray, _volume: float
    ) -> np.ndarray:
        radius = np.linalg.norm(vectors, axis=1)
        return np.column_stack(
            (
                np.exp(-0.2 * radius),
                vectors[:, 0] * np.exp(-0.3 * radius),
                vectors[:, 1] * np.exp(-0.3 * radius),
                vectors[:, 2] * np.exp(-0.3 * radius),
            )
        )


def test_rank_local_atomic_starting_orbitals_match_global_rows() -> None:
    indices = np.array(
        [
            (gx, gy, gz)
            for gx in range(-2, 3)
            for gy in range(-2, 3)
            for gz in range(-2, 3)
            if gx * gx + gy * gy + gz * gz <= 5
        ],
        dtype=np.int32,
    )
    vectors = indices.astype(float)
    basis = PlaneWaveBasis(
        indices,
        vectors,
        0.5 * np.einsum("gi,gi->g", vectors, vectors),
    )
    pw = SimpleNamespace(
        atoms=[SimpleNamespace(label="X", position=np.array([0.2, 0.3, 0.4]))],
        electrons={},
        volume=31.0,
    )
    pseudos = {"X": _SyntheticAtomicOrbitals()}
    global_stream = _QERandom()
    global_trials = _atomic_starting_orbitals(
        pw, pseudos, basis, 4, global_stream
    )
    rows = np.arange(1, len(basis), 3, dtype=np.int32)
    local_stream = _QERandom()
    local_trials = _atomic_starting_orbitals(
        pw,
        pseudos,
        basis,
        4,
        local_stream,
        local_rows=rows,
        mpi=_UsableLocalTrialsMPI(),
        local_vectors=vectors[rows],
        local_kinetic=basis.kinetic[rows],
    )

    np.testing.assert_allclose(
        local_trials, global_trials[rows], rtol=0.0, atol=2.0e-16
    )
    assert local_stream.random() == global_stream.random()


def test_static_kpoint_cache_limit_is_bounded_and_can_be_disabled(
    monkeypatch,
) -> None:
    monkeypatch.delenv("QEPY_STATIC_CACHE_LIMIT_MIB", raising=False)
    assert _static_cache_limit_bytes() == 0
    monkeypatch.setenv("QEPY_STATIC_CACHE_LIMIT_MIB", "0")
    assert _static_cache_limit_bytes() == 0
    monkeypatch.setenv("QEPY_STATIC_CACHE_LIMIT_MIB", "1.5")
    assert _static_cache_limit_bytes() == int(1.5 * 1024**2)
    monkeypatch.setenv("QEPY_STATIC_CACHE_LIMIT_MIB", "invalid")
    with pytest.raises(QEInputError, match="nonnegative number"):
        _static_cache_limit_bytes()
    monkeypatch.setenv("QEPY_STATIC_CACHE_LIMIT_MIB", "-1")
    with pytest.raises(QEInputError, match="nonnegative number"):
        _static_cache_limit_bytes()
    monkeypatch.setenv("QEPY_STATIC_CACHE_LIMIT_MIB", "inf")
    with pytest.raises(QEInputError, match="nonnegative number"):
        _static_cache_limit_bytes()


def _two_species_input(system: str):
    return read_pw_input(io.StringIO(f"""\
&CONTROL
/
&SYSTEM
  ibrav=1, celldm(1)=10, nat=2, ntyp=2, ecutwfc=12,
  {system}
/
&ELECTRONS
/
ATOMIC_SPECIES
H  1.0 H.UPF
Si 28.0 Si.UPF
ATOMIC_POSITIONS crystal
H  0.0 0.0 0.0
Si 0.5 0.5 0.5
K_POINTS gamma
"""))


def test_starting_charge_scales_atomic_valence_and_clips_depletion() -> None:
    pw = _two_species_input("starting_charge(1)=-0.2, starting_charge(2)=5.0")
    pseudos = {
        "H": SimpleNamespace(z_valence=1.0),
        "Si": SimpleNamespace(z_valence=4.0),
    }
    assert _starting_charge_scales(pw, pseudos) == pytest.approx(
        {"H": 1.2, "Si": 0.0}
    )


def test_starting_magnetization_supports_fraction_and_moment_conventions() -> None:
    fractional = _two_species_input(
        "nspin=2, occupations='smearing', degauss=0.01, "
        "starting_magnetization(1)=0.6, starting_magnetization(2)=-0.25"
    )
    pseudos = {
        "H": SimpleNamespace(z_valence=1.0),
        "Si": SimpleNamespace(z_valence=4.0),
    }
    assert _starting_magnetizations(fractional, pseudos) == pytest.approx(
        {"H": 0.6, "Si": -0.25}
    )

    moments = _two_species_input(
        "nspin=2, occupations='smearing', degauss=0.01, "
        "starting_magnetization(1)=2.0, starting_magnetization(2)=0.5"
    )
    assert _starting_magnetizations(moments, pseudos) == pytest.approx(
        {"H": 1.0, "Si": 0.125}
    )


def test_atomic_starting_density_has_requested_charge_and_moment() -> None:
    pw = read_pw_input(io.StringIO(f"""\
&CONTROL
  pseudo_dir='{PSEUDO_DIR.as_posix()}'
/
&SYSTEM
  ibrav=1, celldm(1)=10, nat=2, ntyp=1, ecutwfc=12,
  nspin=2, occupations='fixed', tot_magnetization=0,
  starting_magnetization(1)=0.4
/
&ELECTRONS
/
ATOMIC_SPECIES
H 1.0 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
H 0.25 0.25 0.25
H 0.75 0.75 0.75
K_POINTS gamma
"""))
    pseudo = read_upf(PSEUDO_DIR / "H.pz-vbc.UPF")
    shape = (12, 12, 12)
    geometry = ReciprocalGrid.build(shape, pw.reciprocal, cutoff_ry=48.0)
    workspace = LocalPotentialWorkspace(geometry.charge_indices, shape)
    density, starting_charge = _atomic_starting_density(
        pw,
        {"H": pseudo},
        shape,
        nelec=2.0,
        g2_cutoff=48.0,
        geometry=geometry,
        workspace=workspace,
    )
    scale = pw.volume / np.prod(shape)
    # PP_RHOATOM is numerically integrated on the finite UPF radial mesh;
    # the returned pre-renormalization charge therefore retains that small
    # quadrature deficit, while the real-space starting density is exact.
    assert starting_charge == pytest.approx(2.0, rel=3.0e-5)
    assert scale * np.sum(density) == pytest.approx(2.0, abs=2.0e-12)
    assert scale * np.sum(density[0] - density[1]) == pytest.approx(
        0.8, abs=2.0e-12
    )


def test_scf_force_correction_is_negative_atomic_density_energy_gradient() -> None:
    pw = read_pw_input(io.StringIO(f"""\
&CONTROL
  pseudo_dir='{PSEUDO_DIR.as_posix()}'
/
&SYSTEM
  ibrav=1, celldm(1)=10, nat=2, ntyp=1, ecutwfc=12
/
&ELECTRONS
/
ATOMIC_SPECIES
H 1.0 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
H 0.17 0.23 0.31
H 0.61 0.47 0.73
K_POINTS gamma
"""))
    pseudo = read_upf(PSEUDO_DIR / "H.pz-vbc.UPF")
    shape = (12, 12, 12)
    geometry = ReciprocalGrid.build(
        shape, pw.reciprocal, cutoff_ry=48.0
    )
    workspace = LocalPotentialWorkspace(geometry.charge_indices, shape)
    grid = np.indices(shape, dtype=float)
    potential_residual = (
        0.03 * np.sin(2.0 * np.pi * grid[0] / shape[0])
        + 0.02 * np.cos(4.0 * np.pi * grid[1] / shape[1])
        - 0.01 * np.sin(2.0 * np.pi * grid[2] / shape[2])
    )
    correction = _scf_force_correction(
        pw,
        {"H": pseudo},
        potential_residual,
        workspace,
        geometry.charge_vectors,
        MPIContext.world(),
    )

    residual_g = workspace.grid_to_coefficients(potential_residual)
    g_vectors = geometry.charge_vectors
    radial = pseudo.atomic_density_fourier(
        np.linalg.norm(g_vectors, axis=1), pw.volume
    )

    def interaction(position: np.ndarray) -> float:
        phase = np.exp(-1j * (g_vectors @ position))
        return pw.volume * float(
            np.real(np.vdot(residual_g, radial * phase))
        )

    step = 1.0e-5
    expected = np.zeros_like(correction)
    for atom_index, atom in enumerate(pw.atoms):
        for axis in range(3):
            displacement = np.zeros(3)
            displacement[axis] = step
            expected[atom_index, axis] = -(
                interaction(atom.position + displacement)
                - interaction(atom.position - displacement)
            ) / (2.0 * step)

    np.testing.assert_allclose(correction, expected, atol=2.0e-11)
