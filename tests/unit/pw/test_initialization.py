from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qepy_pw.basis import LocalPotentialWorkspace
from qepy_pw.errors import QEInputError
from qepy_pw.input import read_pw_input
from qepy_pw.pw.scf import (
    ReciprocalGrid,
    _atomic_starting_density,
    _static_cache_limit_bytes,
    _starting_charge_scales,
    _starting_magnetizations,
)
from qepy_pw.upf import read_upf


PSEUDO_DIR = Path(__file__).resolve().parents[2] / "qe_reference" / "upstream" / "pseudo"


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
