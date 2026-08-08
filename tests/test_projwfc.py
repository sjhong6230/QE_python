from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from qepy_pw.errors import QEInputError
from qepy_pw.pp.projwfc import (
    Orbital,
    _dos_kernel,
    _lowdin_basis,
    run_projwfc,
    symmetrize_projection_weights,
)
from qepy_pw.symmetry import SymmetryOperation


def test_lowdin_basis_is_orthonormal_and_preserves_span() -> None:
    atomic = np.asarray([[1.0, 0.3], [0.2j, 1.0], [0.4, -0.1j]], dtype=complex)
    orthogonal, overlap = _lowdin_basis(atomic)
    assert overlap == pytest.approx(atomic.conj().T @ atomic)
    assert orthogonal.conj().T @ orthogonal == pytest.approx(np.eye(2), abs=1.0e-12)
    projector = atomic @ np.linalg.pinv(atomic)
    assert projector @ orthogonal == pytest.approx(orthogonal, abs=1.0e-12)


def test_lowdin_basis_rejects_linear_dependence() -> None:
    atomic = np.asarray([[1.0, 2.0], [0.0, 0.0]], dtype=complex)
    with pytest.raises(QEInputError, match="singular"):
        _lowdin_basis(atomic)


def test_projection_gaussian_kernel_normalizes_to_one_state() -> None:
    grid = np.linspace(-5.0, 5.0, 20001)
    kernel = _dos_kernel(np.asarray([[0.0]]), grid, 0.2, 0)
    assert np.trapezoid(kernel[:, 0, 0], grid) == pytest.approx(1.0, abs=1.0e-6)


def test_projection_kernel_rejects_unknown_smearing() -> None:
    with pytest.raises(QEInputError, match="ngauss"):
        _dos_kernel(np.asarray([[0.0]]), np.asarray([0.0]), 0.1, 3)


def test_cubic_rotation_average_equalizes_p_components() -> None:
    orbitals = tuple(Orbital(1, "X", 1, 1, m, "2P") for m in range(3))
    amplitudes = np.asarray([[[1.0 + 0.2j, 0.0, 0.0]]])
    cyclic = np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=int)
    operations = tuple(
        SymmetryOperation(np.linalg.matrix_power(cyclic, power), np.zeros(3))
        for power in range(3)
    )
    weights = symmetrize_projection_weights(
        amplitudes, orbitals, np.eye(3), (("X", np.zeros(3)),), operations
    )
    assert weights[0, 0] == pytest.approx(np.full(3, 1.04 / 3.0), abs=1.0e-12)
    assert np.sum(weights) == pytest.approx(np.sum(np.abs(amplitudes) ** 2))


def test_projwfc_end_to_end_on_saved_scalar_wavefunctions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).parent / "qe_reference" / "upstream" / "pseudo" / "pwscf.save"
    shutil.copytree(source, tmp_path / "pwscf.save")
    monkeypatch.chdir(tmp_path)
    data, paths = run_projwfc({
        "prefix": "pwscf", "outdir": ".", "filpdos": "si",
        "filproj": "si.proj", "emin": -10.0, "emax": 10.0,
        "deltae": 0.5, "degauss": 0.02, "lwrite_overlaps": True,
    })
    assert data.projections.shape[:2] == data.energies_ev.shape
    assert data.projections.shape[2] == len(data.orbitals) > 0
    assert np.all(data.projections >= 0.0)
    assert np.max(np.sum(data.projections, axis=2)) <= 1.0 + 1.0e-10
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    total = (tmp_path / "si.pdos_tot").read_text(encoding="utf-8").splitlines()
    assert total[0] == "# E (eV) DOS(E) PDOS(E)"
    assert (tmp_path / "pwscf.save" / "atomic_proj.xml").is_file()
