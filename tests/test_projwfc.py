from __future__ import annotations

import io
from pathlib import Path
import shutil

import numpy as np
import pytest

import qepy_pw.pp.projwfc as projwfc_module
from qepy_pw.errors import QEInputError
from qepy_pw.pp.projwfc import (
    Orbital,
    ProjectionData,
    _dos_kernel,
    _lowdin_basis,
    run_projwfc,
    symmetrize_projection_weights,
)
from qepy_pw.pp.namelist import parse_namelist
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


def test_diag_basis_diagonalizes_occupation_block() -> None:
    orbitals = tuple(Orbital(1, "X", 1, 1, m, "2P") for m in range(3))
    amplitudes = np.asarray([[[1.0, 0.0, 0.0], [2**-0.5, 2**-0.5, 0.0]]], dtype=complex)
    occupations = np.asarray([[2.0, 1.0]])
    projected = symmetrize_projection_weights(
        amplitudes, orbitals, np.eye(3), (("X", np.zeros(3)),),
        (SymmetryOperation(np.eye(3, dtype=int), np.zeros(3)),),
        diag_basis=True, kpoint_weights=np.asarray([1.0]), occupations=occupations,
    )
    charges = np.einsum("kb,kbo->o", occupations, projected)
    density = np.einsum("kb,kbi,kbj->ij", occupations, amplitudes.conj(), amplitudes)
    assert np.sort(charges) == pytest.approx(np.linalg.eigvalsh(density), abs=1.0e-12)


def test_projwfc_end_to_end_on_saved_scalar_wavefunctions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).parent / "qe_reference" / "upstream" / "pseudo" / "pwscf.save"
    shutil.copytree(source, tmp_path / "pwscf.save")
    monkeypatch.chdir(tmp_path)
    stdout = io.StringIO()
    data, paths = run_projwfc({
        "prefix": "pwscf", "outdir": ".", "filpdos": "si",
        "filproj": "si.proj", "emin": -10.0, "emax": 10.0,
        "deltae": 0.5, "degauss": 0.02, "lwrite_overlaps": True,
    }, stdout=stdout)
    assert data.projections.shape[:2] == data.energies_ev.shape
    assert data.projections.shape[2] == len(data.orbitals) > 0
    assert np.all(data.projections >= 0.0)
    assert np.max(np.sum(data.projections, axis=2)) <= 1.0 + 1.0e-10
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    total = (tmp_path / "si.pdos_tot").read_text(encoding="utf-8").splitlines()
    assert total[0] == "# E (eV) DOS(E) PDOS(E)"
    assert (tmp_path / "pwscf.save" / "atomic_proj.xml").is_file()
    report = stdout.getvalue()
    assert "     Reading xml data from directory:\n\n" in report
    assert (
        "     Gaussian broadening (read from input): "
        "ngauss,degauss=   0    0.020000\n"
    ) in report


def test_kresolved_projwfc_writes_each_energy_and_k_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).parent / "qe_reference" / "upstream" / "pseudo" / "pwscf.save"
    shutil.copytree(source, tmp_path / "pwscf.save")
    monkeypatch.chdir(tmp_path)
    data, _paths = run_projwfc({
        "prefix": "pwscf",
        "outdir": ".",
        "filpdos": "si-k",
        "emin": -1.0,
        "emax": 1.0,
        "deltae": 1.0,
        "degauss": 0.02,
        "kresolveddos": True,
    })

    lines = (tmp_path / "si-k.pdos_tot").read_text(
        encoding="utf-8"
    ).splitlines()
    assert lines[0] == "# ik E (eV) DOS(E) PDOS(E)"
    assert len(lines) == 1 + len(data.weights) * 3
    assert {int(line.split()[0]) for line in lines[1:]} == set(
        range(1, len(data.weights) + 1)
    )


def test_box_ldos_full_grid_has_unit_state_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).parent / "qe_reference" / "upstream" / "pseudo" / "pwscf.save"
    shutil.copytree(source, tmp_path / "pwscf.save")
    monkeypatch.chdir(tmp_path)
    data, paths = run_projwfc({
        "prefix": "pwscf", "outdir": ".", "tdosinboxes": True,
        "n_proj_boxes": 1, "filpdos": "si-box", "degauss": 0.02,
        "deltae": 0.5, "plotboxes": True, "filproj": "boxes.proj",
    })
    assert data.box_weights == pytest.approx(np.ones_like(data.box_weights), abs=2.0e-12)
    assert (tmp_path / "si-box.ldos_boxes").is_file()
    assert (tmp_path / "box#1.xsf").is_file()
    assert (tmp_path / "boxes.proj").is_file()
    assert all(path.stat().st_size > 0 for path in paths)


def test_box_two_dimensional_indices_parse() -> None:
    options = parse_namelist(
        "&PROJWFC\n n_proj_boxes=2, irmin(1, 2)=3, irmax(3,2)=7\n/",
        "projwfc",
    )
    assert options["irmin(1,2)"] == 3
    assert options["irmax(3,2)"] == 7


def test_projwfc_main_uses_qe_projection_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "projwfc.in"
    source.write_text("&PROJWFC\n/\n", encoding="utf-8")
    data = ProjectionData(
        energies_ev=np.asarray([[0.0]]),
        weights=np.asarray([1.0]),
        projections=np.asarray([[[0.9]]]),
        amplitudes=np.asarray([[[np.sqrt(0.9)]]], dtype=complex),
        occupations=np.asarray([[2.0]]),
        orbitals=(Orbital(1, "Si", 1, 0, 0, "3S"),),
        overlaps=(np.eye(1),),
        fermi_ev=0.0,
    )
    monkeypatch.setattr(
        projwfc_module,
        "run_projwfc",
        lambda _options, stdout=None: (data, []),
    )

    assert projwfc_module.main(["-inp", str(source)]) == 0
    output = capsys.readouterr().out
    assert output.startswith("\n     Program PROJWFC-PY v.")
    assert "     Atomic states used for projection\n" in output
    assert "     state #   1: atom   1 (Si ), wfc  1 (l=0 m= 1)" in output
    assert "\nLowdin Charges: \n" in output
    assert "     Atom #   1: total charge =   1.8000, s =  1.8000, " in output
    assert "     Spilling Parameter:   0.1000" in output
    assert "\n     PROJWFC      : " in output
    assert "\n   JOB DONE.\n" in output
    assert "files written" not in output
