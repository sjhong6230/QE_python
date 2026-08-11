from __future__ import annotations

import io
from pathlib import Path
import shutil

import numpy as np
import pytest

import qepy_pw.pp.projwfc as projwfc_module
from qepy_pw.errors import QEInputError
from qepy_pw.pp.dos import DOSData
from qepy_pw.pp.projwfc import (
    Orbital,
    ProjectionData,
    _dos_kernel,
    _format_projection_summary,
    _lowdin_basis,
    _qe_projection_order,
    _qe_e11_3,
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


def test_lsda_pdos_writes_up_down_columns_and_spin_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save = tmp_path / "iron.save"
    save.mkdir()
    spins = np.asarray([1, 2], dtype=np.int8)
    saved = DOSData(
        np.asarray([[-1.0], [1.0]]),
        np.asarray([1.0, 1.0]),
        0.0,
        "smearing",
        "gaussian",
        0.01,
        None,
        None,
        None,
        spins,
    )
    orbital = Orbital(1, "Fe", 1, 0, 0, "4S")
    projections = ProjectionData(
        energies_ev=saved.eigenvalues_ev,
        weights=saved.weights,
        projections=np.ones((2, 1, 1)),
        amplitudes=np.ones((2, 1, 1), dtype=complex),
        occupations=np.ones((2, 1)),
        orbitals=(orbital,),
        overlaps=(np.eye(1), np.eye(1)),
        fermi_ev=0.0,
        spins=spins,
    )
    monkeypatch.setattr(projwfc_module, "read_saved_dos", lambda *_args: saved)
    monkeypatch.setattr(
        projwfc_module, "compute_projections", lambda *_args, **_kwargs: projections
    )
    monkeypatch.chdir(tmp_path)

    data, _paths = run_projwfc(
        {
            "prefix": "iron",
            "outdir": str(tmp_path),
            "filpdos": "iron",
            "degauss": 0.02,
            "emin": -2.0,
            "emax": 2.0,
            "deltae": 1.0,
        }
    )

    assert data.nspin == 2
    total_lines = (tmp_path / "iron.pdos_tot").read_text(
        encoding="utf-8"
    ).splitlines()
    assert "dosup(E) dosdw(E) pdosup(E) pdosdw(E)" in total_lines[0]
    values = np.loadtxt(tmp_path / "iron.pdos_tot")
    assert values.shape[1] == 5
    assert values[1, 1] > values[1, 2]
    assert values[3, 2] > values[3, 1]
    orbital_values = np.loadtxt(
        tmp_path / "iron.pdos_atm#1(Fe)_wfc#1(s)"
    )
    assert orbital_values.shape[1] == 5
    atomic_text = (save / "atomic_proj.xml").read_text(encoding="utf-8")
    assert 'spin="1"' in atomic_text and 'spin="2"' in atomic_text

    run_projwfc(
        {
            "prefix": "iron",
            "outdir": str(tmp_path),
            "filpdos": "iron-k",
            "degauss": 0.02,
            "emin": -2.0,
            "emax": 2.0,
            "deltae": 1.0,
            "lsym": False,
            "kresolveddos": True,
        }
    )
    kresolved = np.loadtxt(tmp_path / "iron-k.pdos_tot")
    # QE joins the up/down blocks at each spatial k point instead of
    # emitting one zero-padded record for every saved spin k point.
    assert kresolved.shape == (5, 6)
    np.testing.assert_array_equal(kresolved[:, 0], 1)
    assert kresolved[1, 2] > kresolved[1, 3]
    assert kresolved[3, 3] > kresolved[3, 2]


def test_lsda_projection_summary_reports_lowdin_polarization() -> None:
    data = ProjectionData(
        energies_ev=np.asarray([[0.0], [0.0]]),
        weights=np.asarray([1.0, 1.0]),
        projections=np.asarray([[[0.8]], [[0.3]]]),
        amplitudes=np.asarray([[[np.sqrt(0.8)]], [[np.sqrt(0.3)]]]),
        occupations=np.ones((2, 1)),
        orbitals=(Orbital(1, "Fe", 1, 0, 0, "4S"),),
        overlaps=(np.eye(1), np.eye(1)),
        fermi_ev=0.0,
        kpoints=np.zeros((2, 3)),
        spins=np.asarray([1, 2]),
    )

    summary = _format_projection_summary(data)

    assert "(spin up)" in summary and "(spin down)" in summary
    assert "spin up =   0.8000" in summary
    assert "spin down =   0.3000" in summary
    assert "polarization =   0.5000" in summary


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
    source = (
        Path(__file__).resolve().parents[1]
        / "qe_reference"
        / "upstream"
        / "pseudo"
        / "pwscf.save"
    )
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
    assert total[0] == "# E (eV)  dos(E)    pdos(E)"
    assert (tmp_path / "pwscf.save" / "atomic_proj.xml").is_file()
    report = stdout.getvalue()
    assert "     Reading xml data from directory:\n\n" in report
    assert (
        "     Gaussian broadening (read from input): "
        "ngauss,degauss=   0    0.020000\n"
    ) in report
    assert "     IMPORTANT: XC functional enforced from input :\n" in report
    assert "     Exchange-correlation= PZ\n" in report


def test_kresolved_projwfc_writes_each_energy_and_k_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "qe_reference"
        / "upstream"
        / "pseudo"
        / "pwscf.save"
    )
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
    assert lines[0] == "# ik    E (eV)  dos(E)    pdos(E)"
    data_lines = [line for line in lines[1:] if line]
    assert len(data_lines) % len(data.weights) == 0
    assert {int(line.split()[0]) for line in data_lines} == set(
        range(1, len(data.weights) + 1)
    )


def test_qe_pdos_exponent_format() -> None:
    assert _qe_e11_3(5.12e-4) == "  0.512E-03"
    assert _qe_e11_3(-5.12e-4) == " -0.512E-03"
    assert _qe_e11_3(0.0) == "  0.000E+00"


def test_qe_projection_order_preserves_state_order_for_numerical_ties() -> None:
    weights = np.asarray([0.16000, 0.15995, 0.16003, 0.20, 0.01])
    assert _qe_projection_order(weights).tolist() == [3, 0, 1, 2, 4]


def test_box_ldos_full_grid_has_unit_state_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "qe_reference"
        / "upstream"
        / "pseudo"
        / "pwscf.save"
    )
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
        kpoints=np.zeros((1, 3)),
        plane_waves=np.asarray([123]),
        nkb=4,
    )
    def fake_run(_options, stdout=None):
        if stdout is not None:
            print("\n     Calling projwave .... ", file=stdout)
            print(_format_projection_summary(data), file=stdout)
        return data, []

    monkeypatch.setattr(
        projwfc_module,
        "run_projwfc",
        fake_run,
    )

    assert projwfc_module.main(["-inp", str(source)]) == 0
    output = capsys.readouterr().out
    assert output.startswith("\n     Program PROJWFC-PY v.")
    assert "     Atomic states used for projection\n" in output
    assert "     Calling projwave .... " in output
    assert "  Problem Sizes \n" in output
    assert "  natomwfc =            1\n" in output
    assert "  npwx     =          123\n" in output
    assert "==== e(   1) =     0.00000 eV ==== " in output
    assert "    |psi|^2 = 0.900" in output
    assert "     state #   1: atom   1 (Si ), wfc  1 (l=0 m= 1)" in output
    assert "\nLowdin Charges: \n" in output
    assert "     Atom #   1: total charge =   1.8000, s =  1.8000, " in output
    assert "     Spilling Parameter:   0.1000" in output
    assert "\n     PROJWFC      : " in output
    assert "\n   JOB DONE.\n" in output
    assert "files written" not in output
