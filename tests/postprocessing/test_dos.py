from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

import qepy_pw.pp.dos as dos_module
from qepy_pw.constants import EV_PER_HARTREE
from qepy_pw.errors import QEInputError
from qepy_pw.pp.dos import (
    DOSData,
    run_dos,
    smearing_dos,
    smearing_dos_channels,
    tetrahedron_dos,
)


def test_gaussian_smearing_has_two_scalar_spin_states() -> None:
    grid = np.linspace(-5.0, 5.0, 20001)
    density = smearing_dos(np.asarray([[0.0]]), np.asarray([1.0]), grid, 0.2, 0)
    assert np.trapezoid(density, grid) == pytest.approx(2.0, abs=2.0e-6)
    assert density[len(grid) // 2] == pytest.approx(2.0 / (np.sqrt(np.pi) * 0.2), rel=2.0e-8)


def test_smearing_rejects_unsupported_ngauss() -> None:
    with pytest.raises(QEInputError, match="ngauss"):
        smearing_dos(np.asarray([[0.0]]), np.asarray([1.0]), np.asarray([0.0]), 0.1, 2)


def test_lsda_smearing_dos_preserves_up_and_down_state_counts() -> None:
    grid = np.linspace(-4.0, 4.0, 40001)
    data = DOSData(
        np.asarray([[-1.0], [1.0]]),
        np.asarray([1.0, 1.0]),
        0.0,
        "smearing",
        "gaussian",
        0.01,
        None,
        None,
        None,
        np.asarray([1, 2]),
    )

    channels = smearing_dos_channels(data, grid, 0.2, 0)

    assert channels.shape == (2, len(grid))
    np.testing.assert_allclose(
        np.trapezoid(channels, grid, axis=1), [1.0, 1.0], atol=2.0e-6
    )
    assert channels[0, np.argmin(abs(grid + 1.0))] > channels[1].max() * 0.99
    assert channels[1, np.argmin(abs(grid - 1.0))] > channels[0, -1]


def test_run_dos_reads_save_and_includes_energy_endpoint(tmp_path: Path, monkeypatch) -> None:
    save = tmp_path / "tmp" / "si.save"
    save.mkdir(parents=True)
    xml = f"""<?xml version='1.0'?>
<qes:espresso xmlns:qes='http://www.quantum-espresso.org/ns/qes/qes-1.0'>
 <qes:output><qes:band_structure>
  <qes:occupations_kind>smearing</qes:occupations_kind>
  <qes:smearing>gaussian</qes:smearing><qes:degauss>0.01</qes:degauss>
  <qes:fermi_energy>{0.1 / EV_PER_HARTREE}</qes:fermi_energy>
  <qes:ks_energies><qes:k_point weight='1.0'>0 0 0</qes:k_point>
   <qes:eigenvalues>{-0.2 / EV_PER_HARTREE} {0.3 / EV_PER_HARTREE}</qes:eigenvalues>
  </qes:ks_energies>
 </qes:band_structure></qes:output>
</qes:espresso>"""
    (save / "data-file-schema.xml").write_text(xml, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    stdout = io.StringIO()
    output = run_dos({
        "prefix": "si", "outdir": "./tmp", "emin": -1.0, "emax": 1.0,
        "deltae": 0.5, "degauss": 0.02, "ngauss": 0, "fildos": "si.dos",
    }, stdout=stdout)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert "EFermi =    0.100 eV" in lines[0]
    assert len(lines) == 6
    assert float(lines[-1].split()[0]) == pytest.approx(1.0)
    report = stdout.getvalue()
    assert "     Reading xml data from directory:\n\n" in report
    assert (
        "     Gaussian broadening (read from input): "
        "ngauss,degauss=   0    0.020000\n"
    ) in report


def test_linear_tetrahedron_integrates_all_bands() -> None:
    grid_shape = (3, 3, 3)
    count = np.prod(grid_shape)
    # An irrational-looking deterministic sequence avoids accidental
    # vertex degeneracies while retaining a compact synthetic mesh.
    values = np.mod(np.arange(count, dtype=float) * 0.61803398875, 1.0)
    eigenvalues = np.column_stack((values, 2.0 + values))
    data = DOSData(
        eigenvalues, np.full(count, 1.0 / count), None, "tetrahedra",
        "gaussian", 0.0, grid_shape, np.arange(count, dtype=np.int32), np.eye(3),
    )
    energies = np.asarray([-1.0, 4.0])
    density, integrated = tetrahedron_dos(data, energies, "tetrahedra_lin")
    assert density == pytest.approx([0.0, 0.0])
    assert integrated == pytest.approx([0.0, 4.0])


def test_lsda_tetrahedron_uses_both_spin_blocks_without_degeneracy() -> None:
    grid_shape = (2, 2, 2)
    spatial_kpoints = int(np.prod(grid_shape))
    eigenvalues = np.concatenate(
        (
            np.full((spatial_kpoints, 1), -1.0),
            np.full((spatial_kpoints, 1), 1.0),
        )
    )
    data = DOSData(
        eigenvalues,
        np.full(2 * spatial_kpoints, 1.0 / (2 * spatial_kpoints)),
        None,
        "tetrahedra",
        "gaussian",
        0.0,
        grid_shape,
        np.arange(spatial_kpoints, dtype=np.int32),
        np.eye(3),
    )

    _density, integrated = tetrahedron_dos(
        data, np.asarray([-2.0, 0.0, 2.0]), "tetrahedra_lin"
    )

    assert integrated == pytest.approx([0.0, 1.0, 2.0])


def test_tetrahedron_rejects_eigenvalues_inconsistent_with_mapping() -> None:
    data = DOSData(
        np.zeros((3, 1)),
        np.full(3, 1.0 / 3.0),
        None,
        "tetrahedra",
        "gaussian",
        0.0,
        (2, 2, 2),
        np.arange(8, dtype=np.int32),
        np.eye(3),
    )

    with pytest.raises(QEInputError, match="inconsistent"):
        tetrahedron_dos(data, np.asarray([0.0]), "tetrahedra")


def test_dos_main_uses_qe_environment_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "dos.in"
    source.write_text("&DOS\n/\n", encoding="utf-8")
    monkeypatch.setattr(
        dos_module,
        "run_dos",
        lambda _options, stdout=None: tmp_path / "pwscf.dos",
    )

    assert dos_module.main(["-in", str(source)]) == 0
    output = capsys.readouterr().out
    assert output.startswith("\n     Program DOS-PY v.")
    assert "\n     DOS          : " in output
    assert "\n   JOB DONE.\n" in output
    assert "DOS written to file" not in output
