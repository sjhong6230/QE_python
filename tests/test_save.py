from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from qepy_pw.input import read_pw_input
from qepy_pw.errors import QEInputError
from qepy_pw.save import (
    QES_NAMESPACE,
    read_saved_density,
    read_saved_wavefunction,
    resolve_save_directory,
    validate_restart_metadata,
    write_qe_save,
)
from qepy_pw.scf import SCFEnergyTerms, SCFIteration, SCFResult, run_scf
from qepy_pw.timing import TimingEntry


def _result(volume: float) -> SCFResult:
    density = np.full((4, 4, 4), 2.0 / volume)
    miller = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int32)
    coefficients = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 / np.sqrt(2.0)],
            [0.0 + 0.0j, 1.0j / np.sqrt(2.0)],
        ]
    )
    return SCFResult(
        converged=True,
        total_energy_ha=-1.125,
        eigenvalues_ha=[np.array([-0.7, 0.2])],
        density=density,
        iterations=[
            SCFIteration(
                1,
                -1.125,
                1.0e-8,
                cutoff_ry=8.0,
                mixing_beta=0.35,
                davidson_threshold_ha=1.0e-10,
                maximum_eigen_residual_ha=2.0e-11,
            )
        ],
        wall_seconds=1.25,
        energy_terms=SCFEnergyTerms(-0.8, 0.1, -0.2, -0.225),
        plane_waves_per_k=[3],
        peak_rss_bytes_per_rank=1024,
        peak_rss_bytes_all_ranks=1024,
        timings={"electrons": TimingEntry(1.0, 1.2, 1)},
        occupations=[np.array([2.0, 0.0])],
        wavefunctions=[coefficients],
        wavefunction_miller_indices=[miller],
    )


def test_write_qe_save_creates_xml_hdf5_and_pseudopotential(tmp_path: Path):
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.control["outdir"] = str(tmp_path)
    pw.control["prefix"] = "hydrogen"
    prior_save = tmp_path / "hydrogen.save"
    prior_save.mkdir()
    (prior_save / "results.hdf5").write_bytes(b"obsolete qepy result file")
    (prior_save / "wfc2.hdf5").write_bytes(b"obsolete wavefunction file")

    save_directory = write_qe_save(pw, _result(pw.volume))

    assert save_directory == tmp_path / "hydrogen.save"
    assert (save_directory / "H.coulomb.UPF").is_file()
    xml_path = save_directory / "data-file-schema.xml"
    density_path = save_directory / "charge-density.hdf5"
    wavefunction_path = save_directory / "wfc1.hdf5"
    assert xml_path.is_file()
    assert density_path.is_file()
    assert wavefunction_path.is_file()
    assert not (save_directory / "results.hdf5").exists()
    assert not (save_directory / "wfc2.hdf5").exists()

    namespaces = {"qes": QES_NAMESPACE}
    xml = ET.parse(xml_path).getroot()
    assert xml.findtext("qes:output/qes:total_energy/qes:etot", namespaces=namespaces) == (
        "-1.1250000000000000e+00"
    )
    assert xml.find("qes:input/qes:control_variables", namespaces) is not None
    assert xml.find("qes:timing_info", namespaces) is not None
    assert xml.findtext("qes:exit_status", namespaces=namespaces) == "0"
    assert (
        xml.findtext(
            "qes:output/qes:convergence_info/qes:wf_collected",
            namespaces=namespaces,
        )
        == "true"
    )
    assert "github.com/qepy-pw/ns" not in xml_path.read_text(encoding="utf-8")

    with h5py.File(density_path, "r") as h5:
        assert h5.attrs["gamma_only"] == ".TRUE."
        assert h5.attrs["nspin"] == 1
        assert h5.attrs["ngm_g"] == len(h5["MillerIndices"])
        assert h5["MillerIndices"].shape[1] == 3
        assert h5["rhotot_g"].shape == (h5.attrs["ngm_g"],)
        zero = np.flatnonzero(np.all(h5["MillerIndices"][:] == 0, axis=1))[0]
        np.testing.assert_allclose(h5["rhotot_g"][zero].real, 2.0 / pw.volume)

    with h5py.File(wavefunction_path, "r") as h5:
        assert h5.attrs["ik"] == 1
        assert h5.attrs["ispin"] == 1
        assert h5.attrs["gamma_only"] == ".FALSE."
        assert h5.attrs["npol"] == 1
        assert h5.attrs["nbnd"] == 2
        assert h5.attrs["igwx"] == 3
        assert h5.attrs["ngw"] == 3
        assert h5.attrs["scale_factor"] == 1.0
        np.testing.assert_allclose(h5.attrs["xk"], np.zeros(3))
        assert h5["MillerIndices"].shape == (3, 3)
        assert h5["evc"].shape == (2, 3)
        np.testing.assert_allclose(
            h5["evc"][:].T, _result(pw.volume).wavefunctions[0]
        )

def test_disk_io_none_suppresses_save_and_defaults_follow_qe(tmp_path: Path, monkeypatch):
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.control.pop("prefix")
    pw.control.pop("outdir", None)
    monkeypatch.setenv("ESPRESSO_TMPDIR", str(tmp_path))
    assert resolve_save_directory(pw) == tmp_path / "pwscf.save"

    pw.control["disk_io"] = "none"
    assert write_qe_save(pw, _result(pw.volume)) is None
    assert not (tmp_path / "pwscf.save").exists()


def test_converged_scf_wavefunctions_round_trip_through_hdf5(tmp_path: Path):
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.control["outdir"] = str(tmp_path)
    pw.control["prefix"] = "h2_wavefunctions"

    result = run_scf(pw)
    save_directory = write_qe_save(pw, result)

    assert result.converged
    assert save_directory is not None
    assert len(result.wavefunctions) == len(pw.kpoints)
    for index, (coefficients, miller) in enumerate(
        zip(result.wavefunctions, result.wavefunction_miller_indices), start=1
    ):
        with h5py.File(save_directory / f"wfc{index}.hdf5", "r") as h5:
            np.testing.assert_array_equal(h5["MillerIndices"][:], miller)
            np.testing.assert_allclose(h5["evc"][:].T, coefficients)
            stored = h5["evc"][:]
            np.testing.assert_allclose(
                stored.conj() @ stored.T,
                np.eye(stored.shape[0]),
                atol=2.0e-11,
            )


def test_restart_mode_loads_density_and_wavefunctions(tmp_path: Path):
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.control.update(outdir=str(tmp_path), prefix="restart_h2")
    original = run_scf(pw)
    write_qe_save(pw, original)

    pw.control["restart_mode"] = "restart"
    restarted = run_scf(pw)

    assert restarted.converged
    np.testing.assert_allclose(
        restarted.total_energy_ha, original.total_energy_ha, atol=2.0e-7
    )
    # The restarted SCF performs a fresh density update to its requested
    # threshold; pointwise differences are therefore larger than the energy
    # difference in the low-density vacuum region.
    np.testing.assert_allclose(restarted.density, original.density, atol=1.0e-4)


def test_selective_file_readers_reorder_wavefunction_basis(tmp_path: Path):
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.control.update(outdir=str(tmp_path), prefix="selective_h2")
    result = run_scf(pw)
    write_qe_save(pw, result)
    shape = result.density.shape
    nbnd = len(result.eigenvalues_ha[0])

    validate_restart_metadata(pw, shape, nbnd)
    density = read_saved_density(pw, shape, 2.0)
    np.testing.assert_allclose(density, result.density, atol=1.0e-12)
    reversed_miller = result.wavefunction_miller_indices[0][::-1]
    vectors = read_saved_wavefunction(pw, 0, reversed_miller, nbnd)
    np.testing.assert_allclose(vectors, result.wavefunctions[0][::-1])


def test_startingpot_and_startingwfc_file_work_independently(tmp_path: Path):
    root = Path(__file__).parents[1]
    source = root / "examples" / "h2.scf.in"
    saved = read_pw_input(source)
    saved.control.update(outdir=str(tmp_path), prefix="selective_start")
    write_qe_save(saved, run_scf(saved))

    potential_start = read_pw_input(source)
    potential_start.control.update(outdir=str(tmp_path), prefix="selective_start")
    potential_start.electrons["startingpot"] = "file"
    assert run_scf(potential_start).converged

    wavefunction_start = read_pw_input(source)
    wavefunction_start.control.update(outdir=str(tmp_path), prefix="selective_start")
    wavefunction_start.electrons["startingwfc"] = "file"
    assert run_scf(wavefunction_start).converged


def test_restart_rejects_incompatible_fft_grid(tmp_path: Path):
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.control.update(outdir=str(tmp_path), prefix="bad_restart")
    result = run_scf(pw)
    write_qe_save(pw, result)

    with np.testing.assert_raises_regex(QEInputError, "FFT grid"):
        validate_restart_metadata(
            pw, (result.density.shape[0] + 1, *result.density.shape[1:]),
            len(result.eigenvalues_ha[0]),
        )
