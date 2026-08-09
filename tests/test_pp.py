from __future__ import annotations

import io
import numpy as np
import h5py
import hashlib
from pathlib import Path
import shutil
import sys
import pytest

from qepy_pw.errors import QEInputError
from qepy_pw.constants import EV_PER_HARTREE
from qepy_pw.pp.band_data import (
    BandData,
    read_band_file,
    read_saved_bands,
    write_band_file,
    write_gnuplot,
)
from qepy_pw.pp.bands import (
    _read_wavefunctions,
    _symmetry_matrix,
    classify_irreps,
    main as bands_main,
    reorder_by_overlap,
    run_bands,
    write_band_grid_2d,
    write_irrep_file,
)
from qepy_pw.input import read_pw_input
from qepy_pw.output import format_footer
from qepy_pw.pw.output import format_progress
from qepy_pw.pw.buffers import WavefunctionBuffer
from qepy_pw.pw.save import write_qe_save
from qepy_pw.scf import run_scf
from qepy_pw.pp.plotband import (
    high_symmetry_indices,
    main as plotband_main,
    parse_plotband_input,
    run_interactive_plotband,
)
from qepy_pw.pp.p_matrix import momentum_matrices, write_p_avg
from qepy_pw.qe_format import format_qe_opening
from qepy_pw.symmetry import SymmetryOperation


def _data() -> BandData:
    return BandData(
        np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]),
        np.array([[-2.0, 1.0], [-1.0, 2.0], [-0.5, 2.5]]),
    )


def test_qe_band_file_round_trip_and_gnuplot_layout(tmp_path) -> None:
    path = write_band_file(tmp_path / "bands.out", _data())
    restored = read_band_file(path)
    np.testing.assert_allclose(restored.kpoints, _data().kpoints)
    np.testing.assert_allclose(restored.energies_ev, _data().energies_ev)
    gnu = write_gnuplot(tmp_path / "bands.out.gnu", restored)
    assert "\n\n" in gnu.read_text(encoding="utf-8")


def test_overlap_ordering_tracks_crossed_states() -> None:
    data = BandData(
        np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        np.array([[0.0, 1.0], [0.2, 0.8]]),
    )
    miller = np.array([[0, 0, 0], [1, 0, 0]])
    identity = np.eye(2, dtype=complex)
    wavefunctions = [(miller, identity), (miller, identity[:, ::-1])]
    ordered = reorder_by_overlap(data, wavefunctions)
    np.testing.assert_allclose(ordered.energies_ev[1], [0.8, 0.2])


def test_no_overlap_true_preserves_eigenvalue_order_without_wavefunctions(
    tmp_path, monkeypatch
) -> None:
    namespace = "http://www.quantum-espresso.org/ns/qes/qes-1.0"
    save = tmp_path / "plain.save"
    save.mkdir()
    (save / "data-file-schema.xml").write_text(
        f"""<espresso xmlns="{namespace}"><output><band_structure>
        <ks_energies><k_point>0 0 0</k_point><eigenvalues>0 1</eigenvalues></ks_energies>
        <ks_energies><k_point>0.5 0 0</k_point><eigenvalues>0.2 0.8</eigenvalues></ks_energies>
        </band_structure></output></espresso>""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    filband, _gnu = run_bands(
        {
            "prefix": "plain", "outdir": str(tmp_path),
            "filband": "plain.bands", "lsym": False,
            "no_overlap": True,
        }
    )
    restored = read_band_file(filband)
    np.testing.assert_allclose(
        restored.energies_ev[1],
        np.round(np.array([0.2, 0.8]) * EV_PER_HARTREE, 3),
    )


def test_plot_2d_writes_one_rectangular_grid_file_per_band(
    tmp_path,
) -> None:
    origin = np.array([0.1, 0.2, 0.0])
    dkx = np.array([0.25, 0.0, 0.0])
    dky = np.array([0.0, 0.5, 0.0])
    points = np.vstack(
        [origin + i * dkx + j * dky for i in range(2) for j in range(3)]
    )
    energies = np.arange(12, dtype=float).reshape(6, 2)
    outputs = write_band_grid_2d(
        tmp_path / "grid.bands", BandData(points, energies)
    )
    assert [path.name for path in outputs] == ["grid.bands.1", "grid.bands.2"]
    first = np.loadtxt(outputs[0])
    np.testing.assert_allclose(first[:, 0], [0, 0, 0, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(first[:, 1], [0, 0.5, 1.0, 0, 0.5, 1.0])
    np.testing.assert_allclose(first[:, 2], energies[:, 0])


def test_plot_2d_rejects_nonrectangular_kpoint_order(tmp_path) -> None:
    points = np.array(
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1.1, 1, 0]], dtype=float
    )
    with pytest.raises(QEInputError, match="rectangular grid"):
        write_band_grid_2d(
            tmp_path / "bad", BandData(points, np.zeros((4, 1)))
        )


def test_irrep_file_uses_qe_plot_rap_header(tmp_path) -> None:
    path = write_irrep_file(
        tmp_path / "bands.out.rap", _data(), np.ones((3, 2), dtype=int)
    )
    text = path.read_text(encoding="utf-8")
    assert "&plot_rap nbnd_rap=" in text
    assert "nks_rap=" in text
    assert "    T" in text


def test_irrep_file_applies_first_and_last_k_range(tmp_path) -> None:
    path = write_irrep_file(
        tmp_path / "bands.out.rap",
        _data(),
        np.ones((3, 2), dtype=int),
        firstk=2,
        lastk=2,
    )
    assert path.read_text(encoding="utf-8") == (
        " &plot_rap nbnd_rap=   2, nks_rap=   1 /\n"
        "            0.500000  0.000000  0.000000    T\n"
        "       1       1\n"
    )


def test_irrep_file_formats_roundoff_zero_like_qe(tmp_path) -> None:
    data = BandData(np.array([[-1.0e-14, 0.0, 0.0]]), np.array([[0.0]]))
    path = write_irrep_file(
        tmp_path / "bands.out.rap", data, np.ones((1, 1), dtype=int)
    )
    assert "-0.000000" not in path.read_text(encoding="utf-8")


def test_bands_header_contains_qe_environment_opening() -> None:
    text = format_qe_opening(
        "BANDS-PY", "1.0.0", memory_mib=4096
    )
    assert text.startswith("\n     Program BANDS-PY v.1.0.0 starts on ")
    assert "This program is part of the open-source Quantum ESPRESSO suite" in text
    assert "Serial version\n     4096 MiB available memory" in text


def test_upstream_qe_xml_k_points_are_converted_to_crystal_coordinates(
    tmp_path,
) -> None:
    namespace = "http://www.quantum-espresso.org/ns/qes/qes-1.0"
    save = tmp_path / "ref.save"
    save.mkdir()
    (save / "data-file-schema.xml").write_text(
        f"""<qes:espresso xmlns:qes="{namespace}"><output>
        <basis_set><reciprocal_lattice>
        <b1>-1 -1 1</b1><b2>1 1 1</b2><b3>-1 1 -1</b3>
        </reciprocal_lattice></basis_set><band_structure><ks_energies>
        <k_point>-0.25 -0.25 0.25</k_point><eigenvalues>0.5</eigenvalues>
        </ks_energies></band_structure></output></qes:espresso>""",
        encoding="utf-8",
    )
    restored = read_saved_bands("ref", str(tmp_path))
    np.testing.assert_allclose(restored.kpoints, [[0.25, 0.0, 0.0]])
    np.testing.assert_allclose(restored.energies_ev, [[0.5 * EV_PER_HARTREE]])


def test_upstream_qe_interleaved_wavefunction_dataset_is_read(tmp_path) -> None:
    with h5py.File(tmp_path / "wfc1.hdf5", "w") as h5:
        h5.create_dataset(
            "MillerIndices", data=np.array([[0, 0, 0], [1, 0, 0]], np.int32)
        )
        h5.create_dataset(
            "evc", data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
        )
    miller, coefficients = _read_wavefunctions(tmp_path, 1)[0]
    np.testing.assert_array_equal(miller, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(
        coefficients,
        [[1 + 2j, 5 + 6j], [3 + 4j, 7 + 8j]],
    )


def test_plotband_six_line_input_and_vertices() -> None:
    options = parse_plotband_input(
        "bands.out\n-5, 5\nbands.plot\nbands.ps\n0.5\n1.0,0.0\n"
    )
    assert options["emin"] == -5.0
    assert options["emax"] == 5.0
    assert options["delta"] == 1.0
    assert options["reference"] == 0.0
    assert options["fermi"] == 0.5
    assert high_symmetry_indices(_data()) == [0, 1, 2]


def test_plotband_interactive_dialogue_matches_qe(tmp_path) -> None:
    band_file = write_band_file(tmp_path / "bands.out", _data())
    plot_file = tmp_path / "bands.plot"
    ps_file = tmp_path / "bands.ps"
    stdin = io.StringIO(
        f"{band_file}\n-3,3\n{plot_file}\n{ps_file}\n0.5\n1.0, 0.0\n"
    )
    stdout = io.StringIO()

    written_plot, written_ps = run_interactive_plotband(
        stdin=stdin, stdout=stdout
    )

    assert written_plot == plot_file
    assert written_ps == ps_file
    assert plot_file.is_file()
    assert ps_file.is_file()
    dialogue = stdout.getvalue()
    assert dialogue.startswith(
        f"     Input file > Reading    2 bands at      3 k-points\n"
        "Range:   -2.0000    2.5000eV  Emin, Emax, "
        "[firstk, lastk] > "
    )
    assert "high-symmetry point:  0.0000 0.0000 0.0000" in dialogue
    assert "output file (gnuplot/xmgr) > " in dialogue
    assert "output file (ps) > Efermi > " in dialogue
    assert "deltaE, reference E (for tics) " in dialogue


def test_plotband_main_without_input_file_uses_interactive_stdin(
    tmp_path, monkeypatch, capsys
) -> None:
    band_file = write_band_file(tmp_path / "bands.out", _data())
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(f"{band_file}\n-3 3\n\n\n"),
    )

    assert plotband_main([]) == 0
    output = capsys.readouterr().out
    assert output.startswith("     Input file > Reading")
    assert "output file (gnuplot/xmgr) > skipping ..." in output
    assert "output file (ps) > stopping ..." in output


def test_bands_reads_pw_save_and_performs_scalar_irrep_analysis(
    tmp_path, monkeypatch
) -> None:
    namespace = "http://www.quantum-espresso.org/ns/qes/qes-1.0"
    save = tmp_path / "si.save"
    save.mkdir()
    xml = f"""\
<espresso xmlns="{namespace}">
  <output>
    <atomic_structure>
      <cell><a1>1 0 0</a1><a2>0 1 0</a2><a3>0 0 1</a3></cell>
      <atomic_positions><atom name="Si">0 0 0</atom></atomic_positions>
    </atomic_structure>
    <band_structure>
      <ks_energies>
        <k_point weight="1">0 0 0</k_point>
        <eigenvalues>-0.5</eigenvalues>
      </ks_energies>
    </band_structure>
  </output>
</espresso>
"""
    (save / "data-file-schema.xml").write_text(xml, encoding="utf-8")
    with h5py.File(save / "wfc1.hdf5", "w") as h5:
        h5.create_dataset("MillerIndices", data=np.zeros((1, 3), dtype=np.int32))
        h5.create_dataset("evc", data=np.ones((1, 1), dtype=np.complex128))
    monkeypatch.chdir(tmp_path)
    stdout = io.StringIO()
    filband, gnu = run_bands(
        {"prefix": "si", "outdir": str(tmp_path), "filband": "si.bands"},
        stdout=stdout,
    )
    assert filband.is_file()
    assert gnu.is_file()
    assert (tmp_path / "si.bands.rap").read_text(encoding="utf-8").startswith(
        " &plot_rap"
    )
    assert int(
        (tmp_path / "si.bands.rap").read_text(encoding="utf-8").splitlines()[2]
    ) == 1
    assert "A_1g G_1   G_1+" in stdout.getvalue()


def test_irrep_analysis_distinguishes_inversion_parity(tmp_path) -> None:
    namespace = "http://www.quantum-espresso.org/ns/qes/qes-1.0"
    save = tmp_path / "parity.save"
    save.mkdir()
    (save / "data-file-schema.xml").write_text(
        f"""<espresso xmlns="{namespace}"><output>
        <atomic_structure alat="1"><cell><a1>1 0 0</a1><a2>0 1 0</a2>
        <a3>0 0 1</a3></cell><atomic_positions>
        <atom name="X">0 0 0</atom></atomic_positions></atomic_structure>
        <band_structure><symmetry_operations>
        <symmetry><rotation>1 0 0 0 1 0 0 0 1</rotation>
        <fractional_translation>0 0 0</fractional_translation></symmetry>
        <symmetry><rotation>-1 0 0 0 -1 0 0 0 -1</rotation>
        <fractional_translation>0 0 0</fractional_translation></symmetry>
        </symmetry_operations></band_structure></output></espresso>""",
        encoding="utf-8",
    )
    miller = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.int32)
    coefficients = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
    data = BandData(np.zeros((1, 3)), np.array([[0.0, 1.0]]))
    labels = classify_irreps(data, [(miller, coefficients)], save)
    np.testing.assert_array_equal(labels, [[1, 2]])


def test_irrep_analysis_uses_qe_fractional_translation_phase(tmp_path) -> None:
    namespace = "http://www.quantum-espresso.org/ns/qes/qes-1.0"
    save = tmp_path / "translated.save"
    save.mkdir()
    (save / "data-file-schema.xml").write_text(
        f"""<espresso xmlns="{namespace}"><output>
        <atomic_structure alat="1"><cell><a1>1 0 0</a1><a2>0 1 0</a2>
        <a3>0 0 1</a3></cell><atomic_positions>
        <atom name="X">0 0 0</atom></atomic_positions></atomic_structure>
        <band_structure><symmetry_operations>
        <symmetry><rotation>1 0 0 0 1 0 0 0 1</rotation>
        <fractional_translation>0 0 0</fractional_translation></symmetry>
        <symmetry><rotation>-1 0 0 0 -1 0 0 0 -1</rotation>
        <fractional_translation>0.25 0 0</fractional_translation></symmetry>
        </symmetry_operations></band_structure></output></espresso>""",
        encoding="utf-8",
    )
    miller = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.int32)
    coefficients = np.array([[1j, -1j], [1, 1]], dtype=complex) / np.sqrt(2.0)
    data = BandData(np.zeros((1, 3)), np.array([[0.0, 1.0]]))
    labels = classify_irreps(data, [(miller, coefficients)], save)
    np.testing.assert_array_equal(labels, [[2, 1]])


def test_general_k_symmetry_rotates_qe_periodic_part_not_full_bloch_wave() -> None:
    kpoint = np.array([0.25, 0.0, 0.0])
    miller = np.array([[0, 1, 0], [0, -1, 0]], dtype=np.int32)
    coefficients = np.eye(2, dtype=complex)
    glide = SymmetryOperation(
        np.diag([1, -1, 1]), np.array([0.5, 0.0, 0.0])
    )
    matrix = _symmetry_matrix(kpoint, miller, coefficients, glide)
    # The periodic G indices carry no x component, so QE's translation phase
    # is one. Applying it to k+G instead would add exp(-i*pi/4) globally.
    np.testing.assert_allclose(matrix, [[0.0, 1.0], [1.0, 0.0]])


def test_scf_nscf_save_order_and_bands_inp_integration(
    tmp_path, capsys
) -> None:
    root = Path(__file__).parent / "qe_reference"
    source = root / "upstream" / "pw_scf" / "scf.in"
    pseudo_dir = root / "upstream" / "pseudo"

    scf_pw = read_pw_input(source)
    scf_pw.control.update({
        "prefix": "integ", "outdir": str(tmp_path),
        "wfcdir": str(tmp_path / "working-wfc"),
        "pseudo_dir": str(pseudo_dir), "disk_io": "medium",
    })
    scf_result = run_scf(scf_pw)
    assert isinstance(scf_result.wavefunctions, WavefunctionBuffer)
    assert scf_result.wavefunctions.path == (
        tmp_path / "working-wfc" / "integ.wfc"
    )
    assert scf_result.wavefunctions.path.is_file()
    save_directory = write_qe_save(scf_pw, scf_result)
    assert save_directory is not None
    density_path = save_directory / "charge-density.hdf5"
    density_before = hashlib.sha256(density_path.read_bytes()).digest()

    nscf_pw = read_pw_input(source)
    nscf_pw.control.update({
        "calculation": "nscf", "prefix": "integ",
        "outdir": str(tmp_path), "pseudo_dir": str(pseudo_dir),
        "wfcdir": str(tmp_path / "working-wfc"),
        "verbosity": "high", "tstress": False,
    })
    # The source file was SCF and therefore parsed with its default 'low'.
    # Removing that canonicalized value exercises the calculation-dependent
    # library default after changing the calculation programmatically.
    nscf_pw.control.pop("disk_io")
    progress_chunks: list[str] = []
    nscf_result = run_scf(
        nscf_pw,
        progress=lambda kind, payload: progress_chunks.append(
            format_progress(kind, payload)
        ),
    )
    assert isinstance(nscf_result.wavefunctions, WavefunctionBuffer)
    progress_output = "".join(progress_chunks)
    assert progress_output.count("     Computing kpt #:") == len(
        nscf_pw.kpoints
    )
    assert progress_output.count(
        "     total cpu time spent up to now is"
    ) == len(nscf_pw.kpoints)
    for index in range(1, len(nscf_pw.kpoints) + 1):
        assert f"\n     Computing kpt #: {index:5d}\n" in progress_output
    write_qe_save(nscf_pw, nscf_result)
    assert hashlib.sha256(density_path.read_bytes()).digest() == density_before
    nscf_output = format_footer(nscf_pw, nscf_result)
    assert "End of band structure calculation" in nscf_output
    assert "convergence has been achieved" not in nscf_output

    band_input = tmp_path / "bands.in"
    filband = tmp_path / "integ.bands"
    band_input.write_text(
        "&bands\n"
        " prefix='integ',\n"
        f" outdir='{tmp_path}',\n"
        f" filband='{filband}'\n"
        "/\n",
        encoding="utf-8",
    )
    assert bands_main(["-inp", str(band_input)]) == 0
    stdout = capsys.readouterr().out
    assert "Band symmetry" in stdout
    assert "JOB DONE." in stdout
    assert filband.is_file()
    assert Path(f"{filband}.rap").is_file()


def test_plane_wave_momentum_matrix_has_k_plus_g_diagonal() -> None:
    data = BandData(np.array([[0.25, 0.0, 0.0]]), np.array([[0.0, 1.0]]))
    miller = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    coefficients = np.eye(2, dtype=complex)
    matrices = momentum_matrices(
        data,
        [(miller, coefficients)],
        2.0 * np.pi * np.eye(3),
        [],
        {},
    )
    np.testing.assert_allclose(matrices[0, 0], np.diag([0.25, 1.25]))
    np.testing.assert_allclose(matrices[0, 1:], 0.0)


def test_nonlocal_commutator_uses_both_projector_derivatives() -> None:
    class LinearProjector:
        def projector_basis_with_gradient(self, gk, volume):
            beta = gk[:, :1].astype(complex)
            coupling = np.array([[3.0]])
            gradient = np.zeros((len(gk), 1, 3), dtype=complex)
            gradient[:, 0, 0] = 1.0
            return beta, coupling, gradient

    data = BandData(np.array([[0.0, 0.0, 0.0]]), np.array([[0.0]]))
    matrices = momentum_matrices(
        data,
        [(np.array([[2, 0, 0]], dtype=np.int32), np.ones((1, 1), complex))],
        2.0 * np.pi * np.eye(3),
        [("X", np.zeros(3))],
        {"X": LinearProjector()},
    )
    # kinetic 2 plus d(beta*D*beta)/dq = 1*3*2 + 2*3*1 = 12
    np.testing.assert_allclose(matrices[0, 0, 0, 0], 14.0)


def test_p_avg_uses_qe_header_and_squared_conduction_valence_elements(
    tmp_path,
) -> None:
    data = BandData(np.array([[0.0, 0.0, 0.0]]), np.array([[-1.0, 1.0]]))
    matrices = np.zeros((1, 3, 2, 2), dtype=complex)
    matrices[0, 0, 1, 0] = 2.0j
    path = write_p_avg(
        tmp_path / "p_avg.dat", data, matrices, np.array([[2.0, 0.0]])
    )
    text = path.read_text(encoding="utf-8")
    assert text.startswith(" &p_mat nbnd=")
    assert "     4.00000000" in text


def test_bands_lp_writes_p_matrix_from_saved_upf_and_wavefunctions(
    tmp_path, monkeypatch
) -> None:
    namespace = "http://www.quantum-espresso.org/ns/qes/qes-1.0"
    save = tmp_path / "si.save"
    save.mkdir()
    pseudo_source = (
        Path(__file__).parent / "qe_reference" / "upstream" / "pseudo"
        / "Si.pz-vbc.UPF"
    )
    shutil.copy2(pseudo_source, save / pseudo_source.name)
    xml = f"""\
<espresso xmlns="{namespace}">
  <output>
    <atomic_species><species name="Si"><mass>28.085</mass><pseudo_file>{pseudo_source.name}</pseudo_file></species></atomic_species>
    <atomic_structure alat="10.2">
      <cell><a1>-5.1 0 5.1</a1><a2>0 5.1 5.1</a2><a3>-5.1 5.1 0</a3></cell>
      <atomic_positions><atom name="Si">0 0 0</atom></atomic_positions>
    </atomic_structure>
    <band_structure>
      <ks_energies>
        <k_point weight="1">0.25 0 0</k_point>
        <eigenvalues>-0.5 0.2</eigenvalues>
        <occupations>2 0</occupations>
      </ks_energies>
    </band_structure>
  </output>
</espresso>
"""
    (save / "data-file-schema.xml").write_text(xml, encoding="utf-8")
    with h5py.File(save / "wfc1.hdf5", "w") as h5:
        h5.create_dataset(
            "MillerIndices", data=np.array([[0, 0, 0], [1, 0, 0]], np.int32)
        )
        h5.create_dataset("evc", data=np.eye(2, dtype=np.complex128))
    monkeypatch.chdir(tmp_path)
    run_bands(
        {
            "prefix": "si", "outdir": str(tmp_path),
            "filband": "si.bands", "lsym": False,
            "lp": True, "filp": "p_avg.dat",
        }
    )
    text = (tmp_path / "p_avg.dat").read_text(encoding="utf-8")
    assert "&p_mat nbnd=" in text
    assert "nks=" in text
    assert "          -0.250000 -0.250000  0.250000      1" in text
