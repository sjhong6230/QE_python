from __future__ import annotations

import numpy as np
import h5py

from qepy_pw.pp.band_data import BandData, read_band_file, write_band_file, write_gnuplot
from qepy_pw.pp.bands import reorder_by_overlap, run_bands, write_irrep_file
from qepy_pw.pp.plotband import high_symmetry_indices, parse_plotband_input


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


def test_irrep_file_uses_qe_plot_rap_header(tmp_path) -> None:
    path = write_irrep_file(
        tmp_path / "bands.out.rap", _data(), np.ones((3, 2), dtype=int)
    )
    text = path.read_text(encoding="utf-8")
    assert "&plot_rap nbnd_rap=" in text
    assert "nks_rap=" in text
    assert "    T" in text


def test_plotband_six_line_input_and_vertices() -> None:
    options = parse_plotband_input(
        "bands.out\n-5 5\nbands.plot\nbands.ps\n0.5\n1.0 0.0\n"
    )
    assert options["fermi"] == 0.5
    assert high_symmetry_indices(_data()) == [0, 1, 2]


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
    filband, gnu = run_bands(
        {"prefix": "si", "outdir": str(tmp_path), "filband": "si.bands"}
    )
    assert filband.is_file()
    assert gnu.is_file()
    assert (tmp_path / "si.bands.rap").read_text(encoding="utf-8").startswith(
        " &plot_rap"
    )
