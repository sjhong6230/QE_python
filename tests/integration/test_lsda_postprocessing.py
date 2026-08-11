from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from qepy_pw.input import read_pw_input
from qepy_pw.pp.band_data import read_band_file
from qepy_pw.pp.bands import run_bands
from qepy_pw.pp.dos import run_dos
from qepy_pw.pp.pp import extract_plot_grids, read_saved_pp
from qepy_pw.pp.projwfc import ProjectionData, run_projwfc
from qepy_pw.pw.save import write_qe_save
from qepy_pw.scf import run_scf


def test_one_lsda_save_drives_all_postprocessing_programs(
    tmp_path: Path, monkeypatch
) -> None:
    pseudo_dir = (
        Path(__file__).resolve().parents[1]
        / "qe_reference"
        / "upstream"
        / "pseudo"
    )
    pw = read_pw_input(io.StringIO(f"""\
&CONTROL
  calculation='scf', pseudo_dir='{pseudo_dir}',
  outdir='{tmp_path}', prefix='lsda-post', disk_io='medium',
  tstress=.false., tprnfor=.false.
/
&SYSTEM
  ibrav=1, celldm(1)=10.0, nat=2, ntyp=1,
  ecutwfc=10.0, nbnd=1, nspin=2,
  occupations='fixed', tot_magnetization=0,
  starting_magnetization(1)=0.25
/
&ELECTRONS
  conv_thr=2.d-5, electron_maxstep=30
/
ATOMIC_SPECIES
H 1.0008 H.pz-vbc.UPF
ATOMIC_POSITIONS angstrom
H 0.00 0.00 -0.35
H 0.00 0.00  0.35
K_POINTS gamma
"""))
    result = run_scf(pw)
    assert result.converged
    write_qe_save(pw, result)
    monkeypatch.chdir(tmp_path)

    up_path, _ = run_bands({
        "prefix": "lsda-post", "outdir": str(tmp_path),
        "filband": "up.bands", "spin_component": 1,
        "lsym": False, "no_overlap": True,
    })
    down_path, _ = run_bands({
        "prefix": "lsda-post", "outdir": str(tmp_path),
        "filband": "down.bands", "spin_component": 2,
        "lsym": False, "no_overlap": True,
    })
    assert read_band_file(up_path).nks == 1
    assert read_band_file(down_path).nks == 1

    dos_path = run_dos({
        "prefix": "lsda-post", "outdir": str(tmp_path),
        "fildos": "lsda.dos", "degauss": 0.02,
        "emin": -20.0, "emax": 10.0, "deltae": 1.0,
    })
    assert "dosup(E)     dosdw(E)" in dos_path.read_text(
        encoding="utf-8"
    ).splitlines()[0]

    projection_data, _paths = run_projwfc({
        "prefix": "lsda-post", "outdir": str(tmp_path),
        "filpdos": "lsda", "degauss": 0.02, "deltae": 1.0,
        "lsym": False, "kresolveddos": True,
    })
    assert isinstance(projection_data, ProjectionData)
    assert projection_data.nspin == 2
    assert "dosup(E) dosdw(E)" in (tmp_path / "lsda.pdos_tot").read_text(
        encoding="utf-8"
    ).splitlines()[0]

    pp_state = read_saved_pp("lsda-post", str(tmp_path))
    assert pp_state.nspin == 2
    np.testing.assert_allclose(
        pp_state.spin_densities, result.density, rtol=0.0, atol=2.0e-12
    )
    magnetization = extract_plot_grids(
        pp_state, {"plot_num": 6, "filplot": "magnetization.pp"}
    )[0][1]
    np.testing.assert_allclose(
        magnetization.values,
        result.density[0] - result.density[1],
        rtol=0.0,
        atol=2.0e-12,
    )
