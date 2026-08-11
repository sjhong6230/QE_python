"""Create the external bcc-Fe LSDA post-processing validation workflows."""

from __future__ import annotations

from pathlib import Path
import shutil
import textwrap


ROOT = Path("/home/sjhong6230/QE_python_test")
SOURCE_PSEUDO = ROOT / "bcc_Fe_test" / "pseudo" / "Fe.PBE.upf"
TESTS = (
    "lsda_band_test",
    "lsda_dos_test",
    "lsda_projwfc_test",
    "lsda_pp_test",
)


def write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8", newline="\n")
    if executable:
        path.chmod(0o755)


SCF = """
&CONTROL
  calculation = 'scf'
  prefix = 'Fe'
  pseudo_dir = '../pseudo'
  outdir = './outdir'
  disk_io = 'low'
  verbosity = 'high'
  tstress = .false.
  tprnfor = .false.
/
&SYSTEM
  ibrav = 3
  celldm(1) = 5.42000000
  nat = 1
  ntyp = 1
  ecutwfc = 100
  input_dft = 'PBE'
  occupations = 'smearing'
  smearing = 'mp'
  degauss = 0.01
  nspin = 2
  starting_magnetization(1) = 0.50
/
&ELECTRONS
  electron_maxstep = 300
  mixing_mode = 'local-TF'
  mixing_beta = 0.10
  conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  Fe  55.845  Fe.PBE.upf
ATOMIC_POSITIONS crystal
  Fe  0.0  0.0  0.0
K_POINTS automatic
  12 12 12 0 0 0
"""

HIGH_SYMMETRY_NSCF = """
&CONTROL
  calculation = 'bands'
  prefix = 'Fe'
  pseudo_dir = '../pseudo'
  outdir = './outdir'
  disk_io = 'medium'
  verbosity = 'high'
/
&SYSTEM
  ibrav = 3
  celldm(1) = 5.42000000
  nat = 1
  ntyp = 1
  ecutwfc = 100
  input_dft = 'PBE'
  nbnd = 12
  occupations = 'smearing'
  smearing = 'mp'
  degauss = 0.01
  nspin = 2
  starting_magnetization(1) = 0.50
/
&ELECTRONS
  electron_maxstep = 300
  conv_thr = 1.0d-9
  startingpot = 'file'
  startingwfc = 'atomic'
/
ATOMIC_SPECIES
  Fe  55.845  Fe.PBE.upf
ATOMIC_POSITIONS crystal
  Fe  0.0  0.0  0.0
K_POINTS crystal_b
  8
  0.000000  0.000000  0.000000  30 ! Gamma
  0.000000  0.000000  1.000000  30 ! H
  0.000000  0.500000  0.000000  30 ! N
  0.000000  0.000000  0.000000  30 ! Gamma
  0.250000  0.250000  0.250000  30 ! P
  0.000000  0.000000  1.000000  30 ! H
  0.250000  0.250000  0.250000  30 ! P
  0.000000  0.500000  0.000000   1 ! N
"""

DOS_NSCF = """
&CONTROL
  calculation = 'nscf'
  prefix = 'Fe'
  pseudo_dir = '../pseudo'
  outdir = './outdir'
  disk_io = 'low'
  verbosity = 'high'
/
&SYSTEM
  ibrav = 3
  celldm(1) = 5.42000000
  nat = 1
  ntyp = 1
  ecutwfc = 100
  input_dft = 'PBE'
  nbnd = 12
  occupations = 'smearing'
  smearing = 'mp'
  degauss = 0.01
  nspin = 2
  starting_magnetization(1) = 0.50
/
&ELECTRONS
  electron_maxstep = 300
  conv_thr = 1.0d-9
  startingpot = 'file'
  startingwfc = 'atomic'
/
ATOMIC_SPECIES
  Fe  55.845  Fe.PBE.upf
ATOMIC_POSITIONS crystal
  Fe  0.0  0.0  0.0
K_POINTS automatic
  20 20 20 0 0 0
"""


BAND_PLOT = r'''
from pathlib import Path
import io
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def blocks(path):
    chunks = path.read_text(encoding="utf-8").strip().split("\n\n")
    return [np.loadtxt(io.StringIO(chunk)) for chunk in chunks if chunk.strip()]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True,
                         constrained_layout=True)
labels = [r"$\Gamma$", "H", "N", r"$\Gamma$", "P", "H", "P", "N"]
for axis, implementation in zip(axes, ("QE", "python")):
    up = blocks(ROOT / implementation / "Fe.bands.up.dat.gnu")
    down = blocks(ROOT / implementation / "Fe.bands.down.dat.gnu")
    for index, band in enumerate(up):
        axis.plot(band[:, 0], band[:, 1], color="red", lw=0.9,
                  label="spin up" if index == 0 else None)
    for index, band in enumerate(down):
        axis.plot(band[:, 0], band[:, 1], color="blue", lw=0.9,
                  label="spin down" if index == 0 else None)
    x = up[0][:, 0]
    vertices = np.linspace(0, len(x) - 1, len(labels)).round().astype(int)
    axis.set_xticks(x[vertices], labels)
    for value in x[vertices]:
        axis.axvline(value, color="0.75", lw=0.7)
    axis.axhline(0.0, color="0.5", lw=0.7, ls="--")
    axis.set_title(implementation)
    axis.set_xlabel("bcc Fe high-symmetry path")
    axis.grid(alpha=0.15)
axes[0].set_ylabel("Kohn-Sham energy (eV)")
axes[0].legend(loc="best")
fig.suptitle("LSDA bcc Fe bands: red=up, blue=down")
fig.savefig(ROOT / "bands_qe_python.svg", format="svg")
'''


DOS_PLOT = r'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
qe = np.loadtxt(ROOT / "QE" / "Fe.dos.dat")
py = np.loadtxt(ROOT / "python" / "Fe.dos.dat")
py_up = np.interp(qe[:, 0], py[:, 0], py[:, 1])
py_down = np.interp(qe[:, 0], py[:, 0], py[:, 2])

def relative(candidate, reference):
    floor = max(1.0e-12, 1.0e-4 * np.max(np.abs(reference)))
    return (candidate - reference) / np.maximum(np.abs(reference), floor)

errors = np.column_stack((qe[:, 0], relative(py_up, qe[:, 1]),
                          relative(py_down, qe[:, 2])))
np.savetxt(ROOT / "dos_relative_error.dat", errors,
           header="energy_eV rel_error_up rel_error_down")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True,
                         constrained_layout=True)
for axis, column, color, title in zip(
    axes, (1, 2), ("red", "blue"), ("spin up", "spin down")
):
    axis.plot(errors[:, 0], errors[:, column], color=color, lw=1.0)
    axis.axhline(0.0, color="black", lw=0.7)
    axis.set_title(title)
    axis.set_xlabel("Energy (eV)")
    axis.grid(alpha=0.25)
axes[0].set_ylabel("Stabilized relative error (Python-QE)/|QE|")
fig.suptitle("LSDA bcc Fe DOS relative error")
fig.savefig(ROOT / "dos_relative_error.svg", format="svg")
'''


PROJWFC_PLOT = r'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FILES = {
    "Fe 4s": "Fe.pdos.pdos_atm#1(Fe)_wfc#4(s)",
    "Fe 3d": "Fe.pdos.pdos_atm#1(Fe)_wfc#3(d)",
}

def relative(candidate, reference):
    floor = max(1.0e-12, 1.0e-4 * np.max(np.abs(reference)))
    return (candidate - reference) / np.maximum(np.abs(reference), floor)

records = {}
for orbital, filename in FILES.items():
    qe = np.loadtxt(ROOT / "QE" / filename)
    py = np.loadtxt(ROOT / "python" / filename)
    if qe.shape != py.shape or not np.allclose(qe[:, :2], py[:, :2], atol=1.0e-8):
        raise RuntimeError(f"incompatible PDOS grids for {orbital}: {qe.shape} vs {py.shape}")
    ik = qe[:, 0].astype(int)
    nks = int(ik.max())
    ne = len(qe) // nks
    energy = qe[:, 1].reshape(nks, ne)
    up = relative(py[:, 2], qe[:, 2]).reshape(nks, ne)
    down = relative(py[:, 3], qe[:, 3]).reshape(nks, ne)
    records[orbital] = (energy, up, down)
np.savez_compressed(
    ROOT / "pdos_relative_error.npz",
    **{f"{orbital.replace(' ', '_')}_{spin}": values[index]
       for orbital, values in records.items()
       for spin, index in (("energy", 0), ("up", 1), ("down", 2))},
)

limit = max(1.0e-8, max(float(np.nanpercentile(np.abs(item), 99.5))
                         for values in records.values() for item in values[1:]))
fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True,
                         constrained_layout=True)
mesh = None
for row, (orbital, (energy, up, down)) in enumerate(records.items()):
    for column, (spin, error) in enumerate((("up", up), ("down", down))):
        axis = axes[row, column]
        mesh = axis.pcolormesh(np.arange(1, energy.shape[0] + 1), energy[:, 0],
                               error.T, shading="auto", cmap="bwr",
                               vmin=-limit, vmax=limit)
        axis.set_title(f"{orbital}, spin {spin}")
        axis.set_ylabel("Energy (eV)")
        axis.set_xlabel("k-point index along path")
fig.colorbar(mesh, ax=axes, label="Stabilized relative error")
fig.suptitle("LSDA bcc Fe orbital PDOS: Python vs QE")
fig.savefig(ROOT / "pdos_relative_error.svg", format="svg")
'''


PP_PLOT = r'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
QUANTITIES = (
    "charge_up", "charge_down", "potential_up", "potential_down",
    "magnetization", "density_difference_up", "density_difference_down",
    "ildos_up", "ildos_down", "kinetic_up", "kinetic_down",
)

def relative(candidate, reference):
    floor = max(1.0e-12, 1.0e-4 * np.max(np.abs(reference)))
    return (candidate - reference) / np.maximum(np.abs(reference), floor)

errors = {}
summary = []
for name in QUANTITIES:
    qe = np.loadtxt(ROOT / "QE" / f"Fe.{name}.line.dat")
    py = np.loadtxt(ROOT / "python" / f"Fe.{name}.line.dat")
    candidate = np.interp(qe[:, 0], py[:, 0], py[:, -1])
    error = relative(candidate, qe[:, -1])
    errors[name] = (qe[:, 0], error)
    summary.append((name, np.max(np.abs(error)), np.sqrt(np.mean(error**2))))
np.savez_compressed(ROOT / "pp_relative_error.npz",
                    **{name: np.column_stack(values) for name, values in errors.items()})
with (ROOT / "pp_error_summary.csv").open("w", encoding="utf-8") as stream:
    stream.write("quantity,max_abs_relative_error,rms_relative_error\n")
    for name, maximum, rms in summary:
        stream.write(f"{name},{maximum:.16e},{rms:.16e}\n")

fig, axes = plt.subplots(4, 3, figsize=(17, 15), constrained_layout=True)
for axis, name in zip(axes.ravel(), QUANTITIES):
    coordinate, error = errors[name]
    coordinate = (coordinate - coordinate[0]) / max(coordinate[-1] - coordinate[0], 1.0e-30)
    axis.plot(coordinate, error, color="black", lw=0.9)
    axis.axhline(0.0, color="0.5", lw=0.6)
    axis.set_title(name.replace("_", " "))
    axis.set_xlabel("Normalized [100] distance")
    axis.set_ylabel("Relative error")
    axis.grid(alpha=0.2)
for axis in axes.ravel()[len(QUANTITIES):]:
    axis.set_visible(False)
fig.suptitle("LSDA bcc Fe pp.x fields: Python vs QE")
fig.savefig(ROOT / "pp_relative_error.svg", format="svg")
'''


RUN_ALL = r'''
#!/bin/bash
set -euo pipefail

ROOT=/home/sjhong6230/QE_python_test
QE_BIN=/home/sjhong6230/qe-7.5/bin
PY_BIN=/home/sjhong6230/miniconda3/envs/DFT/bin
BAND=$ROOT/lsda_band_test
DOS=$ROOT/lsda_dos_test
PROJ=$ROOT/lsda_projwfc_test
PP=$ROOT/lsda_pp_test

run_qe_pw() { (cd "$1/QE" && OMP_NUM_THREADS=1 mpirun -np 2 "$QE_BIN/pw.x" -in "$2" > "$3"); }
run_py_pw() { (cd "$1/python" && QEPY_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpiexec -n 2 "$PY_BIN/pw.py" -in "$2" > "$3"); }
run_qe_pp() { (cd "$1/QE" && OMP_NUM_THREADS=1 mpirun -np 2 "$QE_BIN/$2" -in "$3" > "$4"); }
run_py_pp() { (cd "$1/python" && QEPY_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY_BIN/$2" -in "$3" > "$4"); }

copy_scf_save() {
  local implementation=$1 destination=$2
  mkdir -p "$destination/$implementation/outdir"
  cp -a "$BAND/$implementation/outdir/Fe.save" "$destination/$implementation/outdir/"
  cp "$BAND/$implementation/Fe.scf.out" "$destination/$implementation/Fe.scf.out"
}

echo "[1/8] QE SCF: bcc Fe, PBE, LSDA, symmetry on, 12x12x12, ecutwfc=100 Ry"
run_qe_pw "$BAND" Fe.scf.in Fe.scf.out
echo "[2/8] Python SCF: 2 MPI processes x 2 qepy threads"
run_py_pw "$BAND" Fe.scf.in Fe.scf.out

for implementation in QE python; do
  copy_scf_save "$implementation" "$DOS"
  copy_scf_save "$implementation" "$PP"
done

echo "[3/8] High-symmetry bands NSCF and bands.x"
run_qe_pw "$BAND" Fe.nscf.in Fe.nscf.out
run_py_pw "$BAND" Fe.nscf.in Fe.nscf.out
for spin in up down; do
  run_qe_pp "$BAND" bands.x "Fe.bands.$spin.in" "Fe.bands.$spin.out"
  run_py_pp "$BAND" bands.py "Fe.bands.$spin.in" "Fe.bands.$spin.out"
done
"$PY_BIN/python" "$BAND/plot_comparison.py"

echo "[4/8] Reuse high-symmetry save for projwfc.x"
for implementation in QE python; do
  mkdir -p "$PROJ/$implementation"
  mv "$BAND/$implementation/outdir" "$PROJ/$implementation/outdir"
  cp "$BAND/$implementation/Fe.scf.out" "$PROJ/$implementation/Fe.scf.out"
  cp "$BAND/$implementation/Fe.nscf.out" "$PROJ/$implementation/Fe.nscf.out"
done
run_qe_pp "$PROJ" projwfc.x Fe.projwfc.in Fe.projwfc.out
run_py_pp "$PROJ" projwfc.py Fe.projwfc.in Fe.projwfc.out
"$PY_BIN/python" "$PROJ/plot_comparison.py"

echo "[5/8] Dense 20x20x20 DOS NSCF"
run_qe_pw "$DOS" Fe.nscf.in Fe.nscf.out
run_py_pw "$DOS" Fe.nscf.in Fe.nscf.out
run_qe_pp "$DOS" dos.x Fe.dos.in Fe.dos.out
run_py_pp "$DOS" dos.py Fe.dos.in Fe.dos.out
"$PY_BIN/python" "$DOS/plot_comparison.py"

echo "[6/8] LSDA pp.x fields, including plot_num=6 magnetization"
for input in "$PP/QE"/Fe.*.pp.in; do
  name=$(basename "$input" .pp.in)
  run_qe_pp "$PP" pp.x "$(basename "$input")" "$name.pp.out"
  run_py_pp "$PP" pp.py "$(basename "$input")" "$name.pp.out"
done
"$PY_BIN/python" "$PP/plot_comparison.py"

echo "[7/8] Verify expected plots and raw comparison data"
test -s "$BAND/bands_qe_python.svg"
test -s "$DOS/dos_relative_error.svg"
test -s "$DOS/dos_relative_error.dat"
test -s "$PROJ/pdos_relative_error.svg"
test -s "$PROJ/pdos_relative_error.npz"
test -s "$PP/pp_relative_error.svg"
test -s "$PP/pp_relative_error.npz"
test -s "$PP/pp_error_summary.csv"

echo "[8/8] Remove calculation outdirs"
for directory in \
  "$BAND/QE/outdir" "$BAND/python/outdir" \
  "$DOS/QE/outdir" "$DOS/python/outdir" \
  "$PROJ/QE/outdir" "$PROJ/python/outdir" \
  "$PP/QE/outdir" "$PP/python/outdir"; do
  case "$directory" in
    "$BAND"/*/outdir|"$DOS"/*/outdir|"$PROJ"/*/outdir|"$PP"/*/outdir)
      rm -rf -- "$directory"
      ;;
    *) echo "Refusing unsafe cleanup target: $directory" >&2; exit 2 ;;
  esac
done
echo "All LSDA post-processing comparisons completed."
'''


def pp_input(name: str, plot_num: int, spin_component: int | None = None) -> str:
    spin = "" if spin_component is None else f"  spin_component = {spin_component}\n"
    energy = "  emin = -100.0\n  emax = 100.0\n" if plot_num == 10 else ""
    return f"""&INPUTPP
  prefix = 'Fe'
  outdir = './outdir'
  filplot = 'Fe.{name}.grid.dat'
  plot_num = {plot_num}
{spin}{energy}/
&PLOT
  nfile = 1
  filepp(1) = 'Fe.{name}.grid.dat'
  weight(1) = 1.0
  iflag = 1
  output_format = 0
  e1(1) = 1.0
  e1(2) = 0.0
  e1(3) = 0.0
  x0(1) = 0.0
  x0(2) = 0.0
  x0(3) = 0.0
  nx = 500
  fileout = 'Fe.{name}.line.dat'
/
"""


def main() -> None:
    if not SOURCE_PSEUDO.is_file():
        raise FileNotFoundError(SOURCE_PSEUDO)
    for name in TESTS:
        root = ROOT / name
        (root / "pseudo").mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_PSEUDO, root / "pseudo" / "Fe.PBE.upf")
        for implementation in ("QE", "python"):
            directory = root / implementation
            directory.mkdir(parents=True, exist_ok=True)
            write(directory / "Fe.scf.in", SCF)

    for implementation in ("QE", "python"):
        band = ROOT / "lsda_band_test" / implementation
        write(band / "Fe.nscf.in", HIGH_SYMMETRY_NSCF)
        for component, name in ((1, "up"), (2, "down")):
            write(band / f"Fe.bands.{name}.in", f"""&BANDS
  prefix = 'Fe'
  outdir = './outdir'
  filband = 'Fe.bands.{name}.dat'
  spin_component = {component}
  lsym = .false.
  no_overlap = .true.
/
""")

        proj = ROOT / "lsda_projwfc_test" / implementation
        write(proj / "Fe.nscf.in", HIGH_SYMMETRY_NSCF)
        write(proj / "Fe.projwfc.in", """&PROJWFC
  prefix = 'Fe'
  outdir = './outdir'
  filpdos = 'Fe.pdos'
  ngauss = 1
  degauss = 0.01
  Emin = -15.0
  Emax = 30.0
  DeltaE = 0.05
  lsym = .false.
  kresolveddos = .true.
/
""")

        dos = ROOT / "lsda_dos_test" / implementation
        write(dos / "Fe.nscf.in", DOS_NSCF)
        write(dos / "Fe.dos.in", """&DOS
  prefix = 'Fe'
  outdir = './outdir'
  fildos = 'Fe.dos.dat'
  bz_sum = 'smearing'
  ngauss = 1
  degauss = 0.01
  DeltaE = 0.02
/
""")

        pp = ROOT / "lsda_pp_test" / implementation
        quantities = (
            ("charge_up", 0, 1), ("charge_down", 0, 2),
            ("potential_up", 1, 1), ("potential_down", 1, 2),
            ("magnetization", 6, None),
            ("density_difference_up", 9, 1),
            ("density_difference_down", 9, 2),
            ("ildos_up", 10, 1), ("ildos_down", 10, 2),
            ("kinetic_up", 22, 1), ("kinetic_down", 22, 2),
        )
        for quantity, number, component in quantities:
            write(
                pp / f"Fe.{quantity}.pp.in",
                pp_input(quantity, number, component),
            )

    write(ROOT / "lsda_band_test" / "plot_comparison.py", BAND_PLOT)
    write(ROOT / "lsda_dos_test" / "plot_comparison.py", DOS_PLOT)
    write(ROOT / "lsda_projwfc_test" / "plot_comparison.py", PROJWFC_PLOT)
    write(ROOT / "lsda_pp_test" / "plot_comparison.py", PP_PLOT)
    write(ROOT / "run_lsda_postprocessing_tests.sh", RUN_ALL, executable=True)


if __name__ == "__main__":
    main()
