"""Convert QE ``bands.x`` output to plottable data and PostScript."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from ..errors import QEInputError, format_qe_error
from .band_data import BandData, read_band_file, write_gnuplot


def _meaningful_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith(("#", "!"))]


def parse_plotband_input(text: str) -> dict[str, object]:
    """Parse QE's six-line interactive ``plotband.x`` input stream."""
    lines = _meaningful_lines(text)
    if len(lines) < 6:
        raise QEInputError(
            "plotband input requires: band file, Emin/Emax, plot file, "
            "PostScript file, Fermi energy, and deltaE/reference energy"
        )
    try:
        emin, emax = (float(value.replace("d", "e").replace("D", "E")) for value in lines[1].split()[:2])
        fermi = float(lines[4].split()[0].replace("d", "e").replace("D", "E"))
        delta, reference = (
            float(value.replace("d", "e").replace("D", "E"))
            for value in lines[5].split()[:2]
        )
    except (ValueError, IndexError) as exc:
        raise QEInputError("invalid numeric value in plotband input") from exc
    if emax <= emin or delta <= 0.0:
        raise QEInputError("plotband requires Emax > Emin and deltaE > 0")
    return {
        "band_file": lines[0], "emin": emin, "emax": emax,
        "plot_file": lines[2], "ps_file": lines[3], "fermi": fermi,
        "delta": delta, "reference": reference,
    }


def high_symmetry_indices(data: BandData, tolerance: float = 1.0e-7) -> list[int]:
    """Locate path endpoints, repeated vertices, and direction changes."""
    if data.nks <= 2:
        return list(range(data.nks))
    steps = np.diff(data.kpoints, axis=0)
    indices = {0, data.nks - 1}
    for index in range(1, data.nks - 1):
        left, right = steps[index - 1], steps[index]
        left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
        if left_norm <= tolerance or right_norm <= tolerance:
            indices.add(index)
            continue
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
        if not np.isclose(cosine, 1.0, atol=tolerance):
            indices.add(index)
    return sorted(indices)


def write_postscript(
    path: str | Path,
    data: BandData,
    emin: float,
    emax: float,
    fermi: float,
    delta: float,
    reference: float,
) -> Path:
    """Write a compact dependency-free PostScript band plot."""
    output = Path(path)
    coordinate = data.path_coordinate()
    xmax = max(float(coordinate[-1]), 1.0)
    left, bottom, width, height = 72.0, 72.0, 468.0, 648.0
    def xmap(value: float) -> float:
        return left + width * value / xmax
    def ymap(value: float) -> float:
        return bottom + height * (value - emin) / (emax - emin)
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 50 50 560 750\n")
        stream.write("/Helvetica findfont 9 scalefont setfont\n0 setgray 0.6 setlinewidth\n")
        stream.write(f"newpath {left} {bottom} moveto {left + width} {bottom} lineto {left + width} {bottom + height} lineto {left} {bottom + height} lineto closepath stroke\n")
        for index in high_symmetry_indices(data):
            x = xmap(float(coordinate[index]))
            stream.write(f"0.85 setgray newpath {x:.3f} {bottom} moveto {x:.3f} {bottom + height} lineto stroke 0 setgray\n")
        first_tick = np.ceil((emin - reference) / delta) * delta + reference
        for energy in np.arange(first_tick, emax + 0.5 * delta, delta):
            y = ymap(float(energy))
            stream.write(f"newpath {left - 4} {y:.3f} moveto {left} {y:.3f} lineto stroke {left - 42} {y - 3:.3f} moveto ({energy:g}) show\n")
        if emin <= fermi <= emax:
            y = ymap(fermi)
            stream.write(f"[4 3] 0 setdash newpath {left} {y:.3f} moveto {left + width} {y:.3f} lineto stroke [] 0 setdash\n")
        stream.write("0.8 setlinewidth\n")
        for band in range(data.nbnd):
            visible = [(xmap(float(x)), ymap(float(e))) for x, e in zip(coordinate, data.energies_ev[:, band])]
            if not visible:
                continue
            stream.write(f"newpath {visible[0][0]:.3f} {visible[0][1]:.3f} moveto\n")
            for x, y in visible[1:]:
                stream.write(f"{x:.3f} {y:.3f} lineto\n")
            stream.write("stroke\n")
        stream.write("showpage\n%%EOF\n")
    return output


def run_plotband(options: dict[str, object]) -> tuple[Path, Path]:
    data = read_band_file(str(options["band_file"]))
    plot_file = write_gnuplot(
        str(options["plot_file"]), data, reference_ev=float(options["fermi"])
    )
    ps_file = write_postscript(
        str(options["ps_file"]), data,
        float(options["emin"]), float(options["emax"]),
        float(options["fermi"]), float(options["delta"]),
        float(options["reference"]),
    )
    return plot_file, ps_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plotband.py")
    parser.add_argument("-i", "-in", "--input", dest="input_file")
    args = parser.parse_args(argv)
    try:
        text = Path(args.input_file).read_text(encoding="utf-8") if args.input_file else sys.stdin.read()
        plot_file, ps_file = run_plotband(parse_plotband_input(text))
        print(f"     Bands in gnuplot/xmgr format written to file {plot_file}")
        print(f"     PostScript plot written to file {ps_file}")
        return 0
    except (QEInputError, OSError, ValueError) as exc:
        print(format_qe_error(exc), end="", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
