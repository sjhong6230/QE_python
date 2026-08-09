"""Convert QE ``bands.x`` output to plottable data and PostScript."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import TextIO

import numpy as np

from ..errors import QEInputError, emit_qe_error
from .band_data import BandData, read_band_file, write_gnuplot


def _meaningful_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith(("#", "!"))]


def _numeric_fields(text: str) -> list[str]:
    """Split a Fortran list-directed numeric record on blanks or commas."""
    normalized = text.replace("d", "e").replace("D", "E").strip()
    return [field for field in re.split(r"[\s,]+", normalized) if field]


def parse_plotband_input(text: str) -> dict[str, object]:
    """Parse QE's six-line interactive ``plotband.x`` input stream."""
    lines = _meaningful_lines(text)
    if len(lines) < 6:
        raise QEInputError(
            "plotband input requires: band file, Emin/Emax, plot file, "
            "PostScript file, Fermi energy, and deltaE/reference energy"
        )
    try:
        emin, emax = (float(value) for value in _numeric_fields(lines[1])[:2])
        fermi = float(_numeric_fields(lines[4])[0])
        delta, reference = (
            float(value) for value in _numeric_fields(lines[5])[:2]
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
        if float(np.dot(data.kpoints[index], data.kpoints[index])) < 1.0e-9:
            indices.add(index)
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
        str(options["plot_file"]), data
    )
    ps_file = write_postscript(
        str(options["ps_file"]), data,
        float(options["emin"]), float(options["emax"]),
        float(options["fermi"]), float(options["delta"]),
        float(options["reference"]),
    )
    return plot_file, ps_file


def _read_response(
    prompt: str,
    *,
    stdin: TextIO,
    stdout: TextIO,
) -> str:
    """Print and flush one QE prompt before waiting for its response."""
    print(prompt, end="", file=stdout, flush=True)
    response = stdin.readline()
    if response == "":
        raise QEInputError("end of file while reading plotband input")
    return response.rstrip("\r\n")


def _interactive_band_file(
    *,
    stdin: TextIO,
    stdout: TextIO,
) -> tuple[str, BandData]:
    while True:
        filename = _read_response(
            "     Input file > ", stdin=stdin, stdout=stdout
        ).strip()
        if not filename:
            continue
        if not Path(filename).is_file():
            print(f"{filename}: file not found", file=stdout, flush=True)
            continue
        return filename, read_band_file(filename)


def _interactive_energy_range(
    data: BandData,
    *,
    stdin: TextIO,
    stdout: TextIO,
) -> tuple[float, float]:
    minimum = float(np.min(data.energies_ev))
    maximum = float(np.max(data.energies_ev))
    response = _read_response(
        f"Range:{minimum:10.4f}{maximum:10.4f}eV  "
        "Emin, Emax, [firstk, lastk] > ",
        stdin=stdin,
        stdout=stdout,
    )
    try:
        fields = _numeric_fields(response)
        emin, emax = float(fields[0]), float(fields[1])
    except (ValueError, IndexError) as exc:
        raise QEInputError("invalid Emin/Emax in plotband input") from exc
    if emax <= emin:
        raise QEInputError("plotband requires Emax > Emin")
    return emin, emax


def _print_high_symmetry_points(data: BandData, stdout: TextIO) -> None:
    coordinate = data.path_coordinate()
    for index in high_symmetry_indices(data):
        point = data.kpoints[index]
        if index == 0:
            suffix = "   x coordinate   0.0000"
        else:
            suffix = f"   x coordinate{coordinate[index]:9.4f}"
        print(
            "high-symmetry point: "
            f"{point[0]:7.4f}{point[1]:7.4f}{point[2]:7.4f}{suffix}",
            file=stdout,
            flush=True,
        )


def run_interactive_plotband(
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> tuple[Path | None, Path | None]:
    """Run the prompt-by-prompt terminal dialogue of QE ``plotband.x``."""
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    _filename, data = _interactive_band_file(
        stdin=input_stream, stdout=output_stream
    )
    print(
        f"Reading {data.nbnd:4d} bands at {data.nks:6d} k-points",
        file=output_stream,
        flush=True,
    )
    emin, emax = _interactive_energy_range(
        data, stdin=input_stream, stdout=output_stream
    )
    _print_high_symmetry_points(data, output_stream)

    plot_name = _read_response(
        "output file (gnuplot/xmgr) > ",
        stdin=input_stream,
        stdout=output_stream,
    ).strip()
    plot_file: Path | None = None
    if plot_name:
        plot_file = write_gnuplot(plot_name, data)
        print(
            "bands in gnuplot/xmgr format written to file "
            f"{plot_file}",
            file=output_stream,
            flush=True,
        )
    else:
        print("skipping ...", file=output_stream, flush=True)

    ps_name = _read_response(
        "output file (ps) > ", stdin=input_stream, stdout=output_stream
    ).strip()
    if not ps_name:
        print("stopping ...", file=output_stream, flush=True)
        return plot_file, None
    fermi_response = _read_response(
        "Efermi > ", stdin=input_stream, stdout=output_stream
    )
    tick_response = _read_response(
        "deltaE, reference E (for tics) ",
        stdin=input_stream,
        stdout=output_stream,
    )
    try:
        fermi = float(_numeric_fields(fermi_response)[0])
        tick_fields = _numeric_fields(tick_response)
        delta, reference = float(tick_fields[0]), float(tick_fields[1])
    except (ValueError, IndexError) as exc:
        raise QEInputError("invalid numeric value in plotband input") from exc
    if delta <= 0.0:
        raise QEInputError("plotband requires deltaE > 0")
    ps_file = write_postscript(
        ps_name, data, emin, emax, fermi, delta, reference
    )
    print(
        f"bands in PostScript format written to file {ps_file}",
        file=output_stream,
        flush=True,
    )
    return plot_file, ps_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plotband.py")
    parser.add_argument(
        "-i", "-in", "-inp", "-input", "--input", dest="input_file"
    )
    args = parser.parse_args(argv)
    try:
        if args.input_file is None:
            run_interactive_plotband()
            return 0
        text = Path(args.input_file).read_text(encoding="utf-8")
        plot_file, ps_file = run_plotband(parse_plotband_input(text))
        print(f"bands in gnuplot/xmgr format written to file {plot_file}")
        print(f"bands in PostScript format written to file {ps_file}")
        return 0
    except (QEInputError, OSError, ValueError) as exc:
        emit_qe_error(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
