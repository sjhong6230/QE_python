"""Convert QE ``bands.x`` output to plottable data and PostScript."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import TextIO

import numpy as np

from ..cli_options import add_input_file_argument
from ..errors import QEInputError, emit_qe_error
from .band_data import BandData, read_band_file, write_gnuplot


_RAP_HEADER = re.compile(
    r"nbnd_rap\s*=\s*(\d+).*nks_rap\s*=\s*(\d+)", re.IGNORECASE
)


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


def _qe_plot_geometry(
    data: BandData,
    inherited_high_symmetry: np.ndarray | None = None,
    tolerance: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return QE plotband.x path coordinates and high-symmetry flags."""
    points = np.asarray(data.kpoints, dtype=np.float32)
    high_symmetry = (
        np.zeros(data.nks, dtype=bool)
        if inherited_high_symmetry is None
        else np.asarray(inherited_high_symmetry, dtype=bool).copy()
    )
    if high_symmetry.shape != (data.nks,):
        raise QEInputError("invalid plotband high-symmetry flags")
    coordinate = np.zeros(data.nks, dtype=np.float32)
    if data.nks == 0:
        return coordinate, high_symmetry
    high_symmetry[0] = True
    high_symmetry[-1] = True
    if data.nks > 2:
        for index in range(1, data.nks - 1):
            left = points[index] - points[index - 1]
            right = points[index + 1] - points[index]
            left_norm = np.float32(np.sqrt(np.sum(left * left)))
            right_norm = np.float32(np.sqrt(np.sum(right * right)))
            if left_norm == 0.0 or right_norm == 0.0:
                high_symmetry[index] = True
            else:
                cosine = np.float32(
                    np.sum(left * right) / (left_norm * right_norm)
                )
                high_symmetry[index] |= abs(float(cosine) - 1.0) > tolerance
            if np.sum(points[index] * points[index]) < np.float32(1.0e-9):
                high_symmetry[index] = True
    if data.nks > 1:
        first_step = points[1] - points[0]
        typical = np.float32(np.sqrt(np.sum(first_step * first_step)))
        for index in range(1, data.nks):
            step = points[index] - points[index - 1]
            distance = np.float32(np.sqrt(np.sum(step * step)))
            if typical > 0.0 and distance > np.float32(10.0) * typical:
                coordinate[index] = coordinate[index - 1]
            else:
                coordinate[index] = np.float32(
                    coordinate[index - 1] + distance
                )
    return coordinate, high_symmetry


def high_symmetry_indices(data: BandData, tolerance: float = 1.0e-4) -> list[int]:
    """Locate the path vertices identified by QE ``plotband.x``."""
    _coordinate, high_symmetry = _qe_plot_geometry(
        data, tolerance=tolerance
    )
    return np.flatnonzero(high_symmetry).tolist()


def _read_rap_high_symmetry(
    band_file: str | Path, data: BandData
) -> np.ndarray | None:
    """Read the optional ``bands.x`` .rap high-symmetry markers."""
    path = Path(f"{band_file}.rap")
    if not path.is_file():
        return None
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        match = _RAP_HEADER.search(lines[0])
        if match is None:
            return None
        nbnd, nks = map(int, match.groups())
        if nbnd != data.nbnd or nks != data.nks:
            return None
        flags = np.zeros(nks, dtype=bool)
        cursor = 1
        for index in range(nks):
            fields = lines[cursor].split()
            cursor += 1
            if len(fields) < 4:
                return None
            point = np.asarray([float(value) for value in fields[:3]])
            if np.sum(np.abs(point - data.kpoints[index])) > 1.0e-4:
                return None
            flags[index] = fields[3].strip(".").upper() in {"T", "TRUE"}
            count = 0
            while count < nbnd:
                count += len(lines[cursor].split())
                cursor += 1
        return flags
    except (IndexError, OSError, ValueError):
        return None


def _qe_spline_interpolate(
    x: np.ndarray, y: np.ndarray, xout: np.ndarray
) -> np.ndarray:
    """Reproduce plotband.x's single-precision natural-end spline."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    xout = np.asarray(xout, dtype=np.float32)
    count = len(x)
    second = np.empty(count, dtype=np.float32)
    work = np.empty(count, dtype=np.float32)
    first_derivative = np.float32((y[1] - y[0]) / (x[1] - x[0]))
    last_derivative = np.float32(0.0)
    second[0] = np.float32(-0.5)
    work[0] = np.float32(
        np.float32(3.0) / (x[1] - x[0])
        * ((y[1] - y[0]) / (x[1] - x[0]) - first_derivative)
    )
    for index in range(1, count - 1):
        sigma = np.float32(
            (x[index] - x[index - 1]) / (x[index + 1] - x[index - 1])
        )
        p = np.float32(sigma * second[index - 1] + np.float32(2.0))
        second[index] = np.float32((sigma - np.float32(1.0)) / p)
        slope_change = np.float32(
            (y[index + 1] - y[index]) / (x[index + 1] - x[index])
            - (y[index] - y[index - 1]) / (x[index] - x[index - 1])
        )
        work[index] = np.float32(
            (
                np.float32(6.0) * slope_change
                / (x[index + 1] - x[index - 1])
                - sigma * work[index - 1]
            )
            / p
        )
    qn = np.float32(0.5)
    un = np.float32(
        np.float32(3.0) / (x[-1] - x[-2])
        * (last_derivative - (y[-1] - y[-2]) / (x[-1] - x[-2]))
    )
    second[-1] = np.float32(
        (un - qn * work[-2]) / (qn * second[-2] + np.float32(1.0))
    )
    for index in range(count - 2, -1, -1):
        second[index] = np.float32(
            second[index] * second[index + 1] + work[index]
        )
    output = np.empty(len(xout), dtype=np.float32)
    lower = 0
    for output_index, value in enumerate(xout):
        upper = lower + 1
        while upper < count and x[upper] < value:
            upper += 1
        if upper >= count:
            upper = count - 1
            lower = upper - 1
        else:
            lower = max(0, upper - 1)
        width = np.float32(x[upper] - x[lower])
        a = np.float32((x[upper] - value) / width)
        b = np.float32((value - x[lower]) / width)
        output[output_index] = np.float32(
            a * y[lower]
            + b * y[upper]
            + (
                (a**3 - a) * second[lower]
                + (b**3 - b) * second[upper]
            )
            * width
            * width
            / np.float32(6.0)
        )
    return output


def _write_fortran_reals(
    stream: TextIO, values: list[np.float32], per_record: int = 9
) -> None:
    for start in range(0, len(values), per_record):
        stream.write(
            " ".join(
                f"{float(value):8.3f}"
                for value in values[start:start + per_record]
            )
            + "\n"
        )


def write_postscript(
    path: str | Path,
    data: BandData,
    emin: float,
    emax: float,
    fermi: float,
    delta: float,
    reference: float,
    high_symmetry: np.ndarray | None = None,
) -> Path:
    """Write the PostScript program emitted by QE ``plotband.x``."""
    output = Path(path)
    coordinate, high_symmetry_flags = _qe_plot_geometry(data, high_symmetry)
    if data.nks < 2 or coordinate[-1] <= 0.0:
        raise QEInputError("plotband requires a nonzero k-point path")
    energies = np.asarray(data.energies_ev, dtype=np.float32)
    emin32 = np.float32(emin)
    emax32 = np.float32(emax)
    fermi32 = np.float32(fermi)
    delta32 = np.float32(delta)
    reference32 = np.float32(reference)
    cm = np.float32(28.453)
    xdim = np.float32(15.0) * cm
    ydim = np.float32(10.0) * cm
    x0 = np.float32(2.0) * cm
    y0 = np.float32(2.0) * cm
    energy_span = np.float32(emax32 - emin32)
    in_range = np.any(
        (energies >= emin32) & (energies <= emax32), axis=0
    )
    vertices = np.flatnonzero(high_symmetry_flags)
    segments = list(zip(vertices[:-1], vertices[1:]))

    def xmap(value: np.float32) -> np.float32:
        return np.float32(value * xdim / coordinate[-1])

    def ymap(value: np.float32) -> np.float32:
        return np.float32((value - emin32) * ydim / energy_span)

    def nint(value: np.float32) -> int:
        number = float(value)
        return int(np.floor(number + 0.5) if number >= 0.0 else np.ceil(number - 0.5))

    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("%! PS-Adobe-1.0\n")
        stream.write(" /localdict 100 dict def\n localdict begin\n")
        stream.write(" % delete next line for insertion in a LaTeX file\n")
        stream.write("  0 0 moveto\n gsave\n")
        stream.write(" /nm  {newpath moveto} def\n")
        stream.write(" /riga {newpath moveto lineto stroke} def\n")
        stream.write(" /banda {3 1 roll moveto {lineto} repeat stroke} def\n")
        stream.write(" /dot {newpath  1 0 360 arc fill} def\n")
        stream.write(" /Times-Roman findfont 12 scalefont setfont\n")
        stream.write(" currentpoint translate\n")
        stream.write(" % Landscape: uncomment next line\n")
        stream.write("  90 rotate 0 21 neg 28.451 mul translate 1.5 1.5 scale\n")
        stream.write(" % Landscape:   comment next line\n")
        stream.write(" % 1.2 1.2 scale\n")
        stream.write(f"{float(x0):8.3f} {float(y0):8.3f}  translate\n")
        stream.write(" 0 setgray 0.5 setlinewidth\n")
        ni = nint(np.float32((reference32 - emin32) / delta32)) + 1
        nf = nint(np.float32((emax32 - reference32) / delta32)) + 1
        for index in range(-ni, nf + 1):
            tick = np.float32(reference32 + np.float32(index) * delta32)
            if tick < emin32 or tick > emax32:
                continue
            y = ymap(tick)
            stream.write(
                f"{0.0:8.3f} {float(y):8.3f}  moveto -5 0 rlineto stroke\n"
            )
            stream.write(
                f"{-30.0:8.3f} {float(y):8.3f}  moveto "
                f"({float(np.float32(tick - reference32)):5.1f}) show\n"
            )
        if fermi32 > emin32 and fermi32 < emax32:
            y = ymap(fermi32)
            stream.write(
                f"[2 4] 0 setdash newpath {0.0:8.3f} {float(y):8.3f}  moveto \n"
            )
            stream.write(
                f"{float(xdim):8.3f} {float(y):8.3f}  lineto stroke [] 0 setdash\n"
            )
        stream.write(" 1 setlinewidth\n")
        _write_fortran_reals(
            stream,
            [
                np.float32(0.0), np.float32(0.0), np.float32(0.0), ydim,
                xdim, ydim, xdim, np.float32(0.0),
            ],
            per_record=8,
        )
        stream.write(" newpath moveto lineto lineto lineto closepath clip stroke\n")
        stream.write(" 0.5 setlinewidth\n")
        for point_index in range(data.nks):
            x = xmap(coordinate[point_index])
            if high_symmetry_flags[point_index]:
                stream.write(
                    f"{float(x):8.3f} {0.0:8.3f} {float(x):8.3f} "
                    f"{float(ydim):8.3f}  riga\n"
                )
            for band in range(data.nbnd):
                if in_range[band]:
                    stream.write(
                        f"{float(x):8.3f} {float(ymap(energies[point_index, band])):8.3f}  dot\n"
                    )
        for band in range(data.nbnd):
            if not in_range[band]:
                continue
            for start, end in segments:
                interpolation_count = 2 * (end - start) + 1
                if interpolation_count < 7:
                    continue
                interpolation_x = np.linspace(
                    coordinate[start], coordinate[end], interpolation_count,
                    dtype=np.float32,
                )
                interpolation_energy = _qe_spline_interpolate(
                    coordinate[start:end + 1],
                    energies[start:end + 1, band],
                    interpolation_x,
                )
                flattened: list[np.float32] = []
                for index in range(interpolation_count - 1, -1, -1):
                    flattened.extend(
                        [xmap(interpolation_x[index]), ymap(interpolation_energy[index])]
                    )
                _write_fortran_reals(stream, flattened)
                stream.write(f"{interpolation_count - 1:4d} banda\n")
        stream.write(" grestore\n")
        stream.write(" % delete next lines for insertion in a tex file\n")
        stream.write("%%Page\n")
        stream.write(" showpage\n")
    return output


def run_plotband(options: dict[str, object]) -> tuple[Path, Path]:
    band_file = str(options["band_file"])
    data = read_band_file(band_file)
    rap_high_symmetry = _read_rap_high_symmetry(band_file, data)
    plot_file = write_gnuplot(
        str(options["plot_file"]), data
    )
    ps_file = write_postscript(
        str(options["ps_file"]), data,
        float(options["emin"]), float(options["emax"]),
        float(options["fermi"]), float(options["delta"]),
        float(options["reference"]),
        high_symmetry=rap_high_symmetry,
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


def _print_high_symmetry_points(
    data: BandData,
    stdout: TextIO,
    inherited_high_symmetry: np.ndarray | None = None,
) -> None:
    coordinate, high_symmetry = _qe_plot_geometry(
        data, inherited_high_symmetry
    )
    for index in np.flatnonzero(high_symmetry):
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
    filename, data = _interactive_band_file(
        stdin=input_stream, stdout=output_stream
    )
    rap_high_symmetry = _read_rap_high_symmetry(filename, data)
    print(
        f"Reading {data.nbnd:4d} bands at {data.nks:6d} k-points",
        file=output_stream,
        flush=True,
    )
    emin, emax = _interactive_energy_range(
        data, stdin=input_stream, stdout=output_stream
    )
    _print_high_symmetry_points(data, output_stream, rap_high_symmetry)

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
        ps_name, data, emin, emax, fermi, delta, reference,
        high_symmetry=rap_high_symmetry,
    )
    print(
        f"bands in PostScript format written to file {ps_file}",
        file=output_stream,
        flush=True,
    )
    return plot_file, ps_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plotband.py")
    add_input_file_argument(parser)
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
        emit_qe_error(exc, routine="plotband.py")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
