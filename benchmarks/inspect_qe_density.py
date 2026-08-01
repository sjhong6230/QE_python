"""Inspect the legacy QE Fortran-record charge-density file."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np


path = Path(sys.argv[1])
records: list[bytes] = []
with path.open("rb") as handle:
    record = 0
    while marker := handle.read(4):
        size = struct.unpack("<i", marker)[0]
        data = handle.read(size)
        trailing = struct.unpack("<i", handle.read(4))[0]
        print(record, size, trailing, data[:24].hex())
        records.append(data)
        record += 1

if len(sys.argv) > 2:
    import os

    from qepy_pw.input import read_pw_input
    from qepy_pw.scf import run_scf

    input_path = Path(sys.argv[2]).resolve()
    os.chdir(input_path.parent)
    pw = read_pw_input(input_path)
    if len(sys.argv) > 3:
        pw.electrons["diago_thr_init"] = float(sys.argv[3])
    if len(sys.argv) > 4:
        pw.electrons["py_davidson_residual_factor"] = float(
            sys.argv[4]
        )
    if len(sys.argv) > 5:
        pw.electrons["py_davidson_residual_energy_scale"] = float(
            sys.argv[5]
        )
    result = run_scf(pw)
    print("Python total energy (Ry)", 2.0 * result.total_energy_ha)
    header = np.frombuffer(records[0], dtype="<i4")
    ngm = int(header[1])
    g_vectors = np.frombuffer(records[2], dtype="<i4").reshape(
        (3, ngm), order="F"
    ).T
    qe_density = np.frombuffer(records[3], dtype="<c16")
    python_grid = np.fft.fftn(result.density) / result.density.size
    slots = tuple(
        g_vectors[:, axis] % python_grid.shape[axis]
        for axis in range(3)
    )
    python_density = python_grid[slots]
    difference = python_density - qe_density
    print("QE G=0", qe_density[0])
    print("Python G=0", python_density[0])
    print("max |delta rho(G)|", np.max(np.abs(difference)))
    print("RMS |delta rho(G)|", np.sqrt(np.mean(np.abs(difference) ** 2)))
    largest = np.argsort(np.abs(difference))[-12:][::-1]
    for index in largest:
        print(
            g_vectors[index],
            qe_density[index],
            python_density[index],
            difference[index],
        )
