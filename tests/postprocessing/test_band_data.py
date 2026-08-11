from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qepy_pw.errors import QEInputError
from qepy_pw.pp.band_data import BandData, read_band_file, resolve_save_directory


def test_path_coordinate_collapses_disconnected_segments() -> None:
    data = BandData(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.1, 0.0, 0.0],
            ]
        ),
        np.zeros((5, 1)),
    )

    np.testing.assert_allclose(data.path_coordinate(), [0.0, 0.1, 0.2, 0.2, 0.3])


@pytest.mark.parametrize(
    ("kpoints", "energies", "message"),
    [
        (np.zeros((2, 2)), np.zeros((2, 1)), "kpoints"),
        (np.zeros((2, 3)), np.zeros((3, 1)), "energies"),
        (np.zeros((1, 3)), np.asarray([[np.nan]]), "non-finite"),
    ],
)
def test_band_data_validates_shapes_and_finite_values(
    kpoints: np.ndarray, energies: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BandData(kpoints, energies)


@pytest.mark.parametrize("prefix", ["../pwscf", "subdir/pwscf", "."])
def test_save_directory_rejects_prefix_path_traversal(prefix: str) -> None:
    with pytest.raises(QEInputError, match="prefix"):
        resolve_save_directory(prefix, None)


def test_band_reader_reports_truncated_records(tmp_path: Path) -> None:
    source = tmp_path / "truncated.dat"
    source.write_text(
        " &plot nbnd=   2, nks=     1 /\n"
        "          0.0 0.0 0.0\n"
        "   -1.000\n",
        encoding="utf-8",
    )

    with pytest.raises(QEInputError, match="truncated"):
        read_band_file(source)
