from __future__ import annotations

import io

import numpy as np
import pytest

from qepy_pw.input import read_pw_input


@pytest.mark.parametrize(
    ("ibrav", "primitive"),
    [
        (
            3,
            [[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]],
        ),
        (
            -3,
            [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
        ),
    ],
)
def test_bcc_primitive_vectors_follow_qe_ibrav_convention(
    ibrav: int,
    primitive: list[list[float]],
) -> None:
    pw = read_pw_input(io.StringIO(f"""\
&SYSTEM
  ibrav={ibrav}, celldm(1)=6.0, nat=1, ntyp=1, ecutwfc=10
/
ATOMIC_SPECIES
H 1.0 H.UPF
ATOMIC_POSITIONS crystal
H 0 0 0
K_POINTS gamma
"""))

    expected = 3.0 * np.asarray(primitive)
    np.testing.assert_array_equal(pw.lattice, expected)
    np.testing.assert_allclose(
        pw.reciprocal,
        2.0 * np.pi * np.linalg.inv(expected).T,
        rtol=0.0,
        atol=1.0e-15,
    )
