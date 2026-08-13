from __future__ import annotations

import pytest

from qepy_pw.pw.output import _format_scf_accuracy_ry, format_iteration
from qepy_pw.pw.scf import SCFIteration


@pytest.mark.parametrize(
    ("accuracy_ry", "expected"),
    [
        (2.0e-8, "       0.00000002"),
        (1.0e-8, "          1.0E-08"),
        (5.0e-9, "          5.0E-09"),
        (9.5e-12, "          9.5E-12"),
    ],
)
def test_scf_accuracy_matches_qe_format_switch(
    accuracy_ry: float, expected: str
) -> None:
    assert _format_scf_accuracy_ry(accuracy_ry) == expected


def test_iteration_uses_scientific_scf_accuracy_below_eps8() -> None:
    step = SCFIteration(
        number=4,
        total_energy_ha=-1.0,
        estimated_accuracy_ha=2.5e-9,
    )

    assert (
        "     estimated scf accuracy    <          5.0E-09 Ry\n"
        in format_iteration(step)
    )
