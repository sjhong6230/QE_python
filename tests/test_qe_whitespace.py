"""Exact whitespace checks for text governed by QE FORMAT statements."""

from __future__ import annotations

from datetime import datetime
import io

import numpy as np

from qepy_pw.pp.bands import _format_group_info
from qepy_pw.pw.output import (
    _format_force_component,
    _format_stress_component,
)
from qepy_pw.qe_format import (
    format_qe_closing,
    format_qe_duration,
    qe_date_and_time,
)


def test_qe_date_time_and_closing_message_fixed_fields() -> None:
    now = datetime(2026, 8, 9, 7, 4, 2)
    assert qe_date_and_time(now) == (" 9Aug2026", " 7: 4: 2")

    lines = format_qe_closing(now=now).splitlines()
    assert lines[0] == ""
    assert lines[1] == (
        "   This run was terminated on:   7: 4: 2  9Aug2026"
        + " " * 13
    )
    assert lines[2] == ""
    assert lines[3] == "=" + "-" * 78 + "="
    assert lines[4] == "   JOB DONE."
    assert lines[5] == "=" + "-" * 78 + "="


def test_qe_primary_clock_duration_fields() -> None:
    assert format_qe_duration(4.2, "CPU") == "     4.20s CPU"
    assert format_qe_duration(64.2, "WALL") == "  1m 4.20s WALL"
    assert format_qe_duration(3664.2, "CPU") == "     1h 1m CPU"
    assert format_qe_duration(90064.2, "WALL") == "  1d 1h 1m WALL"


def test_force_component_uses_qe_9035_layout_without_extra_blank_lines() -> None:
    out = io.StringIO()
    _format_force_component(
        out,
        "The ionic contribution  to forces",
        np.asarray([[0.5, -0.25, 0.0]]),
        [2],
    )
    assert out.getvalue() == (
        "     The ionic contribution  to forces\n"
        "     atom    1 type  2   force = "
        "    1.00000000   -0.50000000    0.00000000\n"
    )


def test_stress_component_uses_qe_9005_continuation_indent() -> None:
    out = io.StringIO()
    _format_stress_component(out, "kinetic stress (kbar)", np.zeros((3, 3)))
    assert out.getvalue() == (
        "     kinetic stress (kbar)      0.00      0.00      0.00\n"
        + " " * 26 + "      0.00      0.00      0.00\n"
        + " " * 26 + "      0.00      0.00      0.00\n\n"
    )


def test_scalar_group_info_uses_qe_a5_and_i3_fields() -> None:
    point_class = type(
        "PointClass",
        (),
        {
            "label": "E",
            "operation_indices": (1,),
            "description": "identity",
        },
    )()
    table = type(
        "PointTable",
        (),
        {
            "schoenflies": "C_1",
            "international": "1",
            "classes": (point_class,),
            "irreps": (("A", (1.0 + 0.0j,)),),
        },
    )()
    assert _format_group_info(table) == (
        "     point group C_1 (1)    \n"
        "     there are  1 classes\n"
        "     the character table:\n\n"
        "       E     \n"
        "A      1.00\n"
        "\n     the symmetry operations in each class and the name "
        "of the first element:\n\n"
        "     E        1\n"
        "          identity\n"
    )
