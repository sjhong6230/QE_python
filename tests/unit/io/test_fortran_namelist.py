from __future__ import annotations

import pytest

from qepy_pw.fortran_namelist import (
    expanded_array_keys,
    split_assignment_values,
    split_namelist_assignments,
    strip_namelist_comment,
)


def test_comment_markers_inside_fortran_strings_are_preserved() -> None:
    assert strip_namelist_comment("title='Fe!bcc#test', ecut=40 ! comment") == (
        "title='Fe!bcc#test', ecut=40 "
    )


def test_assignment_lexer_keeps_arrays_and_quoted_commas_together() -> None:
    body = "title='a,b', e1=1, 0, 0,\n nx=20, flag=.true."
    assert split_namelist_assignments(body) == [
        "title='a,b'",
        "e1=1,0,0",
        "nx=20",
        "flag=.true.",
    ]
    assert split_assignment_values("'a,b', (1,2), 3") == [
        "'a,b'", "(1,2)", "3"
    ]


def test_array_key_expansion_follows_fortran_one_based_continuation() -> None:
    assert expanded_array_keys("celldm", 3, unindexed_array=True) == [
        "celldm(1)", "celldm(2)", "celldm(3)"
    ]
    assert expanded_array_keys("starting_magnetization(2)", 3) == [
        "starting_magnetization(2)",
        "starting_magnetization(3)",
        "starting_magnetization(4)",
    ]
    with pytest.raises(ValueError, match="multi-dimensional"):
        expanded_array_keys("hubbard(1,2)", 2)

