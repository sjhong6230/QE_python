"""Regression tests using official QE inputs and Python output references."""

from __future__ import annotations

import pytest

from qepy_pw.input import read_pw_input
from tests.qe_reference.check import (
    compare_values,
    extract_pw,
    input_path,
    load_manifest,
    reference_path,
    run_case,
    verify_pseudos,
)


MANIFEST = load_manifest()
CASES = {case["id"]: case for case in MANIFEST["cases"]}


def test_qe_fixture_checksums_and_python_references_are_complete():
    verify_pseudos(MANIFEST)
    for case in MANIFEST["cases"]:
        assert input_path(case).is_file()
        assert reference_path(case).is_file()


@pytest.mark.parametrize("case_id", CASES)
def test_selected_qe_inputs_are_within_the_implemented_scf_subset(case_id: str):
    pw = read_pw_input(input_path(CASES[case_id]))
    assert str(pw.control.get("calculation", "scf")).lower() == "scf"
    assert int(pw.system.get("nspin", 1)) == 1
    assert not bool(pw.system.get("noncolin", False))
    assert str(pw.system.get("occupations", "fixed")).lower() != "from_input"


@pytest.mark.parametrize("case_id", CASES)
def test_python_output_reference(case_id: str):
    case = CASES[case_id]
    output, actual = run_case(case)
    assert "JOB DONE." in output
    expected = extract_pw(reference_path(case).read_text(encoding="utf-8"))
    close, comparison = compare_values(
        expected, actual, MANIFEST["reference_tolerances"]
    )
    assert close, comparison
