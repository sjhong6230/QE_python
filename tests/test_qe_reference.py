"""Regression tests using official QE inputs and Python output references."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import spglib

from qepy_pw.input import read_pw_input
from qepy_pw.output import _qe_symmetry_operations, format_header
from qepy_pw.point_group import point_group_character_table
from qepy_pw.symmetry import SymmetryOperation
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


def test_qe_diamond_character_classes_use_the_printed_symmetry_indices():
    """Keep the O_h class listing tied to the preceding ``isym`` numbers."""
    case = CASES["scf_baseline"]
    pw = read_pw_input(input_path(case))
    pw.control["verbosity"] = "high"
    pw.control["pseudo_dir"] = str(input_path(case).parents[1] / "pseudo")

    operations = _qe_symmetry_operations(pw)
    table = point_group_character_table(pw, operations)
    class_indices = [
        index for point_class in table.classes
        for index in point_class.operation_indices
    ]
    assert sorted(class_indices) == list(range(1, len(operations) + 1))
    assert len(class_indices) == len(set(class_indices))

    output = format_header(pw)
    assert "point group O_h (m-3m)" in output
    assert "there are 10 classes" in output
    for point_class in table.classes:
        rendered_indices = "".join(
            f"{index:5d}" for index in point_class.operation_indices
        )
        assert f"     {point_class.label:<6s}{rendered_indices}" in output


@pytest.mark.filterwarnings("ignore:Set OLD_ERROR_HANDLING:DeprecationWarning")
def test_all_32_crystallographic_point_group_character_tables():
    """Exercise every scalar point group represented in QE's tables."""
    representatives = {
        "C_1": 1, "C_i": 2, "C_2": 3, "C_s": 6, "C_2h": 10,
        "D_2": 16, "C_2v": 25, "D_2h": 47, "C_4": 75,
        "S_4": 81, "C_4h": 83, "D_4": 89, "C_4v": 99,
        "D_2d": 111, "D_4h": 123, "C_3": 143, "S_6": 147,
        "D_3": 149, "C_3v": 156, "D_3d": 162, "C_6": 168,
        "C_3h": 174, "C_6h": 175, "D_6": 177, "C_6v": 183,
        "D_3h": 187, "D_6h": 191, "T": 195, "T_h": 200,
        "O": 207, "T_d": 215, "O_h": 221,
    }
    halls = {}
    for hall_number in range(1, 531):
        group_type = spglib.get_spacegroup_type(hall_number)
        halls.setdefault(group_type.number, hall_number)

    dummy_pw = SimpleNamespace(lattice=np.eye(3))
    for expected_name, space_group in representatives.items():
        symmetry = spglib.get_symmetry_from_database(halls[space_group])
        rotations = []
        seen = set()
        for rotation in symmetry["rotations"]:
            matrix = np.asarray(rotation, dtype=int).T
            key = tuple(matrix.ravel())
            if key not in seen:
                seen.add(key)
                rotations.append(
                    SymmetryOperation(matrix, np.zeros(3, dtype=float))
                )
        table = point_group_character_table(dummy_pw, tuple(rotations))
        assert table.schoenflies == expected_name
        assert len(table.classes) == len(table.irreps)
        printed_indices = sorted(
            index
            for point_class in table.classes
            for index in point_class.operation_indices
        )
        assert printed_indices == list(range(1, len(rotations) + 1))
        class_sizes = np.asarray(
            [len(item.operation_indices) for item in table.classes]
        )
        character_rows = np.asarray([row for _name, row in table.irreps])
        gram = (
            character_rows.conj() * class_sizes[None, :]
        ) @ character_rows.T
        assert np.allclose(gram, len(rotations) * np.eye(len(table.irreps)))
