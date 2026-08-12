"""Regression tests using official QE inputs and Python output references."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import spglib

from qepy_pw.input import read_pw_input
from qepy_pw.output import _qe_symmetry_operations, format_header
from qepy_pw.point_group import point_group_character_table
from qepy_pw.scf import run_scf
from qepy_pw.symmetry import ReciprocalDensitySymmetrizer, SymmetryOperation
from tests.qe_reference.check import (
    compare_values,
    extract_pw,
    input_path,
    load_manifest,
    reference_path,
    run_case,
    verify_pseudos,
    verify_saved_fixture,
)


MANIFEST = load_manifest()
CASES = {case["id"]: case for case in MANIFEST["cases"]}


def test_qe_fixture_checksums_and_python_references_are_complete():
    verify_pseudos(MANIFEST)
    verify_saved_fixture(MANIFEST)
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
    assert "deg rotation - cart. axis" in output
    assert "point group O_h (m-3m)" in output
    assert "there are 10 classes" in output
    for point_class in table.classes:
        rendered_indices = "".join(
            f"{index:5d}" for index in point_class.operation_indices
        )
        assert f"     {point_class.label:<5s}{rendered_indices}" in output


def test_c2v_mirror_classes_follow_qe_double_group_axis_order():
    """QE fixes B1/B2 by mirror normals, not discovery order."""
    operations = (
        SymmetryOperation(np.eye(3, dtype=int), np.zeros(3)),
        SymmetryOperation(np.diag([-1, -1, 1]), np.zeros(3)),
        # Deliberately provide the y-normal mirror before the x-normal one.
        SymmetryOperation(np.diag([1, -1, 1]), np.zeros(3)),
        SymmetryOperation(np.diag([-1, 1, 1]), np.zeros(3)),
    )
    table = point_group_character_table(
        SimpleNamespace(lattice=np.eye(3)), operations
    )
    assert table.schoenflies == "C_2v"
    assert [item.label for item in table.classes] == [
        "E", "C2", "s_v", "s_v'",
    ]
    # For principal z, QE is_c2v orders normal x before normal y.
    assert table.classes[2].operation_indices == (4,)
    assert table.classes[3].operation_indices == (3,)


def test_gamma_only_scf_still_projects_density_onto_crystal_symmetry(
    monkeypatch,
):
    """QE calls sym_rho even when Gamma is both full and irreducible."""
    case = CASES["scf_gamma"]
    pw = read_pw_input(input_path(case))
    assert pw.full_kpoint_count == len(pw.kpoints) == 1
    assert len(pw.symmetry_operations) > 1

    applications = 0
    original_apply = ReciprocalDensitySymmetrizer.apply

    def tracked_apply(self, density):
        nonlocal applications
        applications += 1
        return original_apply(self, density)

    monkeypatch.setattr(ReciprocalDensitySymmetrizer, "apply", tracked_apply)
    run_case(case)
    assert applications > 0


def test_gamma_only_scf_preserves_multidimensional_irrep_degeneracies():
    """The Si Gamma T and E irreps retain their 3- and 2-fold dimensions."""
    case = CASES["scf_gamma"]
    pw = read_pw_input(input_path(case))
    pw.control["pseudo_dir"] = str(input_path(case).parents[1] / "pseudo")
    pw.control["disk_io"] = "none"
    pw.system["nbnd"] = 12
    pw.electrons["conv_thr"] = 1.0e-10
    pw.electrons["diago_thr_init"] = 1.0e-10

    result = run_scf(pw)
    eigenvalues = result.eigenvalues_ha[0]
    expected_multiplets = {
        "T valence (bands 2-4)": eigenvalues[1:4],
        "T conduction (bands 5-7)": eigenvalues[4:7],
        "E conduction (bands 10-11)": eigenvalues[9:11],
    }
    for label, multiplet in expected_multiplets.items():
        assert np.ptp(multiplet) < 1.0e-10, label


def test_gamma_nscf_preserves_degeneracies_and_bands_finds_irreps(
    tmp_path, monkeypatch,
):
    """NSCF uses QE's tighter default ethr and bands rotates translations."""
    from qepy_pw.pp.band_data import read_saved_bands
    from qepy_pw.pp.bands import _read_wavefunctions, classify_irreps
    from qepy_pw.pw.save import resolve_save_directory, write_qe_save
    import qepy_pw.pw.scf as scf_module

    case = CASES["scf_gamma"]
    source = input_path(case)
    pseudo_dir = source.parents[1] / "pseudo"
    scf_pw = read_pw_input(source)
    scf_pw.control.update({
        "prefix": "gamma-nscf",
        "outdir": str(tmp_path),
        "pseudo_dir": str(pseudo_dir),
        "disk_io": "medium",
        "tstress": False,
    })
    scf_pw.system["nbnd"] = 12
    scf_pw.electrons.update({
        "conv_thr": 1.0e-10,
        "diago_thr_init": 1.0e-10,
    })
    write_qe_save(scf_pw, run_scf(scf_pw))

    nscf_pw = read_pw_input(source)
    nscf_pw.control.update({
        "calculation": "nscf",
        "prefix": "gamma-nscf",
        "outdir": str(tmp_path),
        "pseudo_dir": str(pseudo_dir),
        "disk_io": "medium",
        "tstress": False,
    })
    nscf_pw.system["nbnd"] = 12
    # Exercise QE's NSCF defaults, independently of the tight SCF settings.
    nscf_pw.electrons.clear()

    thresholds = []
    symmetry_applications = 0
    original_davidson = scf_module.davidson
    original_apply = ReciprocalDensitySymmetrizer.apply

    def tracked_davidson(*args, **kwargs):
        thresholds.append(kwargs["tolerance"])
        solution = original_davidson(*args, **kwargs)
        if len(thresholds) == 1:
            # Exercise QE's outer c_bands retry independently of whether
            # this small Gamma fixture happens to converge in one call.
            return replace(
                solution,
                converged=False,
                number_unconverged=nscf_pw.system["nbnd"],
            )
        return solution

    def tracked_apply(self, density):
        nonlocal symmetry_applications
        symmetry_applications += 1
        return original_apply(self, density)

    monkeypatch.setattr(scf_module, "davidson", tracked_davidson)
    monkeypatch.setattr(ReciprocalDensitySymmetrizer, "apply", tracked_apply)
    result = run_scf(nscf_pw)

    assert len(thresholds) >= 2
    # QE prints ethr in Ry; the solver receives Hartree.
    np.testing.assert_allclose(thresholds, 0.5 * 0.1e-6 / 8.0)
    assert symmetry_applications == 1
    eigenvalues = result.eigenvalues_ha[0]
    for multiplet in (eigenvalues[1:4], eigenvalues[4:7], eigenvalues[9:11]):
        assert np.ptp(multiplet) < 1.0e-8

    write_qe_save(nscf_pw, result)
    save = resolve_save_directory(nscf_pw)
    data = read_saved_bands("gamma-nscf", str(tmp_path))
    wavefunctions = _read_wavefunctions(save, data.nks)
    irreps = classify_irreps(data, wavefunctions, save)
    # O_h rows: A1g, T2g, T1u, A2u, A1g, Eu.
    np.testing.assert_array_equal(
        irreps[0, :11],
        [1, 5, 5, 5, 9, 9, 9, 7, 1, 8, 8],
    )


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


def test_high_verbosity_force_and_stress_decompositions_sum_to_totals():
    from qepy_pw.output import format_footer
    from qepy_pw.scf import run_scf

    case = CASES["scf_baseline"]
    pw = read_pw_input(input_path(case))
    pw.control["pseudo_dir"] = str(input_path(case).parents[1] / "pseudo")
    pw.control["verbosity"] = "high"
    pw.control["tprnfor"] = True
    result = run_scf(pw)

    assert result.force_terms is not None
    assert result.stress_terms is not None
    force_sum = sum((
        result.force_terms.nonlocal_ha_per_bohr,
        result.force_terms.ionic_ha_per_bohr,
        result.force_terms.local_ha_per_bohr,
        result.force_terms.core_correction_ha_per_bohr,
        result.force_terms.scf_correction_ha_per_bohr,
    ))
    stress_sum = sum((
        result.stress_terms.kinetic_ha_per_bohr3,
        result.stress_terms.local_ha_per_bohr3,
        result.stress_terms.nonlocal_ha_per_bohr3,
        result.stress_terms.hartree_ha_per_bohr3,
        result.stress_terms.xc_ha_per_bohr3,
        result.stress_terms.core_correction_ha_per_bohr3,
        result.stress_terms.ewald_ha_per_bohr3,
    ))
    np.testing.assert_allclose(force_sum, result.forces_ha_per_bohr)
    np.testing.assert_allclose(
        stress_sum, result.stress_ha_per_bohr3, atol=1.0e-18
    )
    output = format_footer(pw, result)
    assert "The non-local contrib.  to forces" in output
    assert "The SCF correction term to forces" in output
    assert "kinetic stress (kbar)" in output
    assert "ewald   stress (kbar)" in output

    pw.control["calculation"] = "nscf"
    nscf_output = format_footer(pw, result)
    assert "End of band structure calculation" in nscf_output
    assert "convergence has been achieved" not in nscf_output
