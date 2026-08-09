from __future__ import annotations

import io

import pytest

from qepy_pw.errors import (
    QEInputError,
    QEWarning,
    UnsupportedFeatureError,
    format_qe_error,
    format_qe_warning,
)
from qepy_pw.input import read_pw_input


def _input(system_extra: str = "", electrons_extra: str = "", cards: str = "") -> str:
    if not cards:
        cards = """\
ATOMIC_SPECIES
Si 28.085 Si.UPF
ATOMIC_POSITIONS crystal
Si 0.0 0.0 0.0
K_POINTS gamma
"""
    return f"""\
&CONTROL
  calculation = 'scf'
/
&SYSTEM
  ibrav = 1, celldm(1) = 10.0,
  nat = 1, ntyp = 1, ecutwfc = 20.0
  {system_extra}
/
&ELECTRONS
  {electrons_extra}
/
{cards}
"""


def _read(text: str):
    return read_pw_input(io.StringIO(text))


def test_qe_error_and_warning_rendering() -> None:
    error = QEInputError("ecutwfc not set", routine="set_cutoff")
    rendered = format_qe_error(error)
    assert "Error in routine set_cutoff (1):" in rendered
    assert "ecutwfc not set" in rendered
    assert "stopping ..." in rendered
    assert format_qe_warning(QEWarning("setup", "check")) == (
        "     Message from routine setup:\n     check\n"
    )


def test_cutoff_warning_and_silent_fixed_degauss_reset_match_qe() -> None:
    pw = _read(_input("ecutrho = 60.0, degauss = 0.01"))
    messages = [(item.routine, item.message) for item in pw.warnings]
    assert ("set_cutoff", "ecutrho < 4*ecutwfc, are you sure?") in messages
    assert all(routine != "set_occupations" for routine, _message in messages)
    assert pw.system["degauss"] == 0.0


def test_missing_cutoff_uses_qe_routine_and_message() -> None:
    with pytest.raises(QEInputError) as caught:
        _read(_input().replace("ecutwfc = 20.0", "ecutwfc = 0.0"))
    assert caught.value.routine == "set_cutoff"
    assert str(caught.value) == "ecutwfc not set"

    with pytest.raises(QEInputError) as caught:
        _read(_input().replace("ecutwfc = 20.0", "ecutwfc = 'large'"))
    assert caught.value.routine == "read_namelists"
    assert "invalid value for 'ecutwfc'" in str(caught.value)


def test_smearing_diagnostics_match_qe() -> None:
    with pytest.raises(QEInputError) as caught:
        _read(_input("occupations='smearing', degauss=0.0"))
    assert caught.value.routine == "set_occupations"
    assert str(caught.value) == (
        "smearing requires a value for gaussian broadening (degauss)"
    )

    with pytest.raises(QEInputError, match="smearing mystery unknown"):
        _read(_input("occupations='smearing', degauss=0.01, smearing='mystery'"))


def test_invalid_starting_options_warn_and_use_qe_defaults() -> None:
    pw = _read(_input(electrons_extra="startingpot='bad', startingwfc='bad'"))
    assert pw.electrons["startingpot"] == "atomic"
    assert pw.electrons["startingwfc"] == "atomic+random"
    assert [item.message for item in pw.warnings] == [
        "wrong startingpot: use default (1)",
        "wrong startingwfc: use default (atomic+random)",
    ]


@pytest.mark.parametrize("calculation", ["nscf", "bands"])
def test_fixed_potential_calculations_select_saved_density(
    calculation: str,
) -> None:
    pw = _read(
        _input().replace(
            "calculation = 'scf'", f"calculation = '{calculation}'"
        )
    )
    assert pw.control["calculation"] == calculation
    assert pw.control["disk_io"] == "medium"
    assert pw.electrons["startingpot"] == "file"
    assert pw.electrons.get("startingwfc", "atomic+random") == "atomic+random"


def test_disk_io_medium_and_known_levels_are_accepted() -> None:
    assert _read(_input()).control["disk_io"] == "low"
    for level in ("none", "low", "medium", "high"):
        text = _input().replace(
            "calculation = 'scf'",
            f"calculation = 'scf', disk_io = '{level}'",
        )
        assert _read(text).control["disk_io"] == level

    with pytest.raises(QEInputError, match="unknown disk_io mystery"):
        _read(
            _input().replace(
                "calculation = 'scf'",
                "calculation = 'scf', disk_io = 'mystery'",
            )
        )


def test_wfcdir_is_an_implemented_control_variable() -> None:
    text = _input().replace(
        "calculation = 'scf'",
        "calculation = 'scf', wfcdir = './wavefunctions'",
    )
    assert _read(text).control["wfcdir"] == "./wavefunctions"


def test_unported_qe_variable_is_never_silently_ignored() -> None:
    with pytest.raises(UnsupportedFeatureError) as caught:
        _read(_input(system_extra="assume_isolated='mt'"))
    assert caught.value.routine == "iosys"
    assert str(caught.value) == (
        "&SYSTEM variable 'assume_isolated' is not implemented in PWSCF-PY"
    )

    with pytest.raises(QEInputError) as caught:
        _read(_input(system_extra="ecutwcf=20.0"))
    assert caught.value.routine == "read_namelists"
    assert "unknown variable 'ecutwcf'" in str(caught.value)


def test_duplicate_species_and_overlapping_atoms_match_qe() -> None:
    duplicate_species = """\
ATOMIC_SPECIES
Si 28.085 Si.UPF
Si 28.085 Si2.UPF
ATOMIC_POSITIONS crystal
Si 0 0 0
K_POINTS gamma
"""
    text = _input(cards=duplicate_species).replace("ntyp = 1", "ntyp = 2")
    with pytest.raises(QEInputError) as caught:
        _read(text)
    assert caught.value.routine == "card_atomic_species"
    assert str(caught.value) == "two occurrences of the same atomic label"

    overlapping = """\
ATOMIC_SPECIES
Si 28.085 Si.UPF
ATOMIC_POSITIONS crystal
Si 0 0 0
Si 0 0 0
K_POINTS gamma
"""
    text = _input(cards=overlapping).replace("nat = 1", "nat = 2")
    with pytest.raises(QEInputError) as caught:
        _read(text)
    assert caught.value.routine == "check_atoms"
    assert "atoms #" in str(caught.value)
    assert "overlap!" in str(caught.value)


def test_missing_card_units_emit_qe_deprecation_messages() -> None:
    cards = """\
ATOMIC_SPECIES
Si 28.085 Si.UPF
ATOMIC_POSITIONS
Si 0 0 0
K_POINTS gamma
"""
    pw = _read(_input(cards=cards))
    assert [item.message for item in pw.warnings] == [
        "DEPRECATED: no units specified in ATOMIC_POSITIONS card",
        "ATOMIC_POSITIONS: units set to alat",
    ]
