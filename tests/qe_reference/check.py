"""Generate and validate Python outputs for QE-derived SCF inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"
UPSTREAM = ROOT / "upstream"
REFERENCE = ROOT / "reference"
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text, flags=re.MULTILINE)
    if not values:
        return None
    value = values[-1]
    if isinstance(value, tuple):
        value = value[-1]
    return float(str(value).replace("D", "E").replace("d", "e"))


def extract_pw(text: str) -> dict[str, Any]:
    """Extract stable physical values from a QE-shaped output stream."""
    values: dict[str, Any] = {}
    patterns = {
        "energy_ry": rf"^!\s+total energy\s+=\s+({_FLOAT})\s+Ry",
        "iterations": r"convergence has been achieved in\s+(\d+) iterations",
        "total_force_ry_bohr": rf"Total force\s+=\s+({_FLOAT})",
        "pressure_kbar": rf"P=\s*({_FLOAT})",
        "fermi_ev": rf"^\s*the Fermi energy is\s+({_FLOAT})\s+ev",
        "homo_ev": rf"^\s*highest occupied level \(ev\):\s+({_FLOAT})",
        "num_kpoints": r"number of k points=\s*(\d+)",
    }
    for name, pattern in patterns.items():
        value = _last_float(pattern, text)
        if value is not None:
            values[name] = value

    pairs = re.findall(
        rf"highest occupied, lowest unoccupied level \(ev\):\s+"
        rf"({_FLOAT})\s+({_FLOAT})",
        text,
    )
    if pairs:
        values["homo_lumo_ev"] = [float(value) for value in pairs[-1]]

    bands: list[float] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "bands (ev):" not in line:
            continue
        point_values: list[float] = []
        cursor = index + 1
        while cursor < len(lines) and len(point_values) < 5:
            stripped = lines[cursor].strip()
            cursor += 1
            if not stripped:
                continue
            if re.search(r"[A-Za-z]", stripped):
                break
            try:
                point_values.extend(float(value) for value in stripped.split())
            except ValueError:
                break
        bands.extend(point_values[:5])
    if bands:
        values["bands_ev"] = bands
    diagonalization_iterations = re.findall(
        rf"avg # of iterations\s*=\s*({_FLOAT})", text
    )
    if diagonalization_iterations:
        values["diagonalization_iterations"] = [
            float(value.replace("D", "E").replace("d", "e"))
            for value in diagonalization_iterations
        ]
    return values


def input_path(case: dict[str, Any]) -> Path:
    return UPSTREAM / case["suite"] / case["input"]


def reference_path(case: dict[str, Any]) -> Path:
    return REFERENCE / case["suite"] / f"{case['input']}.out"


def run_case(case: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    from qepy_pw.input import read_pw_input
    from qepy_pw.output import (
        format_footer,
        format_header,
        format_progress,
    )
    from qepy_pw.scf import run_scf

    pw = read_pw_input(input_path(case))
    pw.control["pseudo_dir"] = str(UPSTREAM / "pseudo")
    pw.control["disk_io"] = "none"
    chunks = [format_header(pw)]

    def report(kind, payload) -> None:
        chunks.append(format_progress(kind, payload))

    result = run_scf(pw, progress=report)
    chunks.append(format_footer(pw, result))
    output = "".join(chunks).replace(str(ROOT.parents[1]), ".")
    return output, extract_pw(output)


def compare_values(
    expected: dict[str, Any],
    actual: dict[str, Any],
    tolerances: dict[str, dict[str, float | None]],
) -> tuple[bool, dict[str, Any]]:
    comparisons: dict[str, Any] = {}
    passed = True
    for name, tolerance in tolerances.items():
        if name not in expected or name not in actual:
            continue
        reference = np.atleast_1d(np.asarray(expected[name], dtype=float))
        observed = np.atleast_1d(np.asarray(actual[name], dtype=float))
        if reference.shape != observed.shape:
            comparisons[name] = {"passed": False, "reason": "shape mismatch"}
            passed = False
            continue
        difference = np.abs(observed - reference)
        limit = float(tolerance["absolute"] or 0.0)
        field_passed = bool(np.all(difference <= limit))
        comparisons[name] = {
            "passed": field_passed,
            "max_absolute_difference": float(np.max(difference)),
        }
        passed = passed and field_passed
    return passed, comparisons


def verify_pseudos(manifest: dict[str, Any]) -> None:
    for name, metadata in manifest["pseudopotentials"].items():
        data = (UPSTREAM / "pseudo" / name).read_bytes()
        if "fixture_sha256" in metadata:
            algorithm = "sha256"
            digest = hashlib.sha256(data).hexdigest()
            expected = metadata["fixture_sha256"]
        else:
            # Version-1 manifests recorded MD5. Retain support while old
            # fixture sets are being migrated, but require SHA-256 for new
            # provenance records.
            algorithm = "md5"
            digest = hashlib.md5(data).hexdigest()
            expected = metadata["fixture_md5"]
        if digest != expected:
            raise RuntimeError(
                f"{algorithm} checksum mismatch for {name}: {digest}"
            )


def verify_saved_fixture(manifest: dict[str, Any]) -> None:
    metadata = manifest["saved_fixture"]
    directory = UPSTREAM / metadata["directory"]
    for name, expected in metadata["files_sha256"].items():
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(
                f"sha256 checksum mismatch for {directory.name}/{name}: "
                f"{digest}"
            )


def generate_references(case_ids: set[str] | None = None) -> None:
    manifest = load_manifest()
    verify_pseudos(manifest)
    verify_saved_fixture(manifest)
    for case in manifest["cases"]:
        if case_ids is not None and case["id"] not in case_ids:
            continue
        output, _ = run_case(case)
        target = reference_path(case)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Keep checked-in snapshots friendly to whitespace linters.  The
        # executable output itself intentionally retains QE's trailing A60
        # padding on the termination line and is covered byte-for-byte by
        # ``test_qe_whitespace``.
        snapshot = "\n".join(line.rstrip() for line in output.splitlines())
        if output.endswith("\n"):
            snapshot += "\n"
        target.write_text(snapshot, encoding="utf-8")
        print(f"updated {target.relative_to(ROOT)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="regenerate only this manifest case (repeatable)",
    )
    arguments = parser.parse_args(argv)
    generate_references(set(arguments.cases) if arguments.cases else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
