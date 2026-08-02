from pathlib import Path
from contextlib import contextmanager
import copy

import numpy as np
import pytest

import qepy_pw.scf as scf_module
from qepy_pw.basis import LocalPotentialWorkspace
from qepy_pw.errors import QEInputError, UnsupportedFeatureError
from qepy_pw.input import Atom, KPoint, Species, read_pw_input
from qepy_pw.output import format_output
from qepy_pw.scf import (
    SCFSetup,
    _density_error_ry,
    _hartree,
    _iteration_energy,
    _xc_terms,
    run_scf,
)
from qepy_pw.xc import pz81_unpolarized


def test_atomic_trial_randomization_matches_qe_multiplicative_form():
    trials = np.zeros((12, 3), dtype=complex)
    trials[:4, 0] = 1.0
    trials[4:8, 1] = 2.0
    trials[8:, 2] = 3.0
    kinetic = np.linspace(0.0, 6.0, len(trials))

    randomized = scf_module._randomize_atomic_trials(
        trials,
        kinetic,
        scf_module._QERandom(),
    )

    assert randomized.shape == trials.shape
    assert np.all(randomized[trials == 0.0] == 0.0)
    relative_change = np.linalg.norm(
        randomized - trials, axis=0
    ) / np.linalg.norm(trials, axis=0)
    assert np.all(relative_change < 0.11)


def test_qe_random_batch_matches_scalar_stream_exactly():
    batch_stream = scf_module._QERandom(17)
    scalar_stream = scf_module._QERandom(17)
    first, second = batch_stream.pairs_by_band(11, 4)
    scalar = np.array(
        [scalar_stream.random() for _ in range(2 * 11 * 4)]
    )
    expected_first = scalar[0::2].reshape(4, 11).T
    expected_second = scalar[1::2].reshape(4, 11).T
    assert np.array_equal(first, expected_first)
    assert np.array_equal(second, expected_second)
    assert batch_stream.random() == scalar_stream.random()


def test_xc_functional_family_recognizes_qe_and_rejects_pbe_variants():
    assert scf_module._functional_family("PBE") == "pbe"
    assert scf_module._functional_family("SLA PW PBX PBC") == "pbe"
    assert scf_module._functional_family("SLA PZ NOGX NOGC") == "pz"
    assert scf_module._functional_family("PBEsol") == "unsupported"


def test_h2_end_to_end_converges_and_has_qe_markers():
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    result = run_scf(pw)
    output = format_output(pw, result)
    assert result.converged
    assert np.isfinite(result.total_energy_ha)
    assert abs(np.mean(result.density) * pw.volume - 2.0) < 1e-10
    assert "!    total energy" in output
    assert "JOB DONE." in output
    assert "diagonalization           = david" in output
    assert "number of Kohn-Sham states" in output
    assert "crystal axes:" in output
    assert "Davidson diagonalization with overlap" in output
    assert "ethr =" in output
    assert "highest occupied level (ev)" in output
    assert "The total energy is the sum" in output
    assert "one-electron contribution" in output
    assert "convergence has been achieved" in output
    assert "Real-time Memory Report at c_bands" in output
    assert "Called by electrons:" in output
    assert "Called by c_bands:" in output
    assert "Called by h_psi:" in output
    assert "PWSCF        :" in output
    assert result.timings["electrons"].calls == 1
    assert len(result.wavefunctions) == len(pw.kpoints)
    assert len(result.wavefunction_miller_indices) == len(pw.kpoints)
    for coefficients, miller in zip(
        result.wavefunctions, result.wavefunction_miller_indices
    ):
        assert coefficients.shape[0] == len(miller)
        np.testing.assert_allclose(
            coefficients.conj().T @ coefficients,
            np.eye(coefficients.shape[1]),
            atol=2.0e-11,
        )
    assert result.peak_rss_bytes_per_rank > 0


def test_h2_pbe_end_to_end_uses_gga_energy_and_potential():
    root = Path(__file__).parents[1]
    lda_input = read_pw_input(root / "examples" / "h2.scf.in")
    pbe_input = read_pw_input(root / "examples" / "h2.scf.in")
    pbe_input.system["input_dft"] = "PBE"

    lda = run_scf(lda_input)
    pbe = run_scf(pbe_input)
    output = format_output(pbe_input, pbe)

    assert pbe.converged
    assert np.isfinite(pbe.total_energy_ha)
    assert not np.isclose(pbe.total_energy_ha, lda.total_energy_ha)
    assert "Exchange-correlation      =  SLA  PW   PBX  PBC" in output


def test_disk_io_none_does_not_collect_final_wavefunctions():
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.control["disk_io"] = "none"
    result = run_scf(pw)
    assert result.converged
    assert result.wavefunctions == []
    assert result.wavefunction_miller_indices == []


def test_h2_fermi_dirac_smearing_converges_with_fractional_occupations():
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.system["occupations"] = "smearing"
    pw.system["smearing"] = "fermi-dirac"
    pw.system["degauss"] = 0.2
    pw.system["nbnd"] = 4
    pw.control["tprnfor"] = True
    pw.control["tstress"] = True
    result = run_scf(pw)
    output = format_output(pw, result)

    assert result.converged
    assert result.fermi_energy_ha is not None
    assert len(result.occupations) == len(pw.kpoints)
    electron_count = sum(
        point.weight * np.sum(occupations)
        for point, occupations in zip(pw.kpoints, result.occupations)
    )
    assert electron_count == pytest.approx(2.0, abs=2.0e-11)
    assert any(
        np.any((occupations > 1.0e-8) & (occupations < 2.0 - 1.0e-8))
        for occupations in result.occupations
    )
    assert result.energy_terms is not None
    assert result.forces_ha_per_bohr is not None
    assert np.all(np.isfinite(result.forces_ha_per_bohr))
    assert result.stress_ha_per_bohr3 is not None
    assert np.all(np.isfinite(result.stress_ha_per_bohr3))
    terms = result.energy_terms
    assert result.total_energy_ha == pytest.approx(
        terms.one_electron_ha
        + terms.hartree_ha
        + terms.xc_ha
        + terms.ewald_ha
        + terms.descf_ha
        + terms.smearing_ha,
        abs=1.0e-12,
    )
    assert "the Fermi energy is" in output
    assert "smearing contrib. (-TS)" in output
    assert "internal energy E=F+TS" in output


def test_hellmann_feynman_force_and_stress_match_scf_energy_derivatives():
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.electrons["conv_thr"] = 1.0e-10
    pw.control["tprnfor"] = True
    pw.control["tstress"] = True
    result = run_scf(pw)

    displacement = 1.0e-3
    displaced_energies = []
    for sign in (1.0, -1.0):
        displaced = copy.deepcopy(pw)
        displaced.control["tprnfor"] = False
        displaced.control["tstress"] = False
        position = displaced.atoms[0].position.copy()
        position[0] += sign * displacement
        displaced.atoms[0] = Atom(displaced.atoms[0].label, position)
        displaced_energies.append(run_scf(displaced).total_energy_ha)
    finite_force = -(
        displaced_energies[0] - displaced_energies[1]
    ) / (2.0 * displacement)

    strain = 2.0e-4
    strained_energies = []
    for sign in (1.0, -1.0):
        strained = copy.deepcopy(pw)
        strained.control["tprnfor"] = False
        strained.control["tstress"] = False
        deformation = np.eye(3)
        deformation[0, 0] += sign * strain
        strained.lattice = strained.lattice @ deformation
        strained.atoms = [
            Atom(atom.label, atom.position @ deformation)
            for atom in strained.atoms
        ]
        strained_energies.append(run_scf(strained).total_energy_ha)
    finite_stress = -(
        strained_energies[0] - strained_energies[1]
    ) / (2.0 * strain * pw.volume)

    assert result.forces_ha_per_bohr is not None
    assert result.stress_ha_per_bohr3 is not None
    assert np.isclose(
        result.forces_ha_per_bohr[0, 0], finite_force, atol=3.0e-7
    )
    assert np.isclose(
        result.stress_ha_per_bohr3[0, 0], finite_stress, atol=1.0e-9
    )
    output = format_output(pw, result)
    assert "Forces acting on atoms (cartesian axes, Ry/au)" in output
    assert "total   stress  (Ry/bohr**3)" in output


def _finite_xx_stress(pw, strain=2.0e-4):
    energies = []
    for sign in (1.0, -1.0):
        strained = copy.deepcopy(pw)
        strained.control["tstress"] = False
        deformation = np.eye(3)
        deformation[0, 0] += sign * strain
        strained.lattice = strained.lattice @ deformation
        strained.atoms = [
            Atom(atom.label, atom.position @ deformation)
            for atom in strained.atoms
        ]
        energies.append(run_scf(strained).total_energy_ha)
    return -(energies[0] - energies[1]) / (
        2.0 * strain * pw.volume
    )


def test_pbe_analytic_stress_matches_scf_energy_derivative():
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.system["input_dft"] = "PBE"
    pw.electrons["conv_thr"] = 1.0e-10
    pw.control["tstress"] = True

    result = run_scf(pw)

    assert result.stress_ha_per_bohr3 is not None
    assert np.isclose(
        result.stress_ha_per_bohr3[0, 0],
        _finite_xx_stress(pw),
        atol=1.0e-9,
    )


def test_nlcc_analytic_stress_matches_scf_energy_derivative():
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.control["pseudo_dir"] = str(root / "tests" / "data")
    pw.control["tstress"] = True
    pw.system.update(
        nat=1,
        ntyp=1,
        nbnd=1,
        input_dft="PZ",
    )
    pw.electrons["conv_thr"] = 1.0e-10
    pw.species = [Species("He", 4.0, "He.local-nc.UPF")]
    pw.atoms = [Atom("He", np.array([6.0, 6.0, 6.0]))]

    result = run_scf(pw)

    assert result.stress_ha_per_bohr3 is not None
    assert np.isclose(
        result.stress_ha_per_bohr3[0, 0],
        _finite_xx_stress(pw),
        atol=3.0e-9,
    )


def test_real_qe_si_norm_conserving_upf_converges():
    root = Path(__file__).parents[1]
    pseudo = (
        root
        / "quantum-espresso"
        / "atomic"
        / "examples"
        / "pseudo-LDA-0.5"
        / "reference"
        / "Si.LDA.0.5.UPF"
    )
    if not pseudo.exists():
        pytest.skip("QE source pseudopotential is not checked out")
    pw = read_pw_input(root / "examples" / "si-nc.scf.in")
    result = run_scf(pw)
    assert result.converged
    assert len(result.iterations) <= 60
    assert np.isfinite(result.total_energy_ha)
    assert abs(np.mean(result.density) * pw.volume - 8.0) < 1e-10

def test_nlcc_xc_energy_uses_total_density_but_potential_uses_valence():
    valence = np.array([0.02, 0.04, 0.08])
    core = np.array([0.20, 0.10, 0.05])
    potential, energy, double_counting = _xc_terms(valence, core, volume=12.0)
    assert np.all(potential < 0.0)
    assert energy < 0.0
    assert np.isclose(double_counting, 12.0 * np.mean(valence * potential))


def test_numba_pz81_matches_numpy_piecewise_branches():
    pytest.importorskip("numba")
    density = np.geomspace(1.0e-12, 20.0, 257).reshape(257, 1, 1)
    expected_epsilon, expected_potential = pz81_unpolarized(density)
    actual_epsilon, actual_potential = pz81_unpolarized(
        density, use_numba=True
    )
    assert np.allclose(actual_epsilon, expected_epsilon, atol=2.0e-15)
    assert np.allclose(actual_potential, expected_potential, atol=2.0e-15)


def test_qe_density_error_matches_one_fourier_mode():
    shape = (4, 4, 4)
    grid = np.indices(shape)
    amplitude = 0.03
    density_in = np.zeros(shape)
    density_out = amplitude * np.cos(
        2.0 * np.pi * grid[0] / shape[0]
    )
    volume = 17.0
    error_ry = _density_error_ry(
        density_in,
        density_out,
        2.0 * np.pi * np.eye(3),
        volume,
    )
    assert np.isclose(
        error_ry, volume * amplitude**2 / (2.0 * np.pi)
    )


def test_unconverged_iteration_energy_contains_qe_descf_correction():
    shape = (4, 4, 4)
    grid = np.indices(shape)
    rho_out = 0.12 + 0.02 * np.cos(
        2.0 * np.pi * grid[0] / shape[0]
    )
    rho_mixed = 0.12 + 0.01 * np.cos(
        2.0 * np.pi * grid[0] / shape[0]
    )
    input_hxc = -0.3 + 0.04 * np.cos(
        2.0 * np.pi * grid[1] / shape[1]
    )
    reciprocal = 2.0 * np.pi * np.eye(3)
    volume = 15.0
    band_energy = -1.7
    e_ion = -0.4
    energy, terms = _iteration_energy(
        band_energy,
        e_ion,
        rho_out,
        rho_mixed,
        input_hxc,
        np.zeros(shape),
        reciprocal,
        volume,
    )
    mixed_g = np.fft.fftn(rho_mixed) / np.prod(shape)
    vh_g, eh_density = _hartree(mixed_g, reciprocal)
    vh = np.real(np.fft.ifftn(vh_g * np.prod(shape)))
    vxc, exc, _ = _xc_terms(
        rho_mixed, np.zeros(shape), volume
    )
    expected_descf = -volume * np.mean(
        (rho_mixed - rho_out) * (vh + vxc)
    )
    expected = (
        band_energy
        - volume * np.mean(rho_out * input_hxc)
        + volume * eh_density
        + exc
        + e_ion
        + expected_descf
    )
    assert np.isclose(terms.descf_ha, expected_descf)
    assert np.isclose(
        terms.one_electron_ha,
        band_energy - volume * np.mean(rho_out * input_hxc),
    )
    assert np.isclose(energy, expected)


def test_real_qe_fe_nlcc_upf_converges():
    root = Path(__file__).parents[1]
    pseudo = root / "quantum-espresso" / "pseudo" / "Fe.pz-n-nc.UPF"
    if not pseudo.exists():
        pytest.skip("QE Fe NLCC pseudopotential is not checked out")
    pw = read_pw_input(root / "examples" / "fe-nlcc.scf.in")
    result = run_scf(pw)
    output = format_output(pw, result)
    assert result.converged
    assert np.isfinite(result.total_energy_ha)
    assert "Nonlinear core correction = present" in output
    assert abs(np.mean(result.density) * pw.volume - 8.0) < 1e-10


def test_unreduced_kpoint_limit_reports_setup_before_failing():
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.electrons["diagonalization"] = "dense"
    pw.kpoints = [KPoint(np.zeros(3), 1.0 / 65.0) for _ in range(65)]
    events: list[tuple[str, object]] = []
    with pytest.raises(UnsupportedFeatureError, match="65 active k points"):
        run_scf(pw, progress=lambda kind, payload: events.append((kind, payload)))
    assert events and events[0][0] == "setup"
    assert isinstance(events[0][1], SCFSetup)
    assert events[0][1].kpoints == 65
    assert events[0][1].diagonalization == "dense"
    assert events[0][1].estimated_persistent_bytes_per_rank > 0
    assert events[0][1].estimated_peak_workspace_bytes_per_rank > 0


def test_invalid_python_blas_thread_control_is_rejected():
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.electrons["py_blas_threads"] = 0
    with pytest.raises(QEInputError, match="py_blas_threads"):
        run_scf(pw)


def test_invalid_projector_cache_control_is_rejected():
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.electrons["py_cache_projectors"] = "yes"
    with pytest.raises(QEInputError, match="py_cache_projectors"):
        run_scf(pw)


def test_numerical_library_threads_default_to_one(monkeypatch):
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    observed_limits: list[int] = []

    @contextmanager
    def fake_limits(*, limits: int):
        observed_limits.append(int(limits))
        yield

    sentinel = object()
    monkeypatch.setattr(scf_module, "threadpool_limits", fake_limits)
    monkeypatch.setattr(
        scf_module, "_run_scf", lambda pw, progress, mpi: sentinel
    )
    assert scf_module.run_scf(pw) is sentinel
    assert observed_limits == [1]


def test_implicit_diagonalization_threshold_tracks_scf_accuracy():
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.electrons["conv_thr"] = 2.0e-8
    events: list[tuple[str, object]] = []
    result = run_scf(
        pw,
        progress=lambda kind, payload: events.append((kind, payload)),
    )
    assert result.converged
    first_step = next(
        payload for kind, payload in events if kind == "iteration"
    )
    assert first_step.davidson_threshold_ha <= 5.0e-3
    second_step = [
        payload for kind, payload in events if kind == "iteration"
    ][1]
    expected_second_ha = min(
        5.0e-3,
        0.1 * result.iterations[0].estimated_accuracy_ha / 2.0,
    )
    assert second_step.davidson_threshold_ha == pytest.approx(
        expected_second_ha
    )


def test_scf_defaults_to_qe_davidson_controls(monkeypatch):
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    observed_controls: list[tuple[float | None, int, int]] = []
    original = scf_module.davidson

    def observing_davidson(*args, **kwargs):
        observed_controls.append(
            (
                kwargs["residual_energy_scale"],
                kwargs["subspace_multiplier"],
                kwargs["max_iterations"],
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(scf_module, "davidson", observing_davidson)
    result = run_scf(pw)
    assert result.converged
    assert observed_controls
    assert set(observed_controls) == {(None, 2, 20)}


def test_energy_scaled_safeguard_remains_available_as_opt_in(monkeypatch):
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.electrons["py_davidson_residual_energy_scale"] = 10.0
    observed_scales: list[float | None] = []
    original = scf_module.davidson

    def observing_davidson(*args, **kwargs):
        observed_scales.append(kwargs["residual_energy_scale"])
        return original(*args, **kwargs)

    monkeypatch.setattr(scf_module, "davidson", observing_davidson)
    result = run_scf(pw)
    assert result.converged
    assert observed_scales
    assert set(observed_scales) == {10.0}


def test_scf_rebuilds_nonlocal_projectors_for_each_kpoint_solve(monkeypatch):
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    calls = 0
    original = scf_module._nonlocal_projector_terms

    def counting_projectors(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        scf_module,
        "_nonlocal_projector_terms",
        counting_projectors,
    )
    events: list[tuple[str, object]] = []
    result = run_scf(
        pw,
        progress=lambda kind, payload: events.append((kind, payload)),
    )
    assert result.converged
    assert calls == result.timings["init_us_2"].calls
    assert calls >= len(pw.kpoints) * len(result.iterations)


def test_removed_projector_cache_control_is_rejected():
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    pw.electrons["py_cache_projectors"] = False
    with pytest.raises(QEInputError, match="has been removed"):
        run_scf(pw)


def test_scf_reuses_real_potential_and_reciprocal_geometry(monkeypatch):
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "h2.scf.in")
    prepared = 0
    geometry_ids: list[int] = []
    original_prepare = LocalPotentialWorkspace.prepare_potential
    original_hartree = scf_module._hartree

    def counting_prepare(self, potential_g):
        nonlocal prepared
        prepared += 1
        return original_prepare(self, potential_g)

    def observing_hartree(*args, **kwargs):
        geometry = (
            kwargs.get("geometry")
            if "geometry" in kwargs
            else (args[3] if len(args) > 3 else None)
        )
        assert geometry is not None
        geometry_ids.append(id(geometry))
        return original_hartree(*args, **kwargs)

    monkeypatch.setattr(
        LocalPotentialWorkspace, "prepare_potential", counting_prepare
    )
    monkeypatch.setattr(scf_module, "_hartree", observing_hartree)
    result = run_scf(pw)
    assert result.converged
    assert prepared == 0
    assert geometry_ids and len(set(geometry_ids)) == 1


def test_symmetry_reduced_si_scf_converges():
    root = Path(__file__).parents[1]
    pw = read_pw_input(root / "examples" / "si-symmetry.scf.in")
    result = run_scf(pw)
    assert pw.full_kpoint_count == 8
    assert len(pw.kpoints) == 3
    assert result.converged
    assert abs(np.mean(result.density) * pw.volume - 8.0) < 1e-10
