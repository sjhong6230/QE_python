from __future__ import annotations

from dataclasses import replace
import io
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
from scipy.special import spherical_jn

from qepy_pw.errors import QEInputError, UnsupportedFeatureError
from qepy_pw.occupations import smearing_density, wgauss
from qepy_pw.pp.namelist import parse_namelist
from qepy_pw.pp.pp import (
    PlotGrid,
    _formatted_fortran_e,
    _fortran_e,
    _fourier_regular_values,
    _fourier_values,
    _atomic_density,
    _atomic_magnetization,
    _miller_and_g,
    _qe_local_dos_weights,
    _read_density,
    _spherical_average_values,
    _saved_smearing_metadata,
    _stm_window_weights,
    _spinor_grid_quantity,
    _symmetrize_wave_field,
    _wave_density_sum,
    _wave_grid,
    _read_wavefunction,
    _write_standard_e_values,
    _write_values,
    combine_plot_files,
    extract_plot_grids,
    read_plot_file,
    read_saved_pp,
    run_pp,
    write_output_plot,
    write_plot_file,
)


@pytest.fixture(scope="module")
def saved_state():
    outdir = (
        Path(__file__).resolve().parents[1]
        / "qe_reference"
        / "upstream"
        / "pseudo"
    )
    return read_saved_pp("pwscf", str(outdir))


def test_pp_reads_scalar_nc_save_and_normalizes_charge(saved_state) -> None:
    assert saved_state.shape == saved_state.density.shape
    assert saved_state.energies_ha.shape == saved_state.occupations.shape
    assert np.sum(saved_state.weights) == pytest.approx(1.0)
    charge = np.mean(saved_state.density) * saved_state.volume
    assert charge == pytest.approx(8.0, abs=1.0e-8)
    np.testing.assert_allclose(
        _symmetrize_wave_field(saved_state, saved_state.density),
        saved_state.density,
        atol=1.0e-12,
    )


def test_pp_reads_qe_interleaved_charge_density_hdf5(tmp_path: Path) -> None:
    shape = (4, 4, 4)
    miller = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    coefficients = np.asarray([2.0 + 0.0j, 1.0 + 2.0j])
    with h5py.File(tmp_path / "charge-density.hdf5", "w") as h5:
        h5.attrs["gamma_only"] = np.bytes_(b".FALSE.")
        h5.attrs["nspin"] = 1
        h5.create_dataset("MillerIndices", data=miller)
        h5.create_dataset(
            "rhotot_g",
            data=np.column_stack((coefficients.real, coefficients.imag)).ravel(),
        )
    expected_coefficients = np.zeros(shape, dtype=complex)
    expected_coefficients[tuple(miller.T)] = coefficients
    expected = np.real(np.fft.ifftn(expected_coefficients * np.prod(shape)))
    total, spins, magnetization = _read_density(tmp_path, shape)
    np.testing.assert_allclose(total, expected)
    np.testing.assert_allclose(spins, expected[None, ...])
    np.testing.assert_allclose(magnetization, 0.0)


def test_pp_reconstructs_lsda_spin_density_and_magnetization(tmp_path: Path) -> None:
    shape = (2, 2, 2)
    with h5py.File(tmp_path / "charge-density.hdf5", "w") as h5:
        h5.attrs["gamma_only"] = ".FALSE."
        h5.attrs["nspin"] = 2
        h5.create_dataset("MillerIndices", data=np.asarray([[0, 0, 0]]))
        h5.create_dataset("rhotot_g", data=np.asarray([2.0 + 0.0j]))
        h5.create_dataset("rhodiff_g", data=np.asarray([0.5 + 0.0j]))

    total, spins, magnetization = _read_density(tmp_path, shape)

    np.testing.assert_allclose(total, 2.0)
    np.testing.assert_allclose(spins[0], 1.25)
    np.testing.assert_allclose(spins[1], 0.75)
    np.testing.assert_allclose(spins[0] - spins[1], 0.5)
    np.testing.assert_allclose(magnetization, 0.0)


def test_pp_reconstructs_noncollinear_pauli_density(tmp_path: Path) -> None:
    shape = (2, 2, 2)
    with h5py.File(tmp_path / "charge-density.hdf5", "w") as h5:
        h5.attrs["gamma_only"] = ".FALSE."
        h5.attrs["nspin"] = 4
        h5.create_dataset("MillerIndices", data=np.asarray([[0, 0, 0]]))
        h5.create_dataset("rhotot_g", data=np.asarray([2.0 + 0.0j]))
        h5.create_dataset("m_x", data=np.asarray([0.25 + 0.0j]))
        h5.create_dataset("m_y", data=np.asarray([-0.5 + 0.0j]))
        h5.create_dataset("m_z", data=np.asarray([0.75 + 0.0j]))

    total, spins, magnetization = _read_density(tmp_path, shape)

    np.testing.assert_allclose(total, 2.0)
    np.testing.assert_allclose(spins, total[None, ...])
    np.testing.assert_allclose(
        magnetization,
        np.broadcast_to(
            np.asarray((0.25, -0.5, 0.75))[:, None, None, None],
            (3,) + shape,
        ),
    )


def test_pp_spinor_grid_charge_and_magnetization_components() -> None:
    up = np.asarray([[[1.0 + 0.5j]]])
    down = np.asarray([[[-0.25 + 0.75j]]])
    wave = np.asarray((up, down))
    coherence = np.conjugate(up) * down

    np.testing.assert_allclose(
        _spinor_grid_quantity(wave, 0), np.abs(up) ** 2 + np.abs(down) ** 2
    )
    np.testing.assert_allclose(
        _spinor_grid_quantity(wave, 1), 2.0 * np.real(coherence)
    )
    np.testing.assert_allclose(
        _spinor_grid_quantity(wave, 2), 2.0 * np.imag(coherence)
    )
    np.testing.assert_allclose(
        _spinor_grid_quantity(wave, 3), np.abs(up) ** 2 - np.abs(down) ** 2
    )


def test_pp_noncollinear_magnetization_and_xc_field_plots(saved_state) -> None:
    magnetization = np.asarray(
        (
            0.10 * saved_state.density,
            -0.20 * saved_state.density,
            0.30 * saved_state.density,
        )
    )
    state = replace(
        saved_state,
        noncolin=True,
        spinorbit=True,
        magnetization_density=magnetization,
    )

    magnitude = extract_plot_grids(
        state, {"plot_num": 13, "spin_component": 0}
    )[0][1].values
    y_component = extract_plot_grids(
        state, {"plot_num": 13, "spin_component": 2}
    )[0][1].values
    np.testing.assert_allclose(magnitude, np.linalg.norm(magnetization, axis=0))
    np.testing.assert_allclose(y_component, magnetization[1])

    field_magnitude = extract_plot_grids(
        state, {"plot_num": 18, "spin_component": 0}
    )[0][1].values
    field_components = np.asarray([
        extract_plot_grids(
            state, {"plot_num": 18, "spin_component": component}
        )[0][1].values
        for component in (1, 2, 3)
    ])
    assert np.all(np.isfinite(field_components))
    np.testing.assert_allclose(
        field_magnitude, np.linalg.norm(field_components, axis=0), atol=2.0e-13
    )


def test_pp_reads_degauss_from_qe_smearing_attribute() -> None:
    bands = ET.fromstring(
        '<band_structure><smearing degauss="1.0E-2">gaussian</smearing></band_structure>'
    )
    assert _saved_smearing_metadata(bands) == ("gaussian", 1.0e-2)
    legacy = ET.fromstring(
        "<band_structure><smearing>gaussian</smearing><degauss>2.0E-2</degauss></band_structure>"
    )
    assert _saved_smearing_metadata(legacy) == ("gaussian", 1.0e-2)


def test_pp_plot3_requires_smearing_parent_calculation(saved_state) -> None:
    with pytest.raises(QEInputError, match="gaussian broadening needed"):
        extract_plot_grids(
            saved_state,
            {
                "plot_num": 3,
                "emin": 0.0,
                "emax": 0.0,
                "degauss_ldos": 0.5,
                "use_gauss_ldos": True,
            },
        )


@pytest.mark.parametrize("order", [1, -1], ids=["methfessel-paxton", "marzari-vanderbilt"])
def test_pp_local_dos_omits_negative_smearing_lobes_like_qe(order: int) -> None:
    broadened = smearing_density(np.asarray([0.0, 3.0]), order)
    assert broadened[0] > 0.0
    assert broadened[1] < 0.0
    weights = np.asarray(
        [broadened[0], 0.25 * broadened[0], broadened[1], np.finfo(float).eps / 2.0, 0.0]
    )
    np.testing.assert_array_equal(
        _qe_local_dos_weights(weights),
        np.asarray([broadened[0], 0.25 * broadened[0], 0.0, 0.0, 0.0]),
    )


def test_pp_plot10_symmetrizes_but_plot23_retains_selected_state_density(
    saved_state, monkeypatch
) -> None:
    import qepy_pw.pp.pp as pp_module

    monkeypatch.setattr(
        pp_module,
        "_wave_density_sum",
        lambda _state, _factors: np.ones(saved_state.shape),
    )
    monkeypatch.setattr(
        pp_module,
        "_symmetrize_wave_field",
        lambda _state, values: values + 1.0,
    )
    common = {"emin": -1.0e6, "emax": 1.0e6}
    plot10 = extract_plot_grids(saved_state, {"plot_num": 10, **common})[0][1]
    plot23 = extract_plot_grids(saved_state, {"plot_num": 23, **common})[0][1]
    np.testing.assert_array_equal(plot10.values, np.full(saved_state.shape, 2.0))
    np.testing.assert_array_equal(plot23.values, np.ones(saved_state.shape))


@pytest.mark.parametrize("order", [0, 1, -1], ids=["gaussian", "methfessel-paxton", "marzari-vanderbilt"])
def test_pp_stm_uses_w0gauss_at_bias_window_edges_like_qe(order: int) -> None:
    energies = np.asarray([-0.03, 0.00, 0.04, 0.10, 0.13])
    actual = _stm_window_weights(energies, 0.00, 0.10, 0.01, order)
    expected = np.asarray(
        [
            smearing_density(np.asarray(3.0), order),
            smearing_density(np.asarray(0.0), order),
            1.0,
            smearing_density(np.asarray(0.0), order),
            smearing_density(np.asarray(-3.0), order),
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-15)
    assert actual[1] != pytest.approx(float(wgauss(np.asarray(0.0), order)))


def test_pp_fortran_unscaled_exponential_format() -> None:
    assert _fortran_e(0.00582481, 14, 6) == "  0.582481E-02"
    assert _fortran_e(-12.5, 14, 6) == " -0.125000E+02"
    values = np.asarray([0.0, 0.00582481, -12.5, 0.9999999, -1.0e-20])
    expected = [_fortran_e(value, 25, 14) for value in values]
    assert _formatted_fortran_e(values, 25, 14).tolist() == expected


def test_pp_vectorized_grid_writers_preserve_qe_records() -> None:
    values = np.asarray([0.0, 0.00582481, -12.5, 1.23456789e30, -1.0e-20, 9.9, 7.0])

    output = io.StringIO()
    _write_values(output, values, 3, field_width=14, precision=6)
    expected = "".join(
        "".join(_fortran_e(value, 14, 6) for value in values[begin:begin + 3]) + "\n"
        for begin in range(0, len(values), 3)
    )
    assert output.getvalue() == expected

    output = io.StringIO()
    _write_standard_e_values(output, values, 5, field_width=17, precision=9)
    expected = "".join(
        "".join(f"{value:17.9E}" for value in values[begin:begin + 5]) + "\n"
        for begin in range(0, len(values), 5)
    )
    assert output.getvalue() == expected


def test_pp_regular_fourier_evaluator_matches_arbitrary_points(saved_state) -> None:
    _name, grid = extract_plot_grids(saved_state, {"plot_num": 0})[0]
    origin = np.asarray([0.13, -0.07, 0.19]) * grid.alat
    directions = [
        np.asarray([0.5, 0.1, -0.2]) * grid.alat,
        np.asarray([-0.1, 0.4, 0.3]) * grid.alat,
        np.asarray([0.2, -0.2, 0.6]) * grid.alat,
    ]
    axes = [np.linspace(0.0, 1.0, count) for count in (4, 3, 2)]
    regular = _fourier_regular_values(grid, origin, directions, axes)
    points = (
        origin
        + axes[0][:, None, None, None] * directions[0]
        + axes[1][None, :, None, None] * directions[1]
        + axes[2][None, None, :, None] * directions[2]
    )
    direct = np.real(_fourier_values(grid, points.reshape(-1, 3))).reshape(4, 3, 2)
    np.testing.assert_allclose(regular, direct, rtol=2.0e-13, atol=2.0e-13)


def test_pp_fourier_plot_discards_coefficients_outside_saved_gcutm() -> None:
    shape = (8, 8, 8)
    fractional_x = np.arange(shape[0]) / shape[0]
    values = (
        2.0 * np.cos(2.0 * np.pi * fractional_x)
        + 1.0e-3 * np.cos(6.0 * np.pi * fractional_x)
    )[:, None, None] * np.ones((1, shape[1], shape[2]))
    lattice = 2.0 * np.pi * np.eye(3)
    grid = PlotGrid(
        "cutoff",
        8,
        values,
        lattice,
        2.0 * np.pi,
        (),
        (),
        ecutwfc_ry=1.0,
        ecutrho_ry=1.1,
        gcutm=1.1,
    )
    fractional_points = np.asarray([[0.123, 0.0, 0.0], [0.417, 0.0, 0.0]])
    actual = np.real(_fourier_values(grid, fractional_points @ lattice))
    expected = 2.0 * np.cos(2.0 * np.pi * fractional_points[:, 0])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)


def test_pp_spherical_shell_evaluator_matches_direct_sum(saved_state) -> None:
    _name, grid = extract_plot_grids(saved_state, {"plot_num": 0})[0]
    origin = np.asarray([0.13, -0.07, 0.19]) * grid.alat
    radii = np.linspace(0.0, 1.7 * grid.alat, 11)
    coefficients = np.fft.fftn(grid.values) / grid.values.size
    _miller, g = _miller_and_g(
        grid.values.shape, 2.0 * np.pi * np.linalg.inv(grid.lattice).T
    )
    phase = np.exp(1j * np.einsum("...i,i->...", g, origin))
    expected = np.asarray([
        np.real(np.sum(coefficients * phase * spherical_jn(0, np.linalg.norm(g, axis=-1) * radius)))
        for radius in radii
    ])
    actual = _spherical_average_values(grid, origin, radii)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)


def test_pp_density_intermediate_round_trip(saved_state, tmp_path: Path) -> None:
    _name, density = extract_plot_grids(
        saved_state, {"plot_num": 0, "filplot": "density.pp", "title": "silicon"}
    )[0]
    path = write_plot_file(tmp_path / "density.pp", density)
    restored = read_plot_file(path)
    assert restored.title == "silicon"
    assert restored.plot_num == 0
    assert restored.ibrav == 2
    np.testing.assert_allclose(restored.lattice, saved_state.lattice, atol=1.0e-12)
    np.testing.assert_allclose(restored.values, saved_state.density, rtol=5.0e-10, atol=5.0e-12)


def test_pp_filplot_omits_explicit_cell_for_nonzero_ibrav(saved_state, tmp_path: Path) -> None:
    _name, density = extract_plot_grids(saved_state, {"plot_num": 0})[0]
    lines = write_plot_file(tmp_path / "density.pp", density).read_text(encoding="ascii").splitlines()

    assert int(lines[2].split()[0]) == 2
    # QE plot_io writes the cutoff record immediately after celldm whenever
    # ibrav is nonzero; the three explicit at(:,i) records are ibrav=0 only.
    assert len(lines[3].split()) == 4


def test_pp_scalar_potentials_and_density_descriptors_are_finite(saved_state) -> None:
    for plot_num in (1, 2, 9, 11, 19, 20, 123):
        _name, grid = extract_plot_grids(saved_state, {"plot_num": plot_num})[0]
        assert grid.values.shape == saved_state.shape
        assert np.all(np.isfinite(grid.values))
    for plot_num in (8, 22):
        _name, grid = extract_plot_grids(saved_state, {"plot_num": plot_num})[0]
        assert np.all(np.isfinite(grid.values))
        assert np.min(grid.values) >= 0.0
        if plot_num == 8:
            assert np.max(grid.values) <= 1.0


def test_pp_batched_kinetic_density_matches_state_by_state_sum(saved_state) -> None:
    factors = saved_state.weights[:, None] * saved_state.occupations
    expected = np.zeros(saved_state.shape, dtype=float)
    for ik in range(len(saved_state.weights)):
        miller, coefficients, xk = _read_wavefunction(saved_state, ik)
        gk = miller @ saved_state.reciprocal + xk
        for band in np.flatnonzero(factors[ik]):
            for axis in range(3):
                derivative = _wave_grid(
                    miller,
                    1j * gk[:, axis] * coefficients[:, band],
                    saved_state.shape,
                    saved_state.volume,
                )
                expected += factors[ik, band] * np.abs(derivative) ** 2
    actual = _wave_density_sum(saved_state, factors, kinetic=True)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-15, atol=3.0e-15)


def test_pp_selected_orbital_is_normalized(saved_state) -> None:
    _name, grid = extract_plot_grids(
        saved_state,
        {"plot_num": 7, "kpoint": 1, "kband": 1, "filplot": "orbital.pp"},
    )[0]
    norm = np.mean(grid.values) * saved_state.volume
    assert norm == pytest.approx(1.0, abs=1.0e-9)


def test_pp_combines_formatted_files_and_writes_xsf_and_cube(
    saved_state, tmp_path: Path
) -> None:
    _name, density = extract_plot_grids(saved_state, {"plot_num": 0})[0]
    first = write_plot_file(tmp_path / "first.pp", density)
    second = write_plot_file(tmp_path / "second.pp", density)
    combined = combine_plot_files([first, second], [0.25, 0.75])
    np.testing.assert_allclose(combined.values, density.values, rtol=5.0e-10)

    xsf = write_output_plot(
        combined, {"iflag": 3, "output_format": 5, "fileout": str(tmp_path / "rho.xsf")}
    )
    cube = write_output_plot(
        combined, {"iflag": 3, "output_format": 6, "fileout": str(tmp_path / "rho.cube")}
    )
    assert "BEGIN_DATAGRID_3D" in xsf.read_text(encoding="ascii")
    assert cube.read_text(encoding="ascii").startswith(" Cubefile created from PWScf")


def test_pp_run_supports_extraction_and_plotting_in_one_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "qe_reference"
        / "upstream"
        / "pseudo"
        / "pwscf.save"
    )
    shutil.copytree(source, tmp_path / "pwscf.save")
    monkeypatch.chdir(tmp_path)
    stdout = io.StringIO()
    extracted, output = run_pp(
        """&inputpp
 prefix='pwscf', outdir='.', plot_num=0, filplot='rho.pp'
/
&plot
 nfile=1, filepp(1)='rho.pp', weight(1)=1.0,
 iflag=3, output_format=6, fileout='rho.cube'
/
""",
        stdout=stdout,
    )
    assert extracted == [Path("rho.pp")]
    assert output == Path("rho.cube")
    assert all(path.is_file() for path in (*extracted, output))
    text = stdout.getvalue()
    assert "Reading xml data from directory:" in text
    assert "Calling punch_plot, plot_num =   0" in text


def test_pp_renders_every_multi_orbital_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "qe_reference"
        / "upstream"
        / "pseudo"
        / "pwscf.save"
    )
    shutil.copytree(source, tmp_path / "pwscf.save")
    monkeypatch.chdir(tmp_path)
    extracted, output = run_pp(
        """&inputpp
 prefix='pwscf', outdir='.', plot_num=7, filplot='psi',
 kpoint(1)=1, kpoint(2)=1, kband(1)=1, kband(2)=2
/
&plot
 nfile=1, weight(1)=1.0,
 iflag=3, output_format=6, fileout='.cube'
/
"""
    )
    assert extracted == [Path("psi_K001_B001"), Path("psi_K001_B002")]
    assert output == [Path("psi_K001_B001.cube"), Path("psi_K001_B002.cube")]
    assert all(path.is_file() for path in (*extracted, *output))


def test_pp_rejects_unimplemented_spinor_and_paw_paths(saved_state) -> None:
    for plot_num in (17, 21, 24):
        with pytest.raises(UnsupportedFeatureError):
            extract_plot_grids(saved_state, {"plot_num": plot_num})
    for plot_num in (13, 18):
        with pytest.raises(QEInputError, match="noncollinear"):
            extract_plot_grids(saved_state, {"plot_num": plot_num})

    with pytest.raises(QEInputError, match="LSDA"):
        extract_plot_grids(saved_state, {"plot_num": 6})


@pytest.fixture
def lsda_saved_state(saved_state):
    up = 0.65 * saved_state.density
    down = 0.35 * saved_state.density
    count = len(saved_state.weights)
    return replace(
        saved_state,
        spin_densities=np.asarray((up, down)),
        energies_ha=np.concatenate(
            (saved_state.energies_ha - 0.01, saved_state.energies_ha + 0.01)
        ),
        occupations=np.concatenate(
            (0.5 * saved_state.occupations, 0.5 * saved_state.occupations)
        ),
        weights=np.concatenate((saved_state.weights, saved_state.weights)),
        spins=np.concatenate(
            (np.ones(count, dtype=np.int8), np.full(count, 2, dtype=np.int8))
        ),
    )


def test_pp_lsda_charge_magnetization_and_spin_xc_potentials(lsda_saved_state) -> None:
    total = extract_plot_grids(lsda_saved_state, {"plot_num": 0})[0][1].values
    up = extract_plot_grids(
        lsda_saved_state, {"plot_num": 0, "spin_component": 1}
    )[0][1].values
    down = extract_plot_grids(
        lsda_saved_state, {"plot_num": 0, "spin_component": 2}
    )[0][1].values
    magnetization = extract_plot_grids(
        lsda_saved_state, {"plot_num": 6}
    )[0][1].values

    np.testing.assert_allclose(up + down, total)
    np.testing.assert_allclose(up - down, magnetization)
    moment = np.mean(magnetization) * lsda_saved_state.volume
    assert moment == pytest.approx(0.30 * 8.0, abs=1.0e-8)

    potential_up = extract_plot_grids(
        lsda_saved_state, {"plot_num": 1, "spin_component": 1}
    )[0][1].values
    potential_down = extract_plot_grids(
        lsda_saved_state, {"plot_num": 1, "spin_component": 2}
    )[0][1].values
    assert np.all(np.isfinite(potential_up))
    assert np.all(np.isfinite(potential_down))
    assert not np.allclose(potential_up, potential_down)


def test_pp_lsda_unindexed_namelist_spin_component_selects_up_density(
    lsda_saved_state,
) -> None:
    options = parse_namelist(
        "&INPUTPP plot_num=0, spin_component=1 /", "inputpp"
    )

    assert options["spin_component(1)"] == 1
    selected = extract_plot_grids(lsda_saved_state, options)[0][1].values
    np.testing.assert_allclose(selected, lsda_saved_state.spin_densities[0])


def test_pp_lsda_density_difference_uses_starting_magnetization(
    lsda_saved_state,
) -> None:
    state = replace(lsda_saved_state, starting_magnetizations=(0.5,))
    total = extract_plot_grids(state, {"plot_num": 9})[0][1].values
    up = extract_plot_grids(
        state, {"plot_num": 9, "spin_component": 1}
    )[0][1].values
    down = extract_plot_grids(
        state, {"plot_num": 9, "spin_component": 2}
    )[0][1].values

    np.testing.assert_allclose(up + down, total)
    np.testing.assert_allclose(
        up - down,
        state.magnetization - _atomic_magnetization(state),
    )
    assert np.max(np.abs(_atomic_density(state))) > 0.0


def test_pp_lsda_gga_potential_retains_spin_dependence(lsda_saved_state) -> None:
    state = replace(lsda_saved_state, functional="pbe")

    averaged = extract_plot_grids(
        state, {"plot_num": 1, "spin_component": 0}
    )[0][1].values
    up = extract_plot_grids(
        state, {"plot_num": 1, "spin_component": 1}
    )[0][1].values
    down = extract_plot_grids(
        state, {"plot_num": 1, "spin_component": 2}
    )[0][1].values

    assert np.all(np.isfinite(up)) and np.all(np.isfinite(down))
    np.testing.assert_allclose(averaged, 0.5 * (up + down), atol=2.0e-12)
    assert not np.allclose(up, down)


def test_pp_lsda_ildos_selects_only_requested_spin_rows(
    lsda_saved_state, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qepy_pw.pp.pp as pp_module

    captured: list[np.ndarray] = []

    def fake_density(_state, factors, **_kwargs):
        captured.append(np.asarray(factors).copy())
        return np.full(_state.shape, np.sum(factors))

    monkeypatch.setattr(pp_module, "_wave_density_sum", fake_density)
    bounds = {"plot_num": 10, "emin": -1.0e6, "emax": 1.0e6}
    extract_plot_grids(lsda_saved_state, {**bounds, "spin_component": 1})
    extract_plot_grids(lsda_saved_state, {**bounds, "spin_component": 2})

    up_rows = lsda_saved_state.spins == 1
    assert np.all(captured[0][~up_rows] == 0.0)
    assert np.any(captured[0][up_rows] > 0.0)
    assert np.all(captured[1][up_rows] == 0.0)
    assert np.any(captured[1][~up_rows] > 0.0)
