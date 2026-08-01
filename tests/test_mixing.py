import numpy as np
import pytest

from qepy_pw.mixing import LinearMixer, PlainBroydenMixer


def _manufactured_response(
    density: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """A slowly contracting, charge-conserving dielectric response."""
    error = density - target
    return target + 0.85 * error + 0.03 * np.roll(error, 1, axis=0)


def test_plain_broyden_accelerates_a_manufactured_fixed_point():
    shape = (4, 4, 4)
    grid = np.indices(shape)
    target = (
        0.25
        + 0.03 * np.cos(2.0 * np.pi * grid[0] / shape[0])
        + 0.02 * np.sin(2.0 * np.pi * grid[1] / shape[1])
    )
    initial = (
        target
        + 0.08 * np.cos(2.0 * np.pi * grid[2] / shape[2])
        + 0.05
        * np.cos(2.0 * np.pi * (grid[0] + grid[1]) / shape[0])
    )
    linear = LinearMixer(beta=0.2)
    broyden = PlainBroydenMixer(
        shape, 2.0 * np.pi * np.eye(3), beta=0.2, ndim=8
    )
    linear_density = initial.copy()
    broyden_density = initial.copy()
    for _ in range(6):
        linear_density = linear.mix(
            linear_density,
            _manufactured_response(linear_density, target),
        )
        broyden_density = broyden.mix(
            broyden_density,
            _manufactured_response(broyden_density, target),
        )

    linear_error = np.linalg.norm(linear_density - target)
    broyden_error = np.linalg.norm(broyden_density - target)
    assert broyden_error < 1.0e-10
    assert broyden_error < 1.0e-6 * linear_error


def test_plain_broyden_preserves_charge_and_bounds_history():
    shape = (4, 4, 4)
    reciprocal = np.diag([1.0, 1.2, 1.4])
    mixer = PlainBroydenMixer(shape, reciprocal, beta=0.35, ndim=3)
    grid = np.indices(shape)
    density = 0.4 + 0.02 * np.cos(2.0 * np.pi * grid[0] / shape[0])
    charge = float(np.sum(density))
    for iteration in range(7):
        output = (
            density
            + 0.01
            * np.sin(
                2.0
                * np.pi
                * (grid[1] + iteration * grid[2])
                / shape[1]
            )
            + 0.03
        )
        density = mixer.mix(density, output)
        assert float(np.sum(density)) == pytest.approx(
            charge, abs=1.0e-12
        )
    assert len(mixer.delta_inputs) == 3
    assert len(mixer.delta_residuals) == 3


def test_plain_broyden_stores_qe_consecutive_secants():
    shape = (3, 3, 3)
    mixer = PlainBroydenMixer(shape, np.eye(3), beta=0.3, ndim=4)
    grid = np.indices(shape)
    first = 0.4 + 0.02 * np.cos(2.0 * np.pi * grid[0] / 3.0)
    first_out = first + 0.01 * np.sin(
        2.0 * np.pi * grid[1] / 3.0
    )
    second = mixer.mix(first, first_out)
    second_out = second + 0.015 * np.cos(
        2.0 * np.pi * grid[2] / 3.0
    )
    previous_input = np.fft.fftn(first) / np.prod(shape)
    current_input = np.fft.fftn(second) / np.prod(shape)
    previous_residual = (
        np.fft.fftn(first_out) / np.prod(shape) - previous_input
    )
    current_residual = (
        np.fft.fftn(second_out) / np.prod(shape) - current_input
    )
    mixer.mix(second, second_out)
    assert np.allclose(
        mixer.delta_inputs[-1],
        (previous_input - current_input).ravel()[mixer.active],
    )
    assert np.allclose(
        mixer.delta_residuals[-1],
        (previous_residual - current_residual).ravel()[mixer.active],
    )


def test_plain_broyden_metric_honors_charge_density_cutoff():
    mixer = PlainBroydenMixer(
        (5, 5, 5),
        np.eye(3),
        beta=0.3,
        ndim=4,
        g2_cutoff=1.1,
    )
    low = np.zeros((5, 5, 5), dtype=complex)
    high = np.zeros_like(low)
    low[1, 0, 0] = 1.0
    high[2, 0, 0] = 1.0
    assert mixer._metric(low, low) > 0.0
    assert mixer._metric(high, high) == 0.0


def test_numba_coulomb_metric_matches_numpy():
    pytest.importorskip("numba")
    shape = (6, 5, 4)
    reciprocal = np.diag([1.0, 1.3, 0.8])
    rng = np.random.default_rng(717)
    left = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    right = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    numpy_mixer = PlainBroydenMixer(shape, reciprocal)
    numba_mixer = PlainBroydenMixer(
        shape, reciprocal, use_numba=True
    )
    assert numba_mixer._metric(left, right) == pytest.approx(
        numpy_mixer._metric(left, right), abs=1.0e-12
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"beta": 0.0}, "mixing_beta"),
        ({"beta": 1.1}, "mixing_beta"),
        ({"ndim": 0}, "mixing_ndim"),
        ({"g2_cutoff": 0.0}, "charge-density cutoff"),
    ],
)
def test_plain_broyden_rejects_invalid_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PlainBroydenMixer((2, 2, 2), np.eye(3), **kwargs)
