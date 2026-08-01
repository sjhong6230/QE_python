import numpy as np

from qepy_pw.xc import (
    pbe_unpolarized,
    pw92_unpolarized,
    pz81_unpolarized,
)


def test_pz81_is_finite_and_attractive():
    epsilon, potential = pz81_unpolarized(np.array([1e-8, 0.01, 1.0]))
    assert np.all(np.isfinite(epsilon))
    assert np.all(epsilon < 0)
    assert np.all(potential < 0)


def test_pz81_matches_qe_density_threshold_and_absolute_density():
    density = np.array([-1.0e-4, -1.0e-12, 0.0, 1.0e-12, 1.0e-4])
    epsilon, potential = pz81_unpolarized(density)
    assert np.array_equal(epsilon[1:4], np.zeros(3))
    assert np.array_equal(potential[1:4], np.zeros(3))
    assert epsilon[0] == epsilon[4]
    assert potential[0] == potential[4]


def _spectral_pbe(density: np.ndarray):
    shape = density.shape
    lengths = np.array([5.0, 6.0, 7.0])
    axes = [
        2.0 * np.pi * np.fft.fftfreq(size, d=length / size)
        for size, length in zip(shape, lengths)
    ]
    wavevectors = np.meshgrid(*axes, indexing="ij")
    density_g = np.fft.fftn(density)
    gradient = np.stack(
        [
            np.fft.ifftn(1j * vector * density_g).real
            for vector in wavevectors
        ]
    )

    def divergence(field):
        return sum(
            np.fft.ifftn(
                1j * vector * np.fft.fftn(component)
            ).real
            for vector, component in zip(wavevectors, field)
        )

    return pbe_unpolarized(density, gradient, divergence)


def test_pbe_uniform_limit_is_slater_exchange_plus_pw92_correlation():
    density = np.full((4, 3, 2), 0.07)
    epsilon, potential = _spectral_pbe(density)
    ec, vc = pw92_unpolarized(density)
    rs = 0.6203504908994 / np.cbrt(density)
    exchange = (-0.687247939924714 * (2.0 / 3.0)) / rs
    assert np.allclose(epsilon, exchange + ec, atol=2.0e-15)
    assert np.allclose(
        potential, (4.0 / 3.0) * exchange + vc, atol=2.0e-15
    )


def test_pbe_potential_is_variational_derivative_of_periodic_energy():
    shape = (12, 10, 8)
    grid = np.indices(shape)
    density = (
        0.07
        + 0.012 * np.cos(2.0 * np.pi * grid[0] / shape[0])
        + 0.008 * np.sin(2.0 * np.pi * grid[1] / shape[1])
        + 0.005
        * np.cos(2.0 * np.pi * (grid[0] + grid[2]) / shape[0])
    )
    perturbation = (
        np.cos(4.0 * np.pi * grid[0] / shape[0])
        - 0.4 * np.sin(2.0 * np.pi * grid[2] / shape[2])
    )
    epsilon, potential = _spectral_pbe(density)
    step = 1.0e-6
    epsilon_plus, _ = _spectral_pbe(density + step * perturbation)
    epsilon_minus, _ = _spectral_pbe(density - step * perturbation)
    finite_difference = (
        np.mean((density + step * perturbation) * epsilon_plus)
        - np.mean((density - step * perturbation) * epsilon_minus)
    ) / (2.0 * step)
    analytic = np.mean(potential * perturbation)
    assert np.isclose(finite_difference, analytic, atol=2.0e-9)
