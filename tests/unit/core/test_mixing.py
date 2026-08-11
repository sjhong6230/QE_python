import numpy as np
import pytest

import qepy_pw.mixing as mixing_module
from qepy_pw.mixing import (
    DistributedBroydenMixer,
    LinearMixer,
    PauliDensityMixer,
    PlainBroydenMixer,
    SpinDensityMixer,
)
from qepy_pw.mpi import MPIContext


def _mixing_fields(shape=(4, 4, 4)):
    x, y, z = np.meshgrid(
        *(np.arange(size) / size for size in shape), indexing="ij"
    )
    density = 1.0 + 0.1 * np.cos(2.0 * np.pi * x)
    target = density + 0.03 * np.sin(2.0 * np.pi * (x + y + z))
    reciprocal = 2.0 * np.pi * np.eye(3)
    return density, target, reciprocal


def test_pulay_frequency_one_preserves_default_trajectory():
    density, target, reciprocal = _mixing_fields()
    default = PlainBroydenMixer(density.shape, reciprocal, beta=0.2)
    explicit = PlainBroydenMixer(
        density.shape, reciprocal, beta=0.2, pulay_frequency=1
    )
    left = density.copy()
    right = density.copy()
    for _ in range(5):
        left = default.mix(left, target)
        right = explicit.mix(right, target)
        np.testing.assert_array_equal(left, right)


def test_periodic_pulay_collects_every_step_but_extrapolates_periodically(
    monkeypatch,
):
    density, target, reciprocal = _mixing_fields()
    calls = 0
    original = mixing_module._broyden_projection_and_gram

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        mixing_module, "_broyden_projection_and_gram", counted
    )
    mixer = PlainBroydenMixer(
        density.shape,
        reciprocal,
        beta=0.2,
        ndim=8,
        pulay_frequency=3,
    )
    current = density
    for _ in range(6):
        current = mixer.mix(current, target)
    assert calls == 2
    assert len(mixer.delta_inputs) == 5
    assert len(mixer.delta_residuals) == 5


def test_invalid_pulay_frequency_is_rejected():
    density, _target, reciprocal = _mixing_fields()
    with pytest.raises(ValueError, match="mixing_pulay_frequency"):
        PlainBroydenMixer(
            density.shape, reciprocal, pulay_frequency=0
        )


class _SerialDensityWorkspace:
    def __init__(self, shape):
        self.shape = shape
        self.mpi = MPIContext()
        self.forward_calls = 0

    def grid_to_coefficients(self, grid):
        self.forward_calls += 1
        values = np.asarray(grid)
        transformed = np.fft.fftn(values, axes=(0, 1, 2))
        return np.ascontiguousarray(
            transformed.reshape((-1,) + values.shape[3:])
            / np.prod(self.shape)
        )

    def coefficients_to_grid(self, coefficients, *, use_scratch=False):
        del use_scratch
        values = np.asarray(coefficients)
        trailing = values.shape[1:]
        transformed = values.reshape(self.shape + trailing)
        return np.fft.ifftn(
            transformed * np.prod(self.shape), axes=(0, 1, 2)
        )


def test_lsda_broyden_history_contains_magnetization_channel():
    shape = (4, 4, 4)
    workspace = _SerialDensityWorkspace(shape)
    frequencies = [np.fft.fftfreq(size) * size for size in shape]
    vectors = np.stack(
        np.meshgrid(*frequencies, indexing="ij"), axis=-1
    ) @ (2.0 * np.pi * np.eye(3))
    g2 = np.einsum("...j,...j->...", vectors, vectors).ravel()
    charge = DistributedBroydenMixer(
        workspace, g2, beta=0.35, ndim=8
    )
    mixer = SpinDensityMixer(charge, beta=0.35)
    x, y, z = np.meshgrid(
        *(np.arange(size) / size for size in shape), indexing="ij"
    )
    total = 1.0 + 0.05 * np.cos(2.0 * np.pi * x)
    magnetic_in = 0.08 * np.cos(2.0 * np.pi * y)
    magnetic_out = 0.17 * np.sin(2.0 * np.pi * (x + z))
    current = np.stack((
        0.5 * (total + magnetic_in),
        0.5 * (total - magnetic_in),
    ))
    target = np.stack((
        0.5 * (total + magnetic_out),
        0.5 * (total - magnetic_out),
    ))
    first = mixer.mix(current, target)
    second = mixer.mix(first, target)
    assert workspace.forward_calls == 3
    assert len(mixer.delta_inputs) == 1
    assert len(mixer.delta_residuals) == 1
    # A purely magnetic residual still creates a nonzero Broyden secant and
    # therefore does not follow a second independent linear-mixing step.
    assert np.linalg.norm(mixer.delta_residuals[0]) > 0.0
    first_m = first[0] - first[1]
    second_m = second[0] - second[1]
    linear_second_m = 0.65 * first_m + 0.35 * magnetic_out
    assert not np.allclose(second_m, linear_second_m, atol=1.0e-12)
    np.testing.assert_allclose(
        mixer.metric_weights[mixer.magnetic_positions],
        1.0 / (4.0 * np.pi**2),
    )


def test_lsda_linear_mixing_remains_available_without_history():
    density, target, _reciprocal = _mixing_fields()
    spin_density = np.stack((0.55 * density, 0.45 * density))
    spin_target = np.stack((0.4 * target, 0.6 * target))
    mixer = SpinDensityMixer(LinearMixer(beta=0.25), beta=0.25)
    mixed = mixer.mix(spin_density, spin_target)
    np.testing.assert_allclose(
        mixed, 0.75 * spin_density + 0.25 * spin_target
    )
    assert len(mixer.delta_inputs) == 0


def test_noncollinear_broyden_mixes_all_three_magnetic_components():
    shape = (4, 4, 4)
    workspace = _SerialDensityWorkspace(shape)
    frequencies = [np.fft.fftfreq(size) * size for size in shape]
    vectors = np.stack(
        np.meshgrid(*frequencies, indexing="ij"), axis=-1
    ) @ (2.0 * np.pi * np.eye(3))
    g2 = np.einsum("...j,...j->...", vectors, vectors).ravel()
    mixer = PauliDensityMixer(
        DistributedBroydenMixer(workspace, g2, beta=0.3, ndim=6),
        beta=0.3,
    )
    x, y, z = np.meshgrid(
        *(np.arange(size) / size for size in shape), indexing="ij"
    )
    current = np.stack(
        (
            1.0 + 0.05 * np.cos(2 * np.pi * x),
            0.03 * np.sin(2 * np.pi * y),
            0.04 * np.cos(2 * np.pi * z),
            0.02 * np.sin(2 * np.pi * (x + z)),
        )
    )
    target = current.copy()
    target[1:] += np.stack(
        (
            0.02 * np.cos(2 * np.pi * z),
            0.03 * np.sin(2 * np.pi * x),
            0.04 * np.cos(2 * np.pi * y),
        )
    )
    first = mixer.mix(current, target)
    second = mixer.mix(first, target)
    assert len(mixer.delta_inputs) == 1
    assert np.linalg.norm(mixer.delta_residuals[0]) > 0.0
    component_changes = np.sqrt(
        np.sum((second[1:] - first[1:]) ** 2, axis=(1, 2, 3))
    )
    assert np.all(component_changes > 0.0)
