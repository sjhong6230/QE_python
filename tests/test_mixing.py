import numpy as np
import pytest

import qepy_pw.mixing as mixing_module
from qepy_pw.mixing import PlainBroydenMixer


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
