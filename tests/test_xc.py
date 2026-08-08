"""Numerical regression tests for the memory-bounded Python XC formulas."""

from __future__ import annotations

import numpy as np

from qepy_pw.xc import (
    pbe_unpolarized_components,
    pw92_lda_unpolarized,
    pw92_unpolarized,
    pz81_unpolarized,
)


RHO = np.array([-0.2, 0.0, 1.0e-8, 1.0e-4, 0.03, 0.7])
GRADIENT = np.array(
    [
        [0.01, 0.0, 1.0e-8, 2.0e-4, -0.04, 0.3],
        [-0.02, 0.0, 0.0, 1.0e-4, 0.02, -0.1],
        [0.03, 0.0, -1.0e-8, 0.0, 0.01, 0.2],
    ]
)


def test_lda_family_matches_qe_formula_reference_values():
    expected = {
        "pz": (
            [-0.49027706396402754, -0.049573512285933576, -0.7235267006348448],
            [-0.6413964495954185, -0.06449585183819752, -0.9500803711849825],
        ),
        "pw_c": (
            [-0.05842012792002624, -0.015316229379184998, -0.06827928599739908],
            [-0.06602930545485859, -0.018796908949672595, -0.07640078192269545],
        ),
        "pw": (
            [-0.49033191464275516, -0.049597090609241216, -0.7240485701900751],
            [-0.6419116877518305, -0.06450472392308089, -0.9507598275129301],
        ),
    }
    indices = np.array([0, 3, 5])
    for name, function in (
        ("pz", pz81_unpolarized),
        ("pw_c", pw92_unpolarized),
        ("pw", pw92_lda_unpolarized),
    ):
        epsilon, potential = function(RHO)
        np.testing.assert_allclose(epsilon[indices], expected[name][0], rtol=2e-14)
        np.testing.assert_allclose(potential[indices], expected[name][1], rtol=2e-14)
        assert epsilon[1] == potential[1] == 0.0


def test_pbe_family_matches_qe_formula_reference_values():
    expected_at_index_four = {
        "pbe": (-0.28064136790323385, -0.34161000883284454, -0.28231169563903985),
        "pbesol": (-0.27448506869604195, -0.3505782657202599, -0.09028792341446806),
        "revpbe": (-0.2821315889252751, -0.33663138022597167, -0.3569430167418392),
        "rpbe": (-0.2827270275051642, -0.33471483889025766, -0.38598365362290493),
    }
    for functional, expected in expected_at_index_four.items():
        values = pbe_unpolarized_components(RHO, GRADIENT, functional)
        actual = np.array([value[4] for value in values])
        np.testing.assert_allclose(actual, expected, rtol=2e-14)
        assert all(value[1] == 0.0 for value in values)
        assert values[2][2] == 0.0


def test_pbe_tiling_handles_noncontiguous_mpi_slab_layout():
    """Large moveaxis views must write tiles into their real output owner."""
    rng = np.random.default_rng(314159)
    density_storage = np.abs(rng.normal(size=(128, 129, 2))) * 0.05
    density = np.moveaxis(density_storage, -1, 0)
    gradient_storage = rng.normal(size=(128, 129, 2, 3)) * 0.02
    gradient = gradient_storage.transpose(3, 2, 0, 1)
    assert density.size > 32_768
    assert not density.flags.c_contiguous
    assert not gradient.flags.c_contiguous

    actual = pbe_unpolarized_components(density, gradient, "pbe")
    expected = pbe_unpolarized_components(
        np.ascontiguousarray(density),
        np.ascontiguousarray(gradient),
        "pbe",
    )
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(actual_array, expected_array)

    energy_density = np.empty_like(density)
    epsilon, potential, coefficient = pbe_unpolarized_components(
        density,
        gradient,
        "pbe",
        need_epsilon=False,
        energy_density_out=energy_density,
    )
    assert epsilon is None
    np.testing.assert_array_equal(
        energy_density,
        density * expected[0],
    )
    np.testing.assert_array_equal(potential, expected[1])
    np.testing.assert_array_equal(coefficient, expected[2])
