"""Serial policies and Gamma algebra for the scalable FFT engine."""

from __future__ import annotations

import numpy as np
import pytest

from qepy_pw.errors import QEInputError
from qepy_pw.fft_engine import (
    FFTMemoryPlan,
    GammaHalfSpectrum,
    choose_fft_decomposition,
    choose_process_grid,
)


def _complete_indices(shape: tuple[int, int, int]) -> np.ndarray:
    axes = [np.rint(np.fft.fftfreq(n) * n).astype(np.int32) for n in shape]
    return np.column_stack(
        [axis.ravel() for axis in np.meshgrid(*axes, indexing="ij")]
    )


def test_process_grid_rejects_empty_pencils() -> None:
    assert np.prod(choose_process_grid((12, 10, 8), 4)) == 4
    with pytest.raises(QEInputError, match="cannot form nonempty pencils"):
        choose_process_grid((2, 2, 2), 8)


def test_decomposition_policy_preserves_slab_until_required() -> None:
    shape = (32, 30, 24)
    assert choose_fft_decomposition(shape, 8) == "slab"
    assert (
        choose_fft_decomposition(
            shape,
            8,
            slab_bytes_per_rank=9,
            memory_limit_bytes=8,
        )
        == "pencil"
    )
    assert choose_fft_decomposition(shape, 32) == "pencil"


def test_memory_plan_bounds_band_tile_and_gamma_pairs() -> None:
    plan = FFTMemoryPlan.choose(
        bands=19,
        fixed_bytes=1024,
        bytes_per_band=4096,
        memory_limit_bytes=1024 + 5 * 4096,
        preferred_tile=12,
        gamma_pairs=True,
    )
    assert plan.band_tile == 4
    assert not plan.double_buffer
    with pytest.raises(QEInputError, match="cannot hold one"):
        FFTMemoryPlan.choose(
            bands=2,
            fixed_bytes=1024,
            bytes_per_band=4096,
            memory_limit_bytes=4096,
        )


def test_gamma_half_spectrum_roundtrip_pair_packing_and_metric() -> None:
    shape = (5, 4, 3)
    indices = _complete_indices(shape)
    gamma = GammaHalfSpectrum.build(indices, shape)
    rng = np.random.default_rng(719)
    real_first = rng.normal(size=shape)
    real_second = rng.normal(size=shape)
    first_full = np.fft.fftn(real_first).reshape(-1)
    second_full = np.fft.fftn(real_second).reshape(-1)
    first = gamma.compress(first_full)
    second = gamma.compress(second_full)

    np.testing.assert_allclose(gamma.expand(first), first_full, atol=2.0e-13)
    packed = gamma.pack_two_real(first, second)
    unpacked_first, unpacked_second = gamma.unpack_two_real(packed)
    np.testing.assert_allclose(unpacked_first, first, atol=2.0e-13)
    np.testing.assert_allclose(unpacked_second, second, atol=2.0e-13)

    half_matrix = np.column_stack((first, second))
    full_matrix = np.column_stack((first_full, second_full))
    expected = np.real(np.conj(full_matrix).T @ full_matrix)
    np.testing.assert_allclose(
        gamma.real_inner(half_matrix, half_matrix), expected, atol=5.0e-11
    )


def test_gamma_rejects_nonhermitian_coefficients() -> None:
    shape = (4, 3, 2)
    indices = _complete_indices(shape)
    gamma = GammaHalfSpectrum.build(indices, shape)
    values = np.ones(len(indices), dtype=np.complex128)
    values[1] += 0.5j
    with pytest.raises(ValueError, match="not conjugate symmetric"):
        gamma.compress(values)
