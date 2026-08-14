from __future__ import annotations

import numpy as np
import pytest

from qepy_pw.basis import (
    FFTScratchPool,
    LocalPotentialWorkspace,
    PlaneWaveBasis,
    PlaneWaveCatalog,
    _cutoff_miller_extents,
    _next_fast_len_real,
    apply_local_potential,
    fft_shape,
    make_basis,
    make_bases,
    potential_matrix,
)
from qepy_pw.timing import TimingRegistry


@pytest.mark.parametrize(
    ("target", "expected"),
    [(0, 1), (1, 1), (7, 8), (11, 12), (17, 18), (31, 32)],
)
def test_real_fft_lengths_are_235_smooth(target: int, expected: int) -> None:
    assert _next_fast_len_real(target) == expected


def test_compact_plane_wave_basis_materializes_catalog_mapping() -> None:
    reciprocal = np.diag([1.0, 1.5, 2.0])
    indices = np.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1]], dtype=np.int32)
    catalog = PlaneWaveCatalog(indices, reciprocal)
    basis = PlaneWaveBasis(catalog, np.asarray([2, 0]), np.asarray([0.25, 0.0, 0.0]))
    expected_vectors = indices[[2, 0]] @ reciprocal + np.asarray([0.25, 0.0, 0.0]) @ reciprocal

    np.testing.assert_array_equal(basis.indices, indices[[2, 0]])
    np.testing.assert_allclose(basis.vectors, expected_vectors)
    np.testing.assert_allclose(
        basis.kinetic, 0.5 * np.einsum("gi,gi->g", expected_vectors, expected_vectors)
    )
    materialized = basis.materialize()
    assert materialized.catalog is None
    np.testing.assert_array_equal(materialized.indices, basis.indices)
    np.testing.assert_allclose(materialized.vectors, basis.vectors)

    rows = np.asarray([1], dtype=np.int32)
    np.testing.assert_array_equal(
        basis.indices_for_rows(rows), basis.indices[rows]
    )
    np.testing.assert_allclose(
        basis.vectors_for_rows(rows), basis.vectors[rows]
    )
    np.testing.assert_allclose(
        basis.kinetic_for_rows(rows), basis.kinetic[rows]
    )


def test_plane_wave_mapping_rejects_indices_outside_catalog() -> None:
    catalog = PlaneWaveCatalog(np.zeros((1, 3), dtype=np.int32), np.eye(3))
    with pytest.raises(ValueError, match="mapping is out of range"):
        PlaneWaveBasis(catalog, np.asarray([1]), np.zeros(3))


def test_multiple_kpoint_bases_share_catalog_and_obey_cutoff() -> None:
    reciprocal = 2.0 * np.pi / 5.0 * np.eye(3)
    kpoints = np.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    bases = make_bases(reciprocal, kpoints, ecut_ry=10.0)

    assert bases[0].catalog is bases[1].catalog
    for basis in bases:
        assert np.all(np.diff(basis.kinetic) >= -1.0e-12)
        assert np.max(basis.kinetic) <= 5.0 + 1.0e-12
        assert len(np.unique(basis.indices, axis=0)) == len(basis)


def test_fft_shape_respects_symmetry_translation_factors() -> None:
    reciprocal = 2.0 * np.pi / 6.0 * np.eye(3)
    bases = make_bases(reciprocal, np.zeros(3), ecut_ry=8.0)
    shape = fft_shape(
        bases,
        reciprocal,
        ecutrho_ry=32.0,
        fft_factors=(2, 3, 1),
    )
    assert shape[0] % 2 == 0
    assert shape[1] % 3 == 0
    assert all(_next_fast_len_real(size) == size for size in shape)


@pytest.mark.parametrize(
    "reciprocal",
    [
        np.diag([1.0, 1.3, 1.7]),
        np.asarray(
            [[1.1, 0.2, 0.0], [0.1, 1.4, 0.3], [0.2, 0.0, 1.6]]
        ),
    ],
)
@pytest.mark.parametrize("ecut_ry", [3.0, 11.0, 37.0])
def test_chunked_cutoff_extents_match_materialized_basis(
    reciprocal: np.ndarray, ecut_ry: float
) -> None:
    expected = np.max(
        np.abs(make_basis(reciprocal, np.zeros(3), ecut_ry).indices),
        axis=0,
    )
    actual = _cutoff_miller_extents(
        reciprocal, ecut_ry, chunk_bytes=512
    )
    np.testing.assert_array_equal(actual, expected)


def test_fft_local_potential_matches_explicit_convolution() -> None:
    shape = (6, 6, 6)
    grid = np.indices(shape).sum(axis=0)
    potential_r = 0.3 + 0.2 * np.cos(2.0 * np.pi * grid / shape[0])
    potential_g = np.fft.fftn(potential_r) / np.prod(shape)
    indices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=np.int32
    )
    coefficients = np.asarray([1.0 + 0.2j, -0.3j, 0.4, -0.1 + 0.5j])

    expected = potential_matrix(potential_g, indices) @ coefficients
    actual = apply_local_potential(potential_g, indices, coefficients)
    np.testing.assert_allclose(actual, expected, atol=2.0e-14)


def test_serial_sparse_spatial_fft_matches_full_3d_fft() -> None:
    shape = (12, 10, 8)
    axes = [
        np.rint(np.fft.fftfreq(size) * size).astype(np.int32)
        for size in shape
    ]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij", sparse=True)
    slots = np.column_stack(
        np.nonzero(gx * gx + gy * gy + gz * gz <= 3.5**2)
    )
    indices = np.column_stack(
        [axes[axis][slots[:, axis]] for axis in range(3)]
    ).astype(np.int32, copy=False)
    rows = np.arange(len(indices), dtype=float)[:, None]
    bands = np.arange(3, dtype=float)[None, :]
    coefficients = np.asfortranarray(
        np.sin(0.13 * rows + bands) + 1j * np.cos(0.17 * rows - bands)
    )
    xyz = np.sin(0.07 * np.arange(np.prod(shape))).reshape(shape)
    diagonal = np.linspace(0.1, 0.9, len(indices))

    sparse = LocalPotentialWorkspace(indices, shape).apply(
        np.ascontiguousarray(np.moveaxis(xyz, 2, 0)),
        coefficients,
        native_potential_layout=True,
        diagonal=diagonal,
    )
    full = LocalPotentialWorkspace(indices, shape).apply(
        xyz,
        coefficients,
        native_potential_layout=False,
        diagonal=diagonal,
    )

    np.testing.assert_allclose(sparse, full, atol=8.0e-13)


def test_serial_fftw_timer_counts_qe_logical_band_transforms() -> None:
    shape = (6, 6, 6)
    indices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]],
        dtype=np.int32,
    )
    coefficients = np.asfortranarray(
        np.arange(12, dtype=float).reshape(4, 3) + 0.25j
    )
    timers = TimingRegistry()
    workspace = LocalPotentialWorkspace(
        indices,
        shape,
        timers=timers,
        serial_fft_batch_size=1,
    )

    workspace.apply(np.ones(shape), coefficients)
    # QE counts one inverse and one forward transform for every Hpsi band.
    assert timers.entries["fftw"].calls == 2 * coefficients.shape[1]

    density = np.zeros(shape)
    workspace.accumulate_density(
        density, coefficients, np.ones(coefficients.shape[1])
    )
    # Density construction adds one inverse transform per occupied band.
    assert timers.entries["fftw"].calls == 3 * coefficients.shape[1]


def test_fft_scratch_pool_reuses_and_grows_named_storage() -> None:
    pool = FFTScratchPool()
    first = pool.complex_buffer("work", (4, 5), zero=True, alignment=64)
    pointer = first.ctypes.data
    assert pointer % 64 == 0
    first.fill(3.0)
    reused = pool.complex_buffer("work", (2, 5))
    assert reused.ctypes.data == pointer
    assert np.all(reused == 3.0)
    grown = pool.complex_buffer("work", (100,), alignment=64)
    assert grown.size == 100
    assert grown.ctypes.data % 64 == 0
    assert pool.nbytes >= grown.nbytes

