import numpy as np
import pytest

from qepy_pw.basis import FFTGridDescriptor, LocalPotentialWorkspace
from qepy_pw.diagonalization import _orthonormalize
from qepy_pw.mpi import MPIContext


def test_serial_mpi_context_collectives_are_identity():
    context = MPIContext()
    values = np.arange(12.0).reshape(3, 4)
    assert context.is_root
    assert context.size == 1
    assert context.slab(7) == slice(0, 7)
    assert context.slab_for_rank(7, 0) == slice(0, 7)
    assert np.array_equal(context.sum_array(values), values)
    cube = np.arange(24.0).reshape(2, 3, 4)
    assert np.array_equal(
        context.gather_z_slabs(cube, cube.shape), cube
    )
    assert np.array_equal(
        context.scatter_z_slabs_root(cube, cube.shape), cube
    )
    assert context.sum_scalar(3.25) == 3.25
    assert context.broadcast("input") == "input"
    exchanged = context.exchange_complex(
        [np.array([1.0 + 2.0j, 3.0 - 4.0j])]
    )
    assert np.array_equal(
        exchanged[0], np.array([1.0 + 2.0j, 3.0 - 4.0j])
    )
    indexed = context.gather_indexed_rows_root(
        np.array([[3.0 + 1.0j], [1.0 - 2.0j], [2.0 + 0.5j]]),
        np.array([2, 0, 1]),
        total_rows=3,
    )
    np.testing.assert_allclose(
        indexed,
        np.array([[1.0 - 2.0j], [2.0 + 0.5j], [3.0 + 1.0j]]),
    )


def test_mpi_slab_partition_covers_axis_without_overlap():
    slabs = [
        MPIContext(rank=rank, size=4).slab(11)
        for rank in range(4)
    ]
    covered = [
        index
        for slab in slabs
        for index in range(slab.start, slab.stop)
    ]
    assert covered == list(range(11))


def test_g_vector_ownership_is_disjoint_complete_and_balanced():
    shape = (16, 8, 8)
    frequencies = np.array([0, 1, 2, 3, 13, 14, 15])
    indices = np.array(
        [
            (gx, gy, gz)
            for gx in frequencies
            for gy in (-1, 0, 1)
            for gz in (-1, 0, 1)
        ]
    )
    workspaces = [
        LocalPotentialWorkspace(
            indices,
            shape,
            mpi=MPIContext(rank=rank, size=4),
        )
        for rank in range(4)
    ]
    owners = [
        set(workspace.local_plane_wave_indices.tolist())
        for workspace in workspaces
    ]
    assert set.union(*owners) == set(range(len(indices)))
    assert sum(len(owner) for owner in owners) == len(indices)
    assert max(map(len, owners)) - min(map(len, owners)) <= 9


def test_qe_sticks_are_indexed_by_gx_gy_and_extend_along_gz():
    indices = np.array(
        [
            [1, -2, -3],
            [1, -2, 0],
            [1, -2, 4],
            [0, 1, -1],
            [0, 1, 2],
        ]
    )
    workspace = LocalPotentialWorkspace(indices, (9, 9, 9))
    assert len(workspace.sticks) == 2
    assert {tuple(stick) for stick in workspace.sticks} == {
        (1, 7),
        (0, 1),
    }
    # Serial workspaces intentionally retain only linear slots; reconstruct
    # the pairs here to verify that Gz does not enter the stick identity.
    slots = np.column_stack(
        np.unravel_index(workspace.linear_slots, workspace.shape)
    )
    assert np.all(slots[:3, :2] == slots[0, :2])
    assert np.all(slots[3:, :2] == slots[3, :2])
    assert not np.array_equal(slots[0, :2], slots[3, :2])


@pytest.mark.parametrize("backend", ["scipy", "pyfftw"])
def test_distributed_fft_roundtrip_uses_shared_descriptor(backend):
    mpi = MPIContext.world()
    if mpi.size == 1:
        pytest.skip("run this smoke test under mpiexec -n 2 or more")
    if backend == "pyfftw":
        pytest.importorskip("pyfftw")
    shape = (8, 8, 8)
    frequencies = (-2, -1, 0, 1, 2)
    indices = np.asarray(
        [
            (gx, gy, gz)
            for gx in frequencies
            for gy in (-1, 0, 1)
            for gz in (-1, 0, 1)
        ],
        dtype=np.int32,
    )
    descriptor = FFTGridDescriptor.build(indices, shape, mpi.size)
    workspace = LocalPotentialWorkspace(
        indices,
        shape,
        backend=backend,
        planner_effort="estimate",
        mpi=mpi,
        descriptor=descriptor,
    )
    rows = workspace.local_plane_wave_indices
    global_coefficients = (
        np.arange(len(indices) * 2, dtype=float).reshape(len(indices), 2)
        + 1j
        * np.arange(len(indices) * 2, dtype=float)
        .reshape(len(indices), 2)[::-1]
    )
    local = global_coefficients[rows]
    grid = workspace.coefficients_to_grid(local, use_scratch=True)
    recovered = workspace.grid_to_coefficients(grid)
    np.testing.assert_allclose(recovered, local, atol=2.0e-12)


def test_fused_distributed_orthonormalization_preserves_constraints():
    mpi = MPIContext.world()
    if mpi.size == 1:
        pytest.skip("run this regression under mpiexec -n 2 or more")
    rng = np.random.default_rng(812)
    dimension = 47
    against_global, _ = np.linalg.qr(
        rng.normal(size=(dimension, 4))
        + 1j * rng.normal(size=(dimension, 4))
    )
    candidates_global = (
        rng.normal(size=(dimension, 3))
        + 1j * rng.normal(size=(dimension, 3))
    )
    rows = mpi.slab(dimension)
    result = _orthonormalize(
        candidates_global[rows],
        against=against_global[rows],
        mpi=mpi,
    )
    gram = mpi.sum_array(result.conj().T @ result)
    overlap = mpi.sum_array(
        against_global[rows].conj().T @ result
    )
    np.testing.assert_allclose(gram, np.eye(3), atol=2.0e-12)
    np.testing.assert_allclose(overlap, 0.0, atol=2.0e-12)
