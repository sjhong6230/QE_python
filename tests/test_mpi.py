import numpy as np

from qepy_pw.basis import LocalPotentialWorkspace
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
    first_stick = workspace.stick_indices[0]
    assert np.all(workspace.stick_indices[:3] == first_stick)
    assert workspace.stick_indices[3] == workspace.stick_indices[4]
    assert workspace.stick_indices[3] != first_stick
