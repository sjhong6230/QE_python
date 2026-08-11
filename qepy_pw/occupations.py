"""QE-compatible scalar occupations and Gaussian broadening functions."""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

from .errors import QEInputError, UnsupportedFeatureError
from .special import erf, erfc


_SMEARING_ORDERS = {
    "gauss": 0,
    "gaussian": 0,
    "mp": 1,
    "m-p": 1,
    "methfessel-paxton": 1,
    "methfesselpaxton": 1,
    "mv": -1,
    "m-v": -1,
    "marzari-vanderbilt": -1,
    "marzarivanderbilt": -1,
    "cold": -1,
    "fd": -99,
    "f-d": -99,
    "fermi-dirac": -99,
    "fermidirac": -99,
}

_SMEARING_HELP = (
    "'gaussian'/'gauss', "
    "'methfessel-paxton'/'m-p'/'mp', "
    "'marzari-vanderbilt'/'cold'/'m-v'/'mv', or "
    "'fermi-dirac'/'f-d'/'fd'"
)


def smearing_order(name: object) -> int:
    """Translate QE's ``smearing`` names to its ``ngauss`` value."""
    normalized = str(name).strip().lower().replace("_", "-")
    try:
        return _SMEARING_ORDERS[normalized]
    except KeyError as exc:
        raise UnsupportedFeatureError(
            f"unknown smearing={name!r}; QE accepts {_SMEARING_HELP}"
        ) from exc


def spin_electron_counts(
    nelec: float, tot_magnetization: float = -10000.0
) -> tuple[float, float]:
    """Return QE's ``nelup`` and ``neldw`` for a collinear calculation."""
    if tot_magnetization < -9999.0:
        if abs(nelec - round(nelec)) < 1.0e-8:
            nelup = float(int(round(nelec) + 1) // 2)
            return nelup, float(nelec - nelup)
        return 0.5 * nelec, 0.5 * nelec
    return (
        0.5 * (nelec + tot_magnetization),
        0.5 * (nelec - tot_magnetization),
    )


def default_number_of_bands(
    nelec: float,
    occupations: object,
    *,
    nspin: int = 1,
    tot_magnetization: float = -10000.0,
) -> int:
    """Return QE's ``setup.f90`` default number of bands.

    QE uses Fortran ``NINT`` rather than a ceiling operation.  Electron
    counts are nonnegative here, so ``floor(x + 1/2)`` reproduces its
    round-to-nearest behavior, including half-integer ties away from zero.
    """
    nelup, neldw = spin_electron_counts(nelec, tot_magnetization)
    filled = max(
        int(np.floor(0.5 * nelec + 0.5)),
        int(np.floor(nelup + 0.5)) if nspin == 2 else 0,
        int(np.floor(neldw + 0.5)) if nspin == 2 else 0,
    )
    metallic_modes = {
        "smearing", "tetrahedra", "tetrahedra_lin", "tetrahedra-lin",
        "tetrahedra_opt", "tetrahedra-opt",
    }
    if str(occupations).strip().lower() not in metallic_modes:
        return 2 * filled if nspin == 4 else filled
    metallic = max(
        int(np.floor(1.2 * 0.5 * nelec + 0.5)),
        int(np.floor(1.2 * nelup + 0.5)) if nspin == 2 else 0,
        int(np.floor(1.2 * neldw + 0.5)) if nspin == 2 else 0,
    )
    bands = max(filled + 4, metallic)
    # QE setup.f90 first computes the scalar, twofold-degenerate band count
    # with degspin=2 and only afterwards doubles it for noncollinear spinors,
    # whose bands are not spin degenerate.  Applying the metallic margin to
    # nelec spinor bands directly gives too few empty states (23 rather than
    # QE's 28 for the 19-electron Au regression).
    return 2 * bands if nspin == 4 else bands


def fixed_occupations(
    number_of_bands: int,
    nelup: float,
    neldw: float,
    spatial_kpoints: int,
) -> list[np.ndarray]:
    """Build QE-order fixed LSDA occupations (all up k points, then down)."""
    result: list[np.ndarray] = []
    for electrons in (nelup, neldw):
        if electrons < -1.0e-10 or electrons > number_of_bands + 1.0e-10:
            raise QEInputError("too few spin bands")
        row = np.zeros(number_of_bands, dtype=float)
        whole = min(number_of_bands, int(np.floor(electrons + 1.0e-12)))
        row[:whole] = 1.0
        if whole < number_of_bands:
            row[whole] = max(0.0, min(1.0, electrons - whole))
        result.extend(row.copy() for _ in range(spatial_kpoints))
    return result


_OPTIMIZED_TETRA_WEIGHTS = np.asarray(
    [
        [1440, 0, 30, 0, -38, 7, 17, -28, -56, 9, -46, 9, -38, -28, 17, 7, -18, -18, 12, -18],
        [0, 1440, 0, 30, -28, -38, 7, 17, 9, -56, 9, -46, 7, -38, -28, 17, -18, -18, -18, 12],
        [30, 0, 1440, 0, 17, -28, -38, 7, -46, 9, -56, 9, 17, 7, -38, -28, 12, -18, -18, -18],
        [0, 30, 0, 1440, 7, 17, -28, -38, 9, -46, 9, -56, -28, 17, 7, -38, -18, 12, -18, -18],
    ],
    dtype=float,
) / 1260.0
_IDENTITY_TETRA_INTERPOLATION = np.eye(4)


def _build_tetrahedra(
    grid: tuple[int, int, int],
    full_to_irreducible: np.ndarray,
    reciprocal: np.ndarray,
    shortest_diagonal: bool,
    optimized_weights: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build QE's periodic six-tetrahedra-per-grid-cell connectivity."""
    nx, ny, nz = (int(value) for value in grid)
    mapping = np.asarray(full_to_irreducible, dtype=np.int32)
    if mapping.shape != (nx * ny * nz,):
        raise QEInputError("tetrahedra require a complete automatic k-point grid")
    if shortest_diagonal:
        steps = np.asarray(reciprocal, dtype=float) / np.asarray(grid)[:, None]
        diagonals = np.asarray(
            [
                -steps[0] + steps[1] + steps[2],
                steps[0] - steps[1] + steps[2],
                steps[0] + steps[1] - steps[2],
                steps[0] + steps[1] + steps[2],
            ]
        )
        shaft = int(np.argmin(np.einsum("ij,ij->i", diagonals, diagonals)))
        origin = np.zeros(4, dtype=int)
        directions = np.eye(4, dtype=int)
        origin[shaft] = 1
        directions[shaft, shaft] = -1
        corner_sets = []
        for permutation in itertools.permutations(range(3)):
            corners = [origin[:3].copy()]
            for axis in permutation:
                corners.append(corners[-1] + directions[:3, axis])
            corner_sets.append(np.asarray(corners))
        stencils = []
        for corners in corner_sets:
            v1, v2, v3, v4 = corners
            stencils.append(
                np.asarray(
                    [
                        v1, v2, v3, v4,
                        2*v1-v2, 2*v2-v3, 2*v3-v4, 2*v4-v1,
                        2*v1-v3, 2*v2-v4, 2*v3-v1, 2*v4-v2,
                        2*v1-v4, 2*v2-v1, 2*v3-v2, 2*v4-v3,
                        v4-v1+v2, v1-v2+v3, v2-v3+v4, v3-v4+v1,
                    ]
                )
            )
        if optimized_weights:
            stencil_weights = _OPTIMIZED_TETRA_WEIGHTS
        else:
            stencils = [stencil[:4] for stencil in stencils]
            stencil_weights = np.eye(4)
    else:
        stencils = [
            np.asarray(values, dtype=int)
            for values in (
                ((0,0,0),(1,0,0),(0,1,0),(1,0,1)),
                ((1,0,0),(0,1,0),(1,1,0),(1,0,1)),
                ((0,0,0),(0,1,0),(0,0,1),(1,0,1)),
                ((0,1,0),(1,1,0),(1,0,1),(1,1,1)),
                ((0,1,0),(1,0,1),(0,1,1),(1,1,1)),
                ((0,1,0),(0,0,1),(1,0,1),(0,1,1)),
            )
        ]
        stencil_weights = np.eye(4)
    cells = np.indices((nx, ny, nz), dtype=np.int32).reshape(3, -1).T
    modulo = np.asarray(grid, dtype=np.int32)
    corner_count = len(stencils[0])
    tetrahedra = np.empty(
        (len(cells), len(stencils), corner_count), dtype=np.int32
    )
    # Cap the temporary coordinate block near 24 MiB even for the optimized
    # 20-point stencil. The final connectivity itself is retained and cached.
    cell_block = max(1, 6_000_000 // (3 * corner_count))
    stencil_arrays = [
        np.asarray(stencil, dtype=np.int32) for stencil in stencils
    ]
    for start in range(0, len(cells), cell_block):
        stop = min(len(cells), start + cell_block)
        selected_cells = cells[start:stop]
        for stencil_index, stencil_array in enumerate(stencil_arrays):
            points = (
                selected_cells[:, None, :] + stencil_array[None, :, :]
            ) % modulo
            flat = (
                points[..., 2]
                + nz * (points[..., 1] + ny * points[..., 0])
            )
            tetrahedra[start:stop, stencil_index, :] = mapping[flat]
    return tetrahedra.reshape(-1, corner_count), stencil_weights


@lru_cache(maxsize=16)
def _cached_tetrahedra(
    grid: tuple[int, int, int],
    mapping_bytes: bytes,
    reciprocal_bytes: bytes,
    shortest_diagonal: bool,
    optimized_weights: bool,
) -> tuple[np.ndarray, np.ndarray]:
    mapping = np.frombuffer(mapping_bytes, dtype=np.int32)
    reciprocal = np.frombuffer(reciprocal_bytes, dtype=np.float64).reshape(3, 3)
    return _build_tetrahedra(
        grid,
        mapping,
        reciprocal,
        shortest_diagonal,
        optimized_weights,
    )


def _tetrahedra(
    grid: tuple[int, int, int],
    full_to_irreducible: np.ndarray,
    reciprocal: np.ndarray,
    shortest_diagonal: bool,
    optimized_weights: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return immutable-in-use connectivity cached across SCF iterations."""
    normalized_grid = tuple(int(value) for value in grid)
    mapping = np.ascontiguousarray(full_to_irreducible, dtype=np.int32)
    reciprocal_matrix = np.ascontiguousarray(reciprocal, dtype=np.float64)
    return _cached_tetrahedra(
        normalized_grid,
        mapping.tobytes(),
        reciprocal_matrix.tobytes(),
        shortest_diagonal,
        optimized_weights,
    )


def _tetrahedron_effective_energies(
    band_energies: np.ndarray,
    tetrahedra: np.ndarray,
    interpolation: np.ndarray,
) -> np.ndarray:
    """Gather/interpolate vertices without a full optimized-stencil temporary."""
    if interpolation.shape == (4, 4) and np.array_equal(
        interpolation, _IDENTITY_TETRA_INTERPOLATION
    ):
        return np.transpose(band_energies[tetrahedra], (0, 2, 1))
    tetra_count = len(tetrahedra)
    band_count = band_energies.shape[1]
    point_count = tetrahedra.shape[1]
    effective = np.empty((tetra_count, band_count, 4), dtype=float)
    # The optimized method has a 20-point stencil. Gather it in bounded
    # blocks instead of materializing tetra_count*20*nbnd values at once.
    tetra_block = max(1, 4_000_000 // max(1, point_count * band_count))
    for start in range(0, tetra_count, tetra_block):
        stop = min(tetra_count, start + tetra_block)
        point_energies = band_energies[tetrahedra[start:stop]]
        np.einsum(
            "cp,tpb->tbc",
            interpolation,
            point_energies,
            out=effective[start:stop],
        )
    return effective


def _integrated_tetra_fraction(sorted_energies: np.ndarray, energy: float) -> np.ndarray:
    e1, e2, e3, e4 = np.moveaxis(sorted_energies, -1, 0)
    result = np.zeros_like(e1)
    full = energy >= e4
    result[full] = 1.0
    low = (energy > e1) & (energy < e2)
    result[low] = (energy - e1[low]) ** 3 / (
        (e2[low] - e1[low]) * (e3[low] - e1[low]) * (e4[low] - e1[low])
    )
    high = (energy >= e3) & (energy < e4)
    result[high] = 1.0 - (e4[high] - energy) ** 3 / (
        (e4[high] - e1[high]) * (e4[high] - e2[high]) * (e4[high] - e3[high])
    )
    middle = (energy >= e2) & (energy < e3)
    x = energy - e2[middle]
    result[middle] = (
        (e2[middle] - e1[middle]) ** 2
        + 3.0 * (e2[middle] - e1[middle]) * x
        + 3.0 * x**2
        - (e3[middle] - e1[middle] + e4[middle] - e2[middle])
        / ((e3[middle] - e2[middle]) * (e4[middle] - e2[middle])) * x**3
    ) / ((e3[middle] - e1[middle]) * (e4[middle] - e1[middle]))
    # Completely degenerate tetrahedra are step functions.
    degenerate = np.ptp(sorted_energies, axis=-1) < 1.0e-14
    result[degenerate] = energy >= e1[degenerate]
    return result


def _linear_tetra_moments(energies: np.ndarray, energy: float) -> tuple[np.ndarray, float]:
    """Return normalized vertex moments and DOS for one sorted tetrahedron."""
    e = np.asarray(energies, dtype=float)
    if energy < e[0]:
        return np.zeros(4), 0.0
    if energy >= e[3]:
        return np.full(4, 0.25), 0.0
    if np.ptp(e) < 1.0e-14:
        return (np.full(4, 0.25) if energy >= e[0] else np.zeros(4)), 0.0
    a = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if abs(e[i] - e[j]) > 1.0e-14:
                a[i, j] = (energy - e[j]) / (e[i] - e[j])
    if energy < e[1]:
        c = 0.25 * a[1,0] * a[2,0] * a[3,0]
        moments = c * np.asarray((1+a[0,1]+a[0,2]+a[0,3], a[1,0], a[2,0], a[3,0]))
        dos = 3.0 * (energy-e[0])**2 / ((e[1]-e[0])*(e[2]-e[0])*(e[3]-e[0]))
        return moments, dos
    if energy < e[2]:
        c1 = 0.25 * a[3,0] * a[2,0]
        c2 = 0.25 * a[3,0] * a[2,1] * a[0,2]
        c3 = 0.25 * a[3,1] * a[2,1] * a[0,3]
        moments = np.asarray((
            c1 + (c1+c2)*a[0,2] + (c1+c2+c3)*a[0,3],
            c1+c2+c3 + (c2+c3)*a[1,2] + c3*a[1,3],
            (c1+c2)*a[2,0] + (c2+c3)*a[2,1],
            (c1+c2+c3)*a[3,0] + c3*a[3,1],
        ))
        x = energy-e[1]
        dos = (3*(e[1]-e[0]) + 6*x - 3*(e[2]-e[0]+e[3]-e[1])*x*x/((e[2]-e[1])*(e[3]-e[1]))) / ((e[2]-e[0])*(e[3]-e[0]))
        return moments, dos
    c = a[0,3] * a[1,3] * a[2,3]
    moments = 0.25 * np.asarray((
        1-c*a[0,3], 1-c*a[1,3], 1-c*a[2,3],
        1-c*(1+a[3,0]+a[3,1]+a[3,2]),
    ))
    dos = 3.0 * (e[3]-energy)**2 / ((e[3]-e[0])*(e[3]-e[1])*(e[3]-e[2]))
    return moments, dos


def _linear_tetra_moments_block(
    energies: np.ndarray, energy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ``_linear_tetra_moments`` over tetrahedron/band rows."""
    values = np.asarray(energies, dtype=float)
    if values.ndim < 2 or values.shape[-1] != 4:
        raise ValueError("tetrahedron energies must end in four vertices")
    original_shape = values.shape
    flat = values.reshape(-1, 4)
    e1, e2, e3, e4 = flat.T
    moments = np.zeros_like(flat)
    dos = np.zeros(len(flat), dtype=float)

    full = energy >= e4
    moments[full] = 0.25

    low = (energy >= e1) & (energy < e2)
    if np.any(low):
        selected = flat[low]
        q1 = (energy - selected[:, 0]) / (
            selected[:, 1] - selected[:, 0]
        )
        q2 = (energy - selected[:, 0]) / (
            selected[:, 2] - selected[:, 0]
        )
        q3 = (energy - selected[:, 0]) / (
            selected[:, 3] - selected[:, 0]
        )
        coefficient = 0.25 * q1 * q2 * q3
        moments[low] = coefficient[:, None] * np.column_stack(
            (4.0 - q1 - q2 - q3, q1, q2, q3)
        )
        dos[low] = (
            3.0 * (energy - selected[:, 0]) ** 2
            / (
                (selected[:, 1] - selected[:, 0])
                * (selected[:, 2] - selected[:, 0])
                * (selected[:, 3] - selected[:, 0])
            )
        )

    middle = (energy >= e2) & (energy < e3)
    if np.any(middle):
        selected = flat[middle]
        differences = energy - selected
        # a[i,j]=(E-e[j])/(e[i]-e[j]), matching QE's linear tetrahedron
        # moments. Diagonal elements remain zero.
        denominator = selected[:, :, None] - selected[:, None, :]
        numerator = differences[:, None, :]
        a = np.zeros_like(denominator)
        np.divide(
            numerator,
            denominator,
            out=a,
            where=np.abs(denominator) > 1.0e-14,
        )
        c1 = 0.25 * a[:, 3, 0] * a[:, 2, 0]
        c2 = 0.25 * a[:, 3, 0] * a[:, 2, 1] * a[:, 0, 2]
        c3 = 0.25 * a[:, 3, 1] * a[:, 2, 1] * a[:, 0, 3]
        moments[middle] = np.column_stack(
            (
                c1 + (c1 + c2) * a[:, 0, 2]
                + (c1 + c2 + c3) * a[:, 0, 3],
                c1 + c2 + c3 + (c2 + c3) * a[:, 1, 2]
                + c3 * a[:, 1, 3],
                (c1 + c2) * a[:, 2, 0]
                + (c2 + c3) * a[:, 2, 1],
                (c1 + c2 + c3) * a[:, 3, 0]
                + c3 * a[:, 3, 1],
            )
        )
        x = energy - selected[:, 1]
        dos[middle] = (
            3.0 * (selected[:, 1] - selected[:, 0])
            + 6.0 * x
            - 3.0
            * (selected[:, 2] - selected[:, 0]
               + selected[:, 3] - selected[:, 1])
            * x**2
            / (
                (selected[:, 2] - selected[:, 1])
                * (selected[:, 3] - selected[:, 1])
            )
        ) / (
            (selected[:, 2] - selected[:, 0])
            * (selected[:, 3] - selected[:, 0])
        )

    high = (energy >= e3) & (energy < e4)
    if np.any(high):
        selected = flat[high]
        differences = energy - selected
        denominator = selected[:, :, None] - selected[:, None, :]
        numerator = differences[:, None, :]
        a = np.zeros_like(denominator)
        np.divide(
            numerator,
            denominator,
            out=a,
            where=np.abs(denominator) > 1.0e-14,
        )
        coefficient = a[:, 0, 3] * a[:, 1, 3] * a[:, 2, 3]
        moments[high] = 0.25 * np.column_stack(
            (
                1.0 - coefficient * a[:, 0, 3],
                1.0 - coefficient * a[:, 1, 3],
                1.0 - coefficient * a[:, 2, 3],
                1.0
                - coefficient
                * (1.0 + a[:, 3, 0] + a[:, 3, 1] + a[:, 3, 2]),
            )
        )
        dos[high] = (
            3.0 * (selected[:, 3] - energy) ** 2
            / (
                (selected[:, 3] - selected[:, 0])
                * (selected[:, 3] - selected[:, 1])
                * (selected[:, 3] - selected[:, 2])
            )
        )

    degenerate = np.ptp(flat, axis=1) < 1.0e-14
    moments[degenerate] = (
        0.25 * (energy >= flat[degenerate, 0])[:, None]
    )
    dos[degenerate] = 0.0
    return moments.reshape(original_shape), dos.reshape(original_shape[:-1])


def tetrahedron_occupations(
    eigenvalues: list[np.ndarray],
    irreducible_weights: np.ndarray,
    nelec: float,
    grid: tuple[int, int, int],
    full_to_irreducible: np.ndarray,
    reciprocal: np.ndarray,
    method: str,
    spin_degeneracy: float = 2.0,
    paired_spin_blocks: bool | None = None,
) -> tuple[float, list[np.ndarray]]:
    """QE linear, Blöchl-corrected, or optimized tetrahedron occupations."""
    normalized = str(method).strip().lower().replace("-", "_")
    if normalized not in {"tetrahedra", "tetrahedra_lin", "tetrahedra_opt"}:
        raise QEInputError(f"unknown tetrahedron occupation method {method!r}")
    optimized_connectivity = normalized in {"tetrahedra_lin", "tetrahedra_opt"}
    optimized = normalized == "tetrahedra_opt"
    tetra, interpolation = _tetrahedra(
        grid,
        full_to_irreducible,
        reciprocal,
        optimized_connectivity,
        optimized,
    )
    matrix = np.stack([np.asarray(values, dtype=float) for values in eigenvalues])
    spin_polarized = (
        abs(spin_degeneracy - 1.0) < 1.0e-12
        if paired_spin_blocks is None
        else bool(paired_spin_blocks)
    )
    spatial_points = len(matrix)
    bands_per_spin = matrix.shape[1]
    if spin_polarized:
        if len(matrix) % 2:
            raise QEInputError("LSDA tetrahedra require paired spin k points")
        spatial_points = len(matrix) // 2
        matrix = np.concatenate(
            (matrix[:spatial_points], matrix[spatial_points:]), axis=1
        )
    effective = _tetrahedron_effective_energies(
        matrix, tetra, interpolation
    )
    sorted_effective = np.ascontiguousarray(
        np.sort(effective, axis=-1), dtype=np.float64
    )
    capacity = spin_degeneracy * matrix.shape[1]
    if nelec <= 0.0 or nelec >= capacity - 1.0e-12:
        raise QEInputError(
            f"nbnd provides {capacity:.6g} tetrahedron states for {nelec:.6g} electrons; add unoccupied bands"
        )
    lower = float(np.min(effective))
    upper = float(np.max(effective))
    ntetra = len(tetra)
    from .basis import _load_native_fft
    native = _load_native_fft()
    for _ in range(300):
        fermi = 0.5 * (lower + upper)
        count = (
            spin_degeneracy
            * float(native.tetrahedron_integrated_sum(
                sorted_effective, fermi
            ))
            / ntetra
        )
        if abs(count - nelec) < 1.0e-10:
            break
        if count < nelec:
            lower = fermi
        else:
            upper = fermi
    else:
        raise QEInputError("tetrahedron Fermi-energy search did not converge")
    order = np.argsort(effective, axis=-1, kind="stable")
    sorted_e = np.take_along_axis(effective, order, axis=-1)
    moments, dos = _linear_tetra_moments_block(sorted_e, fermi)
    unsorted = np.empty_like(moments)
    np.put_along_axis(unsorted, order, moments, axis=-1)
    if normalized == "tetrahedra":
        correction = (
            dos[..., None]
            * (np.sum(sorted_e, axis=-1)[..., None] - 4.0 * effective)
            / 40.0
        )
        vertex_weights = unsorted + correction
    else:
        vertex_weights = unsorted
    integrated = native.tetrahedron_accumulate(
        np.ascontiguousarray(tetra, dtype=np.int32),
        np.ascontiguousarray(vertex_weights, dtype=np.float64),
        np.ascontiguousarray(interpolation, dtype=np.float64),
        len(matrix),
    )
    integrated *= spin_degeneracy / ntetra
    point_weights = np.asarray(irreducible_weights, dtype=float)
    if spin_polarized:
        point_weights = point_weights[:spatial_points]
    if np.any(point_weights <= 0.0):
        raise QEInputError("tetrahedron irreducible k-point weights must be positive")
    occupations = integrated / point_weights[:, None]
    # QE averages weights over exactly degenerate bands at each k point.
    sections = (
        ((0, bands_per_spin), (bands_per_spin, 2 * bands_per_spin))
        if spin_polarized
        else ((0, matrix.shape[1]),)
    )
    for ik, values in enumerate(matrix):
        for first, last in sections:
            begin = first
            while begin < last:
                end = begin + 1
                while end < last and abs(values[end] - values[begin]) < 1.0e-6:
                    end += 1
                occupations[ik, begin:end] = np.mean(
                    occupations[ik, begin:end]
                )
                begin = end
    if spin_polarized:
        return float(fermi), [
            *[row[:bands_per_spin].copy() for row in occupations],
            *[row[bands_per_spin:].copy() for row in occupations],
        ]
    return float(fermi), [row.copy() for row in occupations]


def wgauss(x: np.ndarray | float, order: int) -> np.ndarray:
    """QE ``wgauss``: broadened step function for a given ``ngauss``."""
    values = np.asarray(x, dtype=float)
    if order == -99:
        result = np.empty_like(values)
        result[values < -200.0] = 0.0
        result[values > 200.0] = 1.0
        middle = (values >= -200.0) & (values <= 200.0)
        result[middle] = 1.0 / (1.0 + np.exp(-values[middle]))
        return result
    if order == -1:
        shifted = values - 1.0 / np.sqrt(2.0)
        argument = np.minimum(200.0, shifted * shifted)
        return (
            0.5 * erf(shifted)
            + np.exp(-argument) / np.sqrt(2.0 * np.pi)
            + 0.5
        )
    if order < 0:
        raise QEInputError(f"invalid Methfessel-Paxton order {order}")

    argument = np.minimum(200.0, values * values)
    result = 0.5 * erfc(-values)
    if order == 0:
        return result
    hd = np.zeros_like(values)
    hp = np.exp(-argument)
    hermite_order = 0
    coefficient = 1.0 / np.sqrt(np.pi)
    for index in range(1, order + 1):
        hd = 2.0 * values * hp - 2.0 * hermite_order * hd
        hermite_order += 1
        coefficient = -coefficient / (4.0 * index)
        result = result - coefficient * hd
        hp = 2.0 * values * hd - 2.0 * hermite_order * hp
        hermite_order += 1
    return result


def smearing_density(x: np.ndarray | float, order: int) -> np.ndarray:
    """Analytic derivative of QE's supported broadened step functions."""
    values = np.asarray(x, dtype=float)
    if order == -99:
        result = np.zeros_like(values)
        middle = np.abs(values) <= 36.0
        result[middle] = 1.0 / (
            2.0 + np.exp(-values[middle]) + np.exp(values[middle])
        )
        return result
    if order == -1:
        shifted = values - 1.0 / np.sqrt(2.0)
        gaussian = np.exp(-np.minimum(200.0, shifted * shifted)) / np.sqrt(
            np.pi
        )
        return gaussian * (1.0 - np.sqrt(2.0) * shifted)
    gaussian = np.exp(-np.minimum(200.0, values * values)) / np.sqrt(np.pi)
    if order == 0:
        return gaussian
    if order == 1:
        return gaussian * (1.5 - values * values)
    raise QEInputError(
        "analytic smearing density supports ngauss=0, 1, -1, or -99"
    )


def _smearing_density_derivative(
    x: np.ndarray | float, order: int
) -> np.ndarray:
    """QE ``w2gauss``: derivative of the smearing delta function.

    Only the smearing orders accepted by :func:`smearing_order` are needed
    here.  Keeping this analytic is important: ``efermig`` evaluates both
    derivatives repeatedly while refining the Fermi energy.
    """
    values = np.asarray(x, dtype=float)
    if order == -99:
        result = np.zeros_like(values)
        middle = np.abs(values) <= 36.0
        exponent = np.exp(values[middle])
        inverse = 1.0 / exponent
        result[middle] = -(exponent - inverse) / (
            2.0 + exponent + inverse
        ) ** 2
        return result
    if order == -1:
        shifted = values - 1.0 / np.sqrt(2.0)
        gaussian = np.exp(-np.minimum(200.0, shifted * shifted)) / np.sqrt(
            np.pi
        )
        return gaussian * (
            2.0 * np.sqrt(2.0) * values * values
            - 6.0 * values
            + np.sqrt(2.0)
        )
    gaussian = np.exp(-np.minimum(200.0, values * values)) / np.sqrt(np.pi)
    if order == 0:
        return -2.0 * values * gaussian
    if order == 1:
        return values * (2.0 * values * values - 5.0) * gaussian
    raise QEInputError(
        "analytic smearing-density derivative supports ngauss=0, 1, -1, or -99"
    )


def w1gauss(x: np.ndarray | float, order: int) -> np.ndarray:
    """QE ``w1gauss`` variational smearing/free-energy correction."""
    values = np.asarray(x, dtype=float)
    if order == -99:
        result = np.zeros_like(values)
        middle = np.abs(values) <= 36.0
        fermi = 1.0 / (1.0 + np.exp(-values[middle]))
        complement = 1.0 - fermi
        result[middle] = (
            fermi * np.log(fermi)
            + complement * np.log(complement)
        )
        return result
    if order == -1:
        shifted = values - 1.0 / np.sqrt(2.0)
        argument = np.minimum(200.0, shifted * shifted)
        return shifted * np.exp(-argument) / np.sqrt(2.0 * np.pi)
    if order < 0:
        raise QEInputError(f"invalid Methfessel-Paxton order {order}")

    argument = np.minimum(200.0, values * values)
    result = -0.5 * np.exp(-argument) / np.sqrt(np.pi)
    if order == 0:
        return result
    hd = np.zeros_like(values)
    hp = np.exp(-argument)
    hermite_order = 0
    coefficient = 1.0 / np.sqrt(np.pi)
    for index in range(1, order + 1):
        hd = 2.0 * values * hp - 2.0 * hermite_order * hd
        hermite_order += 1
        previous_hp = hp
        hp = 2.0 * values * hd - 2.0 * hermite_order * hp
        hermite_order += 1
        coefficient = -coefficient / (4.0 * index)
        result = result - coefficient * (
            0.5 * hp + hermite_order * previous_hp
        )
    return result


def smeared_occupations(
    eigenvalues: list[np.ndarray],
    weights: np.ndarray,
    nelec: float,
    degauss_ha: float,
    order: int,
    spin_degeneracy: float = 2.0,
    kpoint_weight_sum: float | None = None,
) -> tuple[float, list[np.ndarray], float]:
    """Find the Fermi level, occupations, and QE ``demet`` in Hartree.

    By default returned occupations include scalar spin degeneracy and
    approach two below the Fermi level. Pass ``spin_degeneracy=1`` for LSDA;
    the two spin blocks then each approach one. K-point weights remain
    separate.
    """
    if degauss_ha <= 0.0:
        raise QEInputError("degauss must be positive for smeared occupations")
    if not eigenvalues or any(len(values) == 0 for values in eigenvalues):
        raise QEInputError("cannot determine occupations without bands")
    normalized_weights = np.asarray(weights, dtype=float)
    if normalized_weights.shape != (len(eigenvalues),):
        raise QEInputError("k-point weights do not match eigenvalue arrays")
    if spin_degeneracy <= 0.0:
        raise QEInputError("spin degeneracy must be positive")
    target_weight_sum = (
        2.0 / spin_degeneracy
        if kpoint_weight_sum is None
        else float(kpoint_weight_sum)
    )
    normalized_weights = (
        normalized_weights
        * target_weight_sum
        / np.sum(normalized_weights)
    )
    flat_eigenvalues = np.concatenate(
        [np.asarray(values, dtype=float) for values in eigenvalues]
    )
    flat_weights = np.concatenate(
        [
            np.full(len(values), weight, dtype=float)
            for weight, values in zip(normalized_weights, eigenvalues)
        ]
    )
    capacity = spin_degeneracy * sum(
        weight * len(values)
        for weight, values in zip(normalized_weights, eigenvalues)
    )
    if nelec <= 0.0 or nelec >= capacity - 1.0e-12:
        raise QEInputError(
            f"nbnd provides {capacity:.6g} smeared states for {nelec:.6g} "
            "electrons; add unoccupied bands"
        )

    def electron_count(fermi: float, broadening_order: int = order) -> float:
        return spin_degeneracy * float(
            np.dot(
                flat_weights,
                wgauss(
                    (fermi - flat_eigenvalues) / degauss_ha,
                    broadening_order,
                ),
            )
        )

    minimum = min(float(values[0]) for values in eigenvalues)
    maximum = max(float(values[-1]) for values in eigenvalues)
    lower = minimum - 10.0 * degauss_ha
    upper = maximum + 10.0 * degauss_ha
    while electron_count(lower) > nelec:
        lower -= 10.0 * degauss_ha
    while electron_count(upper) < nelec:
        upper += 10.0 * degauss_ha

    def bisect_fermi(broadening_order: int) -> float:
        """QE-style electron-count bisection without SciPy startup cost."""
        local_lower = lower
        local_upper = upper
        tolerance = 1.0e-12
        if (
            electron_count(local_lower, broadening_order) - nelec > tolerance
            or electron_count(local_upper, broadening_order) - nelec
            < -tolerance
        ):
            raise QEInputError("cannot bracket the Fermi energy")
        for _ in range(300):
            midpoint = 0.5 * (local_lower + local_upper)
            residual = electron_count(midpoint, broadening_order) - nelec
            if abs(residual) < tolerance:
                return midpoint
            if residual < 0.0:
                local_lower = midpoint
            else:
                local_upper = midpoint
        raise QEInputError("Fermi-energy bisection did not converge")

    # QE first obtains a Gaussian estimate for MP/cold smearing.  Its
    # occupation is not necessarily monotonic, so use the same Newton
    # minimization of (N(Ef)-nelec)^2 as QE 7.5 efermig.f90.  The previous
    # implementation sampled 2,001 trial Fermi energies on every SCF step;
    # that dominated otherwise inexpensive LDA calculations with many k
    # points.
    gaussian_fermi = bisect_fermi(-99 if order == -99 else 0)
    if order == 0:
        fermi = gaussian_fermi
    elif order == -99:
        fermi = gaussian_fermi
    else:
        fermi = gaussian_fermi
        residual_tolerance = 1.0e-10
        for _ in range(300):
            scaled = (fermi - flat_eigenvalues) / degauss_ha
            residual = electron_count(fermi) - nelec
            first = spin_degeneracy * float(
                np.dot(flat_weights, smearing_density(scaled, order))
            ) / degauss_ha
            second = spin_degeneracy * float(
                np.dot(
                    flat_weights,
                    _smearing_density_derivative(scaled, order),
                )
            ) / (degauss_ha * degauss_ha)
            numerator = 2.0 * residual * first
            denominator = abs(
                2.0 * (first * first + residual * second)
            )
            if denominator <= residual_tolerance:
                break
            refined = fermi - numerator / denominator
            if (
                abs(refined - fermi) < residual_tolerance
                or abs(electron_count(refined) - nelec) < residual_tolerance
            ):
                fermi = refined
                break
            fermi = refined

        # This is QE's deliberately looser acceptance criterion for MP/cold.
        # Fall back to its legacy actual-smearing bisection only if the local
        # minimization did not locate the physically adjacent root.
        if abs(electron_count(fermi) - nelec) >= 1.0e-2:
            fermi = bisect_fermi(order)

    flat_occupations = spin_degeneracy * wgauss(
        (fermi - flat_eigenvalues) / degauss_ha, order
    )
    boundaries = np.cumsum([len(values) for values in eigenvalues[:-1]])
    occupations = [
        values.copy() for values in np.split(flat_occupations, boundaries)
    ]
    smearing_energy = spin_degeneracy * degauss_ha * float(
        np.dot(
            flat_weights,
            w1gauss(
                (fermi - flat_eigenvalues) / degauss_ha,
                order,
            ),
        )
    )
    return float(fermi), occupations, float(smearing_energy)
