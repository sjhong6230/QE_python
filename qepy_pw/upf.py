"""UPF 2 reader for scalar and fully relativistic NC pseudopotentials.

The reciprocal-space conventions follow QE's ``beta_mod`` and
``init_us_2``: beta radial functions include a factor of ``r`` in UPF,
their transform carries ``4*pi/sqrt(Omega)``, and ``PP_DIJ`` is in Ry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .errors import QEInputError, UnsupportedFeatureError
from .special import erf, spherical_harmonic, spherical_jn

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET


_RADIAL_TABLE_BLOCK = 128


def _complex_spherical_harmonic(
    l: int, m: int, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    return spherical_harmonic(l, m, theta, phi)


@dataclass(frozen=True)
class RadialProjector:
    index: int
    label: str
    angular_momentum: int
    beta: np.ndarray
    cutoff_index: int
    total_angular_momentum: float | None = None


@dataclass(frozen=True)
class AtomicWavefunction:
    index: int
    label: str
    angular_momentum: int
    occupation: float
    chi: np.ndarray
    total_angular_momentum: float | None = None
    principal_quantum_number: int | None = None


@dataclass
class LocalPotential:
    element: str
    z_valence: float
    functional: str
    r: np.ndarray | None
    rab: np.ndarray | None
    vloc_ry: np.ndarray | None
    coulomb: bool
    generated: str = ""
    pseudo_type: str = "NC"
    relativistic: str = "scalar"
    has_spin_orbit: bool = False
    mesh_size: int = 0
    core_density: np.ndarray | None = None
    atomic_density: np.ndarray | None = None
    projectors: tuple[RadialProjector, ...] = ()
    dij_ry: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    atomic_wavefunctions: tuple[AtomicWavefunction, ...] = ()
    _projector_table: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _projector_table_qmax: float = field(
        default=-1.0, init=False, repr=False
    )
    _projector_table_volume: float = field(
        default=-1.0, init=False, repr=False
    )
    _expanded_projector_coupling: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _atomic_wfc_table: tuple[np.ndarray, ...] | None = field(
        default=None, init=False, repr=False
    )
    _atomic_wfc_table_qmax: float = field(
        default=-1.0, init=False, repr=False
    )
    _atomic_wfc_table_volume: float = field(
        default=-1.0, init=False, repr=False
    )
    _core_density_table: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _core_density_table_qmax: float = field(
        default=-1.0, init=False, repr=False
    )
    _core_density_table_volume: float = field(
        default=-1.0, init=False, repr=False
    )

    _local_potential_table: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _local_potential_table_qmax: float = field(
        default=-1.0, init=False, repr=False
    )
    _local_potential_table_volume: float = field(
        default=-1.0, init=False, repr=False
    )

    @property
    def fully_relativistic(self) -> bool:
        """Whether the UPF carries j-resolved, fully relativistic data."""
        return self.has_spin_orbit or self.relativistic.lower() == "full"

    def _ensure_local_potential_table(
        self, maximum_q: float, volume: float
    ) -> np.ndarray:
        """Build QE's dq=.01 short-range Vloc table only when it must grow."""
        assert (
            self.r is not None
            and self.rab is not None
            and self.vloc_ry is not None
        )
        if (
            self._local_potential_table is not None
            and self._local_potential_table_volume == float(volume)
            and self._local_potential_table_qmax + 1.0e-14 >= maximum_q
        ):
            return self._local_potential_table
        beyond = np.flatnonzero(self.r > 10.0)
        count = int(beyond[0] + 1) if len(beyond) else len(self.r)
        if count % 2 == 0:
            count -= 1
        count = max(1, count)
        r = self.r[:count]
        rab = self.rab[:count]
        v_ha = 0.5 * self.vloc_ry[:count]
        dq = 0.01
        table_size = int(maximum_q / dq + 4)
        table_q = np.arange(table_size, dtype=float) * dq
        short_radial = r * v_ha + self.z_valence * erf(r)
        table = np.empty(table_size, dtype=float)
        prefactor = 4.0 * np.pi / volume
        # The complete q-by-r matrix can be tens of MiB at high cutoffs even
        # though the retained QE table is only one dimensional.  Integrate in
        # bounded q blocks so initialization does not create an artificial RSS
        # high-water mark.
        for begin in range(0, table_size, _RADIAL_TABLE_BLOCK):
            end = min(begin + _RADIAL_TABLE_BLOCK, table_size)
            q_block = table_q[begin:end]
            qr = np.multiply.outer(q_block, r)
            sinc = np.ones_like(qr)
            np.divide(
                np.sin(qr),
                q_block[:, None],
                out=sinc,
                where=q_block[:, None] != 0.0,
            )
            if begin == 0:
                sinc[0] = r
            sinc *= short_radial[None, :]
            table[begin:end] = prefactor * _qe_simpson(
                sinc, rab, axis=1
            )
        self._local_potential_table = table
        self._local_potential_table_qmax = maximum_q
        self._local_potential_table_volume = float(volume)
        return self._local_potential_table

    def fourier(self, q: np.ndarray, volume: float) -> np.ndarray:
        """Return periodic V_loc(G), in Hartree, with QE's 1/Omega factor."""
        q_array = np.asarray(q, dtype=float)
        original_shape = q_array.shape
        unique_q, inverse = _unique_radial_arguments(q_array)
        result = np.zeros_like(unique_q)
        if self.coulomb:
            mask = unique_q > 1.0e-12
            result[mask] = -4.0 * np.pi * self.z_valence / (
                volume * unique_q[mask] ** 2
            )
            return result[inverse].reshape(original_shape)
        assert (
            self.r is not None
            and self.rab is not None
            and self.vloc_ry is not None
        )
        beyond = np.flatnonzero(self.r > 10.0)
        count = int(beyond[0] + 1) if len(beyond) else len(self.r)
        if count % 2 == 0:
            count -= 1
        count = max(1, count)
        r = self.r[:count]
        rab = self.rab[:count]
        v_ha = 0.5 * self.vloc_ry[:count]
        prefactor = 4.0 * np.pi / volume

        # QE vloc_mod:init_tab_vloc uses a dq=.01 table on the shortened
        # species mesh.  Index zero is a special G=0 alpha term; positive G
        # values interpolate the short-range transform and then restore the
        # analytic Fourier transform of -Z*erf(r)/r.
        positive = unique_q > 1.0e-12
        maximum_q = float(np.max(unique_q[positive])) if np.any(positive) else 0.0
        dq = 0.01
        table = self._ensure_local_potential_table(maximum_q, volume)
        if np.any(~positive):
            result[~positive] = prefactor * _qe_simpson(
                r * (r * v_ha + self.z_valence), rab
            )
        if np.any(positive):
            scaled = unique_q[positive] / dq
            lower = np.floor(scaled).astype(int)
            fraction = scaled - lower
            u = 1.0 - fraction
            v = 2.0 - fraction
            w = 3.0 - fraction
            interpolated = (
                table[lower] * u * v * w / 6.0
                + table[lower + 1] * fraction * v * w / 2.0
                - table[lower + 2] * fraction * u * w / 2.0
                + table[lower + 3] * fraction * u * v / 6.0
            )
            q_positive = unique_q[positive]
            interpolated -= (
                prefactor
                * self.z_valence
                * np.exp(-0.25 * q_positive**2)
                / q_positive**2
            )
            result[positive] = interpolated
        return result[inverse].reshape(original_shape)

    def fourier_with_derivative(
        self, q: np.ndarray, volume: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``V_loc(q)`` and its analytic radial derivative."""
        q_array = np.asarray(q, dtype=float)
        values = self.fourier(q_array, volume)
        derivative = np.zeros_like(q_array)
        positive = q_array > 1.0e-12
        if self.coulomb:
            derivative[positive] = -2.0 * values[positive] / q_array[positive]
            return values, derivative

        assert (
            self.r is not None
            and self.rab is not None
            and self.vloc_ry is not None
        )
        if not np.any(positive):
            return values, derivative
        prefactor = 4.0 * np.pi / volume
        positive_q = q_array[positive]
        dq = 0.01
        table = self._ensure_local_potential_table(
            float(np.max(positive_q)), volume
        )
        _, table_derivative = _qe_cubic_interpolate_with_derivative(
            table, positive_q, dq
        )
        tail = (
            -prefactor
            * self.z_valence
            * np.exp(-0.25 * positive_q**2)
            / positive_q**2
        )
        derivative[positive] = table_derivative + tail * (
            -0.5 * positive_q - 2.0 / positive_q
        )
        return values, derivative

    @property
    def number_of_projector_channels(self) -> int:
        return sum(
            2 * projector.angular_momentum + 1
            for projector in self.projectors
        )

    def radial_projector_fourier(self, q: np.ndarray, volume: float) -> np.ndarray:
        """Return QE-normalized radial beta transforms, shape ``(nq, nbeta)``."""
        q = np.asarray(q, dtype=float).reshape(-1)
        if not self.projectors:
            return np.empty((len(q), 0))
        assert self.r is not None and self.rab is not None
        if not len(q):
            return np.empty((0, len(self.projectors)))
        # QE beta_mod:init_tab_beta tabulates numerical projectors at dq=.01
        # and init_us_2 obtains beta(|G+k|) with the same four-point cubic
        # interpolation used by its other radial tables.  Direct quadrature
        # at each requested q defines a slightly different Hamiltonian; that
        # error remains even when the SCF conv_thr is made arbitrarily tight.
        dq = 0.01
        maximum_q = float(np.max(q))
        table_size = int(np.floor(maximum_q / dq + 4.0 + 0.5))
        if (
            self._projector_table is None
            or self._projector_table_volume != float(volume)
            or self._projector_table_qmax + 1.0e-14 < maximum_q
        ):
            table_q = np.arange(table_size, dtype=float) * dq
            table = np.empty((table_size, len(self.projectors)))
            prefactor = 4.0 * np.pi / np.sqrt(volume)
            for column, projector in enumerate(self.projectors):
                count = min(
                    projector.cutoff_index,
                    len(self.r),
                    len(projector.beta),
                )
                r = self.r[:count]
                beta = projector.beta[:count]
                radial = beta * r
                for begin in range(0, table_size, _RADIAL_TABLE_BLOCK):
                    end = min(begin + _RADIAL_TABLE_BLOCK, table_size)
                    bessel = spherical_jn(
                        projector.angular_momentum,
                        np.multiply.outer(table_q[begin:end], r),
                    )
                    bessel *= radial[None, :]
                    table[begin:end, column] = prefactor * _qe_simpson(
                        bessel,
                        self.rab[:count],
                        axis=1,
                    )
            self._projector_table = table
            self._projector_table_qmax = maximum_q
            self._projector_table_volume = float(volume)
        table = self._projector_table
        assert table is not None
        from .basis import _load_native_fft

        return _load_native_fft().qe_cubic_interpolate(
            np.ascontiguousarray(table, dtype=np.float64),
            np.ascontiguousarray(q, dtype=np.float64),
            dq,
        )

    @property
    def number_of_spinor_projector_channels(self) -> int:
        """Rank of the j-resolved projector basis for one atom."""
        if not self.fully_relativistic:
            return 2 * self.number_of_projector_channels
        return sum(
            int(round(2.0 * float(projector.total_angular_momentum) + 1.0))
            for projector in self.projectors
        )

    def radial_projector_fourier_with_derivative(
        self, q: np.ndarray, volume: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return radial beta tables and exact cubic-interpolant derivatives."""
        values = self.radial_projector_fourier(q, volume)
        if values.shape[1] == 0:
            return values, np.empty_like(values)
        assert self._projector_table is not None
        _, derivative = _qe_cubic_interpolate_with_derivative(
            self._projector_table, np.asarray(q, dtype=float).reshape(-1), 0.01
        )
        return values, derivative

    @property
    def has_nlcc(self) -> bool:
        return self.core_density is not None

    @property
    def has_atomic_density(self) -> bool:
        return self.atomic_density is not None

    def atomic_density_fourier(
        self, q: np.ndarray, volume: float
    ) -> np.ndarray:
        """QE-tabulated Fourier coefficients of ``PP_RHOATOM`` charge."""
        q_array = np.asarray(q, dtype=float)
        if self.atomic_density is None:
            return np.zeros_like(q_array)
        assert self.r is not None and self.rab is not None
        flat_q = q_array.reshape(-1)
        if not len(flat_q):
            return np.empty_like(q_array)
        # QE read_pseudo introduces msh to suppress numerical noise in the
        # large-r tail.  The first point beyond 10 bohr is included and the
        # number of Simpson points is then forced to be odd.
        beyond = np.flatnonzero(self.r > 10.0)
        count = int(beyond[0] + 1) if len(beyond) else len(self.r)
        if count % 2 == 0:
            count -= 1
        count = max(1, count)
        radial_grid = self.r[:count]
        radial_charge = self.atomic_density[:count]
        rab = self.rab[:count]

        # QE rhoat_mod first constructs tab_rhoat at dq=0.01 and then uses
        # its four-point cubic interpolation, rather than integrating at each
        # requested |G|.  Reproducing the table is important for matching the
        # first SCF iteration.
        dq = 0.01
        maximum_q = float(np.max(flat_q))
        table_size = int(maximum_q / dq + 4)
        table_q = np.arange(table_size, dtype=float) * dq
        table = np.empty(table_size, dtype=float)
        for begin in range(0, table_size, _RADIAL_TABLE_BLOCK):
            end = min(begin + _RADIAL_TABLE_BLOCK, table_size)
            bessel = spherical_jn(
                0, np.multiply.outer(table_q[begin:end], radial_grid)
            )
            bessel *= radial_charge[None, :]
            table[begin:end] = (
                _qe_simpson(
                    bessel,
                    rab,
                    axis=1,
                )
                / volume
            )
        scaled = flat_q / dq
        lower = np.floor(scaled).astype(int)
        fraction = scaled - lower
        u = 1.0 - fraction
        v = 2.0 - fraction
        w = 3.0 - fraction
        transformed = (
            table[lower] * u * v * w / 6.0
            + table[lower + 1] * fraction * v * w / 2.0
            - table[lower + 2] * fraction * u * w / 2.0
            + table[lower + 3] * fraction * u * v / 6.0
        )
        return transformed.reshape(q_array.shape)

    def core_density_fourier(self, q: np.ndarray, volume: float) -> np.ndarray:
        """QE-tabulated frozen-core Fourier coefficients."""
        q_array = np.asarray(q, dtype=float)
        if self.core_density is None:
            return np.zeros_like(q_array)
        flat_q = q_array.reshape(-1)
        if not len(flat_q):
            return np.empty_like(q_array)
        table = self._ensure_core_density_table(flat_q, volume)
        transformed, _ = _qe_cubic_interpolate_with_derivative(
            table, flat_q, 0.01
        )
        return transformed.reshape(q_array.shape)

    def core_density_fourier_derivative(
        self, q: np.ndarray, volume: float
    ) -> np.ndarray:
        """QE derivative of the cubic frozen-core interpolation table."""
        q_array = np.asarray(q, dtype=float)
        if self.core_density is None:
            return np.zeros_like(q_array)
        flat_q = q_array.reshape(-1)
        if not len(flat_q):
            return np.empty_like(q_array)
        table = self._ensure_core_density_table(flat_q, volume)
        _, derivative = _qe_cubic_interpolate_with_derivative(
            table, flat_q, 0.01
        )
        return derivative.reshape(q_array.shape)

    def _ensure_core_density_table(
        self, q: np.ndarray, volume: float
    ) -> np.ndarray:
        """Build QE ``rhoc_mod:init_tab_rhc`` data through requested q."""
        if self.core_density is None:
            raise ValueError("NLCC table requested for a pseudo without NLCC")
        assert self.r is not None and self.rab is not None
        maximum_q = float(np.max(np.asarray(q, dtype=float)))
        if (
            self._core_density_table is None
            or self._core_density_table_volume != float(volume)
            or self._core_density_table_qmax + 1.0e-14 < maximum_q
        ):
            beyond = np.flatnonzero(self.r > 10.0)
            count = int(beyond[0] + 1) if len(beyond) else len(self.r)
            if count % 2 == 0:
                count -= 1
            count = max(1, count)
            dq = 0.01
            table_size = int(maximum_q / dq + 4)
            table_q = np.arange(table_size, dtype=float) * dq
            table = np.empty(table_size, dtype=float)
            radial_core = self.core_density[:count] * self.r[:count] ** 2
            prefactor = 4.0 * np.pi / volume
            for begin in range(0, table_size, _RADIAL_TABLE_BLOCK):
                end = min(begin + _RADIAL_TABLE_BLOCK, table_size)
                bessel = spherical_jn(
                    0,
                    np.multiply.outer(
                        table_q[begin:end], self.r[:count]
                    ),
                )
                bessel *= radial_core[None, :]
                table[begin:end] = prefactor * _qe_simpson(
                    bessel,
                    self.rab[:count],
                    axis=1,
                )
            self._core_density_table = table
            self._core_density_table_qmax = maximum_q
            self._core_density_table_volume = float(volume)
        assert self._core_density_table is not None
        return self._core_density_table

    def projector_basis(
        self, gk: np.ndarray, volume: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the species-centered beta basis and shared coupling matrix."""
        gk = np.asarray(gk, dtype=float)
        if not self.projectors:
            return np.empty((len(gk), 0), dtype=complex), np.empty((0, 0))
        from .basis import _load_native_fft

        q = _load_native_fft().row_norms3(gk)
        radial = self.radial_projector_fourier(q, volume)
        columns: list[np.ndarray] = []
        identities: list[tuple[int, int, int]] | None = (
            [] if self._expanded_projector_coupling is None else None
        )
        projector_columns = [
            (projector_index, projector.angular_momentum, channel)
            for projector_index, projector in enumerate(self.projectors)
            for channel in range(2 * projector.angular_momentum + 1)
        ]
        if all(l_value <= 2 for _, l_value, _ in projector_columns):
            maps = np.asarray(projector_columns, dtype=np.int32)
            beta_matrix = _load_native_fft().assemble_low_l_projectors(
                radial,
                gk,
                q,
                np.ascontiguousarray(maps[:, 0]),
                np.ascontiguousarray(maps[:, 1]),
                np.ascontiguousarray(maps[:, 2]),
            )
            if identities is not None:
                identities.extend(projector_columns)
        else:
            beta_matrix = None
        # UPFs commonly contain several radial projectors with the same l.
        # Their angular factors are identical for a given k point, so build
        # each l block once instead of repeating the SciPy ufunc dispatch.
        if beta_matrix is None:
            harmonics_by_l: dict[int, np.ndarray] = {}
            for projector_index, projector in enumerate(self.projectors):
                angular_momentum = projector.angular_momentum
                harmonics = harmonics_by_l.get(angular_momentum)
                if harmonics is None:
                    harmonics = _qe_real_spherical_harmonics(
                        angular_momentum, gk, q
                    )
                    harmonics_by_l[angular_momentum] = harmonics
                angular_phase = (-1j) ** projector.angular_momentum
                for channel in range(2 * projector.angular_momentum + 1):
                    columns.append(
                        radial[:, projector_index]
                        * harmonics[:, channel]
                        * angular_phase
                    )
                    if identities is not None:
                        identities.append(
                            (projector_index, projector.angular_momentum, channel)
                        )
            beta_matrix = np.asfortranarray(np.column_stack(columns))
        coupling = self._expanded_projector_coupling
        if coupling is None:
            assert identities is not None
            coupling = np.zeros(
                (beta_matrix.shape[1], beta_matrix.shape[1])
            )
            for i, (radial_i, l_i, m_i) in enumerate(identities):
                for j, (radial_j, l_j, m_j) in enumerate(identities):
                    if l_i == l_j and m_i == m_j:
                        coupling[i, j] = (
                            0.5 * self.dij_ry[radial_i, radial_j]
                        )
            coupling.setflags(write=False)
            self._expanded_projector_coupling = coupling
        return beta_matrix, coupling

    def spinor_projector_basis(
        self, gk: np.ndarray, volume: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build j,mj projectors in QE's combined ``(PW, spin)`` layout."""
        if not self.fully_relativistic:
            raise QEInputError("spinor projectors require a fully relativistic UPF")
        vectors = np.asarray(gk, dtype=float)
        q = np.linalg.norm(vectors, axis=1)
        radial = self.radial_projector_fourier(q, volume)
        safe = q > 1.0e-14
        theta = np.zeros(len(q))
        theta[safe] = np.arccos(np.clip(vectors[safe, 2] / q[safe], -1.0, 1.0))
        phi = np.arctan2(vectors[:, 1], vectors[:, 0])
        harmonics: dict[tuple[int, int], np.ndarray] = {}
        columns: list[np.ndarray] = []
        identities: list[tuple[int, int]] = []
        for projector_index, projector in enumerate(self.projectors):
            l_value = projector.angular_momentum
            j_value = float(projector.total_angular_momentum)
            denominator = float(2 * l_value + 1)
            phase = (-1j) ** l_value
            for m_value in range(-l_value - 1, l_value + 1):
                if abs(j_value - l_value - 0.5) < 1.0e-8:
                    up_coefficient = np.sqrt((l_value + m_value + 1.0) / denominator)
                    down_coefficient = np.sqrt((l_value - m_value) / denominator)
                elif abs(j_value - l_value + 0.5) < 1.0e-8:
                    if m_value < -l_value + 1:
                        continue
                    up_coefficient = np.sqrt((l_value - m_value + 1.0) / denominator)
                    down_coefficient = -np.sqrt((l_value + m_value) / denominator)
                else:
                    raise QEInputError(
                        f"incompatible l={l_value}, j={j_value} in relativistic projector"
                    )
                up_harmonic = np.zeros(len(q), dtype=np.complex128)
                if -l_value <= m_value <= l_value:
                    key = (l_value, m_value)
                    if key not in harmonics:
                        harmonics[key] = _complex_spherical_harmonic(
                            l_value, m_value, theta, phi
                        )
                    up_harmonic = harmonics[key]
                down_harmonic = np.zeros(len(q), dtype=np.complex128)
                if -l_value <= m_value + 1 <= l_value:
                    key = (l_value, m_value + 1)
                    if key not in harmonics:
                        harmonics[key] = _complex_spherical_harmonic(
                            l_value, m_value + 1, theta, phi
                        )
                    down_harmonic = harmonics[key]
                column = np.concatenate(
                    (
                        phase * radial[:, projector_index] * up_coefficient * up_harmonic,
                        phase * radial[:, projector_index] * down_coefficient * down_harmonic,
                    )
                )
                columns.append(column)
                identities.append((projector_index, m_value))
        beta = np.asfortranarray(np.column_stack(columns))
        coupling = np.zeros((len(columns), len(columns)), dtype=float)
        for first, (radial_first, m_first) in enumerate(identities):
            projector_first = self.projectors[radial_first]
            for second, (radial_second, m_second) in enumerate(identities):
                projector_second = self.projectors[radial_second]
                if (
                    m_first == m_second
                    and projector_first.angular_momentum
                    == projector_second.angular_momentum
                    and abs(
                        float(projector_first.total_angular_momentum)
                        - float(projector_second.total_angular_momentum)
                    )
                    < 1.0e-8
                ):
                    coupling[first, second] = 0.5 * self.dij_ry[
                        radial_first, radial_second
                    ]
        return beta, coupling

    def spinor_atomic_projectors(
        self, gk: np.ndarray, position: np.ndarray, volume: float
    ) -> tuple[np.ndarray, np.ndarray]:
        beta, coupling = self.spinor_projector_basis(gk, volume)
        phase = np.exp(-1j * (np.asarray(gk) @ np.asarray(position)))
        spinor_phase = np.concatenate((phase, phase))
        return np.asfortranarray(beta * spinor_phase[:, None]), coupling

    def projector_basis_with_gradient(
        self, gk: np.ndarray, volume: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build beta projectors and analytic Cartesian q-space gradients."""
        gk = np.asarray(gk, dtype=float)
        beta, coupling = self.projector_basis(gk, volume)
        if beta.shape[1] == 0:
            return beta, coupling, np.empty((len(gk), 0, 3), dtype=complex)
        q = np.linalg.norm(gk, axis=1)
        radial, radial_derivative = (
            self.radial_projector_fourier_with_derivative(q, volume)
        )
        radial_direction = np.zeros_like(gk)
        nonzero = q > 1.0e-14
        radial_direction[nonzero] = gk[nonzero] / q[nonzero, None]
        gradient_matrix = np.empty(
            (len(gk), beta.shape[1], 3), dtype=np.complex128
        )
        gradient_column = 0
        for projector_index, projector in enumerate(self.projectors):
            harmonics, harmonic_gradient = (
                _qe_real_spherical_harmonics_with_gradient(
                    projector.angular_momentum, gk, q
                )
            )
            angular_phase = (-1j) ** projector.angular_momentum
            for channel in range(2 * projector.angular_momentum + 1):
                gradient_matrix[:, gradient_column, :] = (
                    angular_phase
                    * (
                        radial_derivative[:, projector_index, None]
                        * radial_direction
                        * harmonics[:, channel, None]
                        + radial[:, projector_index, None]
                        * harmonic_gradient[:, channel, :]
                    )
                )
                gradient_column += 1
        if gradient_column != beta.shape[1]:
            raise ValueError("projector gradient channel count is inconsistent")
        return beta, coupling, gradient_matrix

    def atomic_projectors(
        self, gk: np.ndarray, position: np.ndarray, volume: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build one atom's materialized beta matrix for dense diagnostics."""
        gk = np.asarray(gk, dtype=float)
        beta, coupling = self.projector_basis(gk, volume)
        phase = np.exp(-1j * (gk @ np.asarray(position)))
        return np.asfortranarray(beta * phase[:, None]), coupling

    @property
    def number_of_atomic_orbitals(self) -> int:
        return sum(
            2 * wavefunction.angular_momentum + 1
            for wavefunction in self.atomic_wavefunctions
            if wavefunction.occupation >= 0.0
        )

    def atomic_orbitals(
        self, gk: np.ndarray, position: np.ndarray, volume: float
    ) -> np.ndarray:
        """Return QE-normalized reciprocal-space atomic trial orbitals."""
        centered = self.atomic_orbital_basis(gk, volume)
        if centered.shape[1] == 0:
            return centered
        phase = np.exp(-1j * (np.asarray(gk) @ np.asarray(position)))
        return centered * phase[:, None]

    def atomic_orbital_basis(
        self, gk: np.ndarray, volume: float
    ) -> np.ndarray:
        """Return species-centered atomic trial orbitals without atom phases."""
        gk = np.asarray(gk, dtype=float)
        selected = [
            wavefunction
            for wavefunction in self.atomic_wavefunctions
            if wavefunction.occupation >= 0.0
        ]
        if not selected:
            return np.empty((len(gk), 0), dtype=complex)
        assert self.r is not None and self.rab is not None
        q = np.linalg.norm(gk, axis=1)
        prefactor = 4.0 * np.pi / np.sqrt(volume)
        # QE's species mesh ``msh`` is shortened while the UPF is read to
        # suppress the numerically noisy large-r tail.  Both the atomic
        # charge and atomic-wavefunction tables subsequently use that mesh.
        beyond = np.flatnonzero(self.r > 10.0)
        count = int(beyond[0] + 1) if len(beyond) else len(self.r)
        if count % 2 == 0:
            count -= 1
        count = max(1, count)
        radial_grid = self.r[:count]
        rab = self.rab[:count]
        dq = 0.01
        maximum_q = float(np.max(q)) if len(q) else 0.0
        table_size = int(np.floor(maximum_q / dq + 4.0 + 0.5))
        scaled = q / dq
        lower = np.floor(scaled).astype(int)
        fraction = scaled - lower
        u = 1.0 - fraction
        v = 2.0 - fraction
        w = 3.0 - fraction
        if (
            self._atomic_wfc_table is None
            or self._atomic_wfc_table_volume != float(volume)
            or self._atomic_wfc_table_qmax + 1.0e-14 < maximum_q
        ):
            table_q = np.arange(table_size, dtype=float) * dq
            tables = []
            for wavefunction in selected:
                table = np.empty(table_size, dtype=float)
                radial_wfc = wavefunction.chi[:count] * radial_grid
                for begin in range(0, table_size, _RADIAL_TABLE_BLOCK):
                    end = min(begin + _RADIAL_TABLE_BLOCK, table_size)
                    bessel = spherical_jn(
                        wavefunction.angular_momentum,
                        np.multiply.outer(
                            table_q[begin:end], radial_grid
                        ),
                    )
                    bessel *= radial_wfc[None, :]
                    table[begin:end] = prefactor * _qe_simpson(
                        bessel,
                        rab,
                        axis=1,
                    )
                tables.append(table)
            self._atomic_wfc_table = tuple(tables)
            self._atomic_wfc_table_qmax = maximum_q
            self._atomic_wfc_table_volume = float(volume)
        assert self._atomic_wfc_table is not None
        columns: list[np.ndarray] = []
        for wavefunction, table in zip(
            selected, self._atomic_wfc_table
        ):
            radial = (
                table[lower] * u * v * w / 6.0
                + table[lower + 1] * fraction * v * w / 2.0
                - table[lower + 2] * fraction * u * w / 2.0
                + table[lower + 3] * fraction * u * v / 6.0
            )
            harmonics = _qe_real_spherical_harmonics(
                wavefunction.angular_momentum, gk, q
            )
            angular_phase = (1j) ** wavefunction.angular_momentum
            for channel in range(2 * wavefunction.angular_momentum + 1):
                columns.append(
                    radial
                    * harmonics[:, channel]
                    * angular_phase
                )
        return np.column_stack(columns)


def _unique_radial_arguments(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse symmetry-equivalent radial arguments before UPF quadrature."""
    flat = np.asarray(q, dtype=float).reshape(-1)
    # Preserve every argument bit-for-bit. Even physically negligible
    # rounding can select a different basis inside degenerate subspaces and
    # thereby perturb a sensitive nonlinear mixing trajectory.
    return np.unique(flat, return_inverse=True)


def _qe_real_spherical_harmonics(
    angular_momentum: int, vectors: np.ndarray, lengths: np.ndarray | None = None
) -> np.ndarray:
    """Real harmonics in QE ``ylmr2`` order: m=0, cos(1), sin(1), ..."""
    vectors = np.asarray(vectors, dtype=float)
    q = (
        np.linalg.norm(vectors, axis=1)
        if lengths is None
        else np.asarray(lengths)
    )
    # QE's ylmr2 evaluates the low-order real harmonics as Cartesian
    # polynomials. These are the overwhelmingly common norm-conserving
    # projector channels and avoid dispatching SciPy's general complex
    # spherical-harmonic ufunc for every k point and SCF iteration. At q=0,
    # match the former theta=phi=0 convention by choosing the +z direction.
    # The formulas executed by the fused row kernel remain explicit here
    # (x,y,z denote the normalized reciprocal vector):
    #   l=0: [1/sqrt(4*pi)]
    #   l=1: sqrt(3/(4*pi)) * [z, -x, -y]
    #   l=2: [sqrt(5/(16*pi))*(3z^2-1),
    #         -sqrt(15/(4*pi))*xz, -sqrt(15/(4*pi))*yz,
    #          sqrt(15/(16*pi))*(x^2-y^2),
    #          sqrt(15/(4*pi))*xy]
    if angular_momentum <= 2:
        from .basis import _load_native_fft

        return _load_native_fft().low_l_real_harmonics(
            angular_momentum,
            vectors,
            np.ascontiguousarray(q, dtype=np.float64),
        )
    safe = q > 1.0e-14
    theta = np.zeros(len(q))
    theta[safe] = np.arccos(np.clip(vectors[safe, 2] / q[safe], -1.0, 1.0))
    phi = np.arctan2(vectors[:, 1], vectors[:, 0])
    values = [
        np.real(
            _complex_spherical_harmonic(
                angular_momentum, 0, theta, phi
            )
        )
    ]
    for m in range(1, angular_momentum + 1):
        complex_harmonic = _complex_spherical_harmonic(
            angular_momentum, m, theta, phi
        )
        values.extend(
            [np.sqrt(2.0) * np.real(complex_harmonic),
             np.sqrt(2.0) * np.imag(complex_harmonic)]
        )
    return np.column_stack(values)


def _qe_real_spherical_harmonics_with_gradient(
    angular_momentum: int,
    vectors: np.ndarray,
    lengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return QE real harmonics and analytic Cartesian gradients."""
    vectors = np.asarray(vectors, dtype=float)
    q = (
        np.linalg.norm(vectors, axis=1)
        if lengths is None
        else np.asarray(lengths)
    )
    values = _qe_real_spherical_harmonics(angular_momentum, vectors, q)
    gradients = np.zeros((len(q), values.shape[1], 3))
    if angular_momentum == 0:
        return values, gradients

    transverse = np.hypot(vectors[:, 0], vectors[:, 1])
    safe = (q > 1.0e-14) & (transverse > 1.0e-14)
    theta = np.arccos(
        np.clip(vectors[safe, 2] / q[safe], -1.0, 1.0)
    )
    phi = np.arctan2(vectors[safe, 1], vectors[safe, 0])
    theta_gradient = np.column_stack(
        (
            vectors[safe, 2] * vectors[safe, 0]
            / (q[safe] ** 2 * transverse[safe]),
            vectors[safe, 2] * vectors[safe, 1]
            / (q[safe] ** 2 * transverse[safe]),
            -transverse[safe] / q[safe] ** 2,
        )
    )
    phi_gradient = np.column_stack(
        (
            -vectors[safe, 1] / transverse[safe] ** 2,
            vectors[safe, 0] / transverse[safe] ** 2,
            np.zeros(np.count_nonzero(safe)),
        )
    )
    derivative_columns: list[np.ndarray] = []
    for m in range(angular_momentum + 1):
        harmonic = _complex_spherical_harmonic(
            angular_momentum, m, theta, phi
        )
        derivative_theta = (
            m * np.cos(theta) / np.sin(theta) * harmonic
        )
        if m < angular_momentum:
            derivative_theta += (
                np.sqrt(
                    (angular_momentum - m)
                    * (angular_momentum + m + 1)
                )
                * np.exp(-1j * phi)
                * _complex_spherical_harmonic(
                    angular_momentum, m + 1, theta, phi
                )
            )
        derivative_phi = 1j * m * harmonic
        complex_gradient = (
            derivative_theta[:, None] * theta_gradient
            + derivative_phi[:, None] * phi_gradient
        )
        if m == 0:
            derivative_columns.append(np.real(complex_gradient))
        else:
            derivative_columns.extend(
                [
                    np.sqrt(2.0) * np.real(complex_gradient),
                    np.sqrt(2.0) * np.imag(complex_gradient),
                ]
            )
    if np.any(safe):
        gradients[safe] = np.stack(derivative_columns, axis=1)

    # At a polar axis theta/phi coordinates are singular, but the Cartesian
    # limits are finite. Only the |m|=1 channels have nonzero first derivative.
    polar = (q > 1.0e-14) & ~safe
    if np.any(polar):
        coefficient = 0.5 * np.sqrt(
            (2 * angular_momentum + 1)
            * angular_momentum
            * (angular_momentum + 1)
            / (4.0 * np.pi)
        )
        positive_z = vectors[polar, 2] >= 0.0
        derivative_theta = np.where(
            positive_z,
            -coefficient,
            (-1) ** (angular_momentum + 1) * coefficient,
        )
        direction = np.where(positive_z, 1.0, -1.0) / q[polar]
        complex_gradient = np.column_stack(
            (
                direction * derivative_theta,
                1j * direction * derivative_theta,
                np.zeros(np.count_nonzero(polar), dtype=complex),
            )
        )
        # QE order is m=0, Re(m=1), Im(m=1), ... .
        gradients[polar, 1, :] = np.sqrt(2.0) * np.real(complex_gradient)
        gradients[polar, 2, :] = np.sqrt(2.0) * np.imag(complex_gradient)
    return values, gradients


def _qe_cubic_interpolate_with_derivative(
    table: np.ndarray, q: np.ndarray, dq: float
) -> tuple[np.ndarray, np.ndarray]:
    """QE four-point cubic interpolation and its analytic derivative."""
    q = np.asarray(q, dtype=float).reshape(-1)
    scaled = q / dq
    lower = np.floor(scaled).astype(int)
    fraction = scaled - lower
    u = 1.0 - fraction
    v = 2.0 - fraction
    w = 3.0 - fraction
    weights = np.column_stack(
        (
            u * v * w / 6.0,
            fraction * v * w / 2.0,
            -fraction * u * w / 2.0,
            fraction * u * v / 6.0,
        )
    )
    derivatives = np.column_stack(
        (
            -(v * w + u * w + u * v) / 6.0,
            (v * w - fraction * (v + w)) / 2.0,
            -(u * w - fraction * (u + w)) / 2.0,
            (u * v - fraction * (u + v)) / 6.0,
        )
    ) / dq
    samples = np.stack(
        [table[lower + offset] for offset in range(4)], axis=1
    )
    expand = (slice(None), slice(None)) + (None,) * (table.ndim - 1)
    return (
        np.sum(samples * weights[expand], axis=1),
        np.sum(samples * derivatives[expand], axis=1),
    )


def _qe_simpson(values: np.ndarray, rab: np.ndarray, axis: int = -1) -> np.ndarray:
    """QE's ``upflib/simpsn.f90`` quadrature on an arbitrary radial grid."""
    values = np.asarray(values)
    rab = np.asarray(rab)
    size = rab.size
    if size < 3 or values.shape[axis] != size:
        raise QEInputError("radial Simpson integration requires matching arrays of length >= 3")
    weights = np.zeros(size)
    weights[0] = 1.0
    weights[-1] = 1.0
    # Python index 1 is Fortran index 2, so odd Python positions carry the
    # Simpson factor four.  Slice assignment avoids the scalar translation
    # loop while retaining QE's one-based parity exactly.
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    if size % 2 == 0:
        weights[-3] -= 0.25
        weights[-2] += 1.0
        weights[-1] = 1.25
    weighted = rab * weights / 3.0
    normalized_axis = axis % values.ndim
    # Contract the radial axis directly.  The former broadcast multiply
    # materialized another nq-by-mesh array beside the Bessel block; BLAS's
    # matrix-vector path writes only the nq result and is faster for tables.
    radial_last = np.moveaxis(values, normalized_axis, -1)
    return np.tensordot(radial_last, weighted, axes=([-1], [0]))


def _numbers(element: ET.Element | None) -> np.ndarray:
    if element is None or not element.text:
        return np.array([], dtype=float)
    return np.fromstring(
        element.text.replace("D", "E").replace("d", "E"), sep=" "
    )


def read_upf(path: str | Path) -> LocalPotential:
    import re
    import xml.etree.ElementTree as ET

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        # Historical UPF 2 files often place raw Fortran namelists (notably
        # ``&input``) inside PP_INFO. QE treats that section as opaque human
        # documentation, although the unescaped ampersands make it invalid
        # XML. Discard only the PP_INFO payload before parsing machine data.
        machine_text = re.sub(
            r"(<PP_INFO\b[^>]*>).*?(</PP_INFO>)",
            r"\1\2",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        root = ET.fromstring(machine_text)
    except (ET.ParseError, OSError, UnicodeError) as exc:
        raise QEInputError(f"cannot read UPF file {path}: {exc}") from exc
    header = root.find(".//PP_HEADER")
    if header is None:
        raise QEInputError(f"{path} has no PP_HEADER")
    attrs = {key.lower(): value.strip() for key, value in header.attrib.items()}
    flag = lambda key: attrs.get(key, "F").upper().startswith("T")
    if flag("is_ultrasoft") or flag("is_paw"):
        raise UnsupportedFeatureError(f"{path.name}: ultrasoft and PAW potentials are not ported")
    has_spin_orbit = flag("has_so")
    relativistic = attrs.get("relativistic", "scalar").lower()
    nproj = int(float(attrs.get("number_of_proj", "0")))
    coulomb = flag("is_coulomb")
    r = _numbers(root.find(".//PP_R"))
    rab = _numbers(root.find(".//PP_RAB"))
    vloc = _numbers(root.find(".//PP_LOCAL"))
    if not coulomb and (
        not len(r) or len(r) != len(rab) or len(r) != len(vloc)
    ):
        raise QEInputError(f"{path.name}: inconsistent PP_R, PP_RAB, and PP_LOCAL arrays")
    nlcc = _numbers(root.find(".//PP_NLCC")) if flag("core_correction") else None
    if nlcc is not None and len(nlcc) != len(r):
        raise QEInputError(
            f"{path.name}: PP_NLCC has {len(nlcc)} values, expected {len(r)}"
        )
    rhoatom_element = root.find(".//PP_RHOATOM")
    rhoatom = _numbers(rhoatom_element) if rhoatom_element is not None else None
    if rhoatom is not None and len(rhoatom) != len(r):
        raise QEInputError(
            f"{path.name}: PP_RHOATOM has {len(rhoatom)} values, "
            f"expected {len(r)}"
        )
    beta_elements = [
        element
        for element in root.findall(".//PP_NONLOCAL/*")
        if element.tag.upper().startswith("PP_BETA")
    ]
    beta_elements.sort(key=lambda element: int(element.attrib.get("index", "0")))
    if len(beta_elements) != nproj:
        raise QEInputError(
            f"{path.name}: PP_HEADER declares {nproj} projectors, found {len(beta_elements)}"
        )
    spin_orbit = root.find(".//PP_SPIN_ORB")
    relative_beta: dict[int, tuple[int, float]] = {}
    relative_wavefunction: dict[int, tuple[int, int, float]] = {}
    if spin_orbit is not None:
        for element in spin_orbit:
            tag = element.tag.upper()
            index = int(element.attrib.get("index", "0"))
            if tag.startswith("PP_RELBETA"):
                relative_beta[index] = (
                    int(element.attrib.get("lll", "-1")),
                    float(element.attrib.get("jjj", "nan")),
                )
            elif tag.startswith("PP_RELWFC"):
                relative_wavefunction[index] = (
                    int(element.attrib.get("nn", "0")),
                    int(element.attrib.get("lchi", "-1")),
                    float(element.attrib.get("jchi", "nan")),
                )
    if has_spin_orbit and spin_orbit is None:
        raise QEInputError(
            f"{path.name}: PP_HEADER has_so is true but PP_SPIN_ORB is missing"
        )

    projectors: list[RadialProjector] = []
    for position, element in enumerate(beta_elements, start=1):
        beta = _numbers(element)
        if len(beta) != len(r):
            raise QEInputError(
                f"{path.name}: {element.tag} has {len(beta)} values, expected {len(r)}"
            )
        angular_momentum = int(element.attrib.get("angular_momentum", "-1"))
        if angular_momentum < 0:
            raise QEInputError(f"{path.name}: {element.tag} has no angular_momentum")
        cutoff = int(element.attrib.get("cutoff_radius_index", str(len(r))))
        if cutoff <= 1 or cutoff > len(r):
            cutoff = len(r)
        projector_index = int(element.attrib.get("index", position))
        total_angular_momentum = None
        if has_spin_orbit:
            try:
                relative_l, total_angular_momentum = relative_beta[projector_index]
            except KeyError as exc:
                raise QEInputError(
                    f"{path.name}: missing PP_RELBETA for projector {projector_index}"
                ) from exc
            if relative_l != angular_momentum or not np.isfinite(
                total_angular_momentum
            ):
                raise QEInputError(
                    f"{path.name}: inconsistent relativistic projector {projector_index}"
                )
        projectors.append(
            RadialProjector(
                projector_index,
                element.attrib.get("label", f"beta-{position}"),
                angular_momentum,
                beta,
                cutoff,
                total_angular_momentum,
            )
        )
    chi_elements = [
        element
        for element in root.findall(".//PP_PSWFC/*")
        if element.tag.upper().startswith("PP_CHI")
    ]
    chi_elements.sort(key=lambda element: int(element.attrib.get("index", "0")))
    atomic_wavefunctions: list[AtomicWavefunction] = []
    for position, element in enumerate(chi_elements, start=1):
        chi = _numbers(element)
        if len(chi) != len(r):
            raise QEInputError(
                f"{path.name}: {element.tag} has {len(chi)} values, "
                f"expected {len(r)}"
            )
        angular_momentum = int(
            element.attrib.get(
                "l", element.attrib.get("angular_momentum", "-1")
            )
        )
        if angular_momentum < 0:
            raise QEInputError(
                f"{path.name}: {element.tag} has no angular momentum"
            )
        wavefunction_index = int(element.attrib.get("index", position))
        total_angular_momentum = None
        principal_quantum_number = None
        if has_spin_orbit:
            try:
                (
                    principal_quantum_number,
                    relative_l,
                    total_angular_momentum,
                ) = relative_wavefunction[wavefunction_index]
            except KeyError as exc:
                raise QEInputError(
                    f"{path.name}: missing PP_RELWFC for wavefunction {wavefunction_index}"
                ) from exc
            if relative_l != angular_momentum or not np.isfinite(
                total_angular_momentum
            ):
                raise QEInputError(
                    f"{path.name}: inconsistent relativistic wavefunction {wavefunction_index}"
                )
        atomic_wavefunctions.append(
            AtomicWavefunction(
                wavefunction_index,
                element.attrib.get("label", f"chi-{position}"),
                angular_momentum,
                float(element.attrib.get("occupation", "0.0")),
                chi,
                total_angular_momentum,
                principal_quantum_number,
            )
        )
    dij_values = _numbers(root.find(".//PP_DIJ"))
    if nproj:
        if len(dij_values) != nproj * nproj:
            raise QEInputError(
                f"{path.name}: PP_DIJ has {len(dij_values)} values, expected {nproj*nproj}"
            )
        dij = dij_values.reshape((nproj, nproj))
        if not np.allclose(dij, dij.T, atol=1.0e-10):
            raise QEInputError(f"{path.name}: PP_DIJ is not symmetric")
    else:
        dij = np.zeros((0, 0))
    return LocalPotential(
        element=attrs.get("element", path.stem),
        z_valence=float(attrs.get("z_valence", "0")),
        functional=attrs.get("functional", "unknown"),
        r=None if coulomb else r,
        rab=None if coulomb else rab,
        vloc_ry=None if coulomb else vloc,
        coulomb=coulomb,
        generated=attrs.get("generated", ""),
        pseudo_type=attrs.get("pseudo_type", "NC"),
        relativistic=relativistic,
        has_spin_orbit=has_spin_orbit,
        mesh_size=int(float(attrs.get("mesh_size", str(len(r))))),
        core_density=nlcc,
        atomic_density=rhoatom,
        projectors=tuple(projectors),
        dij_ry=dij,
        atomic_wavefunctions=tuple(atomic_wavefunctions),
    )
