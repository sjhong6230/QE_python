# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Native serial and distributed FFT hot paths.

This mandatory extension fuses sparse packing, native MPI_Alltoallv, FFTW
plans, local-potential multiplication, and density accumulation.  All payload
arrays remain NumPy-owned so Python-side nbytes accounting stays complete.
"""

from libc.complex cimport creal, cimag
from libc.math cimport cos, sin, sqrt
from libc.stdint cimport uint16_t, uint32_t, int32_t, int64_t
from libc.string cimport memset
from cython.parallel cimport prange

import numpy as np
cimport numpy as cnp

from mpi4py.MPI cimport Comm
from mpi4py.libmpi cimport (
    MPI_Alltoallv,
    MPI_C_DOUBLE_COMPLEX,
)

cnp.import_array()


cdef extern from *:
    """
    #include <complex.h>
    #include <stdint.h>
    #include <omp.h>
    typedef double qepy_fftw_complex[2];
    typedef struct fftw_plan_s *qepy_fftw_plan;
    extern qepy_fftw_plan fftw_plan_many_dft(
        int rank, const int *n, int howmany,
        qepy_fftw_complex *in, const int *inembed,
        int istride, int idist,
        qepy_fftw_complex *out, const int *onembed,
        int ostride, int odist, int sign, unsigned flags);
    extern void fftw_execute_dft(
        const qepy_fftw_plan p,
        qepy_fftw_complex *in,
        qepy_fftw_complex *out);
    extern void fftw_destroy_plan(qepy_fftw_plan p);
    extern int fftw_init_threads(void);
    extern void fftw_plan_with_nthreads(int nthreads);

    static inline long long qepy_index_value(
        const void *data, int itemsize, long long position) {
        if (itemsize == 2) return ((const uint16_t *)data)[position];
        if (itemsize == 4) return ((const uint32_t *)data)[position];
        return ((const int64_t *)data)[position];
    }

    static void qepy_apply_serial_spatial(
        double _Complex *grid, const double _Complex *vectors,
        const double *potential, const void *slots, int slot_itemsize,
        const void *sticks, int stick_itemsize,
        qepy_fftw_plan z_forward, qepy_fftw_plan z_backward,
        qepy_fftw_plan xy_forward, qepy_fftw_plan xy_backward,
        long long nbands, long long nrows, long long nsticks,
        long long nz, long long plane_size, long long fft_size,
        long long grid_stride, long long vector_stride0,
        long long vector_stride1, double _Complex *result,
        long long result_stride0, long long result_stride1,
        long long result_offset) {
        long long task, b, g, z, stick, xy, point, base;
        /* One persistent team covers the complete bounded band block.  The
           implicit barriers reproduce FFTXlib's phase ordering without
           repeatedly creating a team for every small stick transform. */
        #pragma omp parallel private(task,b,g,z,stick,xy,point,base)
        {
            #pragma omp for schedule(static)
            for (task = 0; task < nbands * nrows; ++task) {
                b = task / nrows;
                g = task - b * nrows;
                base = b * grid_stride;
                point = qepy_index_value(slots, slot_itemsize, g);
                grid[base + point] =
                    vectors[g * vector_stride0 + b * vector_stride1];
            }
            #pragma omp for schedule(static)
            for (task = 0; task < nbands * nsticks; ++task) {
                b = task / nsticks;
                stick = task - b * nsticks;
                base = b * grid_stride;
                xy = qepy_index_value(sticks, stick_itemsize, stick);
                fftw_execute_dft(
                    z_backward, (qepy_fftw_complex *)(grid + base + xy),
                    (qepy_fftw_complex *)(grid + base + xy));
            }
            #pragma omp for schedule(static)
            for (task = 0; task < nbands * nz; ++task) {
                b = task / nz;
                z = task - b * nz;
                base = b * grid_stride;
                point = base + z * plane_size;
                fftw_execute_dft(
                    xy_backward, (qepy_fftw_complex *)(grid + point),
                    (qepy_fftw_complex *)(grid + point));
                for (xy = 0; xy < plane_size; ++xy)
                    grid[point + xy] *= potential[z * plane_size + xy];
                fftw_execute_dft(
                    xy_forward, (qepy_fftw_complex *)(grid + point),
                    (qepy_fftw_complex *)(grid + point));
            }
            #pragma omp for schedule(static)
            for (task = 0; task < nbands * nsticks; ++task) {
                b = task / nsticks;
                stick = task - b * nsticks;
                base = b * grid_stride;
                xy = qepy_index_value(sticks, stick_itemsize, stick);
                fftw_execute_dft(
                    z_forward, (qepy_fftw_complex *)(grid + base + xy),
                    (qepy_fftw_complex *)(grid + base + xy));
            }
            #pragma omp for schedule(static)
            for (task = 0; task < nbands * nrows; ++task) {
                b = task / nrows;
                g = task - b * nrows;
                base = b * grid_stride;
                point = qepy_index_value(slots, slot_itemsize, g);
                result[g * result_stride0 +
                       (result_offset + b) * result_stride1] =
                    grid[base + point] / (double)fft_size;
            }
        }
    }

    static void qepy_accumulate_density_serial_spatial(
        double *density, double _Complex *grid,
        const double _Complex *vectors, const double *weights,
        const void *slots, int slot_itemsize,
        const void *sticks, int stick_itemsize,
        qepy_fftw_plan z_backward, qepy_fftw_plan xy_backward,
        long long nbands, long long nrows, long long nsticks,
        long long nz, long long plane_size, long long fft_size,
        long long grid_stride, long long vector_stride0,
        long long vector_stride1) {
        long long task, b, g, z, stick, xy, point, base;
        double real_part, imag_part, contribution;
        #pragma omp parallel private(task,b,g,z,stick,xy,point,base,real_part,imag_part,contribution)
        {
            #pragma omp for schedule(static)
            for (task = 0; task < nbands * nrows; ++task) {
                b = task / nrows;
                g = task - b * nrows;
                base = b * grid_stride;
                point = qepy_index_value(slots, slot_itemsize, g);
                grid[base + point] =
                    vectors[g * vector_stride0 + b * vector_stride1];
            }
            #pragma omp for schedule(static)
            for (task = 0; task < nbands * nsticks; ++task) {
                b = task / nsticks;
                stick = task - b * nsticks;
                base = b * grid_stride;
                xy = qepy_index_value(sticks, stick_itemsize, stick);
                fftw_execute_dft(
                    z_backward, (qepy_fftw_complex *)(grid + base + xy),
                    (qepy_fftw_complex *)(grid + base + xy));
            }
            #pragma omp for schedule(static)
            for (task = 0; task < nbands * nz; ++task) {
                b = task / nz;
                z = task - b * nz;
                point = b * grid_stride + z * plane_size;
                fftw_execute_dft(
                    xy_backward, (qepy_fftw_complex *)(grid + point),
                    (qepy_fftw_complex *)(grid + point));
            }
            #pragma omp for schedule(static)
            for (point = 0; point < fft_size; ++point) {
                z = point / plane_size;
                xy = point - z * plane_size;
                contribution = 0.0;
                for (b = 0; b < nbands; ++b) {
                    real_part = creal(grid[b * grid_stride + point]);
                    imag_part = cimag(grid[b * grid_stride + point]);
                    contribution += weights[b] *
                        (real_part * real_part + imag_part * imag_part);
                }
                density[xy * nz + z] += contribution;
            }
        }
    }
    """
    ctypedef double qepy_fftw_complex[2]
    ctypedef void* qepy_fftw_plan
    qepy_fftw_plan fftw_plan_many_dft(
        int rank, const int *n, int howmany,
        qepy_fftw_complex *input_array, const int *inembed,
        int istride, int idist,
        qepy_fftw_complex *output_array, const int *onembed,
        int ostride, int odist, int sign, unsigned flags,
    ) noexcept nogil
    void fftw_execute_dft(
        qepy_fftw_plan plan,
        qepy_fftw_complex *input_array,
        qepy_fftw_complex *output_array,
    ) noexcept nogil
    void fftw_destroy_plan(qepy_fftw_plan plan) noexcept nogil
    int fftw_init_threads() noexcept nogil
    void fftw_plan_with_nthreads(int nthreads) noexcept nogil
    void qepy_apply_serial_spatial(
        double complex *grid,
        const double complex *vectors,
        const double *potential,
        const void *slots,
        int slot_itemsize,
        const void *sticks,
        int stick_itemsize,
        qepy_fftw_plan z_forward,
        qepy_fftw_plan z_backward,
        qepy_fftw_plan xy_forward,
        qepy_fftw_plan xy_backward,
        long long nbands,
        long long nrows,
        long long nsticks,
        long long nz,
        long long plane_size,
        long long fft_size,
        long long grid_stride,
        long long vector_stride0,
        long long vector_stride1,
        double complex *result,
        long long result_stride0,
        long long result_stride1,
        long long result_offset,
    ) noexcept nogil
    void qepy_accumulate_density_serial_spatial(
        double *density,
        double complex *grid,
        const double complex *vectors,
        const double *weights,
        const void *slots,
        int slot_itemsize,
        const void *sticks,
        int stick_itemsize,
        qepy_fftw_plan z_backward,
        qepy_fftw_plan xy_backward,
        long long nbands,
        long long nrows,
        long long nsticks,
        long long nz,
        long long plane_size,
        long long fft_size,
        long long grid_stride,
        long long vector_stride0,
        long long vector_stride1,
    ) noexcept nogil


cdef extern from *:
    """
    #include <complex.h>
    #include <dlfcn.h>
    typedef long long qepy_lapack_int;
    typedef qepy_lapack_int (*qepy_zhegvd64_function)(
        int, qepy_lapack_int, char, char, qepy_lapack_int,
        double _Complex *, qepy_lapack_int, double _Complex *,
        qepy_lapack_int, double *);

    static qepy_lapack_int qepy_zhegvd64(
        int matrix_layout, qepy_lapack_int itype, char jobz, char uplo,
        qepy_lapack_int n, double _Complex *a, qepy_lapack_int lda,
        double _Complex *b, qepy_lapack_int ldb, double *w) {
        static qepy_zhegvd64_function function = NULL;
        static int attempted = 0;
        if (!attempted) {
            attempted = 1;
            function = (qepy_zhegvd64_function)dlsym(
                RTLD_DEFAULT, "scipy_LAPACKE_zhegvd64_");
        }
        if (function == NULL) return -1000000;
        return function(
            matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
    }
    """
    ctypedef long long qepy_lapack_int
    qepy_lapack_int qepy_zhegvd64(
        int matrix_layout,
        qepy_lapack_int itype,
        char jobz,
        char uplo,
        qepy_lapack_int n,
        double complex *a,
        qepy_lapack_int lda,
        double complex *b,
        qepy_lapack_int ldb,
        double *w,
    ) noexcept nogil


def generalized_eigh(cnp.ndarray hamiltonian, cnp.ndarray overlap, int roots):
    """Use NumPy's already-loaded LAPACK for one generalized eigensolve.

    Return ``None`` when the NumPy distribution does not expose the ILP64
    OpenBLAS LAPACKE symbol; the Python caller then uses its portable NumPy
    reduction.  No BLAS library is linked into this extension explicitly.
    """
    cdef cnp.ndarray a
    cdef cnp.ndarray b
    cdef cnp.ndarray values
    cdef qepy_lapack_int dimension
    cdef qepy_lapack_int info
    dimension = hamiltonian.shape[0]
    a = np.array(hamiltonian, dtype=np.complex128, order="F", copy=True)
    b = np.array(overlap, dtype=np.complex128, order="F", copy=True)
    values = np.empty(dimension, dtype=np.float64)
    with nogil:
        info = qepy_zhegvd64(
            102, 1, 'V', 'L', dimension,
            <double complex *>cnp.PyArray_DATA(a), dimension,
            <double complex *>cnp.PyArray_DATA(b), dimension,
            <double *>cnp.PyArray_DATA(values),
        )
    if info == -1000000:
        return None
    if info != 0:
        raise np.linalg.LinAlgError(
            f"native LAPACK zhegvd failed with info={info}"
        )
    return values[:roots].copy(), a[:, :roots].copy(order="F")


cdef extern from "omp.h":
    void omp_set_num_threads(int num_threads) noexcept nogil
    int omp_get_max_threads() noexcept nogil
    int omp_get_thread_num() noexcept nogil


def configure_openmp_threads(int nthreads):
    """Set the rank-local OpenMP team used by compiled array kernels."""
    if nthreads < 1:
        raise ValueError("OpenMP thread count must be positive")
    omp_set_num_threads(nthreads)


def qe_precondition(
    cnp.ndarray residuals,
    cnp.ndarray eigenvalues,
    cnp.ndarray diagonal,
):
    """Apply QE's ``g_psi`` formula with PW-row OpenMP decomposition."""
    cdef Py_ssize_t nrows = residuals.shape[0]
    cdef Py_ssize_t nbands = residuals.shape[1]
    cdef Py_ssize_t total = nrows * nbands
    cdef Py_ssize_t index, g, b
    cdef Py_ssize_t rs0 = residuals.strides[0] // 16
    cdef Py_ssize_t rs1 = residuals.strides[1] // 16
    cdef cnp.ndarray output = np.empty(
        (nrows, nbands), dtype=np.complex128, order="F"
    )
    cdef double complex *source = <double complex *>cnp.PyArray_DATA(residuals)
    cdef double complex *target = <double complex *>cnp.PyArray_DATA(output)
    cdef double *values = <double *>cnp.PyArray_DATA(eigenvalues)
    cdef double *kinetic = <double *>cnp.PyArray_DATA(diagonal)
    cdef double x_ry, denominator
    if residuals.ndim != 2 or residuals.dtype != np.complex128:
        raise ValueError("residuals must be a complex128 matrix")
    if (
        eigenvalues.ndim != 1
        or eigenvalues.dtype != np.float64
        or not eigenvalues.flags.c_contiguous
        or eigenvalues.size != nbands
    ):
        raise ValueError("eigenvalues must be contiguous float64")
    if (
        diagonal.ndim != 1
        or diagonal.dtype != np.float64
        or not diagonal.flags.c_contiguous
        or diagonal.size != nrows
    ):
        raise ValueError("diagonal must be contiguous float64")
    with nogil:
        for index in prange(
            total, schedule="static", use_threads_if=total >= 1024
        ):
            b = index // nrows
            g = index - b * nrows
            x_ry = 2.0 * (kinetic[g] - values[b])
            denominator = 0.5 * (
                1.0 + x_ry + sqrt(1.0 + (x_ry - 1.0) * (x_ry - 1.0))
            )
            target[index] = 2.0 * source[g * rs0 + b * rs1] / denominator
    return output


def column_squared_norms(cnp.ndarray vectors):
    """Return rank-local column norms using QE-style PW-row threading."""
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t nbands = vectors.shape[1]
    cdef Py_ssize_t total = nrows * nbands
    cdef Py_ssize_t padded = ((nbands + 7) // 8) * 8
    cdef Py_ssize_t index, g, b
    cdef Py_ssize_t vs0 = vectors.strides[0] // 16
    cdef Py_ssize_t vs1 = vectors.strides[1] // 16
    cdef int nthreads = omp_get_max_threads()
    cdef int thread
    cdef cnp.ndarray partial = np.zeros(
        (nthreads, padded), dtype=np.float64
    )
    cdef cnp.ndarray norms = np.zeros(nbands, dtype=np.float64)
    cdef double complex *source = <double complex *>cnp.PyArray_DATA(vectors)
    cdef double *sums = <double *>cnp.PyArray_DATA(partial)
    cdef double *target = <double *>cnp.PyArray_DATA(norms)
    cdef double real_part, imag_part
    if vectors.ndim != 2 or vectors.dtype != np.complex128:
        raise ValueError("vectors must be a complex128 matrix")
    with nogil:
        for index in prange(
            total, schedule="static", use_threads_if=total >= 1024
        ):
            thread = omp_get_thread_num()
            b = index // nrows
            g = index - b * nrows
            real_part = creal(source[g * vs0 + b * vs1])
            imag_part = cimag(source[g * vs0 + b * vs1])
            sums[thread * padded + b] += (
                real_part * real_part + imag_part * imag_part
            )
        for b in prange(
            nbands, schedule="static", use_threads_if=nbands > 1
        ):
            for thread in range(nthreads):
                target[b] += sums[thread * padded + b]
    return norms


def normalize_selected_columns(
    cnp.ndarray vectors,
    cnp.ndarray squared_norms,
    cnp.ndarray selected,
):
    """Copy and normalize selected Davidson columns without NumPy temporaries."""
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t nbands = selected.size
    cdef Py_ssize_t total = nrows * nbands
    cdef Py_ssize_t index, g, b, source_band
    cdef Py_ssize_t vs0 = vectors.strides[0] // 16
    cdef Py_ssize_t vs1 = vectors.strides[1] // 16
    cdef cnp.ndarray output = np.empty(
        (nrows, nbands), dtype=np.complex128, order="F"
    )
    cdef double complex *source = <double complex *>cnp.PyArray_DATA(vectors)
    cdef double complex *target = <double complex *>cnp.PyArray_DATA(output)
    cdef double *norms = <double *>cnp.PyArray_DATA(squared_norms)
    cdef int64_t *columns = <int64_t *>cnp.PyArray_DATA(selected)
    if vectors.ndim != 2 or vectors.dtype != np.complex128:
        raise ValueError("vectors must be a complex128 matrix")
    if squared_norms.dtype != np.float64 or not squared_norms.flags.c_contiguous:
        raise ValueError("squared_norms must be contiguous float64")
    if selected.dtype != np.int64 or not selected.flags.c_contiguous:
        raise ValueError("selected must be contiguous int64")
    with nogil:
        for index in prange(
            total, schedule="static", use_threads_if=total >= 1024
        ):
            b = index // nrows
            g = index - b * nrows
            source_band = columns[b]
            target[index] = source[g * vs0 + source_band * vs1] / sqrt(
                norms[source_band]
            )
    return output


def row_norms3(cnp.ndarray vectors):
    """Euclidean norms of Cartesian rows, parallelized like QE PW loops."""
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t row
    cdef Py_ssize_t stride0 = vectors.strides[0] // 8
    cdef Py_ssize_t stride1 = vectors.strides[1] // 8
    cdef cnp.ndarray output = np.empty(nrows, dtype=np.float64)
    cdef double *source = <double *>cnp.PyArray_DATA(vectors)
    cdef double *target = <double *>cnp.PyArray_DATA(output)
    cdef double x, y, z
    if vectors.ndim != 2 or vectors.shape[1] != 3 or vectors.dtype != np.float64:
        raise ValueError("vectors must be an N-by-3 float64 matrix")
    with nogil:
        for row in prange(
            nrows, schedule="static", use_threads_if=nrows >= 1024
        ):
            x = source[row * stride0]
            y = source[row * stride0 + stride1]
            z = source[row * stride0 + 2 * stride1]
            target[row] = sqrt(x * x + y * y + z * z)
    return output


def qe_cubic_interpolate(
    cnp.ndarray table,
    cnp.ndarray q,
    double dq,
):
    """Fused four-point QE table interpolation without NumPy temporaries."""
    cdef Py_ssize_t nrows = q.size
    cdef Py_ssize_t ncolumns = table.shape[1]
    cdef Py_ssize_t total = nrows * ncolumns
    cdef Py_ssize_t index, row, column, lower
    cdef cnp.ndarray output = np.empty(
        (nrows, ncolumns), dtype=np.float64, order="C"
    )
    cdef double *samples = <double *>cnp.PyArray_DATA(table)
    cdef double *positions = <double *>cnp.PyArray_DATA(q)
    cdef double *target = <double *>cnp.PyArray_DATA(output)
    cdef double scaled, fraction, u, v, w
    if table.ndim != 2 or table.dtype != np.float64 or not table.flags.c_contiguous:
        raise ValueError("table must be a contiguous float64 matrix")
    if q.ndim != 1 or q.dtype != np.float64 or not q.flags.c_contiguous:
        raise ValueError("q must be contiguous float64")
    if dq <= 0.0:
        raise ValueError("dq must be positive")
    with nogil:
        for index in prange(
            total, schedule="static", use_threads_if=total >= 1024
        ):
            row = index // ncolumns
            column = index - row * ncolumns
            scaled = positions[row] / dq
            lower = <Py_ssize_t>scaled
            fraction = scaled - lower
            u = 1.0 - fraction
            v = 2.0 - fraction
            w = 3.0 - fraction
            target[index] = (
                samples[lower * ncolumns + column] * u * v * w / 6.0
                + samples[(lower + 1) * ncolumns + column]
                * fraction * v * w / 2.0
                - samples[(lower + 2) * ncolumns + column]
                * fraction * u * w / 2.0
                + samples[(lower + 3) * ncolumns + column]
                * fraction * u * v / 6.0
            )
    return output


def low_l_real_harmonics(
    int angular_momentum,
    cnp.ndarray vectors,
    cnp.ndarray lengths,
):
    """QE ``ylmr2`` Cartesian harmonics for l=0,1,2 with row threading."""
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t ncolumns = 2 * angular_momentum + 1
    cdef Py_ssize_t row
    cdef Py_ssize_t stride0 = vectors.strides[0] // 8
    cdef Py_ssize_t stride1 = vectors.strides[1] // 8
    cdef cnp.ndarray output
    cdef double *source
    cdef double *q
    cdef double *target
    cdef double inverse, x, y, z
    cdef double pi_value = np.pi
    cdef double scale_l0, scale_l1, scale0, scale1, scale2
    if angular_momentum < 0 or angular_momentum > 2:
        raise ValueError("native harmonics support l=0,1,2")
    if vectors.ndim != 2 or vectors.shape[1] != 3 or vectors.dtype != np.float64:
        raise ValueError("vectors must be an N-by-3 float64 matrix")
    if lengths.ndim != 1 or lengths.dtype != np.float64 or not lengths.flags.c_contiguous:
        raise ValueError("lengths must be contiguous float64")
    output = np.empty((nrows, ncolumns), dtype=np.float64)
    source = <double *>cnp.PyArray_DATA(vectors)
    q = <double *>cnp.PyArray_DATA(lengths)
    target = <double *>cnp.PyArray_DATA(output)
    scale_l0 = 1.0 / sqrt(4.0 * pi_value)
    scale_l1 = sqrt(3.0 / (4.0 * pi_value))
    scale0 = sqrt(5.0 / (16.0 * pi_value))
    scale1 = sqrt(15.0 / (4.0 * pi_value))
    scale2 = sqrt(15.0 / (16.0 * pi_value))
    with nogil:
        for row in prange(
            nrows, schedule="static", use_threads_if=nrows >= 1024
        ):
            if q[row] > 1.0e-14:
                inverse = 1.0 / q[row]
                x = source[row * stride0] * inverse
                y = source[row * stride0 + stride1] * inverse
                z = source[row * stride0 + 2 * stride1] * inverse
            else:
                x = 0.0
                y = 0.0
                z = 1.0
            if angular_momentum == 0:
                target[row] = scale_l0
            elif angular_momentum == 1:
                target[row * 3] = scale_l1 * z
                target[row * 3 + 1] = -scale_l1 * x
                target[row * 3 + 2] = -scale_l1 * y
            else:
                target[row * 5] = scale0 * (3.0 * z * z - 1.0)
                target[row * 5 + 1] = -scale1 * x * z
                target[row * 5 + 2] = -scale1 * y * z
                target[row * 5 + 3] = scale2 * (x * x - y * y)
                target[row * 5 + 4] = scale1 * x * y
    return output


def phase_matrix(cnp.ndarray vectors, cnp.ndarray positions):
    """Return exp(-i G.r) with a fused PW-row OpenMP loop."""
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t natoms = positions.shape[0]
    cdef Py_ssize_t total = nrows * natoms
    cdef Py_ssize_t index, row, atom
    cdef Py_ssize_t vs0 = vectors.strides[0] // 8
    cdef Py_ssize_t vs1 = vectors.strides[1] // 8
    cdef Py_ssize_t ps0 = positions.strides[0] // 8
    cdef Py_ssize_t ps1 = positions.strides[1] // 8
    cdef cnp.ndarray output = np.empty(
        (nrows, natoms), dtype=np.complex128, order="F"
    )
    cdef double *g_vectors = <double *>cnp.PyArray_DATA(vectors)
    cdef double *atoms = <double *>cnp.PyArray_DATA(positions)
    cdef double complex *target = <double complex *>cnp.PyArray_DATA(output)
    cdef double angle
    if vectors.ndim != 2 or vectors.shape[1] != 3 or vectors.dtype != np.float64:
        raise ValueError("vectors must be an N-by-3 float64 matrix")
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.dtype != np.float64:
        raise ValueError("positions must be an M-by-3 float64 matrix")
    with nogil:
        for index in prange(
            total, schedule="static", use_threads_if=total >= 1024
        ):
            atom = index // nrows
            row = index - atom * nrows
            angle = (
                g_vectors[row * vs0] * atoms[atom * ps0]
                + g_vectors[row * vs0 + vs1]
                * atoms[atom * ps0 + ps1]
                + g_vectors[row * vs0 + 2 * vs1]
                * atoms[atom * ps0 + 2 * ps1]
            )
            target[index] = cos(angle) - 1j * sin(angle)
    return output


def assemble_low_l_projectors(
    cnp.ndarray radial,
    cnp.ndarray vectors,
    cnp.ndarray lengths,
    cnp.ndarray radial_indices,
    cnp.ndarray angular_momenta,
    cnp.ndarray channels,
):
    """Fuse QE l<=2 angular factors and radial projector assembly."""
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t ncolumns = radial_indices.size
    cdef Py_ssize_t total = nrows * ncolumns
    cdef Py_ssize_t index, row, column, radial_column
    cdef Py_ssize_t rs0 = radial.strides[0] // 8
    cdef Py_ssize_t rs1 = radial.strides[1] // 8
    cdef Py_ssize_t vs0 = vectors.strides[0] // 8
    cdef Py_ssize_t vs1 = vectors.strides[1] // 8
    cdef cnp.ndarray output = np.empty(
        (nrows, ncolumns), dtype=np.complex128, order="F"
    )
    cdef double *radial_values = <double *>cnp.PyArray_DATA(radial)
    cdef double *g_vectors = <double *>cnp.PyArray_DATA(vectors)
    cdef double *q = <double *>cnp.PyArray_DATA(lengths)
    cdef int32_t *radial_map = <int32_t *>cnp.PyArray_DATA(radial_indices)
    cdef int32_t *l_map = <int32_t *>cnp.PyArray_DATA(angular_momenta)
    cdef int32_t *channel_map = <int32_t *>cnp.PyArray_DATA(channels)
    cdef double complex *target = <double complex *>cnp.PyArray_DATA(output)
    cdef double pi_value = np.pi
    cdef double scale_l0 = 1.0 / sqrt(4.0 * pi_value)
    cdef double scale_l1 = sqrt(3.0 / (4.0 * pi_value))
    cdef double scale20 = sqrt(5.0 / (16.0 * pi_value))
    cdef double scale21 = sqrt(15.0 / (4.0 * pi_value))
    cdef double scale22 = sqrt(15.0 / (16.0 * pi_value))
    cdef double inverse, x, y, z, harmonic, value
    cdef int l_value, channel
    if radial.ndim != 2 or radial.dtype != np.float64:
        raise ValueError("radial must be a float64 matrix")
    if vectors.ndim != 2 or vectors.shape[1] != 3 or vectors.dtype != np.float64:
        raise ValueError("vectors must be an N-by-3 float64 matrix")
    if lengths.dtype != np.float64 or not lengths.flags.c_contiguous:
        raise ValueError("lengths must be contiguous float64")
    for mapping in (radial_indices, angular_momenta, channels):
        if mapping.dtype != np.int32 or not mapping.flags.c_contiguous:
            raise ValueError("projector maps must be contiguous int32")
    with nogil:
        for index in prange(
            total, schedule="static", use_threads_if=total >= 1024
        ):
            column = index // nrows
            row = index - column * nrows
            radial_column = radial_map[column]
            l_value = l_map[column]
            channel = channel_map[column]
            if q[row] > 1.0e-14:
                inverse = 1.0 / q[row]
                x = g_vectors[row * vs0] * inverse
                y = g_vectors[row * vs0 + vs1] * inverse
                z = g_vectors[row * vs0 + 2 * vs1] * inverse
            else:
                x = 0.0
                y = 0.0
                z = 1.0
            if l_value == 0:
                harmonic = scale_l0
            elif l_value == 1:
                if channel == 0:
                    harmonic = scale_l1 * z
                elif channel == 1:
                    harmonic = -scale_l1 * x
                else:
                    harmonic = -scale_l1 * y
            else:
                if channel == 0:
                    harmonic = scale20 * (3.0 * z * z - 1.0)
                elif channel == 1:
                    harmonic = -scale21 * x * z
                elif channel == 2:
                    harmonic = -scale21 * y * z
                elif channel == 3:
                    harmonic = scale22 * (x * x - y * y)
                else:
                    harmonic = scale21 * x * y
            value = radial_values[row * rs0 + radial_column * rs1] * harmonic
            if l_value == 0:
                target[index] = value
            elif l_value == 1:
                target[index] = -1j * value
            else:
                target[index] = -value
    return output


cdef inline Py_ssize_t _index_value(
    const void *data, int itemsize, Py_ssize_t position
) noexcept nogil:
    if itemsize == 2:
        return (<const uint16_t *>data)[position]
    if itemsize == 4:
        return (<const uint32_t *>data)[position]
    return (<const int64_t *>data)[position]


def project_density_stars(
    cnp.ndarray coefficients,
    cnp.ndarray offsets,
    cnp.ndarray members,
    cnp.ndarray weights,
    cnp.ndarray fill_factors,
):
    """Apply a reciprocal scalar-density star projector without temporaries."""
    cdef Py_ssize_t number_of_stars = offsets.size - 1
    cdef Py_ssize_t star, position, start, stop, row
    cdef int member_itemsize = members.itemsize
    cdef int32_t *star_offsets
    cdef const void *star_members
    cdef double complex *values
    cdef double complex *star_weights
    cdef double complex *star_fills
    cdef double complex averaged
    if coefficients.ndim != 1 or coefficients.dtype != np.complex128:
        raise ValueError("density coefficients must be a complex128 vector")
    if not coefficients.flags.c_contiguous:
        raise ValueError("density coefficients must be contiguous")
    if offsets.ndim != 1 or offsets.dtype != np.int32:
        raise ValueError("star offsets must be an int32 vector")
    if member_itemsize != 2 and member_itemsize != 4:
        raise ValueError("star members must be uint16 or uint32")
    if (
        weights.ndim != 1
        or fill_factors.ndim != 1
        or weights.dtype != np.complex128
        or fill_factors.dtype != np.complex128
        or weights.size != members.size
        or fill_factors.size != members.size
    ):
        raise ValueError("star weights and fill factors must match members")
    star_offsets = <int32_t *>cnp.PyArray_DATA(offsets)
    star_members = cnp.PyArray_DATA(members)
    values = <double complex *>cnp.PyArray_DATA(coefficients)
    star_weights = <double complex *>cnp.PyArray_DATA(weights)
    star_fills = <double complex *>cnp.PyArray_DATA(fill_factors)
    with nogil:
        for star in range(number_of_stars):
            start = star_offsets[star]
            stop = star_offsets[star + 1]
            averaged = 0.0
            for position in range(start, stop):
                row = _index_value(star_members, member_itemsize, position)
                averaged += star_weights[position] * values[row]
            for position in range(start, stop):
                row = _index_value(star_members, member_itemsize, position)
                values[row] = star_fills[position] * averaged


cdef class NativeFFTWPlan:
    """Two in-place FFTW plans tied to one aligned NumPy allocation."""

    cdef qepy_fftw_plan _forward
    cdef qepy_fftw_plan _backward
    cdef object _owner
    cdef int _howmany

    def __cinit__(
        self,
        cnp.ndarray array,
        object lengths,
        int howmany,
        int stride,
        int distance,
        unsigned flags=64,
        int nthreads=1,
    ):
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] dims
        cdef qepy_fftw_complex *pointer
        self._forward = NULL
        self._backward = NULL
        self._howmany = howmany
        if array.dtype != np.complex128 or not array.flags.c_contiguous:
            raise ValueError("native FFTW plans require C-contiguous complex128")
        dims = np.asarray(lengths, dtype=np.int32)
        if dims.size < 1:
            raise ValueError("native FFTW plan needs at least one dimension")
        if nthreads < 1:
            raise ValueError("FFTW thread count must be positive")
        if fftw_init_threads() == 0:
            raise RuntimeError("FFTW failed to initialize thread support")
        fftw_plan_with_nthreads(nthreads)
        self._owner = array
        pointer = <qepy_fftw_complex *>cnp.PyArray_DATA(array)
        self._forward = fftw_plan_many_dft(
            dims.size,
            <int *>cnp.PyArray_DATA(dims),
            howmany,
            pointer,
            NULL,
            stride,
            distance,
            pointer,
            NULL,
            stride,
            distance,
            -1,
            flags,
        )
        if self._forward == NULL:
            raise RuntimeError("FFTW failed to create native forward plan")
        self._backward = fftw_plan_many_dft(
            dims.size,
            <int *>cnp.PyArray_DATA(dims),
            howmany,
            pointer,
            NULL,
            stride,
            distance,
            pointer,
            NULL,
            stride,
            distance,
            1,
            flags,
        )
        if self._backward == NULL:
            fftw_destroy_plan(self._forward)
            self._forward = NULL
            raise RuntimeError("FFTW failed to create native backward plan")

    def __dealloc__(self):
        if self._forward != NULL:
            fftw_destroy_plan(self._forward)
        if self._backward != NULL:
            fftw_destroy_plan(self._backward)

    cdef void forward(self) noexcept nogil:
        cdef qepy_fftw_complex *pointer = <qepy_fftw_complex *>cnp.PyArray_DATA(
            <cnp.ndarray>self._owner
        )
        fftw_execute_dft(self._forward, pointer, pointer)

    cdef void backward(self) noexcept nogil:
        cdef qepy_fftw_complex *pointer = <qepy_fftw_complex *>cnp.PyArray_DATA(
            <cnp.ndarray>self._owner
        )
        fftw_execute_dft(self._backward, pointer, pointer)

    cdef void forward_at(self, double complex *pointer) noexcept nogil:
        fftw_execute_dft(
            self._forward,
            <qepy_fftw_complex *>pointer,
            <qepy_fftw_complex *>pointer,
        )

    cdef void backward_at(self, double complex *pointer) noexcept nogil:
        fftw_execute_dft(
            self._backward,
            <qepy_fftw_complex *>pointer,
            <qepy_fftw_complex *>pointer,
        )

    def execute(self, direction):
        """Execute one cached in-place plan from the Python owner."""
        if direction == "forward":
            with nogil:
                self.forward()
        elif direction == "backward":
            with nogil:
                self.backward()
        else:
            raise ValueError("FFT direction must be 'forward' or 'backward'")


def inverse_real(
    cnp.ndarray reciprocal,
    object lengths,
    unsigned flags=64,
    int nthreads=1,
):
    """Return the real part of one unnormalized inverse FFT."""
    if reciprocal.dtype != np.complex128 or not reciprocal.flags.c_contiguous:
        raise ValueError("native inverse input must be contiguous complex128")
    cdef NativeFFTWPlan plan = NativeFFTWPlan(
        reciprocal, lengths, 1, 1, reciprocal.size, flags, nthreads
    )
    with nogil:
        plan.backward()
    return np.asarray(reciprocal.real).copy()


def inverse_serial(
    cnp.ndarray grid,
    cnp.ndarray vectors,
    cnp.ndarray slots,
    NativeFFTWPlan plan,
    Py_ssize_t fft_size,
    Py_ssize_t grid_stride,
):
    """Scatter reciprocal coefficients and execute serial inverse FFTs."""
    cdef Py_ssize_t nbands = vectors.shape[1]
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t vector_stride0 = vectors.strides[0] // 16
    cdef Py_ssize_t vector_stride1 = vectors.strides[1] // 16
    cdef Py_ssize_t g, b, point
    cdef int slot_itemsize = slots.itemsize
    cdef const void *slot_data = cnp.PyArray_DATA(slots)
    cdef double complex *grid_values = <double complex *>cnp.PyArray_DATA(grid)
    cdef double complex *vector_values = <double complex *>cnp.PyArray_DATA(vectors)
    memset(grid_values, 0, grid.nbytes)
    with nogil:
        for b in prange(
            nbands,
            schedule="static",
            use_threads_if=nbands > 1,
        ):
            for g in range(nrows):
                point = b * grid_stride + _index_value(
                    slot_data, slot_itemsize, g
                )
                grid_values[point] = vector_values[
                    g * vector_stride0 + b * vector_stride1
                ]
        if plan._howmany == 1 and nbands > 1:
            for b in prange(
                nbands,
                schedule="static",
                use_threads_if=nbands > 1,
            ):
                plan.backward_at(grid_values + b * grid_stride)
        else:
            plan.backward()
    return grid


def forward_serial(
    cnp.ndarray grid,
    cnp.ndarray slots,
    NativeFFTWPlan plan,
    Py_ssize_t fft_size,
    Py_ssize_t grid_stride,
):
    """Execute serial forward FFTs and gather reciprocal coefficients."""
    cdef Py_ssize_t nbands = grid.size // grid_stride
    cdef Py_ssize_t nrows = slots.size
    cdef Py_ssize_t g, b, point
    cdef int slot_itemsize = slots.itemsize
    cdef const void *slot_data = cnp.PyArray_DATA(slots)
    cdef double complex *grid_values = <double complex *>cnp.PyArray_DATA(grid)
    cdef cnp.ndarray result = np.empty((nrows, nbands), dtype=np.complex128)
    cdef double complex *result_values = <double complex *>cnp.PyArray_DATA(result)
    with nogil:
        if plan._howmany == 1 and nbands > 1:
            for b in prange(
                nbands, schedule="static", use_threads_if=nbands > 1
            ):
                plan.forward_at(grid_values + b * grid_stride)
        else:
            plan.forward()
        for b in prange(
            nbands, schedule="static", use_threads_if=nbands > 1
        ):
            for g in range(nrows):
                point = _index_value(slot_data, slot_itemsize, g)
                result_values[g * nbands + b] = (
                    grid_values[b * grid_stride + point] / fft_size
                )
    return result


def apply_serial(
    cnp.ndarray grid,
    cnp.ndarray vectors,
    cnp.ndarray potential,
    cnp.ndarray slots,
    NativeFFTWPlan plan,
    Py_ssize_t fft_size,
    Py_ssize_t grid_stride,
    cnp.ndarray result,
    Py_ssize_t result_offset,
):
    """Fused serial scatter, Vloc application, FFT, and gather."""
    cdef Py_ssize_t nbands = vectors.shape[1]
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t vector_stride0 = vectors.strides[0] // 16
    cdef Py_ssize_t vector_stride1 = vectors.strides[1] // 16
    cdef Py_ssize_t result_stride0 = result.strides[0] // 16
    cdef Py_ssize_t result_stride1 = result.strides[1] // 16
    cdef Py_ssize_t g, b, point
    cdef int slot_itemsize = slots.itemsize
    cdef const void *slot_data = cnp.PyArray_DATA(slots)
    cdef double complex *grid_values = <double complex *>cnp.PyArray_DATA(grid)
    cdef double complex *vector_values = <double complex *>cnp.PyArray_DATA(vectors)
    cdef double *potential_values = <double *>cnp.PyArray_DATA(potential)
    cdef double complex *result_values = <double complex *>cnp.PyArray_DATA(result)
    if potential.dtype != np.float64 or not potential.flags.c_contiguous:
        raise ValueError("native local potential must be contiguous float64")
    if result.dtype != np.complex128 or not result.flags.c_contiguous:
        raise ValueError("native local result must be contiguous complex128")
    memset(grid_values, 0, grid.nbytes)
    with nogil:
        if plan._howmany == 1 and nbands > 1:
            for b in prange(
                nbands,
                schedule="static",
                use_threads_if=nbands > 1,
            ):
                for g in range(nrows):
                    point = b * grid_stride + _index_value(
                        slot_data, slot_itemsize, g
                    )
                    grid_values[point] = vector_values[
                        g * vector_stride0 + b * vector_stride1
                    ]
                plan.backward_at(grid_values + b * grid_stride)
                for point in range(fft_size):
                    grid_values[b * grid_stride + point] *= potential_values[point]
                plan.forward_at(grid_values + b * grid_stride)
                for g in range(nrows):
                    point = _index_value(slot_data, slot_itemsize, g)
                    result_values[
                        g * result_stride0
                        + (result_offset + b) * result_stride1
                    ] = (
                        grid_values[b * grid_stride + point] / fft_size
                    )
        else:
            for b in prange(
                nbands,
                schedule="static",
                use_threads_if=nbands > 1,
            ):
                for g in range(nrows):
                    point = b * grid_stride + _index_value(
                        slot_data, slot_itemsize, g
                    )
                    grid_values[point] = vector_values[
                        g * vector_stride0 + b * vector_stride1
                    ]
            plan.backward()
            for b in prange(
                nbands,
                schedule="static",
                use_threads_if=nbands > 1,
            ):
                for point in range(fft_size):
                    grid_values[b * grid_stride + point] *= potential_values[point]
            plan.forward()
            for b in prange(
                nbands,
                schedule="static",
                use_threads_if=nbands > 1,
            ):
                for g in range(nrows):
                    point = _index_value(slot_data, slot_itemsize, g)
                    result_values[
                        g * result_stride0
                        + (result_offset + b) * result_stride1
                    ] = (
                        grid_values[b * grid_stride + point] / fft_size
                    )


def apply_serial_spatial(
    cnp.ndarray grid,
    cnp.ndarray vectors,
    cnp.ndarray potential,
    cnp.ndarray z_major_slots,
    cnp.ndarray stick_positions,
    NativeFFTWPlan z_plan,
    NativeFFTWPlan xy_plan,
    Py_ssize_t fft_size,
    Py_ssize_t grid_stride,
    cnp.ndarray result,
    Py_ssize_t result_offset,
):
    """QE-style shared-memory stick/plane local-potential application."""
    cdef Py_ssize_t nbands = vectors.shape[1]
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t nsticks = stick_positions.size
    cdef Py_ssize_t nz = potential.shape[0]
    cdef Py_ssize_t plane_size = potential.size // nz
    cdef Py_ssize_t vector_stride0 = vectors.strides[0] // 16
    cdef Py_ssize_t vector_stride1 = vectors.strides[1] // 16
    cdef Py_ssize_t result_stride0 = result.strides[0] // 16
    cdef Py_ssize_t result_stride1 = result.strides[1] // 16
    cdef int slot_itemsize = z_major_slots.itemsize
    cdef int stick_itemsize = stick_positions.itemsize
    cdef const void *slot_data = cnp.PyArray_DATA(z_major_slots)
    cdef const void *stick_data = cnp.PyArray_DATA(stick_positions)
    cdef double complex *grid_values = <double complex *>cnp.PyArray_DATA(grid)
    cdef double complex *vector_values = <double complex *>cnp.PyArray_DATA(vectors)
    cdef double complex *result_values = <double complex *>cnp.PyArray_DATA(result)
    cdef double *potential_values = <double *>cnp.PyArray_DATA(potential)
    if potential.dtype != np.float64 or not potential.flags.c_contiguous:
        raise ValueError("spatial serial potential must be contiguous float64")
    if potential.ndim != 3 or potential.size != fft_size:
        raise ValueError("spatial serial potential has the wrong shape")
    memset(grid_values, 0, grid.nbytes)
    with nogil:
        qepy_apply_serial_spatial(
            grid_values,
            vector_values,
            potential_values,
            slot_data,
            slot_itemsize,
            stick_data,
            stick_itemsize,
            z_plan._forward,
            z_plan._backward,
            xy_plan._forward,
            xy_plan._backward,
            nbands,
            nrows,
            nsticks,
            nz,
            plane_size,
            fft_size,
            grid_stride,
            vector_stride0,
            vector_stride1,
            result_values,
            result_stride0,
            result_stride1,
            result_offset,
        )


def accumulate_density_serial_spatial(
    cnp.ndarray density,
    cnp.ndarray grid,
    cnp.ndarray vectors,
    cnp.ndarray band_weights,
    cnp.ndarray z_major_slots,
    cnp.ndarray stick_positions,
    NativeFFTWPlan z_plan,
    NativeFFTWPlan xy_plan,
    Py_ssize_t fft_size,
    Py_ssize_t grid_stride,
):
    """QE-style shared-memory inverse FFT and density accumulation."""
    cdef Py_ssize_t nbands = vectors.shape[1]
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t nsticks = stick_positions.size
    cdef Py_ssize_t nz = density.shape[2]
    cdef Py_ssize_t plane_size = fft_size // nz
    cdef Py_ssize_t vector_stride0 = vectors.strides[0] // 16
    cdef Py_ssize_t vector_stride1 = vectors.strides[1] // 16
    cdef int slot_itemsize = z_major_slots.itemsize
    cdef int stick_itemsize = stick_positions.itemsize
    cdef const void *slot_data = cnp.PyArray_DATA(z_major_slots)
    cdef const void *stick_data = cnp.PyArray_DATA(stick_positions)
    cdef double *density_values = <double *>cnp.PyArray_DATA(density)
    cdef double complex *grid_values = <double complex *>cnp.PyArray_DATA(grid)
    cdef double complex *vector_values = <double complex *>cnp.PyArray_DATA(vectors)
    cdef double *weights = <double *>cnp.PyArray_DATA(band_weights)
    if density.dtype != np.float64 or not density.flags.c_contiguous:
        raise ValueError("density must be contiguous float64")
    memset(grid_values, 0, grid.nbytes)
    with nogil:
        qepy_accumulate_density_serial_spatial(
            density_values,
            grid_values,
            vector_values,
            weights,
            slot_data,
            slot_itemsize,
            stick_data,
            stick_itemsize,
            z_plan._backward,
            xy_plan._backward,
            nbands,
            nrows,
            nsticks,
            nz,
            plane_size,
            fft_size,
            grid_stride,
            vector_stride0,
            vector_stride1,
        )


def accumulate_density_serial(
    cnp.ndarray density,
    cnp.ndarray grid,
    cnp.ndarray vectors,
    cnp.ndarray band_weights,
    cnp.ndarray slots,
    NativeFFTWPlan plan,
    Py_ssize_t fft_size,
    Py_ssize_t grid_stride,
):
    """Accumulate density from one bounded batch of serial FFT grids."""
    cdef Py_ssize_t nbands = vectors.shape[1]
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t vector_stride0 = vectors.strides[0] // 16
    cdef Py_ssize_t vector_stride1 = vectors.strides[1] // 16
    cdef Py_ssize_t g, b, point
    cdef int slot_itemsize = slots.itemsize
    cdef const void *slot_data = cnp.PyArray_DATA(slots)
    cdef double complex *grid_values = <double complex *>cnp.PyArray_DATA(grid)
    cdef double complex *vector_values = <double complex *>cnp.PyArray_DATA(vectors)
    cdef double *density_values = <double *>cnp.PyArray_DATA(density)
    cdef double *weights = <double *>cnp.PyArray_DATA(band_weights)
    cdef double real_part, imag_part, contribution
    memset(grid_values, 0, grid.nbytes)
    with nogil:
        for b in prange(
            nbands,
            schedule="static",
            use_threads_if=nbands > 1,
        ):
            for g in range(nrows):
                point = _index_value(slot_data, slot_itemsize, g)
                grid_values[b * grid_stride + point] = vector_values[
                    g * vector_stride0 + b * vector_stride1
                ]
        if plan._howmany == 1 and nbands > 1:
            for b in prange(
                nbands,
                schedule="static",
                use_threads_if=nbands > 1,
            ):
                plan.backward_at(grid_values + b * grid_stride)
        else:
            plan.backward()
        for point in prange(
            fft_size,
            schedule="static",
            use_threads_if=fft_size * nbands >= 65536,
        ):
            contribution = 0.0
            for b in range(nbands):
                real_part = creal(grid_values[b * grid_stride + point])
                imag_part = cimag(grid_values[b * grid_stride + point])
                contribution += weights[b] * (
                    real_part * real_part + imag_part * imag_part
                )
            density_values[point] += contribution


cdef int _alltoallv(
    Comm comm,
    double complex *send,
    int32_t *send_counts,
    int32_t *send_displacements,
    double complex *receive,
    int32_t *receive_counts,
    int32_t *receive_displacements,
) noexcept:
    return MPI_Alltoallv(
        <void *>send,
        <int *>send_counts,
        <int *>send_displacements,
        MPI_C_DOUBLE_COMPLEX,
        <void *>receive,
        <int *>receive_counts,
        <int *>receive_displacements,
        MPI_C_DOUBLE_COMPLEX,
        comm.ob_mpi,
    )


def apply_streamed(
    cnp.ndarray sticks,
    cnp.ndarray slab,
    cnp.ndarray reverse_send,
    cnp.ndarray inverse_receive,
    cnp.ndarray vectors,
    cnp.ndarray potential,
    cnp.ndarray z_slots,
    cnp.ndarray stick_positions,
    cnp.ndarray slab_points,
    cnp.ndarray forward_send_counts,
    cnp.ndarray forward_send_displacements,
    cnp.ndarray forward_receive_counts,
    cnp.ndarray forward_receive_displacements,
    cnp.ndarray reverse_send_counts,
    cnp.ndarray reverse_send_displacements,
    cnp.ndarray reverse_receive_counts,
    cnp.ndarray reverse_receive_displacements,
    Comm comm,
    NativeFFTWPlan z_plan,
    NativeFFTWPlan xy_plan,
    Py_ssize_t fft_size,
):
    """Fused Vloc application with native packing, MPI, and FFTW."""
    cdef Py_ssize_t nz = sticks.shape[0]
    cdef Py_ssize_t nsticks = sticks.shape[1]
    cdef Py_ssize_t nbands = sticks.shape[2]
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t slab_size = slab.size
    cdef Py_ssize_t transfer_rows = slab_points.size
    cdef Py_ssize_t g, b, point
    cdef Py_ssize_t vector_stride0 = vectors.strides[0] // 16
    cdef Py_ssize_t vector_stride1 = vectors.strides[1] // 16
    cdef int z_itemsize = z_slots.itemsize
    cdef int stick_itemsize = stick_positions.itemsize
    cdef const void *z_data = cnp.PyArray_DATA(z_slots)
    cdef const void *stick_data = cnp.PyArray_DATA(stick_positions)
    cdef int point_itemsize = slab_points.itemsize
    cdef const void *point_data = cnp.PyArray_DATA(slab_points)
    cdef double complex *stick_values = <double complex *>cnp.PyArray_DATA(sticks)
    cdef double complex *slab_values = <double complex *>cnp.PyArray_DATA(slab)
    cdef double complex *vector_values = <double complex *>cnp.PyArray_DATA(vectors)
    cdef double *potential_values = <double *>cnp.PyArray_DATA(potential)
    cdef double complex *send_values = <double complex *>cnp.PyArray_DATA(reverse_send)
    cdef double complex *received_values = <double complex *>cnp.PyArray_DATA(inverse_receive)
    cdef cnp.ndarray result = np.empty((nrows, nbands), dtype=np.complex128)
    cdef double complex *result_values = <double complex *>cnp.PyArray_DATA(result)
    cdef int status

    if sticks.dtype != np.complex128 or slab.dtype != np.complex128:
        raise ValueError("native FFT payloads must be complex128")
    if potential.dtype != np.float64 or not potential.flags.c_contiguous:
        raise ValueError("native local potential must be contiguous float64")
    memset(stick_values, 0, sticks.nbytes)
    with nogil:
        for g in prange(
            nrows, schedule="static", use_threads_if=nrows * nbands >= 65536
        ):
            point = (
                (_index_value(z_data, z_itemsize, g) * nsticks
                 + _index_value(stick_data, stick_itemsize, g)) * nbands
            )
            for b in range(nbands):
                stick_values[point + b] = vector_values[
                    g * vector_stride0 + b * vector_stride1
                ]
        z_plan.backward()
    status = _alltoallv(
        comm,
        stick_values,
        <int32_t *>cnp.PyArray_DATA(forward_send_counts),
        <int32_t *>cnp.PyArray_DATA(forward_send_displacements),
        received_values,
        <int32_t *>cnp.PyArray_DATA(forward_receive_counts),
        <int32_t *>cnp.PyArray_DATA(forward_receive_displacements),
    )
    if status != 0:
        raise RuntimeError(f"native forward MPI_Alltoallv failed: {status}")
    for b in range(nbands):
        memset(slab_values, 0, slab.nbytes)
        with nogil:
            for g in prange(
                transfer_rows,
                schedule="static",
                use_threads_if=transfer_rows >= 65536,
            ):
                point = _index_value(point_data, point_itemsize, g)
                slab_values[point] = received_values[g * nbands + b]
            xy_plan.backward()
            for point in prange(
                slab_size,
                schedule="static",
                use_threads_if=slab_size >= 65536,
            ):
                slab_values[point] *= potential_values[point]
            xy_plan.forward()
            for g in prange(
                transfer_rows,
                schedule="static",
                use_threads_if=transfer_rows >= 65536,
            ):
                point = _index_value(point_data, point_itemsize, g)
                send_values[g * nbands + b] = slab_values[point]
    status = _alltoallv(
        comm,
        send_values,
        <int32_t *>cnp.PyArray_DATA(reverse_send_counts),
        <int32_t *>cnp.PyArray_DATA(reverse_send_displacements),
        stick_values,
        <int32_t *>cnp.PyArray_DATA(reverse_receive_counts),
        <int32_t *>cnp.PyArray_DATA(reverse_receive_displacements),
    )
    if status != 0:
        raise RuntimeError(f"native reverse MPI_Alltoallv failed: {status}")
    with nogil:
        z_plan.forward()
        for g in prange(
            nrows, schedule="static", use_threads_if=nrows * nbands >= 65536
        ):
            point = (
                (_index_value(z_data, z_itemsize, g) * nsticks
                 + _index_value(stick_data, stick_itemsize, g)) * nbands
            )
            for b in range(nbands):
                result_values[g * nbands + b] = (
                    stick_values[point + b] / fft_size
                )
    return result


def accumulate_density_distributed(
    cnp.ndarray density,
    cnp.ndarray sticks,
    cnp.ndarray slab,
    cnp.ndarray inverse_receive,
    cnp.ndarray vectors,
    cnp.ndarray band_weights,
    cnp.ndarray z_slots,
    cnp.ndarray stick_positions,
    cnp.ndarray slab_points,
    cnp.ndarray send_counts,
    cnp.ndarray send_displacements,
    cnp.ndarray receive_counts,
    cnp.ndarray receive_displacements,
    Comm comm,
    NativeFFTWPlan z_plan,
    NativeFFTWPlan xy_plan,
):
    """Fused inverse FFT and occupied-density accumulation."""
    cdef Py_ssize_t nsticks = sticks.shape[1]
    cdef Py_ssize_t nbands = sticks.shape[2]
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t slab_size = slab.size
    cdef Py_ssize_t local_z = slab.shape[0]
    cdef Py_ssize_t plane_size = slab.shape[1] * slab.shape[2]
    cdef Py_ssize_t transfer_rows = slab_points.size
    cdef Py_ssize_t vector_stride0 = vectors.strides[0] // 16
    cdef Py_ssize_t vector_stride1 = vectors.strides[1] // 16
    cdef Py_ssize_t g, b, point, z, xy, grid_point
    cdef int z_itemsize = z_slots.itemsize
    cdef int stick_itemsize = stick_positions.itemsize
    cdef const void *z_data = cnp.PyArray_DATA(z_slots)
    cdef const void *stick_data = cnp.PyArray_DATA(stick_positions)
    cdef int point_itemsize = slab_points.itemsize
    cdef const void *point_data = cnp.PyArray_DATA(slab_points)
    cdef double complex *stick_values = <double complex *>cnp.PyArray_DATA(sticks)
    cdef double complex *slab_values = <double complex *>cnp.PyArray_DATA(slab)
    cdef double complex *vector_values = <double complex *>cnp.PyArray_DATA(vectors)
    cdef double complex *received_values = <double complex *>cnp.PyArray_DATA(inverse_receive)
    cdef double *density_values = <double *>cnp.PyArray_DATA(density)
    cdef double *weights = <double *>cnp.PyArray_DATA(band_weights)
    cdef double real_part, imag_part
    cdef int status

    memset(stick_values, 0, sticks.nbytes)
    with nogil:
        for g in prange(
            nrows, schedule="static", use_threads_if=nrows * nbands >= 65536
        ):
            point = (
                (_index_value(z_data, z_itemsize, g) * nsticks
                 + _index_value(stick_data, stick_itemsize, g)) * nbands
            )
            for b in range(nbands):
                stick_values[point + b] = vector_values[
                    g * vector_stride0 + b * vector_stride1
                ]
        z_plan.backward()
    status = _alltoallv(
        comm,
        stick_values,
        <int32_t *>cnp.PyArray_DATA(send_counts),
        <int32_t *>cnp.PyArray_DATA(send_displacements),
        received_values,
        <int32_t *>cnp.PyArray_DATA(receive_counts),
        <int32_t *>cnp.PyArray_DATA(receive_displacements),
    )
    if status != 0:
        raise RuntimeError(f"native density MPI_Alltoallv failed: {status}")
    for b in range(nbands):
        memset(slab_values, 0, slab.nbytes)
        with nogil:
            for g in prange(
                transfer_rows,
                schedule="static",
                use_threads_if=transfer_rows >= 65536,
            ):
                point = _index_value(point_data, point_itemsize, g)
                slab_values[point] = received_values[g * nbands + b]
            xy_plan.backward()
            for z in prange(
                local_z,
                schedule="static",
                use_threads_if=slab_size >= 65536,
            ):
                for xy in range(plane_size):
                    point = z * plane_size + xy
                    grid_point = xy * local_z + z
                    real_part = creal(slab_values[point])
                    imag_part = cimag(slab_values[point])
                    density_values[grid_point] += weights[b] * (
                        real_part * real_part + imag_part * imag_part
                    )
