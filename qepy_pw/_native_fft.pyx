# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Native serial and distributed FFT hot paths.

This mandatory extension fuses sparse packing, native MPI_Alltoallv, FFTW
plans, local-potential multiplication, and density accumulation.  All payload
arrays remain NumPy-owned so Python-side nbytes accounting stays complete.
"""

from libc.complex cimport conj, creal, cimag
from libc.math cimport cos, sin, sqrt
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int32_t, int64_t
from libc.stddef cimport ptrdiff_t
from libc.string cimport memset
from cython.parallel cimport prange

import numpy as np
cimport numpy as cnp

from mpi4py.MPI cimport Comm
from mpi4py.libmpi cimport (
    MPI_Alltoallv,
    MPI_C_DOUBLE_COMPLEX,
    MPI_Comm,
)

cnp.import_array()


cdef extern from *:
    """
    #include <complex.h>
    #include <mpi.h>
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
        const double *potential, const double *diagonal,
        const void *slots, int slot_itemsize,
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
                    grid[base + point] / (double)fft_size
                    + (diagonal == NULL ? 0.0 :
                       diagonal[g] * vectors[
                           g * vector_stride0 + b * vector_stride1]);
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

    static void qepy_apply_slab_bands(
        double _Complex *slab, const double _Complex *received,
        double _Complex *send, const double *potential,
        const void *points, int point_itemsize,
        qepy_fftw_plan xy_backward, qepy_fftw_plan xy_forward,
        long long nbands, long long transfer_rows,
        long long local_z, long long plane_size) {
        long long b, g, point, z;
        const long long slab_size = local_z * plane_size;
        /* Keep one OpenMP team alive across every band and FFT phase.  This
           is the shared-memory analogue of FFTXlib's plane loop and avoids
           creating two FFTW worker teams plus three packing teams per band. */
        #pragma omp parallel private(b,g,point,z)
        {
            for (b = 0; b < nbands; ++b) {
                #pragma omp for schedule(static)
                for (point = 0; point < slab_size; ++point)
                    slab[point] = 0.0;
                #pragma omp for schedule(static)
                for (g = 0; g < transfer_rows; ++g) {
                    point = qepy_index_value(points, point_itemsize, g);
                    slab[point] = received[g * nbands + b];
                }
                #pragma omp for schedule(static)
                for (z = 0; z < local_z; ++z)
                    fftw_execute_dft(
                        xy_backward,
                        (qepy_fftw_complex *)(slab + z * plane_size),
                        (qepy_fftw_complex *)(slab + z * plane_size));
                #pragma omp for schedule(static)
                for (point = 0; point < slab_size; ++point)
                    slab[point] *= potential[point];
                #pragma omp for schedule(static)
                for (z = 0; z < local_z; ++z)
                    fftw_execute_dft(
                        xy_forward,
                        (qepy_fftw_complex *)(slab + z * plane_size),
                        (qepy_fftw_complex *)(slab + z * plane_size));
                #pragma omp for schedule(static)
                for (g = 0; g < transfer_rows; ++g) {
                    point = qepy_index_value(points, point_itemsize, g);
                    send[g * nbands + b] = slab[point];
                }
            }
        }
    }

    static void qepy_accumulate_slab_bands(
        double *density, double _Complex *slab,
        const double _Complex *received, const double *weights,
        const void *points, int point_itemsize,
        qepy_fftw_plan xy_backward, long long nbands,
        long long transfer_rows, long long local_z, long long plane_size) {
        long long b, g, point, z, xy, grid_point;
        double real_part, imag_part;
        const long long slab_size = local_z * plane_size;
        #pragma omp parallel private(b,g,point,z,xy,grid_point,real_part,imag_part)
        {
            for (b = 0; b < nbands; ++b) {
                #pragma omp for schedule(static)
                for (point = 0; point < slab_size; ++point)
                    slab[point] = 0.0;
                #pragma omp for schedule(static)
                for (g = 0; g < transfer_rows; ++g) {
                    point = qepy_index_value(points, point_itemsize, g);
                    slab[point] = received[g * nbands + b];
                }
                #pragma omp for schedule(static)
                for (z = 0; z < local_z; ++z)
                    fftw_execute_dft(
                        xy_backward,
                        (qepy_fftw_complex *)(slab + z * plane_size),
                        (qepy_fftw_complex *)(slab + z * plane_size));
                #pragma omp for schedule(static)
                for (point = 0; point < slab_size; ++point) {
                    z = point / plane_size;
                    xy = point - z * plane_size;
                    grid_point = xy * local_z + z;
                    real_part = creal(slab[point]);
                    imag_part = cimag(slab[point]);
                    density[grid_point] += weights[b] *
                        (real_part * real_part + imag_part * imag_part);
                }
            }
        }
    }

    static void qepy_inverse_z_sticks(
        double _Complex *sticks, const double _Complex *vectors,
        const void *z_slots, int z_itemsize,
        const void *stick_positions, int stick_itemsize,
        qepy_fftw_plan z_backward, long long nz, long long nsticks,
        long long nbands, long long nrows,
        long long vector_stride0, long long vector_stride1) {
        long long task, g, b, point, stick;
        const long long stick_stride = nsticks * nbands;
        const long long total = nz * stick_stride;
        #pragma omp parallel private(task,g,b,point,stick)
        {
            #pragma omp for schedule(static)
            for (point = 0; point < total; ++point)
                sticks[point] = 0.0;
            #pragma omp for schedule(static)
            for (task = 0; task < nrows * nbands; ++task) {
                g = task / nbands;
                b = task - g * nbands;
                point = (
                    (qepy_index_value(z_slots, z_itemsize, g) * nsticks
                     + qepy_index_value(
                         stick_positions, stick_itemsize, g)) * nbands + b
                );
                sticks[point] = vectors[
                    g * vector_stride0 + b * vector_stride1];
            }
            #pragma omp for schedule(static)
            for (task = 0; task < nsticks * nbands; ++task) {
                stick = task / nbands;
                b = task - stick * nbands;
                fftw_execute_dft(
                    z_backward,
                    (qepy_fftw_complex *)(sticks + stick * nbands + b),
                    (qepy_fftw_complex *)(sticks + stick * nbands + b));
            }
        }
    }

    static void qepy_forward_z_gather(
        double _Complex *result, double _Complex *sticks,
        const double _Complex *vectors, const double *diagonal,
        const void *z_slots, int z_itemsize,
        const void *stick_positions, int stick_itemsize,
        qepy_fftw_plan z_forward, long long nz, long long nsticks,
        long long nbands, long long nrows, long long fft_size,
        long long result_stride0, long long result_stride1,
        long long vector_stride0, long long vector_stride1) {
        long long task, g, b, point, stick;
        #pragma omp parallel private(task,g,b,point,stick)
        {
            #pragma omp for schedule(static)
            for (task = 0; task < nsticks * nbands; ++task) {
                stick = task / nbands;
                b = task - stick * nbands;
                fftw_execute_dft(
                    z_forward,
                    (qepy_fftw_complex *)(sticks + stick * nbands + b),
                    (qepy_fftw_complex *)(sticks + stick * nbands + b));
            }
            #pragma omp for schedule(static)
            for (task = 0; task < nrows * nbands; ++task) {
                g = task / nbands;
                b = task - g * nbands;
                point = (
                    (qepy_index_value(z_slots, z_itemsize, g) * nsticks
                     + qepy_index_value(
                         stick_positions, stick_itemsize, g)) * nbands + b
                );
                result[g * result_stride0 + b * result_stride1] =
                    sticks[point] / (double)fft_size
                    + (diagonal == NULL ? 0.0 :
                       diagonal[g] * vectors[
                           g * vector_stride0 + b * vector_stride1]);
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
        const double *diagonal,
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
    void qepy_apply_slab_bands(
        double complex *slab,
        const double complex *received,
        double complex *send,
        const double *potential,
        const void *points,
        int point_itemsize,
        qepy_fftw_plan xy_backward,
        qepy_fftw_plan xy_forward,
        long long nbands,
        long long transfer_rows,
        long long local_z,
        long long plane_size,
    ) noexcept nogil
    void qepy_accumulate_slab_bands(
        double *density,
        double complex *slab,
        const double complex *received,
        const double *weights,
        const void *points,
        int point_itemsize,
        qepy_fftw_plan xy_backward,
        long long nbands,
        long long transfer_rows,
        long long local_z,
        long long plane_size,
    ) noexcept nogil
    void qepy_inverse_z_sticks(
        double complex *sticks,
        const double complex *vectors,
        const void *z_slots,
        int z_itemsize,
        const void *stick_positions,
        int stick_itemsize,
        qepy_fftw_plan z_backward,
        long long nz,
        long long nsticks,
        long long nbands,
        long long nrows,
        long long vector_stride0,
        long long vector_stride1,
    ) noexcept nogil
    void qepy_forward_z_gather(
        double complex *result,
        double complex *sticks,
        const double complex *vectors,
        const double *diagonal,
        const void *z_slots,
        int z_itemsize,
        const void *stick_positions,
        int stick_itemsize,
        qepy_fftw_plan z_forward,
        long long nz,
        long long nsticks,
        long long nbands,
        long long nrows,
        long long fft_size,
        long long result_stride0,
        long long result_stride1,
        long long vector_stride0,
        long long vector_stride1,
    ) noexcept nogil
cdef extern from *:
    """
    #include <stddef.h>
    #include <mpi.h>
    #ifdef QEPY_HAVE_FFTW_MPI
    #define FFTW_NO_Complex
    #include <fftw3-mpi.h>
    static int qepy_fftw_mpi_available(void) { return 1; }
    static ptrdiff_t qepy_fftw_mpi_local_size_3d(
        ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2, ptrdiff_t block0,
        MPI_Comm comm, ptrdiff_t *local_n0, ptrdiff_t *local_0_start) {
        ptrdiff_t dimensions[3] = {n0, n1, n2};
        fftw_mpi_init();
        return fftw_mpi_local_size_many(
            3, dimensions, 1, block0, comm, local_n0, local_0_start);
    }
    static fftw_plan qepy_fftw_mpi_plan_3d(
        double _Complex *values, ptrdiff_t n0, ptrdiff_t n1,
        ptrdiff_t n2, ptrdiff_t block0, MPI_Comm comm,
        int sign, unsigned flags, int nthreads) {
        ptrdiff_t dimensions[3] = {n0, n1, n2};
        fftw_init_threads();
        fftw_plan_with_nthreads(nthreads);
        return fftw_mpi_plan_many_dft(
            3, dimensions, 1, block0, FFTW_MPI_DEFAULT_BLOCK,
            (fftw_complex *)values, (fftw_complex *)values,
            comm, sign, flags);
    }
    static void qepy_fftw_mpi_execute(fftw_plan plan) {
        fftw_execute(plan);
    }
    #else
    static int qepy_fftw_mpi_available(void) { return 0; }
    static ptrdiff_t qepy_fftw_mpi_local_size_3d(
        ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2, ptrdiff_t block0,
        MPI_Comm comm, ptrdiff_t *local_n0, ptrdiff_t *local_0_start) {
        *local_n0 = 0; *local_0_start = 0; return -1;
    }
    static void *qepy_fftw_mpi_plan_3d(
        double _Complex *values, ptrdiff_t n0, ptrdiff_t n1,
        ptrdiff_t n2, ptrdiff_t block0, MPI_Comm comm,
        int sign, unsigned flags, int nthreads) { return NULL; }
    static void qepy_fftw_mpi_execute(void *plan) { (void)plan; }
    #endif
    """
    int qepy_fftw_mpi_available() noexcept nogil
    ptrdiff_t qepy_fftw_mpi_local_size_3d(
        ptrdiff_t n0,
        ptrdiff_t n1,
        ptrdiff_t n2,
        ptrdiff_t block0,
        MPI_Comm comm,
        ptrdiff_t *local_n0,
        ptrdiff_t *local_0_start,
    ) noexcept nogil
    qepy_fftw_plan qepy_fftw_mpi_plan_3d(
        double complex *values,
        ptrdiff_t n0,
        ptrdiff_t n1,
        ptrdiff_t n2,
        ptrdiff_t block0,
        MPI_Comm comm,
        int sign,
        unsigned flags,
        int nthreads,
    ) noexcept nogil
    void qepy_fftw_mpi_execute(qepy_fftw_plan plan) noexcept nogil


cdef extern from *:
    """
    #include <complex.h>
    #include <dlfcn.h>
    #include <limits.h>
    typedef long long qepy_lapack_int;
    typedef qepy_lapack_int (*qepy_zhegvd64_function)(
        int, qepy_lapack_int, char, char, qepy_lapack_int,
        double _Complex *, qepy_lapack_int, double _Complex *,
        qepy_lapack_int, double *);
    typedef int (*qepy_zhegvd32_function)(
        int, int, char, char, int,
        double _Complex *, int, double _Complex *, int, double *);

    static qepy_lapack_int qepy_zhegvd64(
        int matrix_layout, qepy_lapack_int itype, char jobz, char uplo,
        qepy_lapack_int n, double _Complex *a, qepy_lapack_int lda,
        double _Complex *b, qepy_lapack_int ldb, double *w) {
        static qepy_zhegvd64_function function64 = NULL;
        static qepy_zhegvd32_function function32 = NULL;
        static int attempted = 0;
        if (!attempted) {
            attempted = 1;
            function32 = (qepy_zhegvd32_function)dlsym(
                RTLD_DEFAULT, "LAPACKE_zhegvd");
            function64 = (qepy_zhegvd64_function)dlsym(
                RTLD_DEFAULT, "scipy_LAPACKE_zhegvd64_");
        }
        if (function32 != NULL && n <= INT_MAX && lda <= INT_MAX && ldb <= INT_MAX)
            return function32(
                matrix_layout, (int)itype, jobz, uplo, (int)n,
                a, (int)lda, b, (int)ldb, w);
        if (function64 != NULL)
            return function64(
                matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
        return -1000000;
    }

    typedef void (*qepy_zgemm64_function)(
        int, int, int, long long, long long, long long,
        const void *, const void *, long long, const void *, long long,
        const void *, void *, long long);
    typedef void (*qepy_zgemm32_function)(
        int, int, int, int, int, int,
        const void *, const void *, int, const void *, int,
        const void *, void *, int);

    static int qepy_zgemm64(
        int layout, int transa, int transb,
        long long m, long long n, long long k,
        const double _Complex *alpha, const double _Complex *a,
        long long lda, const double _Complex *b, long long ldb,
        const double _Complex *beta, double _Complex *c, long long ldc) {
        static qepy_zgemm64_function function64 = NULL;
        static qepy_zgemm32_function function32 = NULL;
        static int attempted = 0;
        if (!attempted) {
            attempted = 1;
            function32 = (qepy_zgemm32_function)dlsym(
                RTLD_DEFAULT, "cblas_zgemm");
            function64 = (qepy_zgemm64_function)dlsym(
                RTLD_DEFAULT, "scipy_cblas_zgemm64_");
        }
        if (function32 != NULL && m <= INT_MAX && n <= INT_MAX && k <= INT_MAX
            && lda <= INT_MAX && ldb <= INT_MAX && ldc <= INT_MAX) {
            function32(
                layout, transa, transb, (int)m, (int)n, (int)k,
                alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc);
            return 0;
        }
        if (function64 != NULL) {
            function64(
                layout, transa, transb, m, n, k, alpha, a, lda,
                b, ldb, beta, c, ldc);
            return 0;
        }
        return -1;
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
    int qepy_zgemm64(
        int layout,
        int transa,
        int transb,
        long long m,
        long long n,
        long long k,
        const double complex *alpha,
        const double complex *a,
        long long lda,
        const double complex *b,
        long long ldb,
        const double complex *beta,
        double complex *c,
        long long ldc,
    ) noexcept nogil


def generalized_eigh(cnp.ndarray hamiltonian, cnp.ndarray overlap, int roots):
    """Use build-selected LP64 LAPACKE or NumPy's ILP64 LAPACKE."""
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


def projector_overlaps(cnp.ndarray beta, cnp.ndarray vectors):
    """Return beta^H vectors through BLAS without conjugating either input."""
    cdef long long nrows
    cdef long long nchannels
    cdef long long nbands
    cdef cnp.ndarray output
    cdef double complex alpha = 1.0
    cdef double complex zero = 0.0
    cdef int status
    if (
        beta.ndim != 2
        or vectors.ndim != 2
        or beta.dtype != np.complex128
        or vectors.dtype != np.complex128
        or beta.shape[0] != vectors.shape[0]
    ):
        raise ValueError("projector and vector matrices disagree")
    # Davidson's persistent basis and packed projectors are column-major.
    # Decline the fast path instead of silently copying a large input.
    if not beta.flags.f_contiguous or not vectors.flags.f_contiguous:
        return None
    nrows = beta.shape[0]
    nchannels = beta.shape[1]
    nbands = vectors.shape[1]
    output = np.empty((nchannels, nbands), dtype=np.complex128, order="F")
    with nogil:
        status = qepy_zgemm64(
            102, 113, 111,
            nchannels, nbands, nrows,
            &alpha,
            <double complex *>cnp.PyArray_DATA(beta), nrows,
            <double complex *>cnp.PyArray_DATA(vectors), nrows,
            &zero,
            <double complex *>cnp.PyArray_DATA(output), nchannels,
        )
    if status != 0:
        return None
    return output


def accumulate_projector_diagonal(
    cnp.ndarray diagonal,
    cnp.ndarray coupled,
    cnp.ndarray beta,
    double multiplicity,
):
    """Accumulate ``m * Re[(beta D) beta^H]`` row by row."""
    cdef Py_ssize_t nrows = beta.shape[0]
    cdef Py_ssize_t nchannels = beta.shape[1]
    cdef Py_ssize_t row, i
    cdef Py_ssize_t bs0 = beta.strides[0] // 16
    cdef Py_ssize_t bs1 = beta.strides[1] // 16
    cdef Py_ssize_t cs0 = coupled.strides[0] // 16
    cdef Py_ssize_t cs1 = coupled.strides[1] // 16
    cdef double complex *projectors
    cdef double complex *products
    cdef double *target
    cdef double value
    if (
        diagonal.ndim != 1
        or coupled.ndim != 2
        or beta.ndim != 2
        or diagonal.dtype != np.float64
        or coupled.dtype != np.complex128
        or beta.dtype != np.complex128
        or diagonal.size != nrows
        or coupled.shape[0] != nrows
        or coupled.shape[1] != nchannels
        or not diagonal.flags.c_contiguous
    ):
        raise ValueError("projector diagonal operands disagree")
    projectors = <double complex *>cnp.PyArray_DATA(beta)
    products = <double complex *>cnp.PyArray_DATA(coupled)
    target = <double *>cnp.PyArray_DATA(diagonal)
    with nogil:
        for row in prange(
            nrows, schedule="static", use_threads_if=nrows >= 1024
        ):
            value = 0.0
            for i in range(nchannels):
                value = value + creal(
                    products[row * cs0 + i * cs1]
                    * conj(projectors[row * bs0 + i * bs1])
                )
            target[row] = target[row] + multiplicity * value
    return diagonal


def add_projector_product(
    cnp.ndarray result, cnp.ndarray beta, cnp.ndarray coupled
):
    """Accumulate beta*coupled into a C-order Hpsi result without scratch."""
    cdef long long nrows
    cdef long long nchannels
    cdef long long nbands
    cdef double complex alpha = 1.0
    cdef double complex one = 1.0
    cdef int status
    if (
        result.ndim != 2
        or beta.ndim != 2
        or coupled.ndim != 2
        or result.dtype != np.complex128
        or beta.dtype != np.complex128
        or coupled.dtype != np.complex128
        or result.shape[0] != beta.shape[0]
        or beta.shape[1] != coupled.shape[0]
        or result.shape[1] != coupled.shape[1]
    ):
        raise ValueError("projector product matrices disagree")
    if not beta.flags.f_contiguous:
        return False
    nrows = beta.shape[0]
    nchannels = beta.shape[1]
    nbands = coupled.shape[1]
    if result.strides[1] == 16 and coupled.flags.c_contiguous:
        # A Fortran beta(nrows,nchannels) has the same byte layout as the
        # row-major transpose beta^T(nchannels,nrows).
        with nogil:
            status = qepy_zgemm64(
                101, 112, 111,
                nrows, nbands, nchannels,
                &alpha,
                <double complex *>cnp.PyArray_DATA(beta), nrows,
                <double complex *>cnp.PyArray_DATA(coupled), nbands,
                &one,
                <double complex *>cnp.PyArray_DATA(result),
                result.strides[0] // 16,
            )
    elif result.flags.f_contiguous and coupled.flags.f_contiguous:
        with nogil:
            status = qepy_zgemm64(
                102, 111, 111,
                nrows, nbands, nchannels,
                &alpha,
                <double complex *>cnp.PyArray_DATA(beta), nrows,
                <double complex *>cnp.PyArray_DATA(coupled), nchannels,
                &one,
                <double complex *>cnp.PyArray_DATA(result), nrows,
            )
    else:
        return False
    return status == 0


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


def qe_precondition_normalized_selected(
    cnp.ndarray residuals,
    cnp.ndarray eigenvalues,
    cnp.ndarray diagonal,
    cnp.ndarray selected,
):
    """Precondition and normalize selected roots for serial Davidson."""
    cdef Py_ssize_t nrows = residuals.shape[0]
    cdef Py_ssize_t nbands = selected.size
    cdef Py_ssize_t source_bands = residuals.shape[1]
    cdef Py_ssize_t g, b, source_band
    cdef Py_ssize_t rs0 = residuals.strides[0] // 16
    cdef Py_ssize_t rs1 = residuals.strides[1] // 16
    cdef cnp.ndarray output = np.empty(
        (nrows, nbands), dtype=np.complex128, order="F"
    )
    cdef cnp.ndarray squared_norms = np.zeros(nbands, dtype=np.float64)
    cdef double complex *source
    cdef double complex *target
    cdef double *values
    cdef double *kinetic
    cdef double *norms
    cdef int64_t *columns
    cdef double complex value
    cdef double x_ry, denominator, squared_norm, inverse_norm
    if residuals.ndim != 2 or residuals.dtype != np.complex128:
        raise ValueError("residuals must be a complex128 matrix")
    if (
        eigenvalues.ndim != 1
        or eigenvalues.dtype != np.float64
        or not eigenvalues.flags.c_contiguous
        or eigenvalues.size != source_bands
    ):
        raise ValueError("eigenvalues must match residual columns")
    if (
        diagonal.ndim != 1
        or diagonal.dtype != np.float64
        or not diagonal.flags.c_contiguous
        or diagonal.size != nrows
    ):
        raise ValueError("diagonal must be contiguous float64")
    if selected.dtype != np.int64 or not selected.flags.c_contiguous:
        raise ValueError("selected roots must be contiguous int64")
    source = <double complex *>cnp.PyArray_DATA(residuals)
    target = <double complex *>cnp.PyArray_DATA(output)
    values = <double *>cnp.PyArray_DATA(eigenvalues)
    kinetic = <double *>cnp.PyArray_DATA(diagonal)
    norms = <double *>cnp.PyArray_DATA(squared_norms)
    columns = <int64_t *>cnp.PyArray_DATA(selected)
    # For the small-band serial path a plain loop is materially cheaper than
    # entering three OpenMP regions for every k point and Davidson step.  It
    # also traverses each selected column only twice instead of the separate
    # precondition, norm, and normalized-copy passes used by the generic MPI
    # path.
    with nogil:
        for b in range(nbands):
            source_band = columns[b]
            if source_band < 0 or source_band >= source_bands:
                with gil:
                    raise ValueError("selected root is out of range")
            squared_norm = 0.0
            for g in range(nrows):
                x_ry = 2.0 * (kinetic[g] - values[source_band])
                denominator = 0.5 * (
                    1.0
                    + x_ry
                    + sqrt(1.0 + (x_ry - 1.0) * (x_ry - 1.0))
                )
                value = (
                    2.0 * source[g * rs0 + source_band * rs1] / denominator
                )
                target[g + b * nrows] = value
                squared_norm = squared_norm + (
                    creal(value) * creal(value) + cimag(value) * cimag(value)
                )
            norms[b] = squared_norm
            if squared_norm > 0.0:
                inverse_norm = 1.0 / sqrt(squared_norm)
                for g in range(nrows):
                    target[g + b * nrows] = (
                        target[g + b * nrows] * inverse_norm
                    )
    return output, squared_norms


def davidson_projected_rows(
    cnp.ndarray new_basis,
    cnp.ndarray applied_basis,
    cnp.ndarray basis,
):
    """Return ``new_basis**H @ (applied_basis, basis)`` without copies."""
    cdef long long nrows
    cdef long long nnew
    cdef long long nactive
    cdef cnp.ndarray h_rows
    cdef cnp.ndarray s_rows
    cdef double complex alpha = 1.0
    cdef double complex zero = 0.0
    cdef int h_status
    cdef int s_status
    if (
        new_basis.ndim != 2
        or applied_basis.ndim != 2
        or basis.ndim != 2
        or new_basis.dtype != np.complex128
        or applied_basis.dtype != np.complex128
        or basis.dtype != np.complex128
        or new_basis.shape[0] != applied_basis.shape[0]
        or new_basis.shape[0] != basis.shape[0]
        or applied_basis.shape[0] != basis.shape[0]
        or applied_basis.shape[1] != basis.shape[1]
    ):
        raise ValueError("Davidson projected-matrix operands disagree")
    # The Davidson owner arrays are column-major.  Calling CBLAS with
    # ConjTrans consumes that layout directly and, unlike ``a.conj().T @ b``,
    # does not allocate and populate an Npw-by-Nnew conjugated temporary.
    if (
        not new_basis.flags.f_contiguous
        or not applied_basis.flags.f_contiguous
        or not basis.flags.f_contiguous
    ):
        return None
    nrows = new_basis.shape[0]
    nnew = new_basis.shape[1]
    nactive = basis.shape[1]
    h_rows = np.empty((nnew, nactive), dtype=np.complex128, order="F")
    s_rows = np.empty_like(h_rows, order="F")
    with nogil:
        h_status = qepy_zgemm64(
            102, 113, 111,
            nnew, nactive, nrows,
            &alpha,
            <double complex *>cnp.PyArray_DATA(new_basis), nrows,
            <double complex *>cnp.PyArray_DATA(applied_basis), nrows,
            &zero,
            <double complex *>cnp.PyArray_DATA(h_rows), nnew,
        )
        s_status = qepy_zgemm64(
            102, 113, 111,
            nnew, nactive, nrows,
            &alpha,
            <double complex *>cnp.PyArray_DATA(new_basis), nrows,
            <double complex *>cnp.PyArray_DATA(basis), nrows,
            &zero,
            <double complex *>cnp.PyArray_DATA(s_rows), nnew,
        )
    if h_status != 0 or s_status != 0:
        return None
    return h_rows, s_rows


def davidson_residual(
    cnp.ndarray basis,
    cnp.ndarray applied_basis,
    cnp.ndarray coefficients,
    cnp.ndarray eigenvalues,
):
    """Form ``Hbasis*C - basis*C*e`` in one BLAS-owned result."""
    cdef long long nrows
    cdef long long nactive
    cdef long long nroots
    cdef Py_ssize_t column, row
    cdef cnp.ndarray scaled
    cdef cnp.ndarray output
    cdef double complex *source
    cdef double complex *target
    cdef double *roots
    cdef double complex one = 1.0
    cdef double complex minus_one = -1.0
    cdef double complex zero = 0.0
    cdef int first_status
    cdef int second_status
    if (
        basis.ndim != 2
        or applied_basis.ndim != 2
        or coefficients.ndim != 2
        or eigenvalues.ndim != 1
        or basis.dtype != np.complex128
        or applied_basis.dtype != np.complex128
        or coefficients.dtype != np.complex128
        or eigenvalues.dtype != np.float64
        or basis.shape[0] != applied_basis.shape[0]
        or basis.shape[1] != applied_basis.shape[1]
        or basis.shape[1] != coefficients.shape[0]
        or coefficients.shape[1] != eigenvalues.size
    ):
        raise ValueError("Davidson residual operands disagree")
    if (
        not basis.flags.f_contiguous
        or not applied_basis.flags.f_contiguous
        or not coefficients.flags.f_contiguous
        or not eigenvalues.flags.c_contiguous
    ):
        return None
    nrows = basis.shape[0]
    nactive = basis.shape[1]
    nroots = coefficients.shape[1]
    scaled = np.empty_like(coefficients, order="F")
    output = np.empty((nrows, nroots), dtype=np.complex128, order="F")
    source = <double complex *>cnp.PyArray_DATA(coefficients)
    target = <double complex *>cnp.PyArray_DATA(scaled)
    roots = <double *>cnp.PyArray_DATA(eigenvalues)
    with nogil:
        for column in range(nroots):
            for row in range(nactive):
                target[row + column * nactive] = (
                    source[row + column * nactive] * roots[column]
                )
        first_status = qepy_zgemm64(
            102, 111, 111,
            nrows, nroots, nactive,
            &one,
            <double complex *>cnp.PyArray_DATA(applied_basis), nrows,
            <double complex *>cnp.PyArray_DATA(coefficients), nactive,
            &zero,
            <double complex *>cnp.PyArray_DATA(output), nrows,
        )
        second_status = qepy_zgemm64(
            102, 111, 111,
            nrows, nroots, nactive,
            &minus_one,
            <double complex *>cnp.PyArray_DATA(basis), nrows,
            <double complex *>cnp.PyArray_DATA(scaled), nactive,
            &one,
            <double complex *>cnp.PyArray_DATA(output), nrows,
        )
    if first_status != 0 or second_status != 0:
        return None
    return output


def tetrahedron_integrated_sum(cnp.ndarray sorted_energies, double energy):
    """Sum QE's integrated linear-tetrahedron fractions without temporaries."""
    cdef Py_ssize_t count, index
    cdef double *values
    cdef double e1, e2, e3, e4, x, fraction
    cdef double total = 0.0
    if (
        sorted_energies.dtype != np.float64
        or not sorted_energies.flags.c_contiguous
        or sorted_energies.ndim < 2
        or sorted_energies.shape[sorted_energies.ndim - 1] != 4
    ):
        raise ValueError(
            "sorted_energies must be a contiguous float64 array ending in 4"
        )
    count = sorted_energies.size // 4
    values = <double *>cnp.PyArray_DATA(sorted_energies)
    with nogil:
        for index in prange(
            count, schedule="static", use_threads_if=count >= 4096
        ):
            e1 = values[4 * index]
            e2 = values[4 * index + 1]
            e3 = values[4 * index + 2]
            e4 = values[4 * index + 3]
            fraction = 0.0
            if e4 - e1 < 1.0e-14:
                if energy >= e1:
                    fraction = 1.0
            elif energy >= e4:
                fraction = 1.0
            elif energy > e1 and energy < e2:
                x = energy - e1
                fraction = x * x * x / (
                    (e2 - e1) * (e3 - e1) * (e4 - e1)
                )
            elif energy >= e3 and energy < e4:
                x = e4 - energy
                fraction = 1.0 - x * x * x / (
                    (e4 - e1) * (e4 - e2) * (e4 - e3)
                )
            elif energy >= e2 and energy < e3:
                x = energy - e2
                fraction = (
                    (e2 - e1) * (e2 - e1)
                    + 3.0 * (e2 - e1) * x
                    + 3.0 * x * x
                    - (e3 - e1 + e4 - e2)
                    / ((e3 - e2) * (e4 - e2)) * x * x * x
                ) / ((e3 - e1) * (e4 - e1))
            total += fraction
    return total


cdef inline void _tetrahedron_dos_point(
    double *values,
    Py_ssize_t count,
    double energy,
    double *density_result,
    double *integrated_result,
) noexcept nogil:
    cdef Py_ssize_t index
    cdef double e1, e2, e3, e4, x, fraction, density
    cdef double density_total = 0.0
    cdef double integrated_total = 0.0
    for index in range(count):
        e1 = values[4 * index]
        e2 = values[4 * index + 1]
        e3 = values[4 * index + 2]
        e4 = values[4 * index + 3]
        fraction = 0.0
        density = 0.0
        if e4 - e1 < 1.0e-14:
            if energy >= e1:
                fraction = 1.0
        elif energy >= e4:
            fraction = 1.0
        elif energy >= e1 and energy < e2:
            x = energy - e1
            fraction = x * x * x / (
                (e2 - e1) * (e3 - e1) * (e4 - e1)
            )
            density = 3.0 * x * x / (
                (e2 - e1) * (e3 - e1) * (e4 - e1)
            )
        elif energy >= e3 and energy < e4:
            x = e4 - energy
            fraction = 1.0 - x * x * x / (
                (e4 - e1) * (e4 - e2) * (e4 - e3)
            )
            density = 3.0 * x * x / (
                (e4 - e1) * (e4 - e2) * (e4 - e3)
            )
        elif energy >= e2 and energy < e3:
            x = energy - e2
            fraction = (
                (e2 - e1) * (e2 - e1)
                + 3.0 * (e2 - e1) * x
                + 3.0 * x * x
                - (e3 - e1 + e4 - e2)
                / ((e3 - e2) * (e4 - e2)) * x * x * x
            ) / ((e3 - e1) * (e4 - e1))
            density = (
                3.0 * (e2 - e1)
                + 6.0 * x
                - 3.0 * (e3 - e1 + e4 - e2) * x * x
                / ((e3 - e2) * (e4 - e2))
            ) / ((e3 - e1) * (e4 - e1))
        density_total += density
        integrated_total += fraction
    density_result[0] = density_total
    integrated_result[0] = integrated_total


def tetrahedron_dos_sums(
    cnp.ndarray sorted_energies, cnp.ndarray energy_grid
):
    """Return unnormalized DOS and integrated sums for many energies."""
    cdef Py_ssize_t count, energy_count, energy_index
    cdef double *values
    cdef double *energies
    cdef double *density_values
    cdef double *integrated_values
    cdef cnp.ndarray density
    cdef cnp.ndarray integrated
    if (
        sorted_energies.dtype != np.float64
        or not sorted_energies.flags.c_contiguous
        or sorted_energies.ndim < 2
        or sorted_energies.shape[sorted_energies.ndim - 1] != 4
    ):
        raise ValueError(
            "sorted_energies must be a contiguous float64 array ending in 4"
        )
    if (
        energy_grid.dtype != np.float64
        or not energy_grid.flags.c_contiguous
        or energy_grid.ndim != 1
    ):
        raise ValueError("energy_grid must be a contiguous float64 vector")
    count = sorted_energies.size // 4
    energy_count = energy_grid.size
    density = np.empty(energy_count, dtype=np.float64)
    integrated = np.empty(energy_count, dtype=np.float64)
    values = <double *>cnp.PyArray_DATA(sorted_energies)
    energies = <double *>cnp.PyArray_DATA(energy_grid)
    density_values = <double *>cnp.PyArray_DATA(density)
    integrated_values = <double *>cnp.PyArray_DATA(integrated)
    with nogil:
        for energy_index in prange(
            energy_count,
            schedule="static",
            use_threads_if=energy_count >= 2 and count >= 4096,
        ):
            _tetrahedron_dos_point(
                values,
                count,
                energies[energy_index],
                density_values + energy_index,
                integrated_values + energy_index,
            )
    return density, integrated


def tetrahedron_accumulate(
    cnp.ndarray connectivity,
    cnp.ndarray vertex_weights,
    cnp.ndarray interpolation,
    Py_ssize_t point_count,
):
    """Interpolate/scatter tetrahedron weights without large temporaries."""
    cdef Py_ssize_t tetra_count, band_count, corner_count, vertex_count
    cdef Py_ssize_t tetra_index, band, corner, vertex, point
    cdef bint identity_interpolation
    cdef int32_t *indices
    cdef double *source
    cdef double *coefficients
    cdef double *target
    cdef double contribution
    cdef cnp.ndarray result
    if (
        connectivity.dtype != np.int32
        or not connectivity.flags.c_contiguous
        or connectivity.ndim != 2
    ):
        raise ValueError("connectivity must be a contiguous int32 matrix")
    if (
        vertex_weights.dtype != np.float64
        or not vertex_weights.flags.c_contiguous
        or vertex_weights.ndim != 3
        or vertex_weights.shape[0] != connectivity.shape[0]
    ):
        raise ValueError(
            "vertex_weights must be a contiguous matching float64 tensor"
        )
    if (
        interpolation.dtype != np.float64
        or not interpolation.flags.c_contiguous
        or interpolation.ndim != 2
        or interpolation.shape[0] != vertex_weights.shape[2]
        or interpolation.shape[1] != connectivity.shape[1]
    ):
        raise ValueError(
            "interpolation must map vertex weights to connectivity points"
        )
    if point_count < 0:
        raise ValueError("point_count must be nonnegative")
    if connectivity.size and (
        np.min(connectivity) < 0 or np.max(connectivity) >= point_count
    ):
        raise ValueError("tetrahedron connectivity index is out of range")
    tetra_count = connectivity.shape[0]
    band_count = vertex_weights.shape[1]
    corner_count = connectivity.shape[1]
    vertex_count = vertex_weights.shape[2]
    identity_interpolation = (
        vertex_count == corner_count
        and np.array_equal(interpolation, np.eye(vertex_count))
    )
    result = np.zeros((point_count, band_count), dtype=np.float64)
    indices = <int32_t *>cnp.PyArray_DATA(connectivity)
    source = <double *>cnp.PyArray_DATA(vertex_weights)
    coefficients = <double *>cnp.PyArray_DATA(interpolation)
    target = <double *>cnp.PyArray_DATA(result)
    with nogil:
        for band in prange(
            band_count,
            schedule="static",
            use_threads_if=band_count >= 2 and tetra_count >= 1024,
        ):
            for tetra_index in range(tetra_count):
                for corner in range(corner_count):
                    point = indices[tetra_index * corner_count + corner]
                    if identity_interpolation:
                        contribution = source[
                            (tetra_index * band_count + band) * vertex_count
                            + corner
                        ]
                    else:
                        contribution = 0.0
                        for vertex in range(vertex_count):
                            contribution += coefficients[
                                vertex * corner_count + corner
                            ] * source[
                                (tetra_index * band_count + band) * vertex_count
                                + vertex
                            ]
                    target[point * band_count + band] += contribution
    return result


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


def subtract_band_energies(
    cnp.ndarray result,
    cnp.ndarray vectors,
    cnp.ndarray energies,
):
    """In-place ``result[:,b] -= energies[b] * vectors[:,b]``."""
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t nbands = vectors.shape[1]
    cdef Py_ssize_t total = nrows * nbands
    cdef Py_ssize_t index, g, b
    cdef Py_ssize_t vs0 = vectors.strides[0] // 16
    cdef Py_ssize_t vs1 = vectors.strides[1] // 16
    cdef Py_ssize_t rs0 = result.strides[0] // 16
    cdef Py_ssize_t rs1 = result.strides[1] // 16
    cdef double complex *source
    cdef double complex *target
    cdef double *values
    if (
        result.ndim != 2
        or vectors.ndim != 2
        or result.dtype != np.complex128
        or vectors.dtype != np.complex128
        or result.shape[0] != nrows
        or result.shape[1] != nbands
    ):
        raise ValueError("residual matrices disagree")
    if (
        energies.dtype != np.float64
        or not energies.flags.c_contiguous
        or energies.size != nbands
    ):
        raise ValueError("band energies must be contiguous float64")
    source = <double complex *>cnp.PyArray_DATA(vectors)
    target = <double complex *>cnp.PyArray_DATA(result)
    values = <double *>cnp.PyArray_DATA(energies)
    with nogil:
        for index in prange(
            total, schedule="static", use_threads_if=total >= 1024
        ):
            b = index // nrows
            g = index - b * nrows
            target[g * rs0 + b * rs1] -= (
                values[b] * source[g * vs0 + b * vs1]
            )
    return result


def column_inner_products(cnp.ndarray left, cnp.ndarray right):
    """Rank-local columnwise ``left^H right`` without matrix temporaries."""
    cdef Py_ssize_t nrows = left.shape[0]
    cdef Py_ssize_t nbands = left.shape[1]
    cdef Py_ssize_t total = nrows * nbands
    cdef Py_ssize_t padded = ((nbands + 7) // 8) * 8
    cdef Py_ssize_t index, g, b
    cdef Py_ssize_t ls0 = left.strides[0] // 16
    cdef Py_ssize_t ls1 = left.strides[1] // 16
    cdef Py_ssize_t rs0 = right.strides[0] // 16
    cdef Py_ssize_t rs1 = right.strides[1] // 16
    cdef int nthreads = omp_get_max_threads()
    cdef int thread
    cdef cnp.ndarray partial = np.zeros(
        (nthreads, padded), dtype=np.complex128
    )
    cdef cnp.ndarray output = np.zeros(nbands, dtype=np.complex128)
    cdef double complex *lvalues
    cdef double complex *rvalues
    cdef double complex *sums
    cdef double complex *target
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.dtype != np.complex128
        or right.dtype != np.complex128
        or left.shape[0] != right.shape[0]
        or left.shape[1] != right.shape[1]
    ):
        raise ValueError("inner-product matrices disagree")
    lvalues = <double complex *>cnp.PyArray_DATA(left)
    rvalues = <double complex *>cnp.PyArray_DATA(right)
    sums = <double complex *>cnp.PyArray_DATA(partial)
    target = <double complex *>cnp.PyArray_DATA(output)
    with nogil:
        for index in prange(
            total, schedule="static", use_threads_if=total >= 1024
        ):
            thread = omp_get_thread_num()
            b = index // nrows
            g = index - b * nrows
            sums[thread * padded + b] += (
                conj(lvalues[g * ls0 + b * ls1])
                * rvalues[g * rs0 + b * rs1]
            )
        for b in prange(
            nbands, schedule="static", use_threads_if=nbands > 1
        ):
            for thread in range(nthreads):
                target[b] += sums[thread * padded + b]
    return output


def column_diagonal_expectations(
    cnp.ndarray vectors, cnp.ndarray diagonal
):
    """Rank-local real ``sum_g diagonal[g] |vectors[g,b]|^2``."""
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t nbands = vectors.shape[1]
    cdef Py_ssize_t total = nrows * nbands
    cdef Py_ssize_t padded = ((nbands + 7) // 8) * 8
    cdef Py_ssize_t index, g, b
    cdef Py_ssize_t vs0 = vectors.strides[0] // 16
    cdef Py_ssize_t vs1 = vectors.strides[1] // 16
    cdef int nthreads = omp_get_max_threads()
    cdef int thread
    cdef cnp.ndarray partial = np.zeros((nthreads, padded), dtype=np.float64)
    cdef cnp.ndarray output = np.zeros(nbands, dtype=np.float64)
    cdef double complex *source
    cdef double *dvalues
    cdef double *sums
    cdef double *target
    cdef double real_part, imag_part
    if vectors.ndim != 2 or vectors.dtype != np.complex128:
        raise ValueError("vectors must be complex128")
    if (
        diagonal.dtype != np.float64
        or not diagonal.flags.c_contiguous
        or diagonal.size != nrows
    ):
        raise ValueError("diagonal must be contiguous float64")
    source = <double complex *>cnp.PyArray_DATA(vectors)
    dvalues = <double *>cnp.PyArray_DATA(diagonal)
    sums = <double *>cnp.PyArray_DATA(partial)
    target = <double *>cnp.PyArray_DATA(output)
    with nogil:
        for index in prange(
            total, schedule="static", use_threads_if=total >= 1024
        ):
            thread = omp_get_thread_num()
            b = index // nrows
            g = index - b * nrows
            real_part = creal(source[g * vs0 + b * vs1])
            imag_part = cimag(source[g * vs0 + b * vs1])
            sums[thread * padded + b] += dvalues[g] * (
                real_part * real_part + imag_part * imag_part
            )
        for b in prange(
            nbands, schedule="static", use_threads_if=nbands > 1
        ):
            for thread in range(nthreads):
                target[b] += sums[thread * padded + b]
    return output


def rmm_kinetic_precondition(
    cnp.ndarray residuals,
    cnp.ndarray kinetic,
    cnp.ndarray expectations,
    cnp.ndarray selected,
):
    """Apply the Kresse--Furthmueller diagonal to selected residuals.

    The caller retains the analytic expression in Python.  This kernel only
    evaluates it without materializing the three real Npw-by-Nactive arrays
    formerly used for x, x**2, and the final diagonal.
    """
    cdef Py_ssize_t nrows = residuals.shape[0]
    cdef Py_ssize_t nbands = selected.size
    cdef Py_ssize_t total = nrows * nbands
    cdef Py_ssize_t index, g, b, source_band
    cdef Py_ssize_t rs0 = residuals.strides[0] // 16
    cdef Py_ssize_t rs1 = residuals.strides[1] // 16
    cdef cnp.ndarray output = np.empty(
        (nrows, nbands), dtype=np.complex128, order="F"
    )
    cdef double complex *source
    cdef double complex *target
    cdef double *kinetic_values
    cdef double *energy_values
    cdef int64_t *columns
    cdef double x, x2, numerator, denominator, scale
    if residuals.ndim != 2 or residuals.dtype != np.complex128:
        raise ValueError("residuals must be a complex128 matrix")
    if (
        kinetic.dtype != np.float64
        or not kinetic.flags.c_contiguous
        or kinetic.size != nrows
        or expectations.dtype != np.float64
        or not expectations.flags.c_contiguous
        or expectations.size != nbands
        or selected.dtype != np.int64
        or not selected.flags.c_contiguous
    ):
        raise ValueError("invalid RMM preconditioner metadata")
    source = <double complex *>cnp.PyArray_DATA(residuals)
    target = <double complex *>cnp.PyArray_DATA(output)
    kinetic_values = <double *>cnp.PyArray_DATA(kinetic)
    energy_values = <double *>cnp.PyArray_DATA(expectations)
    columns = <int64_t *>cnp.PyArray_DATA(selected)
    with nogil:
        for index in prange(
            total, schedule="static", use_threads_if=total >= 1024
        ):
            b = index // nrows
            g = index - b * nrows
            source_band = columns[b]
            x = kinetic_values[g] / (1.5 * energy_values[b])
            x2 = x * x
            numerator = 27.0 + x * (18.0 + x * (12.0 + 8.0 * x))
            denominator = numerator + 16.0 * x2 * x2
            scale = (-4.0 / 3.0) * numerator / (
                energy_values[b] * denominator
            )
            target[index] = scale * source[
                g * rs0 + source_band * rs1
            ]
    return output


def qe_random_pairs(
    Py_ssize_t plane_waves,
    Py_ssize_t bands,
    cnp.ndarray table,
    long long current,
    long long seed,
):
    """Advance QE randy() in band-major order and retain every PW row."""
    cdef cnp.ndarray first = np.empty(
        (plane_waves, bands), dtype=np.float64, order="F"
    )
    cdef cnp.ndarray second = np.empty_like(first, order="F")
    cdef double *first_values = <double *>cnp.PyArray_DATA(first)
    cdef double *second_values = <double *>cnp.PyArray_DATA(second)
    cdef int64_t *state
    cdef Py_ssize_t band, row, destination, table_index
    if table.dtype != np.int64 or not table.flags.c_contiguous or table.size != 97:
        raise ValueError("QE random table must contain 97 contiguous int64 values")
    state = <int64_t *>cnp.PyArray_DATA(table)
    with nogil:
        for band in range(bands):
            for row in range(plane_waves):
                destination = row + band * plane_waves
                table_index = (97 * current) // 714025
                current = state[table_index]
                first_values[destination] = current / 714025.0
                seed = (1366 * seed + 150889) % 714025
                state[table_index] = seed
                table_index = (97 * current) // 714025
                current = state[table_index]
                second_values[destination] = current / 714025.0
                seed = (1366 * seed + 150889) % 714025
                state[table_index] = seed
    return first, second, current, seed


def qe_random_pairs_rows(
    Py_ssize_t plane_waves,
    Py_ssize_t bands,
    cnp.ndarray rows,
    cnp.ndarray table,
    long long current,
    long long seed,
):
    """Advance the global QE stream while retaining sorted local PW rows."""
    cdef Py_ssize_t local_rows = rows.size
    cdef cnp.ndarray first = np.empty(
        (local_rows, bands), dtype=np.float64, order="F"
    )
    cdef cnp.ndarray second = np.empty_like(first, order="F")
    cdef double *first_values = <double *>cnp.PyArray_DATA(first)
    cdef double *second_values = <double *>cnp.PyArray_DATA(second)
    cdef int64_t *selected
    cdef int64_t *state
    cdef Py_ssize_t band, row, local_row, destination, table_index
    cdef double amplitude, phase
    if rows.dtype != np.int64 or not rows.flags.c_contiguous:
        raise ValueError("selected QE random rows must be contiguous int64")
    if table.dtype != np.int64 or not table.flags.c_contiguous or table.size != 97:
        raise ValueError("QE random table must contain 97 contiguous int64 values")
    selected = <int64_t *>cnp.PyArray_DATA(rows)
    state = <int64_t *>cnp.PyArray_DATA(table)
    with nogil:
        for band in range(bands):
            local_row = 0
            for row in range(plane_waves):
                table_index = (97 * current) // 714025
                current = state[table_index]
                amplitude = current / 714025.0
                seed = (1366 * seed + 150889) % 714025
                state[table_index] = seed
                table_index = (97 * current) // 714025
                current = state[table_index]
                phase = current / 714025.0
                seed = (1366 * seed + 150889) % 714025
                state[table_index] = seed
                if local_row < local_rows and row == selected[local_row]:
                    destination = local_row + band * local_rows
                    first_values[destination] = amplitude
                    second_values[destination] = phase
                    local_row += 1
    return first, second, current, seed


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
    if itemsize == 1:
        return (<const uint8_t *>data)[position]
    if itemsize == 2:
        return (<const uint16_t *>data)[position]
    if itemsize == 4:
        return (<const uint32_t *>data)[position]
    return (<const int64_t *>data)[position]


def project_density_stars(
    cnp.ndarray coefficients,
    cnp.ndarray offsets,
    cnp.ndarray members,
    cnp.ndarray weight_table,
    cnp.ndarray weight_indices,
    cnp.ndarray fill_table,
    cnp.ndarray fill_indices,
):
    """Apply a reciprocal scalar-density star projector without temporaries."""
    cdef Py_ssize_t number_of_stars = offsets.size - 1
    cdef Py_ssize_t star, position, start, stop, row
    cdef int member_itemsize = members.itemsize
    cdef int weight_index_itemsize = weight_indices.itemsize
    cdef int fill_index_itemsize = fill_indices.itemsize
    cdef int32_t *star_offsets
    cdef const void *star_members
    cdef double complex *values
    cdef double complex *star_weight_table
    cdef double complex *star_fill_table
    cdef const void *star_weight_indices
    cdef const void *star_fill_indices
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
        weight_table.ndim != 1
        or fill_table.ndim != 1
        or weight_table.dtype != np.complex128
        or fill_table.dtype != np.complex128
        or weight_indices.ndim != 1
        or fill_indices.ndim != 1
        or weight_indices.size != members.size
        or fill_indices.size != members.size
        or weight_index_itemsize not in (1, 2, 4)
        or fill_index_itemsize not in (1, 2, 4)
    ):
        raise ValueError("star factor tables and indices must match members")
    star_offsets = <int32_t *>cnp.PyArray_DATA(offsets)
    star_members = cnp.PyArray_DATA(members)
    values = <double complex *>cnp.PyArray_DATA(coefficients)
    star_weight_table = <double complex *>cnp.PyArray_DATA(weight_table)
    star_fill_table = <double complex *>cnp.PyArray_DATA(fill_table)
    star_weight_indices = cnp.PyArray_DATA(weight_indices)
    star_fill_indices = cnp.PyArray_DATA(fill_indices)
    with nogil:
        for star in range(number_of_stars):
            start = star_offsets[star]
            stop = star_offsets[star + 1]
            averaged = 0.0
            for position in range(start, stop):
                row = _index_value(star_members, member_itemsize, position)
                averaged += star_weight_table[
                    _index_value(
                        star_weight_indices,
                        weight_index_itemsize,
                        position,
                    )
                ] * values[row]
            for position in range(start, stop):
                row = _index_value(star_members, member_itemsize, position)
                values[row] = star_fill_table[
                    _index_value(
                        star_fill_indices,
                        fill_index_itemsize,
                        position,
                    )
                ] * averaged


def hartree_coefficients(
    cnp.ndarray density_coefficients,
    cnp.ndarray g2,
    cnp.ndarray potential_coefficients,
):
    """Fill 4*pi*rho(G)/|G|^2 without Boolean/fancy-index temporaries."""
    cdef Py_ssize_t size, index
    cdef double complex *density
    cdef double complex *potential
    cdef double *squared_norm
    if (
        density_coefficients.ndim != 1
        or density_coefficients.dtype != np.complex128
        or not density_coefficients.flags.c_contiguous
        or potential_coefficients.ndim != 1
        or potential_coefficients.size != density_coefficients.size
        or potential_coefficients.dtype != np.complex128
        or not potential_coefficients.flags.c_contiguous
        or g2.ndim != 1
        or g2.size != density_coefficients.size
        or g2.dtype != np.float64
        or not g2.flags.c_contiguous
    ):
        raise ValueError("Hartree kernel needs matching contiguous vectors")
    size = density_coefficients.size
    density = <double complex *>cnp.PyArray_DATA(density_coefficients)
    potential = <double complex *>cnp.PyArray_DATA(potential_coefficients)
    squared_norm = <double *>cnp.PyArray_DATA(g2)
    with nogil:
        for index in range(size):
            if squared_norm[index] > 1.0e-14:
                potential[index] = (
                    (4.0 * 3.141592653589793238462643383279502884)
                    * density[index]
                    / squared_norm[index]
                )
            else:
                potential[index] = 0.0


def hartree_residual_metric(
    cnp.ndarray residual_coefficients,
    cnp.ndarray g2,
):
    """Return sum(4*pi*|drho(G)|^2/G^2) without vector temporaries.

    Neumaier compensation keeps the scalar reduction at least as stable as
    NumPy's pairwise sum while avoiding the Boolean mask, indexed copies,
    absolute-value vector, square vector, and quotient vector formerly made
    once per SCF iteration.
    """
    cdef Py_ssize_t size, index
    cdef double complex *residual
    cdef double *squared_norm
    cdef double real_part, imaginary_part, term
    cdef double total = 0.0
    cdef double correction = 0.0
    cdef double updated
    if (
        residual_coefficients.ndim != 1
        or residual_coefficients.dtype != np.complex128
        or not residual_coefficients.flags.c_contiguous
        or g2.ndim != 1
        or g2.size != residual_coefficients.size
        or g2.dtype != np.float64
        or not g2.flags.c_contiguous
    ):
        raise ValueError("Hartree residual metric needs matching contiguous vectors")
    size = residual_coefficients.size
    residual = <double complex *>cnp.PyArray_DATA(residual_coefficients)
    squared_norm = <double *>cnp.PyArray_DATA(g2)
    with nogil:
        for index in range(size):
            if squared_norm[index] <= 1.0e-14:
                continue
            real_part = creal(residual[index])
            imaginary_part = cimag(residual[index])
            term = (
                (4.0 * 3.141592653589793238462643383279502884)
                * (real_part * real_part + imaginary_part * imaginary_part)
                / squared_norm[index]
            )
            updated = total + term
            if total >= term:
                correction += (total - updated) + term
            else:
                correction += (term - updated) + total
            total = updated
    return total + correction


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


def fftw_mpi_available():
    """Whether this extension was linked to an ABI-compatible FFTW-MPI."""
    return bool(qepy_fftw_mpi_available())


def fftw_mpi_layout(object lengths, Comm comm, Py_ssize_t block0=0):
    """Return FFTW-MPI allocation and first-dimension slab metadata."""
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] dimensions = np.asarray(
        lengths, dtype=np.int64
    )
    cdef ptrdiff_t local_n0 = 0
    cdef ptrdiff_t local_0_start = 0
    cdef ptrdiff_t allocation = 0
    if dimensions.size != 3:
        raise ValueError("FFTW-MPI layout requires exactly three dimensions")
    if not qepy_fftw_mpi_available():
        raise RuntimeError("qepy-pw was built without FFTW-MPI")
    with nogil:
        allocation = qepy_fftw_mpi_local_size_3d(
            dimensions[0], dimensions[1], dimensions[2], block0,
            comm.ob_mpi, &local_n0, &local_0_start,
        )
    if allocation < 0:
        raise RuntimeError("FFTW-MPI failed to determine the local layout")
    return int(allocation), int(local_n0), int(local_0_start)


cdef class NativeFFTWMPIPlan:
    """Collective in-place 3-D FFTW-MPI plans bound to one NumPy buffer."""

    cdef qepy_fftw_plan _forward
    cdef qepy_fftw_plan _backward
    cdef object _owner

    def __cinit__(
        self,
        cnp.ndarray array,
        object lengths,
        Comm comm,
        Py_ssize_t block0=0,
        unsigned flags=64,
        int nthreads=1,
    ):
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] dimensions
        cdef ptrdiff_t local_n0 = 0
        cdef ptrdiff_t local_0_start = 0
        cdef ptrdiff_t allocation = 0
        cdef double complex *pointer
        self._forward = NULL
        self._backward = NULL
        if not qepy_fftw_mpi_available():
            raise RuntimeError("qepy-pw was built without FFTW-MPI")
        if nthreads < 1:
            raise ValueError("FFTW-MPI thread count must be positive")
        if array.dtype != np.complex128 or not array.flags.c_contiguous:
            raise ValueError("FFTW-MPI plans require contiguous complex128")
        dimensions = np.asarray(lengths, dtype=np.int64)
        if dimensions.size != 3:
            raise ValueError("FFTW-MPI plan requires exactly three dimensions")
        with nogil:
            allocation = qepy_fftw_mpi_local_size_3d(
                dimensions[0], dimensions[1], dimensions[2], block0,
                comm.ob_mpi, &local_n0, &local_0_start,
            )
        if array.size < allocation:
            raise ValueError("FFTW-MPI buffer is smaller than alloc_local")
        self._owner = array
        pointer = <double complex *>cnp.PyArray_DATA(array)
        with nogil:
            self._forward = qepy_fftw_mpi_plan_3d(
                pointer, dimensions[0], dimensions[1], dimensions[2],
                block0, comm.ob_mpi, -1, flags, nthreads,
            )
            self._backward = qepy_fftw_mpi_plan_3d(
                pointer, dimensions[0], dimensions[1], dimensions[2],
                block0, comm.ob_mpi, 1, flags, nthreads,
            )
        if self._forward == NULL or self._backward == NULL:
            raise RuntimeError("FFTW-MPI failed to create a collective plan")

    def execute(self, direction):
        if direction == "forward":
            with nogil:
                qepy_fftw_mpi_execute(self._forward)
        elif direction == "backward":
            with nogil:
                qepy_fftw_mpi_execute(self._backward)
        else:
            raise ValueError("FFT direction must be 'forward' or 'backward'")

    def __dealloc__(self):
        if self._forward != NULL:
            fftw_destroy_plan(self._forward)
            self._forward = NULL
        if self._backward != NULL:
            fftw_destroy_plan(self._backward)
            self._backward = NULL


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


def add_diagonal_product(
    cnp.ndarray result,
    cnp.ndarray diagonal,
    cnp.ndarray vectors,
):
    """Accumulate ``diag * vectors`` without a matrix-sized temporary.

    ``PlaneWaveHamiltonian`` already owns the local-potential result.  Doing
    this elementary kinetic contribution here avoids NumPy materialising a
    second Npw-by-nband array before the in-place addition.
    """
    cdef Py_ssize_t nrows = vectors.shape[0]
    cdef Py_ssize_t nbands = vectors.shape[1]
    cdef Py_ssize_t total = nrows * nbands
    cdef Py_ssize_t index, row, band
    cdef Py_ssize_t vector_stride0 = vectors.strides[0] // 16
    cdef Py_ssize_t vector_stride1 = vectors.strides[1] // 16
    cdef Py_ssize_t result_stride0 = result.strides[0] // 16
    cdef Py_ssize_t result_stride1 = result.strides[1] // 16
    cdef double complex *result_values
    cdef double complex *vector_values
    cdef double *diagonal_values
    if (
        result.ndim != 2
        or result.dtype != np.complex128
        or vectors.ndim != 2
        or vectors.dtype != np.complex128
        or result.shape[0] != nrows
        or result.shape[1] != nbands
    ):
        raise ValueError("result and vectors must be matching complex128 matrices")
    if (
        diagonal.ndim != 1
        or diagonal.dtype != np.float64
        or not diagonal.flags.c_contiguous
        or diagonal.size != nrows
    ):
        raise ValueError("diagonal must be contiguous float64")
    result_values = <double complex *>cnp.PyArray_DATA(result)
    vector_values = <double complex *>cnp.PyArray_DATA(vectors)
    diagonal_values = <double *>cnp.PyArray_DATA(diagonal)
    with nogil:
        for index in prange(
            total, schedule="static", use_threads_if=total >= 65536
        ):
            row = index // nbands
            band = index - row * nbands
            result_values[
                row * result_stride0 + band * result_stride1
            ] += diagonal_values[row] * vector_values[
                row * vector_stride0 + band * vector_stride1
            ]


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
    diagonal=None,
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
    cdef cnp.ndarray diagonal_array
    cdef double *diagonal_values = NULL
    cdef double complex *result_values = <double complex *>cnp.PyArray_DATA(result)
    if potential.dtype != np.float64 or not potential.flags.c_contiguous:
        raise ValueError("native local potential must be contiguous float64")
    if result.dtype != np.complex128 or result.ndim != 2:
        raise ValueError("native local result must be a complex128 matrix")
    if diagonal is not None:
        diagonal_array = diagonal
        if (
            diagonal_array.dtype != np.float64
            or not diagonal_array.flags.c_contiguous
            or diagonal_array.ndim != 1
            or diagonal_array.size != nrows
        ):
            raise ValueError("kinetic diagonal must be contiguous float64")
        diagonal_values = <double *>cnp.PyArray_DATA(diagonal_array)
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
                        + (
                            diagonal_values[g]
                            * vector_values[
                                g * vector_stride0 + b * vector_stride1
                            ]
                            if diagonal_values != NULL else 0.0
                        )
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
                        + (
                            diagonal_values[g]
                            * vector_values[
                                g * vector_stride0 + b * vector_stride1
                            ]
                            if diagonal_values != NULL else 0.0
                        )
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
    diagonal=None,
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
    cdef cnp.ndarray diagonal_array
    cdef double *diagonal_values = NULL
    if potential.dtype != np.float64 or not potential.flags.c_contiguous:
        raise ValueError("spatial serial potential must be contiguous float64")
    if potential.ndim != 3 or potential.size != fft_size:
        raise ValueError("spatial serial potential has the wrong shape")
    if diagonal is not None:
        diagonal_array = diagonal
        if (
            diagonal_array.dtype != np.float64
            or not diagonal_array.flags.c_contiguous
            or diagonal_array.ndim != 1
            or diagonal_array.size != nrows
        ):
            raise ValueError("kinetic diagonal must be contiguous float64")
        diagonal_values = <double *>cnp.PyArray_DATA(diagonal_array)
    memset(grid_values, 0, grid.nbytes)
    with nogil:
        qepy_apply_serial_spatial(
            grid_values,
            vector_values,
            potential_values,
            diagonal_values,
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
    cnp.ndarray result,
    Py_ssize_t fft_size,
    diagonal=None,
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
    cdef Py_ssize_t result_stride0 = result.strides[0] // 16
    cdef Py_ssize_t result_stride1 = result.strides[1] // 16
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
    cdef cnp.ndarray diagonal_array
    cdef double *diagonal_values = NULL
    cdef double complex *send_values = <double complex *>cnp.PyArray_DATA(reverse_send)
    cdef double complex *received_values = <double complex *>cnp.PyArray_DATA(inverse_receive)
    cdef double complex *result_values = <double complex *>cnp.PyArray_DATA(result)
    cdef int status

    if sticks.dtype != np.complex128 or slab.dtype != np.complex128:
        raise ValueError("native FFT payloads must be complex128")
    if potential.dtype != np.float64 or not potential.flags.c_contiguous:
        raise ValueError("native local potential must be contiguous float64")
    if (
        result.dtype != np.complex128
        or result.ndim != 2
        or result.shape[0] != nrows
        or result.shape[1] != nbands
    ):
        raise ValueError("native local result has the wrong shape or dtype")
    if diagonal is not None:
        diagonal_array = diagonal
        if (
            diagonal_array.dtype != np.float64
            or not diagonal_array.flags.c_contiguous
            or diagonal_array.ndim != 1
            or diagonal_array.size != nrows
        ):
            raise ValueError("kinetic diagonal must be contiguous float64")
        diagonal_values = <double *>cnp.PyArray_DATA(diagonal_array)
    if z_plan._howmany == 1 and omp_get_max_threads() > 1:
        with nogil:
            qepy_inverse_z_sticks(
                stick_values, vector_values,
                z_data, z_itemsize, stick_data, stick_itemsize,
                z_plan._backward, nz, nsticks, nbands, nrows,
                vector_stride0, vector_stride1,
            )
    else:
        memset(stick_values, 0, sticks.nbytes)
        with nogil:
            for g in prange(
                nrows, schedule="static",
                use_threads_if=nrows * nbands >= 65536
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
    if xy_plan._howmany == 1 and omp_get_max_threads() > 1:
        with nogil:
            qepy_apply_slab_bands(
                slab_values,
                received_values,
                send_values,
                potential_values,
                point_data,
                point_itemsize,
                xy_plan._backward,
                xy_plan._forward,
                nbands,
                transfer_rows,
                slab.shape[0],
                slab.shape[1] * slab.shape[2],
            )
    else:
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
    if z_plan._howmany == 1 and omp_get_max_threads() > 1:
        with nogil:
            qepy_forward_z_gather(
                result_values, stick_values,
                vector_values, diagonal_values,
                z_data, z_itemsize, stick_data, stick_itemsize,
                z_plan._forward, nz, nsticks, nbands, nrows, fft_size,
                result_stride0, result_stride1,
                vector_stride0, vector_stride1,
            )
    else:
        with nogil:
            z_plan.forward()
            for g in prange(
                nrows, schedule="static",
                use_threads_if=nrows * nbands >= 65536
            ):
                point = (
                    (_index_value(z_data, z_itemsize, g) * nsticks
                     + _index_value(stick_data, stick_itemsize, g)) * nbands
                )
                for b in range(nbands):
                    result_values[
                        g * result_stride0 + b * result_stride1
                    ] = (
                        stick_values[point + b] / fft_size
                        + (
                            diagonal_values[g]
                            * vector_values[
                                g * vector_stride0 + b * vector_stride1
                            ]
                            if diagonal_values != NULL else 0.0
                        )
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
    cdef Py_ssize_t nz = sticks.shape[0]
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

    if z_plan._howmany == 1 and omp_get_max_threads() > 1:
        with nogil:
            qepy_inverse_z_sticks(
                stick_values, vector_values,
                z_data, z_itemsize, stick_data, stick_itemsize,
                z_plan._backward, nz, nsticks, nbands, nrows,
                vector_stride0, vector_stride1,
            )
    else:
        memset(stick_values, 0, sticks.nbytes)
        with nogil:
            for g in prange(
                nrows, schedule="static",
                use_threads_if=nrows * nbands >= 65536
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
    if xy_plan._howmany == 1 and omp_get_max_threads() > 1:
        with nogil:
            qepy_accumulate_slab_bands(
                density_values,
                slab_values,
                received_values,
                weights,
                point_data,
                point_itemsize,
                xy_plan._backward,
                nbands,
                transfer_rows,
                local_z,
                plane_size,
            )
    else:
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
