"""Setuptools entry point for the mandatory Cython/MPI/FFTW extension."""

from __future__ import annotations

from ctypes.util import find_library
import glob
import os
from pathlib import Path
import shutil

from Cython.Build import cythonize
import mpi4py
from mpi4py import MPI
import numpy as np
from setuptools import Extension, setup


include_dirs = [np.get_include(), mpi4py.get_include()]
libraries: list[str] = []
library_dirs: list[str] = []
runtime_library_dirs: list[str] = []
extra_objects: list[str] = []
macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
numeric_runtime_link_args: list[str] = []


def _library_root_candidates(*environment_names: str) -> list[Path]:
    roots = [Path("/usr"), Path("/usr/local")]
    for name in environment_names:
        value = os.environ.get(name)
        if value:
            root = Path(value).expanduser().resolve()
            if root not in roots:
                roots.insert(0, root)
    return roots


fftw_roots = _library_root_candidates("FFTW_ROOT", "FFTW3_ROOT")
fftw_include = next(
    (root / "include" for root in fftw_roots if (root / "include/fftw3.h").exists()),
    None,
)
fftw_library_directory = next(
    (
        directory
        for root in fftw_roots
        for directory in (root / "lib64", root / "lib")
        if any(directory.glob("libfftw3.so*"))
    ),
    None,
)

# Prefer one coherent system numerical stack. This is particularly important
# for MPI: mpi4py, FFTW-MPI, and the compiler wrapper must use the same MPI
# ABI. A pyFFTW wheel remains a portable fallback when system FFTW headers or
# shared libraries are absent.
system_fftw = (
    fftw_include is not None
    and (
        find_library("fftw3") is not None
        or bool(fftw_library_directory)
    )
    and (
        find_library("fftw3_omp") is not None
        or bool(
            fftw_library_directory
            and any(fftw_library_directory.glob("libfftw3_omp.so*"))
        )
    )
)
if system_fftw:
    if str(fftw_include) not in include_dirs:
        include_dirs.append(str(fftw_include))
    if fftw_library_directory is not None:
        library_dirs.append(str(fftw_library_directory))
        runtime_library_dirs.append(str(fftw_library_directory))
    libraries.extend(["fftw3_omp", "fftw3"])
    if (
        (fftw_include / "fftw3-mpi.h").exists()
        and (
            find_library("fftw3_mpi") is not None
            or bool(
                fftw_library_directory
                and any(fftw_library_directory.glob("libfftw3_mpi.so*"))
            )
        )
    ):
        libraries.insert(0, "fftw3_mpi")
        macros.append(("QEPY_HAVE_FFTW_MPI", "1"))
    print("qepy-pw build: using system FFTW" + (
        " with FFTW-MPI" if any(name == "fftw3_mpi" for name in libraries)
        else ""
    ))
else:
    import pyfftw

    site_packages = Path(pyfftw.__file__).resolve().parent.parent
    fftw_candidates = sorted(
        glob.glob(str(site_packages / "pyfftw.libs" / "libfftw3-*.so*"))
    )
    fftw_openmp_candidates = sorted(
        glob.glob(
            str(site_packages / "pyfftw.libs" / "libfftw3_omp-*.so*")
        )
    )
    if not fftw_candidates or not fftw_openmp_candidates:
        raise RuntimeError(
            "neither system FFTW nor pyFFTW's bundled FFTW was found"
        )
    extra_objects.extend(
        [fftw_candidates[-1], fftw_openmp_candidates[-1]]
    )
    fftw_directory = str(Path(fftw_candidates[-1]).parent)
    runtime_library_dirs.append(fftw_directory)
    print("qepy-pw build: using pyFFTW bundled FFTW fallback")

# Link an explicitly installed accelerated BLAS/LAPACKE provider when one is
# available. Do not replace NumPy's optimized BLAS with the system reference
# libblas/liblapack pair, which is normally much slower.
mkl_roots = []
if os.environ.get("MKLROOT"):
    mkl_roots.append(Path(os.environ["MKLROOT"]).expanduser().resolve())
mkl_roots.extend(
    [
        Path("/opt/intel/oneapi/mkl/latest"),
        Path("/opt/intel/mkl"),
        Path("/usr/local/intel/mkl"),
    ]
)
mkl_library_dir = next(
    (
        directory
        for root in mkl_roots
        for directory in (root / "lib/intel64", root / "lib")
        if any(directory.glob("libmkl_rt.so*"))
    ),
    None,
)
mkl_available = mkl_library_dir is not None
if mkl_available:
    library_dirs.append(str(mkl_library_dir))
    runtime_library_dirs.append(str(mkl_library_dir))
    numeric_runtime_link_args.extend(
        ["-Wl,--no-as-needed", "-lmkl_rt", "-Wl,--as-needed"]
    )
    print("qepy-pw build: using system Intel MKL")
else:
    numpy_library_dir = Path(np.__file__).resolve().parent.parent / "numpy.libs"
    numpy_bundles_openblas = bool(
        list(numpy_library_dir.glob("*openblas*.so*"))
    )
    force_system_blas = os.environ.get(
        "QEPY_LINK_SYSTEM_BLAS", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}

if not mkl_available and find_library("openblas") is not None and (
    force_system_blas or not numpy_bundles_openblas
):
    numeric_libraries = ["-lopenblas"]
    if find_library("lapacke") is not None:
        numeric_libraries.append("-llapacke")
    # The native wrappers resolve both LP64 system symbols and NumPy's ILP64
    # private symbols with dlsym.  Keep an explicitly selected system runtime
    # in DT_NEEDED even under the linker's usual --as-needed default.
    numeric_runtime_link_args.extend(
        ["-Wl,--no-as-needed", *numeric_libraries, "-Wl,--as-needed"]
    )
    print("qepy-pw build: using system OpenBLAS")
elif not mkl_available:
    print(
        "qepy-pw build: using NumPy BLAS/LAPACK runtime "
        "(avoids a second OpenBLAS ABI in the process)"
    )

explicit_mpicc = (
    os.environ.get("MPICC")
    or os.environ.get("MPI4PY_BUILD_MPICC")
)
mpi_vendor = MPI.get_vendor()[0].strip().lower()
if explicit_mpicc:
    mpicc = explicit_mpicc
elif "intel" in mpi_vendor:
    # oneAPI/current C, classic C, then the C++ wrappers in explicit C mode.
    mpicc = next(
        (
            wrapper
            for wrapper in ("mpiicx", "mpiicc")
            if shutil.which(wrapper)
        ),
        None,
    )
    if mpicc is None:
        cxx_wrapper = next(
            (
                wrapper
                for wrapper in ("mpiicpx", "mpiicpc")
                if shutil.which(wrapper)
            ),
            None,
        )
        mpicc = f"{cxx_wrapper} -x c" if cxx_wrapper else "mpicc"
else:
    # For Open MPI, MPICH, and derivatives, mpicc is the ABI-matched C
    # wrapper even when Intel compilers are selected underneath it.
    mpicc = shutil.which("mpicc") or "mpicc"
print(f"qepy-pw build: MPI vendor={MPI.get_vendor()[0]}, compiler={mpicc}")
os.environ.setdefault("CC", mpicc)
os.environ.setdefault("LDSHARED", f"{mpicc} -shared")

link_args = ["-fopenmp", "-ldl", *numeric_runtime_link_args]
if not system_fftw:
    link_args.append("-Wl,-rpath,$ORIGIN/../pyfftw.libs")

extension = Extension(
    "qepy_pw._native_fft",
    ["qepy_pw/_native_fft.pyx"],
    include_dirs=include_dirs,
    libraries=libraries,
    library_dirs=library_dirs,
    runtime_library_dirs=runtime_library_dirs,
    extra_objects=extra_objects,
    extra_compile_args=["-O3", "-fopenmp"],
    extra_link_args=link_args,
    define_macros=macros,
)

setup(
    ext_modules=cythonize(
        [extension],
        compiler_directives={"language_level": "3"},
    ),
)
