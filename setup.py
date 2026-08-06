"""Setuptools entry point for the mandatory Cython/MPI/FFTW extension."""

from __future__ import annotations

import glob
import os
from pathlib import Path

from Cython.Build import cythonize
import mpi4py
import numpy as np
import pyfftw
from setuptools import Extension, setup


site_packages = Path(pyfftw.__file__).resolve().parent.parent
fftw_candidates = sorted(
    glob.glob(str(site_packages / "pyfftw.libs" / "libfftw3-*.so*"))
)
fftw_openmp_candidates = sorted(
    glob.glob(str(site_packages / "pyfftw.libs" / "libfftw3_omp-*.so*"))
)
if not fftw_candidates or not fftw_openmp_candidates:
    raise RuntimeError(
        "cannot locate pyFFTW's double-precision FFTW/OpenMP libraries"
    )
fftw_library = fftw_candidates[-1]
fftw_openmp_library = fftw_openmp_candidates[-1]
fftw_directory = str(Path(fftw_library).parent)
os.environ.setdefault("CC", "mpicc")
os.environ.setdefault("LDSHARED", "mpicc -shared")

extension = Extension(
    "qepy_pw._native_fft",
    ["qepy_pw/_native_fft.pyx"],
    include_dirs=[np.get_include(), mpi4py.get_include()],
    extra_objects=[fftw_library, fftw_openmp_library],
    extra_compile_args=["-O3", "-fopenmp"],
    extra_link_args=[
        "-fopenmp",
        "-ldl",
        f"-Wl,-rpath,{fftw_directory}",
        "-Wl,-rpath,$ORIGIN/../pyfftw.libs",
    ],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    ext_modules=cythonize(
        [extension],
        compiler_directives={"language_level": "3"},
    ),
)
