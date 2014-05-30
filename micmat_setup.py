import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension(
    "micmat_wrap",
    sources=["micmat_wrap.pyx"],
        # extra_objects=["micmat.o"], #, "micmatMIC.o"
        # extra_link_args = ["-openmp"],
        # library_dirs = ["/global/homes/r/rippel/sota/micmat"],
        library_dirs = [os.getcwd()],
        libraries=["micmat"],
        # include_dirs = [numpy.get_include(), "/global/homes/r/rippel/sota/micmat"]
        include_dirs = [numpy.get_include(), os.getcwd()]
    )]


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
    )