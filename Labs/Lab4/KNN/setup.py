from distutils.core import setup
from distutils.extension import Extension
import os

import numpy
from Cython.Build import cythonize

extensions = [
    Extension("KNN", ["KNN.pyx"]),
]
setup(
    name="KNN",
    ext_modules=cythonize(["KNN.pyx"], annotate=True, language_level="3"),
    include_dirs=[numpy.get_include()]
)