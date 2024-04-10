from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(name='Alogs app',
      ext_modules=cythonize('algos.pyx'),
      include_dirs=[numpy.get_include()])
