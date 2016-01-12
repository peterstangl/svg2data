# bootstrap: download setuptools 3.3 if needed
from ez_setup import use_setuptools
use_setuptools()

from setuptools import find_packages
from numpy.distutils.core import setup, Extension

setup(name='svg2data',
      packages=find_packages(),
     )
