# Based on https://raw.githubusercontent.com/ADicksonLab/geomm/master/build.py
import os
import sys

from Cython.Build import cythonize

import numpy as np

# use cythonize to build the extensions
modules = ["corrscope/wave.pyx", ]

extensions = cythonize(modules, compiler_directives=dict(language_level='3'))


def build(setup_kwargs):
    """Needed for the poetry building interface."""

    setup_kwargs.update({
        'ext_modules': extensions,
        'include_dirs': [np.get_include()],
    })
