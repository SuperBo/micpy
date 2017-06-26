from __future__ import division, absolute_import, print_function

try:
    __MICPY_SETUP__
except NameError:
    __MICPY_SETUP__ = False

if __MICPY_SETUP__:
    import sys
    sys.stderr.write('Running from micpy source directory.\n')
else:
    from .multiarray import *
    from .umath import *
