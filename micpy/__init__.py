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
    from .numeric import (asarray, rollaxis, moveaxis, argmax, argmin)
    from .shape_base import (expand_dims)
    from numpy import (int, int_, int8, int16, int32, int64,
                       uint, uint8, uint16, uint32, uint64,
                       float, float_, float16, float32, float64,
                       complex, complex_, complex64, complex128,
                       byte, short, long, longlong, intp, double, longdouble)
