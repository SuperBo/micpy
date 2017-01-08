import numbers
import numpy

from .. import util
from ..core import empty

def __call_distribution(name, size, *args):
    dtype = util.dtype_to_c(numpy.float64)
    kernel = name + '_distribution'

    if size is None:
        _size = 1
    elif isinstance(size, tuple):
        _size = size
    elif isinstance(size, numbers.Integral):
        _size = 1 if size <= 0 else size
    else:
        raise ValueError("Invalid size argument: {} type {}".format(
                            size, type(size)))

    out = empty(_size, dtype=numpy.float64)

    new_args = args + (out, out.size)
    util.invoke_kernel(kernel, dtype, *new_args)
    return out


def rand(*args):
    if len(args) == 0:
        return uniform(0.0, 1,0, 1)
    return uniform(0.0, 1.0, args)


def randn(*args):
    if len(args) == 0:
        return normal(0.0, 1.0, 1)
    return normal(0.0, 1,0, args)


def random(size=None):
    return uniform(0.0, 1.0, size)


def uniform(low=0.0, high=1.0, size=None):
    return __call_distribution('uniform', size, low, high)


def normal(loc=0.0, scale=1.0, size=None):
    return __call_distribution('gaussian', size, loc, scale)


def standard_normal(size):
    return normal(0.0, 1.0, size)

