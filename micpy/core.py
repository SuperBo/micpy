import sys
import numpy

from . import ndarray
from .runtime import RUNTIME

__micpy__ = sys.modules[__name__]

__all__ = ['asarray', 'empty', 'empty_like', 'zeros', 'zeros_like', 'ones', 'ones_like',
            'full', 'full_like', 'get_array_module']

def get_array_module(*args):
    for arg in args:
        if isinstance(arg, micpy.ndarray):
            return __micpy__

    return numpy

def dtype_to_c(dtype):
    if (dtype == numpy.int32):
        return 0
    if (dtype == numpy.int64):
        return 1
    if (dtype == numpy.float32):
        return 2
    if (dtype == numpy.float64):
        return 3
    if (dtype == numpy.uint64):
        return 5
    if (dtype == numpy.complex):
        return 4

# Numpy like functions
def asarray(a):
    #TODO: add other type
    if isinstance(a, numpy.ndarray):
        stream = RUNTIME.get_stream()
        return stream.bind(a)
    if isinstance(a, micpy.ndarray):
        return a
    return None


def empty(shape, dtype=numpy.float64):
    stream = RUNTIME.get_stream()
    return stream.empty(shape, dtype, update_host=False)

def empty_like(array):
    stream = RUNTIME.get_stream()
    return stream.empty_like(array, update_host=False)

def zeros(shape, dtype=numpy.float64):
    stream = RUNTIME.get_stream()
    return stream.zeros(shape, dtype, update_host=False)

def zeros_like(array):
    stream = RUNTIME.get_stream()
    return stream.zeros_like(array)

def ones(shape, dtype=numpy.float64):
    stream = RUNTIME.get_stream()
    return stream.ones(shape, dtype, update_host=False)

def ones_like(array):
    stream = RUNTIME.get_stream()
    return stream.ones_like(array, update_host=False)

def full(shape, fill_value, dtype=numpy.float64):
    stream = RUNTIME.get_stream()
    return stream.bcast(fill_value, shape, dtype, update_host=False)

def full_like(array, fill_value):
    stream = RUNTIME.get_stream()
    return stream.bcast_like(fill_value, array, update_host=False)
