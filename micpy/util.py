import numpy
from .device import RUNTIME

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


def get_axis(array, axis):
    __axis = array.ndim + axis if axis < 0 else axis
    if __axis < 0 or __axis >= array.ndim:
        raise ValueError('\'axis\' entry is out of bounds')
    return __axis


def invoke_kernel(name, *args):
    lib = RUNTIME.get_lib()
    kernel = getattr(lib, name)

    #TODO(superbo): check stream of args and current stream
    stream = RUNTIME.get_stream()
    stream.invoke(kernel, *args)
    stream.sync()

