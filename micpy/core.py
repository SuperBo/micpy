import sys
import numbers
import numpy

from .ndarray import ndarray
from .device import RUNTIME

__micpy__ = sys.modules[__name__]

__all__ = ['sync', 'asarray', 'native_array', 'empty', 'empty_like', 'zeros', 'zeros_like',
        'ones', 'ones_like', 'full', 'full_like']


def _wrap(array):
    array.__class__ = ndarray
    return array


def sync():
    RUNTIME.get_stream().sync()


# Numpy like functions
def asarray(a, dtype=None, update_device=True):
    if isinstance(a, numbers.Number):
        b = numpy.asarray(a, dtype)
        stream = RUNTIME.get_stream()
        off_a = ndarray(b.shape, b.dtype, alloc_arr=False,
                device=stream._device, stream=stream)
        off_a.array = b
        off_a._device_ptr = stream.allocate_device_memory(b.itemsize, 0)
        if update_device:
            off_a.update_device()
        return off_a
    if isinstance(a, numpy.ndarray):
        stream = RUNTIME.get_stream()
        if dtype is None:
            return _wrap(stream.bind(a, update_device=update_device))
        else:
            return _wrap(stream.bind(a.astype(dtype), update_device=update_device))
    if isinstance(a, ndarray):
        #TODO(sueprbo): cast type
        return a
    return None


def array(*args):
    #TODO(superbo): implement this
    pass


def native_array(shape, dtype=numpy.int64):
    stream = RUNTIME.get_stream()
    arr = ndarray(shape, dtype, alloc_arr=False,
            device=stream._device, stream=stream)
    #Do not need to align for single item array
    if arr.size == 1:
        arr._device_ptr = stream.allocate_device_memory(arr._nbytes, 0)
    else:
        arr._device_ptr = stream.allocate_device_memory(arr._nbytes)
    return arr


def empty(shape, dtype=numpy.float64):
    stream = RUNTIME.get_stream()
    return _wrap(stream.empty(shape, dtype, update_host=False))


def empty_like(array, dtype=None):
    stream = RUNTIME.get_stream()
    return _wrap(stream.empty_like(array, dtype, update_host=False))


def zeros(shape, dtype=numpy.float64):
    stream = RUNTIME.get_stream()
    array = stream.zeros(shape, dtype, update_host=False)
    stream.sync()
    return _wrap(array)


def zeros_like(array, dtype=None):
    stream = RUNTIME.get_stream()
    array = stream.zeros_like(array, dtype, update_host=False)
    stream.sync()
    return _wrap(array)


def ones(shape, dtype=numpy.float64):
    stream = RUNTIME.get_stream()
    array = stream.ones(shape, dtype, update_host=False)
    stream.sync()
    return _wrap(array)


def ones_like(array, dtype=None):
    stream = RUNTIME.get_stream()
    array = stream.ones_like(array, dtype, update_host=False)
    stream.sync()
    return _wrap(array)


def full(shape, fill_value, dtype=numpy.float64):
    stream = RUNTIME.get_stream()
    array = stream.bcast(fill_value, shape, dtype, update_host=False)
    stream.sync()
    return _wrap(array)


def full_like(array, fill_value):
    stream = RUNTIME.get_stream()
    array = stream.bcast_like(fill_value, array, update_host=False)
    stream.sync()
    return _wrap(array)

