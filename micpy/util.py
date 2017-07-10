from __future__ import absolute_import

import numpy, micpy


def get_array_module(*args):
    for arg in args:
        if isinstance(arg, micpy.ndarray):
            return micpy

    return numpy


def get_axis(array, axis):
    __axis = array.ndim + axis if axis < 0 else axis
    if __axis < 0 or __axis >= array.ndim:
        raise ValueError('\'axis\' entry is out of bounds')
    return __axis
