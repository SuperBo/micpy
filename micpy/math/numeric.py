import numpy

from ..device import RUNTIME
from ..core import empty
from .. import util
from .reduce import _reduce


def sum(array, axis=None):
    return _reduce(array, axis, 'sum')


def argmax(array, axis=None):
    return _reduce(array, axis, 'argmax', numpy.int64)


def argmin(array, axis=None):
    return _reduce(array, axis, 'argmin', numpy.int64)


def vdot(array_a, array_b):
    if array_a.size != array_b.size:
        raise ValueError('Shape mismatch')

    n = array_a.size
    #TODO: find common dtype
    dtype = util.dtype_to_c(array_a.dtype)

    #TODO: check stream of two array

    out = empty((), array_a.dtype)

    util.invoke_kernel('vector_dot', dtype, n,
                    array_a, 1,
                    array_b, 1,
                    out, 1)
    return out


def dot(array_a, array_b, out=None):
    a_ndim = len(array_a.shape)
    b_ndim = len(array_b.shape)

    dtype = util.dtype_to_c(array_a.dtype)

    if a_ndim == 1 and b_ndim == 1:
        return vdot(array_a, array_b)
    if a_ndim == 2 and b_ndim == 2 and array_a.shape[1] == array_b.shape[0]:
        m = array_a.shape[0]
        k = array_a.shape[1]
        n = array_b.shape[1]
        array_out = empty((m, n), array_a.dtype)

        util.invoke_kernel('matrix_mul', dtype, m, n, k,
                        array_a, array_b, array_out)
        return array_out

    #TODO: implement other cases
    first_dim = a_ndim-1
    match_dim = b_ndim-2 if b_ndim > 1 else 0
    if array_a.shape[first_dim] != array_b.shape[match_dim]:
        raise ValueError('dot alignment error: dim({}) does not match dim({})'.format(first_dim, match_dim))

    ndim = a_ndim + b_ndim - 2

    return None


def matmul_transB(array_a, array_b):
    """Compute C = A * B.T
    """
    if array_a.ndim != 2 or array_b.ndim != 2:
        raise ValueError('not supported type')
        return
    if array_a.shape[1] != array_b.shape[1]:
        raise ValueError('dim mismatched')

    dtype = util.dtype_to_c(array_a.dtype)
    m = array_a.shape[0]
    k = array_a.shape[1]
    n = array_b.shape[0]
    array_out = empty((m, n), array_a.dtype)

    util.invoke_kernel('matrix_mul_transB', dtype, m, n, k,
                        array_a, array_b, array_out)
    return array_out


def matmul_transA(array_a, array_b):
    """Compute C = A.T * B
    """
    if array_a.ndim != 2 or array_b.ndim != 2:
        raise ValueError('not supported type')
        return
    if array_a.shape[0] != array_b.shape[0]:
        raise ValueError('dim mismatched')

    dtype = util.dtype_to_c(array_a.dtype)
    m = array_a.shape[1]
    k = array_a.shape[0]
    n = array_b.shape[1]
    array_out = empty((m, n), array_a.dtype)

    util.invoke_kernel('matrix_mul_transA', dtype, m, n, k,
                        array_a, array_b, array_out)
    return array_out

