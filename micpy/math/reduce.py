import numpy

from .. import util
from ..core import empty, sync


def _reduce(array, axis, func, dtype=None):
    _dtype = array.dtype if dtype is None else dtype
    ctype = util.dtype_to_c(array.dtype)

    if axis is None:
        out = empty((1), _dtype)
        util.invoke_kernel(func, ctype, array, 1, array.size, array.size, 1, out)
        return out

    __axis = util.get_axis(array, axis)
    if __axis != 0 and __axis != array.ndim -1:
        raise ValueError("Only support first and last axis")

    out_shape = tuple([array.shape[i] for i in range(array.ndim) if i != axis])
    out = empty(out_shape, _dtype)

    stride = 1
    for s in array.shape[1:]:
        stride *=s

    if axis == 0:
        niter = stride
        inciter = 1
        n = array.shape[0]
        inca = stride
    else:
        #Last axis
        niter = array.shape[0]
        inciter = stride
        n = stride
        inca = 1
    util.invoke_kernel(func, ctype, array, niter, inciter, n, inca, out)
    return out

