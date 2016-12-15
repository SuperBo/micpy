import numpy
from .runtime import RUNTIME
from .core import dtype_to_c

def vdot(array_a, array_b):
    if array_a.size != array_b.size:
        raise ValueError('Shape mismatch')

    n = array_a.size
    #TODO: find common dtype
    ctype = dtype_to_c(array_a.dtype)

    #TODO: check stream of two array
    stream = array_a.stream
    lib = RUNTIME.get_lib()

    out = stream.empty((), array_a.dtype, update_host=False)

    stream.invoke(lib.vector_dot, ctype, n,
                    array_a, 1,
                    array_b, 1,
                    out, 1)

    return out

def dot(array_a, array_b, out=None):
    stream = RUNTIME.get_stream()
    lib = RUNTIME.get_lib()
    a_ndim = len(array_a.shape)
    b_ndim = len(array_b.shape)

    ctype = dtype_to_c(array_a.dtype)

    if a_ndim == 1 and b_ndim == 1:
        return vdot(array_a, array_b)
    if a_ndim == 2 and b_ndim == 2 and array_a.shape[1] == array_b.shape[0]:
        #TODO: check stream of a and b
        stream = array_a.stream
        lib = RUNTIME.get_lib()

        m = array_a.shape[0]
        k = array_a.shape[1]
        n = array_b.shape[1]
        array_out = stream.empty((m, n), array_a.dtype, update_host=False)

        stream.invoke(lib.matrix_mul, ctype, m, n, k,
                        array_a, array_b, array_out)

        return array_out

    #TODO: implement other case
    first_dim = a_ndim-1
    match_dim = b_ndim-2 if b_ndim > 1 else 0
    if array_a.shape[first_dim] != array_b.shape[match_dim]:
        raise ValueError('dot alignment error: dim({}) does not match dim({})'.format(first_dim, match_dim))

    ndim = a_ndim + b_ndim - 2

    return None

