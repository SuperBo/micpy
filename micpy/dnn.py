import numpy

from . import ndarray
from . import util
from .core import empty_like, asarray, native_array
from .math.numeric import argmax

__variant = {numpy.float32: '_float32', numpy.float64: '_float64'}


def __get_variant(dtype):
    if dtype.type in __variant:
        return __variant[dtype.type]
    raise ValueError('Not supported type')


def __call_generic(name, *args):
    a = args[0]
    size = a.size
    for arg in args[1:]:
        if not hasattr(arg, 'size'):
            continue
        assert(arg.size == size)

    kernel = name + __get_variant(a.dtype)
    out = empty_like(a)
    new_args = args + (out, size)
    util.invoke_kernel(kernel, *new_args)
    return out


def __call_generic_type(name, dtype=None, *args):
    a = args[0]
    size = a.size
    for arg in args[1:]:
        if not hasattr(arg, 'size'):
            continue
        assert(arg.size == size)

    kernel = name + __get_variant(a.dtype)
    out = empty_like(a, dtype)
    new_args = args + (out, size)
    util.invoke_kernel(kernel, *new_args)
    return out


def __call_generic2d(name, *args):
    a = args[0]
    m, n = a.shape
    for arg in args[1:]:
        if not hasattr(arg, 'size'):
            continue
        assert(arg.size == size)

    kernel = name + __get_variant(a.dtype)
    out = empty_like(a)
    new_args = args + (out, m, n)
    util.invoke_kernel(kernel, *new_args)
    return out


def grad_decrease(params, grads, learning_rate):
    if not isinstance(params, ndarray) or not isinstance(grads, ndarray):
        raise ValueError('Not supported type')

    dtype = util.dtype_to_c(params.dtype)
    util.invoke_kernel('grad_decrease', dtype, params.size,
            params, 1, grads, 1, learning_rate)


def normalize_coeff(t, ignored_label=-1):
    #TODO: check type of parameters
    if (t.dtype != numpy.int32):
        raise ValueError('Only support array of type numpy.int32')

    coeff = asarray((1.0), numpy.float64)
    util.invoke_kernel('normalize_coefficient', t.size, t, ignored_label, coeff)
    return coeff


def predict(y):
    return argmax(y, axis=1)


def bin_predict(y, threshold=0.0):
    return __call_generic_type('bin_predict', numpy.int64, y, threshold)


def accuracy(y, t, ignored_label=None):
    if (y.shape[0] != t.size):
        raise ValueError("Dim size does not match {} != {}".format(
                        y.shape[0], t.size))
    accu = asarray((1.0), numpy.float64)
    pred = predict(y)
    util.invoke_kernel('accuracy', t.size, pred, t, ignored_label, accu)
    return accu


def bin_accuracy(y, t, ignored_label=None):
    if y.size != t.size:
        raise ValueError("Array size does not match {} != {}".format(
                            y.size, t.size))
    pred = bin_predict(y)
    accu = asarray((1.0), numpy.float64)
    util.invoke_kernel('accuracy', t.size, pred, t, ignored_label, accu)
    return accu


#Activation function

def relu(array):
    return __call_generic('relu_forward', array)


def relu_grad(array, grad):
    return __call_generic('relu_backward', array, grad)


def sigmoid(array):
    return __call_generic('sigmoid_forward', array)


def sigmoid_grad(array, grad):
    return __call_generic('sigmoid_backward', array, grad)


def tanh(array):
    return __call_generic('tanh_forward', array)


def tanh_grad(array, grad):
    return __call_generic('tanh_backward', array, gray)


def softmax(array):
    return __call_generic2d('softmax_forward', array, )


def softmax_grad(array, grad):
    return __call_generic2d('softmax_backward', array, grad)


def softmax_cross_entropy(x, t, ignored_label=-1, coeff=1.0, cache=True):
    m, n = x.shape
    assert(x.ndim == 2)
    assert(t.size == m)

    y = empty_like(x) if cache else None

    if x.dtype == numpy.float32:
        loss = asarray(0, dtype=numpy.float32)
        kernel = 'softmax_cross_entropy_forward_float32'
    elif x.dtype == numpy.float64:
        loss = asarray(0, dtype=numpy.float64)
        kernel = 'softmax_cross_entropy_forward_float64'
    else:
        raise ValueError("Not supported type {}".format(x.dtype))

    util.invoke_kernel(kernel, m, n, coeff, ignored_label, x, t, y, loss)
    return (loss, y)


def softmax_cross_entropy_grad(x, t, y, grad, ignored_label=-1, coeff=None):
    m, n = x.shape
    assert(x.ndim == 2)
    assert(t.size == m)

    if x.dtype == numpy.float32:
        kernel = 'softmax_cross_entropy_backward_float32'
    elif x.dtype == numpy.float64:
        kernel = 'softmax_cross_entropy_backward_float64'
    else:
        raise ValueError("Not supported type {}".format(x.dtype))

    gx = empty_like(x)
    util.invoke_kernel(kernel, m, n, coeff, ignored_label, x, t, y, grad, gx)
    return gx

